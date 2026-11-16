"""Leitura do CSV de entrada, preparada para arquivos vindos do Excel.

- Codificação: UTF-8 (com ou sem BOM), UTF-16 ("Texto Unicode" do Excel),
  Windows-1252 e, em último caso, Latin-1, que aceita qualquer byte.
- Separador: ",", ";" (Excel em português) ou tabulação, detectado pelo cabeçalho.
- Nomes de coluna podem vir com maiúscula ou espaço ("ID", " Texto").
- Textos muito longos são cortados em LIMITE_TEXTO caracteres, para não gastar a API à toa.
Qualquer problema vira uma mensagem clara (ErroEntrada), nunca um erro do Python.
"""

import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path

COLUNAS = ("id", "texto")
LIMITE_TEXTO = 4000
LIMITE_CAMPO_CSV = 10_000_000  # o padrão do Python (128 KB) derruba a leitura de um texto gigante
CONTROLE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b\ufeff]")


class ErroEntrada(ValueError):
    pass


@dataclass(frozen=True)
class Linha:
    id: str
    texto: str


def _decodificar(bruto: bytes) -> str:
    tentativas = ["utf-16"] if bruto.startswith((b"\xff\xfe", b"\xfe\xff")) else []
    for codificacao in [*tentativas, "utf-8-sig", "cp1252"]:
        try:
            return bruto.decode(codificacao)
        except UnicodeDecodeError:
            continue
    return bruto.decode("latin-1")  # nunca falha


def _limpar(texto: str | None) -> str:
    return CONTROLE.sub("", texto or "").strip()


def _abrir(caminho: Path) -> str:
    """Lê e decodifica o arquivo, sem as linhas em branco antes do cabeçalho."""
    if not caminho.exists():
        raise ErroEntrada(f"Arquivo não encontrado: {caminho}")
    if not caminho.is_file():
        raise ErroEntrada(f"{caminho} não é um arquivo")
    try:
        bruto = caminho.read_bytes()
    except OSError as erro:
        raise ErroEntrada(f"Não consegui ler {caminho}: {erro.strerror or erro}") from erro
    conteudo = re.sub(r"\A(?:[ \t]*\r?\n)+", "", _decodificar(bruto))
    if not conteudo.strip():
        raise ErroEntrada(f"O arquivo está vazio: {caminho}")
    return conteudo


def _mapear_colunas(nomes: list[str]) -> dict[str, str]:
    """{"id": nome real da coluna, "texto": nome real da coluna}."""
    colunas = {_limpar(c).lower(): c for c in nomes}
    if any(c not in colunas for c in COLUNAS):
        encontradas = ", ".join(_limpar(c) for c in nomes) or "nenhuma"
        raise ErroEntrada(f"O CSV precisa das colunas id e texto. Colunas encontradas: {encontradas}")
    return {c: colunas[c] for c in COLUNAS}


def _texto(registro: dict, coluna: str, separador: str, texto_e_ultima: bool, aviso) -> str:
    texto = registro.get(coluna) or ""
    sobras = registro.get(None)  # campos além do cabeçalho: separador dentro do texto, sem aspas
    if sobras and texto_e_ultima:
        aviso(f"o texto tinha '{separador}' sem aspas; as partes foram juntadas")
        return _limpar(separador.join([texto, *sobras]))
    if sobras:
        aviso("mais colunas que o cabeçalho; confira se o texto está entre aspas")
    return _limpar(texto)


def ler_csv(caminho: Path) -> tuple[list[Linha], list[str]]:
    """Devolve as linhas válidas e uma lista de avisos (linhas ignoradas, juntadas ou cortadas)."""
    conteudo = _abrir(Path(caminho))
    separador = max([",", ";", "\t"], key=conteudo.splitlines()[0].count)
    csv.field_size_limit(max(csv.field_size_limit(), LIMITE_CAMPO_CSV))
    leitor = csv.DictReader(io.StringIO(conteudo, newline=""), delimiter=separador)

    linhas: list[Linha] = []
    avisos: list[str] = []
    repetidos: set[str] = set()
    vistos: set[str] = set()
    try:
        nomes = list(leitor.fieldnames or [])
        colunas = _mapear_colunas(nomes)
        texto_e_ultima = nomes[-1] == colunas["texto"]
        for registro in leitor:
            id_ = _limpar(registro.get(colunas["id"]))
            prefixo = f"linha {leitor.line_num}" + (f" ({id_})" if id_ else "")

            def aviso(mensagem: str, prefixo: str = prefixo) -> None:
                avisos.append(f"{prefixo}: {mensagem}")

            texto = _texto(registro, colunas["texto"], separador, texto_e_ultima, aviso)
            if not id_ or not texto:
                aviso("sem id ou sem texto, ignorada")
            elif id_ in vistos:
                repetidos.add(id_)
            else:
                if len(texto) > LIMITE_TEXTO:
                    aviso(f"texto com {len(texto)} caracteres, cortado em {LIMITE_TEXTO}")
                vistos.add(id_)
                linhas.append(Linha(id_, texto[:LIMITE_TEXTO]))
    except csv.Error as erro:
        raise ErroEntrada(f"CSV malformado perto da linha {leitor.line_num}: {erro}") from erro

    if repetidos:
        raise ErroEntrada(f"IDs repetidos no CSV: {', '.join(sorted(repetidos))}")
    if not linhas:
        raise ErroEntrada("Nenhuma linha com id e texto preenchidos.")
    return linhas, avisos
