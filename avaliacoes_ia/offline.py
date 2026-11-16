"""Classificador por regras, para rodar sem chave de API.

Responde no mesmo formato do LLM (uma chamada à ferramenta registrar_analise),
então o resto do sistema roda igual. Serve como demonstração e como linha de base:
a avaliação com gabarito mostra o quanto o LLM ganha dele.

Regras:
- Procura palavras de cada categoria no texto sem acento.
- Uma parte da frase com "não", "nem", "nunca" ou "nada" é tratada como negada:
  "não demorou" não conta como atraso, e "não estava boa" não conta como elogio.
- Expressões fortes ("péssimo", "nunca mais", "não recomendo") tornam o sentimento negativo.

- Palavras de casos graves (saúde, higiene, cobrança, ofensa) levam a urgência 5, como no prompt do LLM.

Limitações conhecidas: não entende ironia nem frases sem as palavras da lista.
"""

import hashlib
import json
import re

from .analisador import FERRAMENTA, texto_da_mensagem
from .normalizacao import sem_acento

# A ordem importa: categorias de problema vêm antes do elogio, e cada termo achado
# é apagado da frase, para "mal educado" não contar também como elogio ("educado").
TERMOS = {
    "atraso": [
        "atraso",
        "atrasou",
        "atrasad",
        "atrasa",
        "demor",
        "esperei",
        "espera",
        "1h",
        "duas horas",
        "uma hora",
    ],  # não "atras": pegaria "lá atrás"
    "comida_fria": ["fria", "frio", "gelad", "morna", "morno"],
    "pedido_errado": ["errad", "faltou", "faltando", "veio sem", "trocad", "outro pedido", "de outra pessoa"],
    "qualidade": [
        "cru",
        "sem gosto",
        "sem sal",
        "queimad",
        "salgad",
        "murch",
        "cabelo",
        "passei mal",
        "estragad",
        "azed",
        "ruim",
    ],
    "embalagem": ["vazou", "vazad", "derram", "amassad", "rasgad"],
    "preco": ["caro", "taxa alta", "abusiv", "cobra"],
    "atendimento": ["grosso", "grosseir", "mal educad", "rude", "ninguem respond", "nao respond", "ignorou"],
    "elogio": [
        "delicios",
        "otim",
        "perfeit",
        "maravilh",
        "excelente",
        "recomendo",
        "rapid",
        "educad",
        "atencios",
        "parabens",
        "adorei",
        "amei",
        "fresquinh",
        "quentinh",
    ],
    # não "melhor": "podia ser melhor" é reclamação
}
PROBLEMAS = set(TERMOS) - {"elogio"}
NEGATIVO_FORTE = ["pessim", "horrivel", "nunca mais", "nao recomendo", "decepcion", "lamentavel", "absurd"]
# Mesma regra do prompt do LLM: saúde, higiene, cobrança indevida e ofensa são urgência 5.
GRAVE = [
    "cabelo",
    "barata",
    "inseto",
    "passei mal",
    "intoxic",
    "vomit",
    "estragad",
    "duas vezes",
    "cobranca indevida",
    "grosso",
    "grosseir",
    "xing",
    "rude",
]
# Pedido inteiro errado ou perdido: urgência 4, como no prompt.
PEDIDO_PERDIDO = ["tudo errado", "de outra pessoa", "nao chegou", "nunca chegou", "nao entregaram", "nao foi entregue"]
NEGACOES = {"nao", "nem", "nunca", "jamais", "nada"}  # "nada rápida"
SEPARADORES = re.compile(r"[.,;:!?\n]+|\s+(?:e|mas|porem)\s+")


def _achar(termo: str, texto: str) -> bool:
    return re.search(r"\b" + re.escape(termo), texto) is not None


def _apagar(termo: str, texto: str) -> str:
    return re.sub(r"\b" + re.escape(termo) + r"\w*", " ", texto)


def _categorias(normalizado: str) -> list[str]:
    """Categorias citadas, ignorando os trechos negados ("não demorou")."""
    categorias: list[str] = []
    for trecho in SEPARADORES.split(normalizado):
        negada = bool(NEGACOES & set(re.findall(r"[a-z]+", trecho)))
        restante = trecho  # cada termo achado é apagado daqui
        for categoria, termos in TERMOS.items():
            for termo in termos:
                if not _achar(termo, restante):
                    continue
                restante = _apagar(termo, restante)
                termo_ja_e_negativo = bool(NEGACOES & set(termo.split()))  # ex.: "nao respond"
                if (not negada or termo_ja_e_negativo) and categoria not in categorias:
                    categorias.append(categoria)
    return categorias


def _sentimento_e_urgencia(categorias: list[str], forte: bool) -> tuple[str, int]:
    """Mesma escala do prompt: vários problemas comuns são 3; insatisfação forte é 4."""
    problemas = [c for c in categorias if c in PROBLEMAS]
    elogio = "elogio" in categorias
    if problemas and elogio and not forte:
        return "neutro", 2
    if problemas or forte:
        return "negativo", 4 if forte else 3
    if elogio:
        return "positivo", 1
    return "neutro", 1


def classificar(texto: str) -> dict:
    normalizado = sem_acento(texto.lower())
    grave = any(_achar(termo, normalizado) for termo in GRAVE)
    perdido = any(_achar(termo, normalizado) for termo in PEDIDO_PERDIDO)
    forte = any(_achar(termo, normalizado) for termo in NEGATIVO_FORTE)
    for termo in NEGATIVO_FORTE:  # apaga para "não recomendo" não contar como elogio ("recomendo")
        normalizado = _apagar(termo, normalizado)

    categorias = _categorias(normalizado)
    sentimento, urgencia = _sentimento_e_urgencia(categorias, forte)
    if perdido:
        sentimento, urgencia = "negativo", max(urgencia, 4)
    if grave:
        sentimento, urgencia = "negativo", 5

    return {
        "sentimento": sentimento,
        "categorias": categorias,
        "urgencia": urgencia,
        "resumo": " ".join(texto.split())[:120] or "(sem texto)",
    }


def versao_das_regras() -> str:
    regras = [TERMOS, NEGATIVO_FORTE, GRAVE, PEDIDO_PERDIDO, sorted(NEGACOES)]
    return hashlib.sha256(json.dumps(regras, sort_keys=True).encode("utf-8")).hexdigest()[:10]


class ClienteOffline:
    @property
    def origem(self) -> str:  # calculada na hora: se as regras mudarem, a versão muda junto
        return f"offline (regras {versao_das_regras()})"

    def enviar(self, sistema: str, mensagens: list[dict], ferramentas: list[dict], escolha: dict) -> dict:
        texto = texto_da_mensagem(mensagens[0]["content"])
        return {
            "stop_reason": "tool_use",
            "content": [
                {"type": "tool_use", "id": "toolu_offline", "name": FERRAMENTA["name"], "input": classificar(texto)}
            ],
        }
