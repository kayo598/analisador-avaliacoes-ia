import random

import pytest

from avaliacoes_ia.entrada import LIMITE_CAMPO_CSV, LIMITE_TEXTO, ErroEntrada, Linha, ler_csv


def escrever(tmp_path, conteudo: str, codificacao: str = "utf-8"):
    arquivo = tmp_path / "entrada.csv"
    arquivo.write_bytes(conteudo.encode(codificacao))
    return arquivo


@pytest.mark.parametrize(
    "conteudo, codificacao",
    [
        ("id,texto\nA1,Demorou\nA2,Chegou frio\n", "utf-8"),
        ("\ufeffid,texto\nA1,Demorou\nA2,Chegou frio\n", "utf-8"),  # Excel: UTF-8 com BOM
        ("id;texto\nA1;Demorou\nA2;Chegou frio\n", "utf-8"),  # Excel em português
        ("id;texto\nA1;Demorou\nA2;Chegou frio\n", "cp1252"),  # Excel antigo no Windows
        ("ID , Texto \nA1,Demorou\nA2,Chegou frio\n", "utf-8"),
        ("id\ttexto\nA1\tDemorou\nA2\tChegou frio\n", "utf-8"),
    ],
)
def test_formatos_comuns(tmp_path, conteudo, codificacao):
    linhas, avisos = ler_csv(escrever(tmp_path, conteudo, codificacao))
    assert linhas == [Linha("A1", "Demorou"), Linha("A2", "Chegou frio")]
    assert avisos == []


def test_acentos_em_windows_1252(tmp_path):
    linhas, _ = ler_csv(escrever(tmp_path, "id;texto\nA1;Péssimo, não gostei\n", "cp1252"))
    assert linhas[0].texto == "Péssimo, não gostei"


def test_texto_com_virgula_aspas_e_quebra_de_linha(tmp_path):
    linhas, _ = ler_csv(escrever(tmp_path, 'id,texto\nA1,"Demorou, e veio ""frio""\nnunca mais"\n'))
    assert linhas[0].texto == 'Demorou, e veio "frio"\nnunca mais'


def test_linhas_incompletas_viram_aviso(tmp_path):
    linhas, avisos = ler_csv(escrever(tmp_path, "id,texto\nA1,Demorou\nA2\n,sem id\nA4,   \n"))
    assert [linha.id for linha in linhas] == ["A1"]
    assert len(avisos) == 3


@pytest.mark.parametrize(
    "conteudo, mensagem",
    [
        ("codigo,texto\nA1,Demorou\n", "precisa das colunas id e texto"),
        ("", "vazio"),
        ("id,texto\nA1,\n", "Nenhuma linha"),
        ("id,texto\nA1,um\nA2,dois\nA1,três\n", "IDs repetidos no CSV: A1"),
    ],
)
def test_erros_com_mensagem_clara(tmp_path, conteudo, mensagem):
    with pytest.raises(ErroEntrada, match=mensagem):
        ler_csv(escrever(tmp_path, conteudo))


def test_arquivo_inexistente(tmp_path):
    with pytest.raises(ErroEntrada, match="não encontrado"):
        ler_csv(tmp_path / "nao_existe.csv")


# --- segunda rodada de revisão ---------------------------------------------------------------


def escrever_bytes(tmp_path, bruto: bytes):
    arquivo = tmp_path / "entrada.csv"
    arquivo.write_bytes(bruto)
    return arquivo


def test_bytes_que_nao_existem_em_windows_1252(tmp_path):
    linhas, _ = ler_csv(escrever_bytes(tmp_path, b"id;texto\nA1;caf\x81 \x9d frio\n"))
    assert linhas[0].id == "A1"


def test_utf16_do_excel(tmp_path):
    bruto = "id\ttexto\nA1\tPéssimo atendimento\n".encode("utf-16")
    linhas, _ = ler_csv(escrever_bytes(tmp_path, bruto))
    assert linhas == [Linha("A1", "Péssimo atendimento")]


def test_texto_gigante_e_cortado_com_aviso(tmp_path):
    linhas, avisos = ler_csv(escrever(tmp_path, 'id,texto\nA1,"' + "a" * 200_000 + '"\n'))
    assert len(linhas[0].texto) == LIMITE_TEXTO
    assert "cortado" in avisos[0]


def test_linhas_em_branco_antes_do_cabecalho(tmp_path):
    linhas, _ = ler_csv(escrever(tmp_path, "\n  \n\nid,texto\nA1,Demorou\n"))
    assert linhas == [Linha("A1", "Demorou")]


def test_caracteres_de_controle_sao_removidos(tmp_path):
    linhas, _ = ler_csv(escrever(tmp_path, "id,texto\nA\u200b1,De\x00mo\x07rou\n"))
    assert linhas == [Linha("A1", "Demorou")]


def test_pasta_no_lugar_do_arquivo(tmp_path):
    with pytest.raises(ErroEntrada, match="não é um arquivo"):
        ler_csv(tmp_path)


def test_fuzz_bytes_aleatorios_so_geram_erro_tratado(tmp_path):
    """500 arquivos com lixo aleatório: o leitor pode recusar, mas nunca quebrar com erro do Python."""
    gerador = random.Random(2)
    for _ in range(500):
        bruto = gerador.choice([b"id,texto\n", b"id;texto\n", b"\xff\xfe", b""]) + bytes(
            gerador.randint(0, 255) for _ in range(gerador.randint(0, 300))
        )
        try:
            linhas, _ = ler_csv(escrever_bytes(tmp_path, bruto))
        except ErroEntrada:
            continue
        assert all(linha.id and linha.texto for linha in linhas)


def test_separador_dentro_do_texto_sem_aspas_nao_perde_texto(tmp_path):
    linhas, avisos = ler_csv(escrever(tmp_path, "id;texto\nA1;Faltou o molho; que pena; nunca mais\n"))
    assert linhas[0].texto == "Faltou o molho; que pena; nunca mais"
    assert "juntadas" in avisos[0]


def test_colunas_a_mais_quando_texto_nao_e_a_ultima(tmp_path):
    _, avisos = ler_csv(escrever(tmp_path, "texto,id\nDemorou, e frio,A1\n"))
    assert "mais colunas que o cabeçalho" in avisos[0]


# --- quarta rodada: lacunas de cobertura -----------------------------------------------------


def test_arquivo_sem_permissao_de_leitura(tmp_path, monkeypatch):
    arquivo = escrever(tmp_path, "id,texto\nA1,x\n")

    def negar(self):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(type(arquivo), "read_bytes", negar)
    with pytest.raises(ErroEntrada, match=r"Não consegui ler.*Permission denied"):
        ler_csv(arquivo)


def test_campo_acima_do_limite_do_csv_vira_mensagem_clara(tmp_path):
    arquivo = escrever(tmp_path, 'id,texto\nA1,"' + "a" * (LIMITE_CAMPO_CSV + 10) + '"\n')
    with pytest.raises(ErroEntrada, match="CSV malformado"):
        ler_csv(arquivo)
