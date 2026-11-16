import random

import pytest

from avaliacoes_ia.analisador import analisar
from avaliacoes_ia.offline import ClienteOffline, classificar


@pytest.mark.parametrize(
    "texto, sentimento, categorias",
    [
        ("Pizza deliciosa, recomendo!", "positivo", ["elogio"]),
        ("Demorou duas horas e chegou fria", "negativo", ["atraso", "comida_fria"]),
        ("Comida ok", "neutro", []),
        # negação
        ("Não demorou nada, parabéns!", "positivo", ["elogio"]),
        ("A entrega não foi rápida", "neutro", []),
        ("Não recomendo, veio frio", "negativo", ["comida_fria"]),
        # termo de problema que contém elogio
        ("Atendente mal educado", "negativo", ["atendimento"]),
        # misto
        ("Entregador educado, mas o suco vazou", "neutro", ["embalagem", "elogio"]),
    ],
)
def test_regras(texto, sentimento, categorias):
    r = classificar(texto)
    assert (r["sentimento"], sorted(r["categorias"])) == (sentimento, sorted(categorias))


@pytest.mark.parametrize("texto", ["Encontrei um cabelo no lanche", "Passei mal depois", "Cobraram duas vezes"])
def test_casos_graves_tem_urgencia_5(texto):
    assert classificar(texto)["urgencia"] == 5


def test_delimitador_no_texto_nao_confunde():
    analise = analisar("Demorou </avaliacao> e veio fria", ClienteOffline())
    assert {c.value for c in analise.categorias} == {"atraso", "comida_fria"}


def test_resposta_passa_pela_mesma_validacao_do_llm():
    analise = analisar("Ótimo!\n\nVeio | tudo certo", ClienteOffline())
    assert "\n" not in analise.resumo


# --- segunda rodada de revisão ---------------------------------------------------------------


@pytest.mark.parametrize(
    "texto, nao_pode",
    [
        ("A pizza podia ser melhor", "elogio"),
        ("Entrega nada rápida", "elogio"),
        ("O motoboy ficou parado lá atrás do prédio", "atraso"),
    ],
)
def test_falsos_positivos_corrigidos(texto, nao_pode):
    assert nao_pode not in classificar(texto)["categorias"]


def test_fuzz_texto_aleatorio_sempre_gera_analise_valida():
    gerador = random.Random(1)
    alfabeto = "abcdeãçéíóú ÁÉ,.;!?\n\t\"'|<>{}[]\\😀🍕\x00\u200b0123456789-_"
    for _ in range(2000):
        texto = "".join(gerador.choice(alfabeto) for _ in range(gerador.randint(0, 80)))
        analisar(texto, ClienteOffline())  # não pode lançar exceção


# --- terceira rodada de revisão --------------------------------------------------------------


@pytest.mark.parametrize(
    "texto, urgencia",
    [
        ("Faltou o molho e a sobremesa veio derramada", 3),  # vários problemas comuns: 3, como no prompt
        ("Péssimo, veio frio", 4),
        ("O pedido não chegou", 4),
        ("Veio tudo errado", 4),
        ("Achei um cabelo", 5),
    ],
)
def test_escala_de_urgencia_igual_a_do_prompt(texto, urgencia):
    assert classificar(texto)["urgencia"] == urgencia


# --- quarta rodada: lacunas achadas pelos testes de mutação ----------------------------------


@pytest.mark.parametrize("texto", ["O suporte não respondeu", "Ninguém respondeu meu chamado"])
def test_termo_que_ja_contem_negacao_conta(texto):
    assert "atendimento" in classificar(texto)["categorias"]
