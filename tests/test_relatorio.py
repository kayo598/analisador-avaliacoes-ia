import random

import pytest

from avaliacoes_ia.modelos import Analise, AvaliacaoAnalisada
from avaliacoes_ia.relatorio import gerar_markdown, porcentagens, resumir


def item(id_, sentimento, categorias, urgencia, resumo="r"):
    a = Analise(sentimento=sentimento, categorias=categorias, urgencia=urgencia, resumo=resumo)
    return AvaliacaoAnalisada(id=id_, texto="t", analise=a)


def test_resumo_conta_problemas_sem_elogio_e_ordena_urgentes():
    itens = [
        item("1", "negativo", ["atraso", "comida_fria"], 4),
        item("2", "negativo", ["atraso"], 5),
        item("3", "positivo", ["elogio"], 1),
    ]
    r = resumir(itens)
    assert r["problemas"] == [("atraso", 2), ("comida_fria", 1)]
    assert [i.id for i in r["urgentes"]] == ["2", "1"]
    assert r["sentimentos"] == {"positivo": 1, "neutro": 0, "negativo": 2}


def test_barra_vertical_nao_quebra_a_tabela():
    md = gerar_markdown([item("A|1", "negativo", ["atraso"], 5, resumo="veio | errado")])
    linha = next(x for x in md.splitlines() if x.startswith("| A"))
    assert linha == "| A\\|1 | 5 | veio \\| errado |"


def test_sem_itens():
    md = gerar_markdown([])
    assert "Avaliações analisadas: **0**" in md
    assert "Nenhuma avaliação urgente." in md


# --- terceira rodada de revisão --------------------------------------------------------------


@pytest.mark.parametrize(
    "quantidades, esperado", [([1, 1, 1], [34, 33, 33]), ([0, 0, 0], [0, 0, 0]), ([2, 0, 1], [67, 0, 33])]
)
def test_porcentagens_somam_100(quantidades, esperado):
    assert porcentagens(quantidades) == esperado


def test_porcentagens_somam_100_sempre():
    gerador = random.Random(4)
    for _ in range(2000):
        quantidades = [gerador.randint(0, 50) for _ in range(3)]
        resultado = porcentagens(quantidades)
        assert sum(resultado) == (100 if any(quantidades) else 0)
        assert all(abs(r - 100 * q / max(sum(quantidades), 1)) < 1 for r, q in zip(resultado, quantidades, strict=True))
