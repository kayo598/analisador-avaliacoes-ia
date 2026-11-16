import pytest
from pydantic import ValidationError

from avaliacoes_ia.modelos import LIMITE_RESUMO, Analise, Categoria, Sentimento

from .conftest import VALIDA


@pytest.mark.parametrize(
    "campo, valor, esperado",
    [
        ("sentimento", "Negativo", Sentimento.NEGATIVO),
        ("sentimento", " NEUTRO ", Sentimento.NEUTRO),
        ("sentimento", "Negative", Sentimento.NEGATIVO),  # o Gemini às vezes responde em inglês
        ("sentimento", "neutral", Sentimento.NEUTRO),
        ("sentimento", "POSITIVE", Sentimento.POSITIVO),
        ("urgencia", 4.0, 4),
        ("categorias", ["Comida Fria"], [Categoria.COMIDA_FRIA]),
        ("categorias", ["preço", "pedido-errado"], [Categoria.PRECO, Categoria.PEDIDO_ERRADO]),
        ("categorias", "atraso", [Categoria.ATRASO]),
        ("urgencia", "4", 4),
    ],
)
def test_aceita_variacoes_de_forma(campo, valor, esperado):
    assert getattr(Analise.model_validate({**VALIDA, campo: valor}), campo) == esperado


def test_categorias_repetidas_viram_uma():
    a = Analise.model_validate({**VALIDA, "categorias": ["atraso", "Atraso", "comida_fria", "atraso"]})
    assert a.categorias == [Categoria.ATRASO, Categoria.COMIDA_FRIA]


def test_resumo_longo_e_cortado_e_quebras_de_linha_somem():
    a = Analise.model_validate({**VALIDA, "resumo": "linha 1\nlinha 2  " + "x" * 300})
    assert len(a.resumo) == LIMITE_RESUMO
    assert a.resumo.startswith("linha 1 linha 2 x")
    assert a.resumo.endswith("…")


@pytest.mark.parametrize(
    "campo, valor",
    [
        ("urgencia", 9),
        ("urgencia", 0),
        ("urgencia", 2.5),
        ("sentimento", "furioso"),
        ("categorias", ["alienigenas"]),
        ("resumo", "   "),
    ],
)
def test_rejeita_o_que_decide_algo(campo, valor):
    with pytest.raises(ValidationError):
        Analise.model_validate({**VALIDA, campo: valor})


# --- segunda rodada de revisão ---------------------------------------------------------------


def test_pontuacao_em_volta_do_valor():
    a = Analise.model_validate({**VALIDA, "sentimento": "Negativo.", "categorias": ["(atraso)"]})
    assert (a.sentimento, a.categorias) == (Sentimento.NEGATIVO, [Categoria.ATRASO])


def test_categorias_nulas_viram_lista_vazia():
    assert Analise.model_validate({**VALIDA, "categorias": None}).categorias == []


@pytest.mark.parametrize("valor", [True, False])
def test_booleano_nao_vira_urgencia(valor):
    with pytest.raises(ValidationError):
        Analise.model_validate({**VALIDA, "urgencia": valor})


# --- terceira rodada de revisão --------------------------------------------------------------


@pytest.mark.parametrize("categorias", [[["atraso"]], [{"a": 1}], ["atraso", ["atraso"]], [None]])
def test_itens_que_nao_sao_texto_viram_erro_de_validacao(categorias):
    with pytest.raises(ValidationError):  # e não TypeError, que derrubaria o programa
        Analise.model_validate({**VALIDA, "categorias": categorias})
