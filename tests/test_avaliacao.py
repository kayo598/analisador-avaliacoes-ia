import json
from pathlib import Path

import pytest

from avaliacoes_ia.avaliacao import avaliar, intervalo_wilson
from avaliacoes_ia.entrada import ErroEntrada
from avaliacoes_ia.llm import ErroLLM
from avaliacoes_ia.offline import ClienteOffline

from .conftest import ClienteFalso, chamada

GABARITO = Path(__file__).resolve().parent.parent / "avaliacoes_ia" / "gabarito.json"


def gabarito(tmp_path, casos):
    arquivo = tmp_path / "g.json"
    arquivo.write_text(json.dumps(casos), encoding="utf-8")
    return arquivo


def caso(id_, sentimento, categorias, urgente):
    return {"id": id_, "texto": id_, "sentimento": sentimento, "categorias": categorias, "urgente": urgente}


def resposta(sentimento, categorias, urgencia):
    return chamada({"sentimento": sentimento, "categorias": categorias, "urgencia": urgencia, "resumo": "r"})


def test_metricas_calculadas_a_mao(tmp_path):
    casos = [
        caso("a", "negativo", ["atraso", "comida_fria"], True),
        caso("b", "positivo", ["elogio"], False),
    ]
    cliente = ClienteFalso(
        [
            resposta("negativo", ["atraso", "preco"], 5),  # 1 categoria certa, 1 a mais, 1 faltando
            resposta("neutro", ["elogio"], 4),  # sentimento errado, alarme falso de urgência
        ]
    )
    m = avaliar(cliente, gabarito(tmp_path, casos))
    assert m.acuracia_sentimento == 0.5
    assert (m.precisao_categorias, m.revocacao_categorias) == (2 / 3, 2 / 3)
    assert (m.revocacao_urgentes, m.precisao_urgentes) == (1.0, 0.5)
    assert len(m.erros) == 2


def test_falha_conta_como_erro_e_nao_interrompe(tmp_path):
    casos = [caso("a", "negativo", ["atraso"], True), caso("b", "positivo", ["elogio"], False)]
    cliente = ClienteFalso([ErroLLM("529"), resposta("positivo", ["elogio"], 1)])
    m = avaliar(cliente, gabarito(tmp_path, casos))
    assert m.falhas == 1
    assert m.revocacao_urgentes == 0.0 and m.revocacao_categorias == 0.5


def test_erro_fatal_interrompe(tmp_path):
    cliente = ClienteFalso([ErroLLM("401", fatal=True)])
    with pytest.raises(ErroLLM):
        avaliar(cliente, gabarito(tmp_path, [caso("a", "neutro", [], False)]))


def test_linha_de_base_offline_nao_piora():
    """Se uma mudança nas regras piorar o modo offline, este teste avisa."""
    m = avaliar(ClienteOffline(), GABARITO)
    assert m.total == 30
    assert m.acuracia_sentimento >= 0.8
    assert m.f1_categorias >= 0.85
    assert m.revocacao_urgentes >= 0.8


@pytest.mark.parametrize("conteudo", ["[]", '{"id": "a"}'])
def test_gabarito_precisa_ser_lista_com_casos(tmp_path, conteudo):
    arquivo = tmp_path / "g.json"
    arquivo.write_text(conteudo, encoding="utf-8")
    with pytest.raises(ErroEntrada, match="lista JSON"):
        avaliar(ClienteOffline(), arquivo)


# --- quinta rodada ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "acertos, total, esperado",
    [(25, 30, (0.664, 0.927)), (30, 30, (0.886, 1.0)), (0, 10, (0.0, 0.278)), (5, 10, (0.237, 0.763))],
)
def test_intervalo_de_wilson(acertos, total, esperado):
    baixo, alto = intervalo_wilson(acertos, total)
    assert baixo == pytest.approx(esperado[0], abs=0.001)
    assert alto == pytest.approx(esperado[1], abs=0.001)


def test_intervalo_sem_casos():
    assert intervalo_wilson(0, 0) == (0.0, 1.0)


def test_contagens_para_o_intervalo():
    m = avaliar(ClienteOffline(), GABARITO)
    assert m.acertos_sentimento == round(m.acuracia_sentimento * m.total)
    assert m.urgentes_total == 11
    assert m.urgentes_achadas == round(m.revocacao_urgentes * m.urgentes_total)
