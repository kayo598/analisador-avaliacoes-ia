import copy

import pytest


@pytest.fixture(autouse=True)
def sem_chaves_do_computador(monkeypatch):
    """Os testes não podem depender das chaves do computador de quem roda: com a GEMINI_API_KEY
    definida, um teste de "sem chave" escolheria o Gemini e poderia chamar a API de verdade."""
    for nome in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_BASE_URL"):
        monkeypatch.delenv(nome, raising=False)


def chamada(entrada: dict, id_: str = "toolu_1") -> dict:
    """Resposta da API em que o modelo chamou registrar_analise."""
    return {
        "stop_reason": "tool_use",
        "content": [{"type": "tool_use", "id": id_, "name": "registrar_analise", "input": entrada}],
    }


def texto(t: str) -> dict:
    return {"stop_reason": "end_turn", "content": [{"type": "text", "text": t}]}


VALIDA = {"sentimento": "negativo", "categorias": ["atraso"], "urgencia": 4, "resumo": "Atrasou muito"}


class ClienteFalso:
    """Devolve respostas pré-definidas e guarda uma cópia do que recebeu em cada chamada."""

    def __init__(self, respostas):
        self.respostas = list(respostas)
        self.envios = []

    def enviar(self, sistema, mensagens, ferramentas, escolha):
        self.envios.append(
            {"sistema": sistema, "mensagens": copy.deepcopy(mensagens), "ferramentas": ferramentas, "escolha": escolha}
        )
        resposta = self.respostas.pop(0)
        if isinstance(resposta, Exception):
            raise resposta
        return resposta


@pytest.fixture
def falso():
    return ClienteFalso
