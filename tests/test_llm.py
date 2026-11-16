import email.message
import http.client
import io
import json
import urllib.error

import pytest

from avaliacoes_ia import llm
from avaliacoes_ia.analisador import ESCOLHA, FERRAMENTA

from .conftest import VALIDA, chamada


class Resposta:
    def __init__(self, corpo: bytes):
        self.corpo = corpo

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self.corpo


def http_erro(codigo: int, retry_after: str | None = None):
    cabecalhos = email.message.Message()
    if retry_after:
        cabecalhos["retry-after"] = retry_after
    return urllib.error.HTTPError(llm.URL_API, codigo, "erro", cabecalhos, io.BytesIO(b'{"error": "x"}'))


@pytest.fixture
def api(monkeypatch):
    """Simula a API: cada item da lista é uma resposta (dict) ou uma exceção a lançar."""
    estado = {"roteiro": [], "pedidos": []}

    def urlopen(req, timeout):
        estado["pedidos"].append({"corpo": json.loads(req.data), "headers": dict(req.header_items())})
        proximo = estado["roteiro"].pop(0)
        if isinstance(proximo, BaseException):
            raise proximo
        return Resposta(json.dumps(proximo).encode())

    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    return estado


def cliente(esperas: list | None = None, **kw):
    return llm.ClienteClaude(
        chave="chave-teste", espera=(esperas.append if esperas is not None else lambda s: None), **kw
    )


def enviar(c):
    return c.enviar("sistema", [{"role": "user", "content": "oi"}], [FERRAMENTA], ESCOLHA)


def test_monta_a_requisicao(api):
    api["roteiro"] = [chamada(VALIDA)]
    enviar(cliente())
    pedido = api["pedidos"][0]
    assert pedido["headers"]["X-api-key"] == "chave-teste"
    assert pedido["corpo"]["tool_choice"] == ESCOLHA  # o Haiku 5.5 (padrão) aceita a ferramenta forçada
    assert pedido["corpo"]["tools"] == [FERRAMENTA]
    # Os modelos 5.x recusam "temperature" (o Sonnet 5.5 responde 400; o Haiku 5.5 pede para omitir).
    # O teste antigo exigia temperature == 0: o erro estava gravado no próprio teste.
    assert "temperature" not in pedido["corpo"] and "top_p" not in pedido["corpo"]
    assert pedido["corpo"]["output_config"] == {"effort": "low"}
    assert pedido["corpo"]["max_tokens"] >= 2048  # o raciocínio dos modelos 5.x conta neste limite
    assert "thinking" not in pedido["corpo"]


@pytest.mark.parametrize(
    "modelo, forcada, esforco",
    [
        ("claude-haiku-5-5", True, True),
        ("claude-opus-5-5", False, True),  # recusa forçada com 400; o README sugere este modelo
        ("claude-sonnet-5-5", False, True),
        ("claude-sonnet-5", True, True),
        ("claude-opus-5", True, True),
        ("claude-haiku-4-5", True, False),
        ("claude-haiku-4-5-20251001", True, False),
        ("claude-sonnet-4-6", True, False),
        ("claude-fable-5-2", False, True),
        ("claude-mythos-5-1", False, True),
        ("claude-opus-6", False, True),  # modelo futuro: "auto" funciona em todos
        ("claude-3-5-haiku-20241022", False, False),
        ("modelo-do-proxy", False, False),
    ],
)
def test_ferramenta_forcada_so_onde_o_modelo_aceita(api, modelo, forcada, esforco):
    api["roteiro"] = [chamada(VALIDA)]
    enviar(cliente(modelo=modelo))
    corpo = api["pedidos"][0]["corpo"]
    assert corpo["tool_choice"] == (ESCOLHA if forcada else {"type": "auto"})
    assert ("output_config" in corpo) is esforco


@pytest.mark.parametrize("escolha", [{"type": "auto"}, {"type": "none"}])
@pytest.mark.parametrize("modelo", ["claude-haiku-5-5", "claude-opus-5-5"])
def test_so_a_escolha_forcada_e_trocada(api, modelo, escolha):
    api["roteiro"] = [chamada(VALIDA)]
    cliente(modelo=modelo).enviar("s", [{"role": "user", "content": "oi"}], [FERRAMENTA], escolha)
    assert api["pedidos"][0]["corpo"]["tool_choice"] == escolha


def test_tenta_de_novo_em_erro_temporario_com_espera_crescente(api):
    esperas = []
    api["roteiro"] = [http_erro(529), http_erro(500), TimeoutError("read timed out"), chamada(VALIDA)]
    assert enviar(cliente(esperas))["content"][0]["input"] == VALIDA
    assert esperas == [1.0, 2.0, 4.0]


def test_respeita_retry_after(api):
    esperas = []
    api["roteiro"] = [http_erro(429, retry_after="7"), chamada(VALIDA)]
    enviar(cliente(esperas))
    assert esperas == [7.0]


def test_desiste_depois_do_limite_sem_ser_fatal(api):
    api["roteiro"] = [TimeoutError("read timed out")] * 4
    with pytest.raises(llm.ErroLLM, match="4 tentativas") as e:
        enviar(cliente(max_retentativas=3))
    assert not e.value.fatal


@pytest.mark.parametrize("codigo, fatal", [(401, True), (404, True), (400, False)])
def test_erro_definitivo_nao_tenta_de_novo(api, codigo, fatal):
    api["roteiro"] = [http_erro(codigo)]
    with pytest.raises(llm.ErroLLM) as e:
        enviar(cliente())
    assert e.value.fatal is fatal
    assert len(api["pedidos"]) == 1


def test_queda_de_conexao_tem_nova_tentativa(api):
    api["roteiro"] = [urllib.error.URLError("sem rede"), ConnectionResetError(), chamada(VALIDA)]
    enviar(cliente())
    assert len(api["pedidos"]) == 3


def test_resposta_que_nao_e_json(api, monkeypatch):
    monkeypatch.setattr(llm.urllib.request, "urlopen", lambda req, timeout: Resposta(b"<html>erro</html>"))
    with pytest.raises(llm.ErroLLM, match="não é JSON"):
        enviar(cliente())


def test_sem_chave_e_fatal(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(llm.ErroLLM) as e:
        llm.ClienteClaude()
    assert e.value.fatal


# --- segunda rodada de revisão ---------------------------------------------------------------


@pytest.mark.parametrize("valor", ["-5", "nan", "inf-ish", "Wed, 21 Oct 2026 07:28:00 GMT"])
def test_retry_after_invalido_usa_a_espera_padrao(api, valor):
    esperas = []
    api["roteiro"] = [http_erro(429, retry_after=valor), chamada(VALIDA)]
    enviar(cliente(esperas))
    assert esperas == [1.0]


@pytest.mark.parametrize(
    "falha", [http.client.IncompleteRead(b""), OSError("ssl eof"), http.client.RemoteDisconnected("fechou")]
)
def test_outras_falhas_de_rede_tem_nova_tentativa(api, falha):
    api["roteiro"] = [falha, chamada(VALIDA)]
    assert enviar(cliente())["content"][0]["input"] == VALIDA


def test_queda_ao_ler_o_corpo_do_erro(api):
    class CorpoQuebrado(io.BytesIO):
        def read(self, *a):
            raise http.client.IncompleteRead(b"")

    erro = urllib.error.HTTPError(llm.URL_API, 529, "x", email.message.Message(), CorpoQuebrado())
    api["roteiro"] = [erro, chamada(VALIDA)]
    assert enviar(cliente())["content"][0]["input"] == VALIDA


# --- quarta rodada: lacunas achadas pelos testes de mutação ----------------------------------


def test_retry_after_tem_teto_de_60_segundos(api):
    esperas = []
    api["roteiro"] = [http_erro(429, retry_after="999"), chamada(VALIDA)]
    enviar(cliente(esperas))
    assert esperas == [60.0]


def test_nao_espera_depois_da_ultima_tentativa(api):
    esperas = []
    api["roteiro"] = [TimeoutError()] * 3
    with pytest.raises(llm.ErroLLM):
        enviar(cliente(esperas, max_retentativas=2))
    assert esperas == [1.0, 2.0]  # 3 tentativas, 2 esperas: esperar depois da última seria tempo perdido


def test_cabecalhos_obrigatorios_da_api(api):
    api["roteiro"] = [chamada(VALIDA)]
    enviar(cliente())
    cabecalhos = {k.lower(): v for k, v in api["pedidos"][0]["headers"].items()}
    assert cabecalhos["anthropic-version"] == "2023-06-01"
    assert cabecalhos["content-type"] == "application/json"


@pytest.mark.parametrize(
    "codigo, corpo, esperado",
    [
        (
            404,
            '{"error": {"type": "not_found_error", "message": "model: x"}}',
            "API respondeu 404 (not_found_error: model: x). Dica: verifique o nome do modelo",
        ),
        (
            400,
            '{"error": {"type": "invalid_request_error", "message": "ruim"}}',
            "API respondeu 400 (invalid_request_error: ruim)",
        ),
        (400, "<html>\n  erro  </html>", "API respondeu 400 (<html> erro </html>)"),
        (400, "", "API respondeu 400 (sem detalhes)"),
        (400, "[1, 2]", "API respondeu 400 ([1, 2])"),
    ],
)
def test_mensagem_de_erro_legivel(codigo, corpo, esperado):
    assert llm.mensagem_de_erro(codigo, corpo).startswith(esperado)


def test_argumentos_opcionais_so_por_nome():
    with pytest.raises(TypeError):
        llm.ClienteClaude("chave", "modelo")  # evita trocar timeout por max_retentativas sem perceber


# --- quinta rodada ---------------------------------------------------------------------------


def test_soma_os_tokens_gastos(api):
    com_uso = {**chamada(VALIDA), "usage": {"input_tokens": 900, "output_tokens": 60}}
    api["roteiro"] = [com_uso, com_uso, {**chamada(VALIDA), "usage": {"input_tokens": "x", "output_tokens": True}}]
    c = cliente()
    for _ in range(3):
        enviar(c)
    assert c.uso == {"input_tokens": 1800, "output_tokens": 120}  # valores inválidos são ignorados


def test_origem_e_o_modelo():
    assert llm.ClienteClaude(chave="k", modelo="claude-opus-5-5").origem == "claude-opus-5-5"


@pytest.mark.parametrize("falha", [TimeoutError("read timed out"), urllib.error.URLError(TimeoutError("timed out"))])
def test_tempo_esgotado_avisa_na_hora_que_vai_tentar_de_novo(api, falha):
    api["roteiro"] = [falha, chamada(VALIDA)]
    avisos = []
    c = llm.ClienteClaude(chave="k", espera=lambda s: None, avisar=avisos.append)
    c.enviar("s", [{"role": "user", "content": "oi"}], [FERRAMENTA], ESCOLHA)
    assert avisos == ["(Claude API: sem resposta em 60 s; tentando de novo...)"]
