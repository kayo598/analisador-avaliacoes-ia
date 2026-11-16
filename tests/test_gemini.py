"""Testes do cliente do Gemini.

A API do Gemini não é chamada de verdade: um servidor local imita a compatibilidade OpenAI do
Gemini, inclusive as regras que mais quebram integrações (assinatura do raciocínio obrigatória,
resposta de ferramenta logo depois da chamada, com o mesmo id e o nome).
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

import pytest

from avaliacoes_ia import __main__ as cli
from avaliacoes_ia import llm
from avaliacoes_ia.analisador import ESCOLHA, FERRAMENTA, PROMPT_SISTEMA, analisar, origem_de
from avaliacoes_ia.gemini import ClienteGemini, ferramenta_para_gemini, mensagens_para_gemini

from .conftest import VALIDA


def resposta_gemini(texto=None, chamadas=(), motivo="stop", uso=(100, 20)):
    mensagem: dict = {"role": "assistant"}
    if texto is not None:
        mensagem["content"] = texto
    if chamadas:
        mensagem["tool_calls"] = list(chamadas)
    return {
        "choices": [{"index": 0, "message": mensagem, "finish_reason": motivo}],
        "usage": {"prompt_tokens": uso[0], "completion_tokens": uso[1], "total_tokens": sum(uso)},
    }


def registrar(id_, argumentos, assinatura=None):
    c = {"id": id_, "type": "function", "function": {"name": "registrar_analise", "arguments": json.dumps(argumentos)}}
    if assinatura:
        c["extra_content"] = {"google": {"thought_signature": assinatura}}
    return c


class GeminiFalso(BaseHTTPRequestHandler):
    """Imita /chat/completions do Gemini e confere o histórico como o Gemini 3 confere."""

    roteiro: ClassVar[list] = []
    recebidos: ClassVar[list] = []
    emitidas: ClassVar[dict] = {}  # id da chamada -> assinatura que o servidor mandou
    recusa_required: ClassVar[bool] = False  # imita um Gemini que não aceite tool_choice "required"

    def log_message(self, *args):
        pass

    def do_POST(self):  # noqa: RET503 (as respostas saem por _responder)
        corpo = json.loads(self.rfile.read(int(self.headers["content-length"])))
        self.recebidos.append({"caminho": self.path, "cabecalhos": self.headers, "corpo": corpo})
        problema = validar_como_gemini(corpo, self.emitidas)
        if self.recusa_required and corpo.get("tool_choice") == "required":
            problema = "Invalid value for tool_choice: required is not supported."
        if problema:
            erro = [{"error": {"code": 400, "message": problema, "status": "INVALID_ARGUMENT"}}]
            return self._responder(400, erro)
        item = self.roteiro.pop(0) if self.roteiro else resposta_gemini(chamadas=[registrar("padrao", VALIDA)])
        if isinstance(item, tuple):  # (código, corpo do erro)
            return self._responder(*item)
        escolhas = item.get("choices") or [{}]
        for c in escolhas[0].get("message", {}).get("tool_calls", []):
            if "extra_content" in c:
                self.emitidas[c["id"]] = c["extra_content"]
        self._responder(200, item)

    def _responder(self, codigo, dados):
        bruto = json.dumps(dados).encode()
        self.send_response(codigo)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(bruto)))
        self.end_headers()
        self.wfile.write(bruto)


def validar_como_gemini(corpo: dict, emitidas: dict) -> str:
    """As regras do Gemini que a tradução precisa respeitar. Devolve "" ou o problema."""
    mensagens = corpo["messages"]
    if not mensagens or mensagens[0]["role"] != "system":
        return "a primeira mensagem deve ser o system"
    if corpo.get("tool_choice") not in (None, "auto", "required"):
        return "tool_choice inválido"
    pendentes: list[str] = []
    for i, m in enumerate(mensagens[1:], start=1):
        if pendentes and m["role"] != "tool":
            return f"mensagem {i}: faltou responder as chamadas {pendentes}"
        if m["role"] == "assistant":
            chamadas = m.get("tool_calls", [])
            primeira = chamadas[0] if chamadas else {}
            if primeira.get("id") in emitidas and primeira.get("extra_content") != emitidas[primeira["id"]]:
                return f"Function call is missing a thought_signature (mensagem {i})"
            pendentes = [c["id"] for c in chamadas]
        elif m["role"] == "tool":
            if m["tool_call_id"] not in pendentes or not m.get("name"):
                return f"mensagem {i}: resposta de ferramenta sem chamada ou sem nome"
            pendentes.remove(m["tool_call_id"])
    return f"faltou responder as chamadas {pendentes}" if pendentes else ""


@pytest.fixture
def gemini(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    GeminiFalso.roteiro, GeminiFalso.recebidos, GeminiFalso.emitidas = [], [], {}
    GeminiFalso.recusa_required = False
    servidor = ThreadingHTTPServer(("127.0.0.1", 0), GeminiFalso)
    threading.Thread(target=servidor.serve_forever, daemon=True).start()
    monkeypatch.setenv("GEMINI_BASE_URL", f"http://127.0.0.1:{servidor.server_port}/v1beta/openai")
    yield GeminiFalso
    servidor.shutdown()
    servidor.server_close()


def cliente(**kw):
    return ClienteGemini(chave="chave-gemini", espera=lambda s: None, avisar=lambda t: None, **kw)


# --- análise inteira, pela rede ---


def test_analise_pela_rede(gemini):
    gemini.roteiro = [resposta_gemini(chamadas=[registrar("c1", VALIDA, "SIG")], uso=(250, 30))]
    c = cliente()
    analise = analisar("Demorou duas horas", c)
    assert analise.urgencia == 4 and analise.categorias[0].value == "atraso"
    pedido = gemini.recebidos[0]
    assert pedido["caminho"] == "/v1beta/openai/chat/completions"
    assert pedido["cabecalhos"]["authorization"] == "Bearer chave-gemini"
    corpo = pedido["corpo"]
    assert corpo["model"] == "gemini-3.8-flash" and corpo["reasoning_effort"] == "low"
    assert corpo["tool_choice"] == "required"  # forçada: a resposta vem sempre pela ferramenta
    assert corpo["messages"][0] == {"role": "system", "content": PROMPT_SISTEMA}
    assert corpo["tools"][0]["function"]["parameters"] == FERRAMENTA["input_schema"]
    assert c.uso == {"input_tokens": 250, "output_tokens": 30}
    assert origem_de(c).startswith("gemini-3.8-flash | prompt ")


def test_resposta_em_texto_e_depois_invalida_sao_corrigidas_na_mesma_conversa(gemini):
    gemini.roteiro = [
        resposta_gemini("É negativo, urgência 4."),  # não chamou a ferramenta
        resposta_gemini(chamadas=[registrar("c2", {**VALIDA, "urgencia": 9}, "SIG-2")]),  # fora do esquema
        resposta_gemini(chamadas=[registrar("c3", VALIDA, "SIG-3")]),
    ]
    assert analisar("Demorou", cliente()).urgencia == 4
    terceiro = gemini.recebidos[2]["corpo"]["messages"]
    assert terceiro[2] == {"role": "assistant", "content": "É negativo, urgência 4."}
    assert terceiro[3]["role"] == "user" and "Chame a ferramenta" in terceiro[3]["content"]
    assert terceiro[4]["tool_calls"][0]["extra_content"] == {"google": {"thought_signature": "SIG-2"}}
    assert terceiro[5]["role"] == "tool" and terceiro[5]["name"] == "registrar_analise"
    assert terceiro[5]["tool_call_id"] == "c2" and "urgencia" in terceiro[5]["content"]


def test_sem_guardar_a_assinatura_o_servidor_falso_recusa(gemini):
    """Prova que o servidor falso confere a assinatura: sem ela, o teste acima falharia."""

    class Esquecido(dict):
        def __setitem__(self, chave, valor):
            pass

    gemini.roteiro = [resposta_gemini(chamadas=[registrar("c2", {**VALIDA, "urgencia": 9}, "SIG-2")])]
    c = cliente()
    c.assinaturas = Esquecido()
    with pytest.raises(llm.ErroLLM, match="thought_signature"):
        analisar("Demorou", c)


def test_lote_pela_linha_de_comando(gemini, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GEMINI_API_KEY", "chave-do-usuario")
    entrada = tmp_path / "e.csv"
    entrada.write_text("id;texto\nA1;Demorou e veio frio\nA2;Delicioso\n", encoding="utf-8")
    gemini.roteiro = [
        resposta_gemini(chamadas=[registrar("c1", VALIDA, "S1")]),
        resposta_gemini(chamadas=[registrar("c2", {**VALIDA, "sentimento": "positivo", "urgencia": 1})]),
    ]
    assert cli.main(["analisar", str(entrada), "--saida", str(tmp_path / "s")]) == 0
    saida = capsys.readouterr().out
    assert "Usando Google Gemini (gemini-3.8-flash)." in saida
    assert "[A1] negativo urgência 4" in saida and "[A2] positivo urgência 1" in saida
    assert "Tokens gastos: 200 de entrada, 40 de saída" in saida
    assert gemini.recebidos[0]["cabecalhos"]["authorization"] == "Bearer chave-do-usuario"


def test_limite_diario_para_o_lote(gemini, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    entrada = tmp_path / "e.csv"
    entrada.write_text("id;texto\nA1;um\nA2;dois\nA3;três\n", encoding="utf-8")
    diario = {
        "error": {
            "code": 429,
            "status": "RESOURCE_EXHAUSTED",
            "message": "Quota exceeded for metric 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'",
        }
    }
    gemini.roteiro = [(429, diario)]
    assert cli.main(["analisar", str(entrada), "--saida", str(tmp_path / "s"), "--provedor", "gemini"]) == 1
    erro = capsys.readouterr().err
    assert "amanhã" in erro and "Parando" in erro
    assert len(gemini.recebidos) == 1  # não insistiu nas outras avaliações


def test_limite_por_minuto_espera_e_avisa(gemini):
    corpo = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": [{"retryDelay": "33s"}]}}
    gemini.roteiro = [(429, corpo), resposta_gemini(chamadas=[registrar("c1", VALIDA)])]
    esperas, avisos = [], []
    c = ClienteGemini(chave="k", espera=esperas.append, avisar=avisos.append)
    assert analisar("x", c).urgencia == 4
    assert esperas == [33.0] and "esperando 33 s" in avisos[0]


def test_chave_invalida_para_na_hora(gemini):
    corpo = [{"error": {"code": 400, "message": "API key not valid.", "status": "INVALID_ARGUMENT"}}]
    gemini.roteiro = [(400, corpo)]
    with pytest.raises(llm.ErroLLM) as erro:
        analisar("x", cliente())
    assert erro.value.fatal and "GEMINI_API_KEY" in str(erro.value)


# --- tradução ---


def test_esquema_da_ferramenta_vai_inteiro():
    convertida = ferramenta_para_gemini(FERRAMENTA)
    assert convertida == {
        "type": "function",
        "function": {
            "name": "registrar_analise",
            "description": FERRAMENTA["description"],
            "parameters": FERRAMENTA["input_schema"],
        },
    }


def test_escolha_forcada_vira_required_e_a_automatica_vira_auto():
    c = cliente()
    pedido = c.pedido("S", [{"role": "user", "content": "oi"}], [FERRAMENTA], ESCOLHA)
    assert pedido["tool_choice"] == "required" and "temperature" not in pedido
    assert c.pedido("S", [{"role": "user", "content": "oi"}], [FERRAMENTA], {"type": "auto"})["tool_choice"] == "auto"


def test_gemini_que_recusa_required_passa_a_usar_auto(gemini):
    gemini.recusa_required = True
    gemini.roteiro = [
        resposta_gemini(chamadas=[registrar("c1", VALIDA)]),
        resposta_gemini(chamadas=[registrar("c2", VALIDA)]),
    ]
    c = cliente()
    assert analisar("Demorou", c).urgencia == 4
    assert analisar("Demorou de novo", c).urgencia == 4
    escolhas = [r["corpo"]["tool_choice"] for r in gemini.recebidos]
    assert escolhas == ["required", "auto", "auto"]  # só um pedido perdido no lote inteiro


def test_mensagens_traduzidas_sem_assinatura_quando_nao_houve():
    historico = [
        {"role": "user", "content": "classifique"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "registrar_analise", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "is_error": True, "content": "erro"}]},
    ]
    saida = mensagens_para_gemini("S", historico, {})
    assert "extra_content" not in saida[2]["tool_calls"][0] and "content" not in saida[2]
    assert saida[3] == {"role": "tool", "tool_call_id": "a", "name": "registrar_analise", "content": "erro"}


@pytest.mark.parametrize(
    ("motivo", "esperado"), [("length", "max_tokens"), ("content_filter", "refusal"), ("stop", "end_turn")]
)
def test_motivo_de_parada_sem_chamada(motivo, esperado):
    assert cliente().traduzir(resposta_gemini("t", motivo=motivo))["stop_reason"] == esperado


def test_filtro_de_conteudo_vira_recusa_no_analisador(gemini):
    gemini.roteiro = [resposta_gemini("", motivo="content_filter")]
    with pytest.raises(Exception, match="recusa de segurança"):
        analisar("x", cliente())


def test_chamada_com_stop_e_sem_id(gemini):
    r = cliente().traduzir(resposta_gemini(chamadas=[{**registrar("", VALIDA), "id": None}]))
    assert r["stop_reason"] == "tool_use" and r["content"][0]["id"].startswith("gemini-")
    assert r["content"][0]["input"] == VALIDA


@pytest.mark.parametrize("ruim", [None, [], {}, {"choices": []}, {"choices": [{"message": "x"}]}])
def test_resposta_malformada_nao_quebra(ruim):
    assert cliente().traduzir(ruim)["content"] == []


def test_escolha_automatica_do_provedor(monkeypatch):
    assert llm.escolher_provedor() == "claude"
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    assert isinstance(llm.criar_cliente(), ClienteGemini)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    assert type(llm.criar_cliente()) is llm.ClienteClaude
    assert isinstance(llm.criar_cliente("gemini", "gemini-3.5-flash-lite"), ClienteGemini)


def test_offline_nao_aceita_provedor(capsys):
    with pytest.raises(SystemExit):
        cli.main(["avaliar", "--offline", "--provedor", "gemini"])
    assert "só valem com a API" in capsys.readouterr().err


# --- regras da tradução que valem para os três projetos (o gemini.py é o mesmo) ---


def test_traducao_cobre_os_casos_de_borda():
    vazia = ferramenta_para_gemini({"name": "agora", "description": "d", "input_schema": {"type": "object"}})
    assert "parameters" not in vazia["function"]  # o Gemini recusa objeto sem propriedades
    historico = [
        {"role": "user", "content": "oi"},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "...", "signature": "s"}]},
        {"role": "user", "content": [{"type": "text", "text": "e agora?"}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "agora", "input": {"t": "ação"}}]},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "a", "content": "ok"},
                {"type": "text", "text": "e depois?"},
            ],
        },
    ]
    saida = mensagens_para_gemini("S", historico, {})
    assert saida[2] == {"role": "assistant", "content": ""}  # nunca sem conteúdo e sem chamada
    assert saida[3] == {"role": "user", "content": "e agora?"}
    assert saida[4]["tool_calls"][0]["function"]["arguments"] == '{"t": "ação"}'  # sem \\u00e7
    assert saida[5] == {"role": "tool", "tool_call_id": "a", "name": "agora", "content": "ok"}
    assert saida[6] == {"role": "user", "content": "e depois?"}


def test_chamadas_de_ferramenta_na_resposta_e_de_volta_no_historico():
    c = ClienteGemini(chave="k", espera=lambda s: None, avisar=lambda t: None)
    resposta = {
        "choices": [
            {
                "finish_reason": "stop",  # o Gemini às vezes diz "stop" junto com chamadas
                "message": {
                    "tool_calls": [
                        {
                            "id": "a",
                            "function": {"name": "f", "arguments": '{"x": 1}'},
                            "extra_content": {"google": {"thought_signature": "SIG"}},
                        },
                        {"id": "", "function": {"name": "f", "arguments": "{quebrado"}},
                        {"function": {"name": "f", "arguments": "[1, 2]"}},
                    ]
                },
            }
        ]
    }
    r = c.traduzir(resposta)
    assert r["stop_reason"] == "tool_use"
    ids = [b["id"] for b in r["content"]]
    assert ids[0] == "a" and len(set(ids)) == 3  # ids gerados não se repetem
    assert [b["input"] for b in r["content"]] == [{"x": 1}, {}, {}]  # JSON inválido ou lista viram {}
    historico = [
        {"role": "user", "content": "oi"},
        {"role": "assistant", "content": r["content"]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": i, "content": "ok"} for i in ids]},
    ]
    saida = mensagens_para_gemini("S", historico, c.assinaturas)
    assert saida[2]["tool_calls"][0]["extra_content"] == {"google": {"thought_signature": "SIG"}}
    assert all("extra_content" not in t for t in saida[2]["tool_calls"][1:])


def test_erro_do_gemini_legivel_e_espera_com_teto():
    corpo = '[{"error": {"code": 503, "message": "The model is overloaded.", "status": "UNAVAILABLE"}}]'
    assert llm.mensagem_de_erro(503, corpo).startswith("API respondeu 503 (UNAVAILABLE: The model is overloaded.)")
    avisos = []
    c = ClienteGemini(chave="k", espera=lambda s: None, avisar=avisos.append)
    assert c.pausa(429, {}, '{"retryDelay": "999s"}', 1.0) == 60.0
    assert c.pausa(429, {}, "{}", 2.0) == 2.0 and len(avisos) == 1


# --- erros possíveis no uso de verdade (revisão; o gemini.py é o mesmo nos três projetos) ---


def _http(codigo: int, mensagem: str):
    import io  # noqa: PLC0415
    import urllib.error  # noqa: PLC0415

    corpo = json.dumps([{"error": {"code": codigo, "message": mensagem, "status": "X"}}]).encode()
    return urllib.error.HTTPError("u", codigo, "erro", {}, io.BytesIO(corpo))  # type: ignore[arg-type]


def _roteiro(c, itens):
    enviados = []

    def requisitar(corpo):
        enviados.append(json.loads(corpo))
        item = itens.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    c._requisitar = requisitar
    return enviados


OK = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
EXTRA = ([], {})  # ferramentas e escolha da ferramenta


@pytest.mark.parametrize("colada", ["  AIza-chave\n", '"AIza-chave"', "“AIza-chave”"])
def test_chave_colada_com_sobras_e_limpa(monkeypatch, colada):
    monkeypatch.setenv("GEMINI_API_KEY", colada)
    assert ClienteGemini().chave == "AIza-chave"


def test_chave_com_acento_no_meio_da_erro_claro_sem_mostrar_a_chave(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIzá-chave")
    with pytest.raises(llm.ErroLLM) as erro:
        ClienteGemini()
    assert erro.value.fatal and "GEMINI_API_KEY" in str(erro.value) and "AIz" not in str(erro.value)


def test_google_api_key_tambem_vale(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "AIza-google")
    assert llm.escolher_provedor() == "gemini" and llm.criar_cliente().chave == "AIza-google"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "  ")  # só espaços: não conta
    assert llm.escolher_provedor() == "gemini"


def test_sem_chave_no_windows_mostra_o_comando_do_powershell(monkeypatch):
    monkeypatch.setattr(llm.sys, "platform", "win32")
    with pytest.raises(llm.ErroLLM, match=r"\$env:GEMINI_API_KEY"):
        ClienteGemini()


@pytest.mark.parametrize("motivo", ["SAFETY", "RECITATION", "PROHIBITED_CONTENT"])
def test_bloqueios_do_gemini_viram_recusa(motivo):
    c = ClienteGemini(chave="k")
    resposta = {"choices": [{"message": {"content": None}, "finish_reason": motivo}]}
    assert c.traduzir(resposta)["stop_reason"] == "refusal"


def test_inteiro_que_chega_como_decimal_vira_inteiro():
    chamada_ = {"id": "a", "function": {"name": "f", "arguments": '{"n": 4.0, "l": [1.0, 2.5], "x": NaN}'}}
    r = ClienteGemini(chave="k").traduzir({"choices": [{"message": {"tool_calls": [chamada_]}}]})
    assert r["content"][0]["input"] == {}  # NaN não é JSON válido
    chamada_["function"]["arguments"] = '{"n": 4.0, "l": [1.0, 2.5]}'
    r = ClienteGemini(chave="k").traduzir({"choices": [{"message": {"tool_calls": [chamada_]}}]})
    assert r["content"][0]["input"] == {"n": 4, "l": [1, 2.5]}
    assert isinstance(r["content"][0]["input"]["n"], int)


def test_plano_b_sem_raciocinio_e_com_assinatura_ficticia():
    c = ClienteGemini(chave="k", espera=lambda s: None, avisar=lambda t: None)
    enviados = _roteiro(c, [_http(400, "Thinking is not supported for this model."), OK, OK])
    c.enviar("s", [{"role": "user", "content": "oi"}], *EXTRA)
    assert ["reasoning_effort" in e for e in enviados] == [True, False]
    historico = [
        {"role": "user", "content": "oi"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "f", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "x"}]},
    ]
    _roteiro(c, [_http(400, "Function call is missing a thought_signature"), OK])
    enviados = []
    original = c._requisitar

    def guardar(corpo):
        enviados.append(json.loads(corpo))
        return original(corpo)

    c._requisitar = guardar
    c.enviar("s", historico, *EXTRA)
    assert "extra_content" not in enviados[0]["messages"][2]["tool_calls"][0]
    assert enviados[1]["messages"][2]["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == (
        "skip_thought_signature_validator"
    )


def test_outro_400_e_chave_invalida_nao_usam_plano_b():
    c = ClienteGemini(chave="k", espera=lambda s: None, avisar=lambda t: None)
    enviados = _roteiro(c, [_http(400, "API key not valid. thinking")])
    with pytest.raises(llm.ErroLLM) as erro:
        c.enviar("s", [{"role": "user", "content": "oi"}], *EXTRA)
    assert erro.value.fatal and len(enviados) == 1 and c.usar_esforco


def test_desiste_com_dica_de_sobrecarga_sem_aviso_falso():
    avisos: list[str] = []
    c = ClienteGemini(chave="k", espera=lambda s: None, avisar=avisos.append)
    _roteiro(c, [_http(503, "The model is overloaded.") for _ in range(4)])
    with pytest.raises(llm.ErroLLM, match=r"desisti após 4 tentativas\. Dica: o Gemini está sobrecarregado"):
        c.enviar("s", [{"role": "user", "content": "oi"}], *EXTRA)
    assert avisos == []  # esperas de 1, 2 e 4 s: curtas, sem aviso; e nenhum aviso de espera que não houve


def test_inteiros_em_lista_viram_int_de_verdade():
    chamada_ = {"id": "a", "function": {"name": "f", "arguments": '{"n": 4.0, "l": [1.0, 2.5]}'}}
    entrada = ClienteGemini(chave="k").traduzir({"choices": [{"message": {"tool_calls": [chamada_]}}]})["content"][0][
        "input"
    ]
    assert type(entrada["n"]) is int and type(entrada["l"][0]) is int and entrada["l"][1] == 2.5


def test_mensagem_de_erro_longa_e_encurtada():
    corpo = json.dumps({"error": {"code": 429, "message": "x " * 1000, "status": "RESOURCE_EXHAUSTED"}})
    assert len(llm.mensagem_de_erro(429, corpo)) < 400


def test_tempo_de_espera_na_mensagem_e_aviso_certo_para_cada_erro():
    avisos: list[str] = []
    c = ClienteGemini(chave="k", espera=lambda s: None, avisar=avisos.append)
    assert c.pausa(429, {}, '{"message": "Please retry in 7.5s."}', 1.0) == 7.5
    assert c.pausa(503, {}, '{"retryDelay": "10s"}', 1.0) == 10.0
    assert "Limite por minuto" in avisos[0] and "respondeu 503" in avisos[1] and "Limite" not in avisos[1]


def test_os_dois_planos_b_e_depois_desiste_sem_laco_infinito():
    c = ClienteGemini(chave="k", espera=lambda s: None, avisar=lambda t: None)
    enviados = _roteiro(
        c,
        [
            _http(400, "thinking is not supported"),
            _http(400, "missing thought_signature"),
            _http(400, "thinking again"),
        ],
    )
    with pytest.raises(llm.ErroLLM, match="thinking again"):
        c.enviar("s", [{"role": "user", "content": "oi"}], *EXTRA)
    assert len(enviados) == 3


# --- terceira revisão: respostas fora do padrão e ids repetidos ---


def test_texto_em_partes_e_motivo_estranho():
    c = ClienteGemini(chave="k")
    partes = {"choices": [{"message": {"content": [{"type": "text", "text": "Olá, "}, {"text": "tudo bem?"}, 3]}}]}
    assert c.traduzir(partes)["content"] == [{"type": "text", "text": "Olá, tudo bem?"}]
    estranho = {"choices": [{"message": {"content": "oi"}, "finish_reason": ["stop"]}]}
    assert c.traduzir(estranho)["stop_reason"] == "end_turn"


def test_ids_repetidos_viram_ids_unicos():
    c = ClienteGemini(chave="k")

    def resposta(*ids):
        chamadas = [{"id": i, "function": {"name": "f", "arguments": "{}"}} for i in ids]
        return {"choices": [{"message": {"tool_calls": chamadas}}]}

    primeiro = [b["id"] for b in c.traduzir(resposta("function-call-1", "function-call-1"))["content"]]
    segundo = [b["id"] for b in c.traduzir(resposta("function-call-1", ""))["content"]]
    todos = primeiro + segundo
    assert primeiro[0] == "function-call-1" and len(set(todos)) == 4


def test_resposta_aleatoria_nunca_quebra_a_traducao():
    import random  # noqa: PLC0415

    gerador = random.Random(11)
    folhas = [None, 0, 1.5, -1, True, "", "x", "stop", "length", "SAFETY", [], {}, "{bad", '{"a": 2.0}', float("nan")]

    def valor(profundidade=0):
        if profundidade > 3 or gerador.random() < 0.4:
            return gerador.choice(folhas)
        if gerador.random() < 0.5:
            return [valor(profundidade + 1) for _ in range(gerador.randint(0, 3))]
        chaves = [
            "choices",
            "message",
            "content",
            "tool_calls",
            "function",
            "name",
            "arguments",
            "id",
            "extra_content",
            "finish_reason",
            "usage",
            "prompt_tokens",
            "refusal",
            "type",
            "text",
        ]
        return {gerador.choice(chaves): valor(profundidade + 1) for _ in range(gerador.randint(0, 4))}

    c = ClienteGemini(chave="k")
    for _ in range(3000):
        r = c.traduzir(valor())
        assert isinstance(r["content"], list)
        for bloco in r["content"]:
            assert bloco["type"] in ("text", "tool_use")
            if bloco["type"] == "tool_use":
                assert isinstance(bloco["id"], str) and bloco["id"] and isinstance(bloco["input"], dict)
                json.dumps(bloco["input"], allow_nan=False)  # volta ao Gemini como JSON válido


def test_argumentos_ja_em_objeto_com_nan_viram_vazio():
    chamada_ = {"id": "a", "function": {"name": "f", "arguments": {"x": float("nan")}}}
    r = ClienteGemini(chave="k").traduzir({"choices": [{"message": {"tool_calls": [chamada_]}}]})
    assert r["content"][0]["input"] == {}
    chamada_["function"]["arguments"] = {"n": 3.0}
    r = ClienteGemini(chave="k").traduzir({"choices": [{"message": {"tool_calls": [chamada_]}}]})
    assert r["content"][0]["input"] == {"n": 3} and type(r["content"][0]["input"]["n"]) is int


def test_provedor_aceita_maiusculas(monkeypatch):
    recebido = {}

    def fabrica(provedor, modelo):
        recebido["provedor"] = provedor
        raise llm.ErroLLM("parar aqui", fatal=True)

    monkeypatch.setattr(cli, "criar_cliente_api", fabrica)
    assert cli.main(["avaliar", "--provedor", "Gemini"]) == 1
    assert recebido == {"provedor": "gemini"}


def test_gemini_espera_mais_que_a_claude():
    """No primeiro teste real, o plano grátis do Gemini passou de 60 s para responder."""
    assert ClienteGemini(chave="k").timeout == 120
    assert llm.ClienteClaude(chave="k").timeout == 60
    assert ClienteGemini(chave="k", timeout=30).timeout == 30


def test_analisador_nunca_troca_de_modelo_no_meio_do_lote():
    """Cada análise grava o modelo que a fez: trocar no meio misturaria modelos no mesmo lote."""
    c = ClienteGemini(chave="k", espera=lambda s: None, avisar=lambda t: None)
    assert c.reserva is None
    enviados = _roteiro(c, [_http(503, "overloaded") for _ in range(4)])
    with pytest.raises(llm.ErroLLM, match="desisti após 4 tentativas"):
        c.enviar("s", [{"role": "user", "content": "oi"}], *EXTRA)
    assert {e["model"] for e in enviados} == {"gemini-3.8-flash"}
