"""Testes de integração pela rede de verdade.

Um servidor HTTP local imita a Claude API. Diferente de test_llm.py, aqui nada é
substituído: a requisição sai pelo urllib, passa por um socket real e volta.
Isso pega problemas de cabeçalho, URL, tempo esgotado e conexão cortada.
"""

import io
import json
import socket
import threading
import time
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

import pytest

from avaliacoes_ia import __main__ as cli
from avaliacoes_ia import llm
from avaliacoes_ia.analisador import analisar

from .conftest import VALIDA, chamada


COM_USO = {**chamada(VALIDA), "usage": {"input_tokens": 1200, "output_tokens": 80}}


class APIFalsa(BaseHTTPRequestHandler):
    roteiro: ClassVar[list] = []
    recebidos: ClassVar[list] = []

    def log_message(self, *args):  # silencia o log do servidor
        pass

    def do_POST(self):
        corpo = self.rfile.read(int(self.headers["content-length"]))
        self.recebidos.append({"caminho": self.path, "cabecalhos": self.headers, "corpo": json.loads(corpo)})
        acao, *args = self.roteiro.pop(0) if self.roteiro else ("ok",)
        try:
            if acao == "ok":
                self._responder(200, json.dumps(COM_USO).encode())
            elif acao == "status":
                codigo, corpo_erro, extras = args
                self._responder(codigo, json.dumps(corpo_erro).encode(), extras)
            elif acao == "lento":
                time.sleep(args[0])
                self._responder(200, json.dumps(chamada(VALIDA)).encode())
            elif acao == "corta":  # promete 500 bytes e fecha a conexão no meio
                self.wfile.write(b'HTTP/1.1 200 OK\r\nContent-Length: 500\r\n\r\n{"content": [')
                self.wfile.flush()
                self.connection.shutdown(socket.SHUT_RDWR)
            elif acao == "html":
                self._responder(200, b"<html>gateway</html>", tipo="text/html")
        except (BrokenPipeError, ConnectionResetError):  # o cliente já desistiu (tempo esgotado)
            pass

    def _responder(self, codigo, corpo, extras=None, tipo="application/json"):
        self.send_response(codigo)
        self.send_header("content-type", tipo)
        self.send_header("content-length", str(len(corpo)))
        for nome, valor in (extras or {}).items():
            self.send_header(nome, valor)
        self.end_headers()
        self.wfile.write(corpo)


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")  # não mandar o teste local para um proxy
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    APIFalsa.roteiro, APIFalsa.recebidos = [], []
    servidor = ThreadingHTTPServer(("127.0.0.1", 0), APIFalsa)
    threading.Thread(target=servidor.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{servidor.server_port}", APIFalsa
    servidor.shutdown()
    servidor.server_close()


def cliente(base, esperas=None, **kw):
    kw.setdefault("timeout", 1.0)
    return llm.ClienteClaude(
        chave="sk-teste", url_base=base, espera=esperas.append if esperas is not None else (lambda s: None), **kw
    )


def test_requisicao_real_tem_url_cabecalhos_e_corpo_certos(api):
    base, servidor = api
    assert analisar("Demorou", cliente(base)).urgencia == 4
    pedido = servidor.recebidos[0]
    assert pedido["caminho"] == "/v1/messages"
    assert pedido["cabecalhos"]["x-api-key"] == "sk-teste"
    assert pedido["cabecalhos"]["anthropic-version"] == "2023-06-01"
    assert pedido["cabecalhos"]["content-type"] == "application/json"
    assert pedido["corpo"]["tool_choice"] == {"type": "tool", "name": "registrar_analise"}


def test_url_base_vem_da_variavel_de_ambiente(api, monkeypatch):
    base, servidor = api
    monkeypatch.setenv("ANTHROPIC_BASE_URL", base + "/")  # barra no fim também funciona
    analisar("x", llm.ClienteClaude(chave="k", espera=lambda s: None))
    assert servidor.recebidos[0]["caminho"] == "/v1/messages"


def test_tempo_esgotado_de_verdade(api):
    base, servidor = api
    servidor.roteiro = [("lento", 1.5), ("ok",)]
    esperas = []
    assert analisar("x", cliente(base, esperas, timeout=0.3)).urgencia == 4
    assert esperas == [1.0]


def test_conexao_cortada_no_meio_da_resposta(api):
    base, servidor = api
    servidor.roteiro = [("corta",), ("ok",)]
    assert analisar("x", cliente(base)).urgencia == 4
    assert len(servidor.recebidos) == 2


def test_sobrecarga_respeita_retry_after_com_teto(api):
    base, servidor = api
    servidor.roteiro = [("status", 529, {}, {"retry-after": "999"}), ("ok",)]
    esperas = []
    analisar("x", cliente(base, esperas))
    assert esperas == [60.0]


def test_erro_da_api_vira_mensagem_legivel(api):
    base, servidor = api
    corpo = {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}
    servidor.roteiro = [("status", 401, corpo, {})]
    with pytest.raises(llm.ErroLLM) as e:
        analisar("x", cliente(base))
    assert e.value.fatal
    assert str(e.value) == (
        "API respondeu 401 (authentication_error: invalid x-api-key). Dica: verifique a ANTHROPIC_API_KEY"
        " (para usar o Gemini, que é grátis: --provedor gemini)"
    )


def test_pagina_html_de_um_proxy(api):
    base, servidor = api
    servidor.roteiro = [("html",)]
    with pytest.raises(llm.ErroLLM, match="não é JSON"):
        analisar("x", cliente(base))


def test_porta_fechada_tem_nova_tentativa_e_depois_desiste():
    with socket.socket() as s:  # pega uma porta livre e fecha: ninguém escuta nela
        s.bind(("127.0.0.1", 0))
        porta = s.getsockname()[1]
    esperas = []
    with pytest.raises(llm.ErroLLM, match="2 tentativas") as e:
        analisar("x", cliente(f"http://127.0.0.1:{porta}", esperas, max_retentativas=1))
    assert not e.value.fatal
    assert esperas == [1.0]


def test_linha_de_comando_inteira_pela_rede(api, monkeypatch, tmp_path):
    base, servidor = api
    monkeypatch.setenv("ANTHROPIC_BASE_URL", base)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-teste")
    entrada = tmp_path / "e.csv"
    entrada.write_text("id,texto\nA1,Demorou\nA2,Frio\n", encoding="utf-8")
    saida = io.StringIO()
    with redirect_stdout(saida), redirect_stderr(io.StringIO()):
        assert cli.main(["analisar", str(entrada), "--saida", str(tmp_path / "s")]) == 0
    assert len(servidor.recebidos) == 2
    assert "Tokens gastos: 2.400 de entrada, 160 de saída" in saida.getvalue()
    assert (tmp_path / "s" / "analises.jsonl").read_text(encoding="utf-8").count("\n") == 2


@pytest.mark.parametrize(
    "base", ["https://proxy.empresa.com", "https://proxy.empresa.com/", "https://proxy.empresa.com//"]
)
def test_url_montada_sem_barra_dupla(base):
    # O servidor de teste do Python corrige "//" sozinho (proteção contra redirecionamento),
    # então a URL é conferida aqui, no cliente: a API real poderia responder 404.
    assert llm.ClienteClaude(chave="k", url_base=base).url == "https://proxy.empresa.com/v1/messages"
