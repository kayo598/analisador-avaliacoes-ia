"""Testes da linha de comando, do começo ao fim."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from avaliacoes_ia import __main__ as cli
from avaliacoes_ia import analisador, offline
from avaliacoes_ia.analisador import ErroAnalise
from avaliacoes_ia.llm import ErroLLM

from .conftest import VALIDA

CSV = "id;texto\nA1;Demorou e veio frio\nA2;Delicioso, recomendo\nA3;Faltou o molho\n"


@pytest.fixture
def entrada(tmp_path):
    arquivo = tmp_path / "entrada.csv"
    arquivo.write_text(CSV, encoding="utf-8")
    return arquivo


def ids_gravados(saida):
    linhas = (saida / "analises.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(registro)["id"] for registro in linhas]


def rodar(*args):
    return cli.main([str(a) for a in args])


def test_analisa_tudo_e_gera_relatorio(entrada, tmp_path, capsys):
    saida = tmp_path / "saida"
    assert rodar("analisar", entrada, "--offline", "--saida", saida) == 0
    assert ids_gravados(saida) == ["A1", "A2", "A3"]
    assert "Avaliações analisadas: **3**" in (saida / "relatorio.md").read_text(encoding="utf-8")


def test_falha_parcial_grava_o_resto_e_continuar_refaz_so_a_que_falhou(entrada, tmp_path, monkeypatch, capsys):
    saida = tmp_path / "saida"
    original = cli.analisar

    def falha_no_a2(texto, cliente):
        if "Delicioso" in texto:
            raise ErroAnalise("formato inválido")
        return original(texto, cliente)

    monkeypatch.setattr(cli, "analisar", falha_no_a2)
    assert rodar("analisar", entrada, "--offline", "--saida", saida) == 2
    assert ids_gravados(saida) == ["A1", "A3"]
    assert "--continuar" in capsys.readouterr().err

    chamados = []
    monkeypatch.setattr(cli, "analisar", lambda t, c: chamados.append(t) or original(t, c))
    assert rodar("analisar", entrada, "--offline", "--saida", saida, "--continuar") == 0
    assert chamados == ["Delicioso, recomendo"]
    assert sorted(ids_gravados(saida)) == ["A1", "A2", "A3"]


def test_erro_fatal_para_o_lote_sem_perder_o_que_foi_feito(entrada, tmp_path, monkeypatch):
    saida = tmp_path / "saida"
    original = cli.analisar
    vistos = []

    def chave_revogada_no_meio(texto, cliente):
        vistos.append(texto)
        if len(vistos) == 2:
            raise ErroLLM("API respondeu 401", fatal=True)
        return original(texto, cliente)

    monkeypatch.setattr(cli, "analisar", chave_revogada_no_meio)
    assert rodar("analisar", entrada, "--offline", "--saida", saida) == 1
    assert len(vistos) == 2  # não tentou a terceira
    assert ids_gravados(saida) == ["A1"]
    assert "Avaliações analisadas: **1**" in (saida / "relatorio.md").read_text(encoding="utf-8")


def test_ctrl_c_salva_o_que_ja_foi_feito(entrada, tmp_path, monkeypatch):
    saida = tmp_path / "saida"
    original = cli.analisar
    vistos = []

    def interrompe(texto, cliente):
        vistos.append(texto)
        if len(vistos) == 2:
            raise KeyboardInterrupt
        return original(texto, cliente)

    monkeypatch.setattr(cli, "analisar", interrompe)
    assert rodar("analisar", entrada, "--offline", "--saida", saida) == 130
    assert ids_gravados(saida) == ["A1"]


def test_linha_corrompida_no_jsonl_e_refeita(entrada, tmp_path, capsys):
    saida = tmp_path / "saida"
    rodar("analisar", entrada, "--offline", "--saida", saida)
    arquivo = saida / "analises.jsonl"
    linhas = arquivo.read_text(encoding="utf-8").splitlines()
    arquivo.write_text("\n".join([*linhas[:2], linhas[2][:20]]) + "\n", encoding="utf-8")  # queda no meio

    assert rodar("analisar", entrada, "--offline", "--saida", saida, "--continuar") == 0
    assert "corrompida" in capsys.readouterr().err
    assert "Avaliações analisadas: **3**" in (saida / "relatorio.md").read_text(encoding="utf-8")


def test_nao_apaga_resultados_sem_pedir(entrada, tmp_path, capsys):
    saida = tmp_path / "saida"
    rodar("analisar", entrada, "--offline", "--saida", saida)
    antes = (saida / "analises.jsonl").read_text(encoding="utf-8")
    assert rodar("analisar", entrada, "--offline", "--saida", saida) == 1
    assert "--sobrescrever" in capsys.readouterr().err
    assert (saida / "analises.jsonl").read_text(encoding="utf-8") == antes


def test_sobrescrever_comeca_do_zero(entrada, tmp_path):
    saida = tmp_path / "saida"
    rodar("analisar", entrada, "--offline", "--saida", saida)
    assert rodar("analisar", entrada, "--offline", "--saida", saida, "--sobrescrever") == 0
    assert ids_gravados(saida) == ["A1", "A2", "A3"]


def test_continuar_refaz_avaliacao_com_texto_alterado(entrada, tmp_path, capsys):
    saida = tmp_path / "saida"
    rodar("analisar", entrada, "--offline", "--saida", saida)
    entrada.write_text(CSV.replace("Demorou e veio frio", "Chegou quentinho, delicioso"), encoding="utf-8")
    assert rodar("analisar", entrada, "--offline", "--saida", saida, "--continuar") == 0
    assert "serão refeitas: A1" in capsys.readouterr().out
    relatorio = (saida / "relatorio.md").read_text(encoding="utf-8")
    assert "| positivo | 2 |" in relatorio  # A1 agora é positivo


def test_saida_que_e_arquivo_da_mensagem_clara(entrada, tmp_path, capsys):
    arquivo = tmp_path / "ocupado"
    arquivo.write_text("x")
    assert rodar("analisar", entrada, "--offline", "--saida", arquivo) == 1
    assert "Erro ao ler ou gravar arquivos" in capsys.readouterr().err


@pytest.mark.parametrize(
    "conteudo, mensagem",
    [
        (None, "Gabarito não encontrado"),
        ("{nao é json", "ilegível"),
        (
            '[{"id": "a", "texto": "x", "sentimento": "negativo", "categorias": ["atrazo"], "urgente": true}]',
            "Caso 1 do gabarito inválido: categorias.0",
        ),
        (
            '[{"id": "a", "texto": "x", "sentimento": "neutro", "categorias": [], "urgente": false},'
            ' {"id": "a", "texto": "y", "sentimento": "neutro", "categorias": [], "urgente": false}]',
            "IDs repetidos",
        ),
    ],
)
def test_gabarito_invalido(tmp_path, capsys, conteudo, mensagem):
    arquivo = tmp_path / "g.json"
    if conteudo is not None:
        arquivo.write_text(conteudo, encoding="utf-8")
    assert rodar("avaliar", "--offline", "--gabarito", arquivo) == 1
    assert mensagem in capsys.readouterr().err


@pytest.mark.parametrize(
    "conteudo, mensagem",
    [("codigo,texto\nA1,x\n", "precisa das colunas id e texto"), ("id;texto\nA1;x\nA1;y\n", "IDs repetidos")],
)
def test_erro_de_entrada_sai_com_1_e_mensagem(tmp_path, capsys, conteudo, mensagem):
    arquivo = tmp_path / "ruim.csv"
    arquivo.write_text(conteudo, encoding="utf-8")
    assert rodar("analisar", arquivo, "--offline", "--saida", tmp_path / "s") == 1
    err = capsys.readouterr().err
    assert err.startswith("Erro: ") and mensagem in err


def test_sem_chave_sai_com_1(entrada, tmp_path, capsys):
    assert rodar("analisar", entrada, "--saida", tmp_path / "s") == 1
    erro = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY" in erro and "GEMINI_API_KEY" in erro


def test_avaliar_offline(capsys):
    assert rodar("avaliar", "--offline") == 0
    assert "Sentimento   acurácia" in capsys.readouterr().out


# --- terceira rodada de revisão --------------------------------------------------------------


def test_terminal_do_windows_nao_derruba_o_lote(tmp_path):
    """Saída redirecionada no Windows usa cp1252: um emoji no id não pode quebrar o programa."""
    entrada = tmp_path / "e.csv"
    entrada.write_text("id,texto\nA😀1,Demorou\nA2,Ótimo\n", encoding="utf-8")
    raiz = Path(__file__).resolve().parent.parent
    resultado = subprocess.run(
        [sys.executable, "-m", "avaliacoes_ia", "analisar", str(entrada), "--offline", "--saida", str(tmp_path / "s")],
        capture_output=True,
        env={**os.environ, "PYTHONIOENCODING": "cp1252", "PYTHONPATH": str(raiz)},
        check=False,
    )
    assert resultado.returncode == 0, resultado.stderr
    assert b"2/2" in resultado.stdout


def test_jsonl_salvo_com_bom_continua_valendo(entrada, tmp_path, capsys):
    saida = tmp_path / "saida"
    rodar("analisar", entrada, "--offline", "--saida", saida)
    arquivo = saida / "analises.jsonl"
    arquivo.write_bytes(b"\xef\xbb\xbf" + arquivo.read_bytes())  # como o Bloco de Notas faz
    capsys.readouterr()
    assert rodar("analisar", entrada, "--offline", "--saida", saida, "--continuar") == 0
    saida_texto = capsys.readouterr()
    assert "3 já analisadas, 0 pendentes" in saida_texto.out
    assert "corrompida" not in saida_texto.err


# --- quarta rodada: lacunas achadas pelos testes de mutação e de cobertura -------------------


def test_erro_temporario_da_api_nao_para_o_lote(entrada, tmp_path, monkeypatch, capsys):
    original = cli.analisar

    def instavel(texto, cliente):
        if "Delicioso" in texto:
            raise ErroLLM("API respondeu 529; desisti após 4 tentativas")
        return original(texto, cliente)

    monkeypatch.setattr(cli, "analisar", instavel)
    assert rodar("analisar", entrada, "--offline", "--saida", tmp_path / "s") == 2
    assert ids_gravados(tmp_path / "s") == ["A1", "A3"]
    assert "[A2] falhou: API respondeu 529" in capsys.readouterr().err


def test_api_fora_do_ar_para_o_lote(tmp_path, monkeypatch, capsys):
    entrada = tmp_path / "e.csv"
    entrada.write_text("id,texto\n" + "".join(f"A{i},texto {i}\n" for i in range(20)), encoding="utf-8")
    chamadas = []

    def fora_do_ar(texto, cliente):
        chamadas.append(texto)
        raise ErroLLM("falha de conexão")

    monkeypatch.setattr(cli, "analisar", fora_do_ar)
    assert rodar("analisar", entrada, "--offline", "--saida", tmp_path / "s") == 3
    assert len(chamadas) == cli.LIMITE_FALHAS_SEGUIDAS  # não insistiu nas outras 15
    assert "parece fora do ar" in capsys.readouterr().err


def test_contador_de_falhas_seguidas_zera_quando_a_api_volta(tmp_path, monkeypatch):
    entrada = tmp_path / "e.csv"
    entrada.write_text("id,texto\n" + "".join(f"A{i},t{i}\n" for i in range(12)), encoding="utf-8")
    original = cli.analisar

    def intermitente(texto, cliente):  # falha 4, acerta 1, falha 4...: nunca 5 seguidas
        if int(texto[1:]) % 5 != 4:
            raise ErroLLM("instável")
        return original(texto, cliente)

    monkeypatch.setattr(cli, "analisar", intermitente)
    assert rodar("analisar", entrada, "--offline", "--saida", tmp_path / "s") == 2
    assert ids_gravados(tmp_path / "s") == ["A4", "A9"]


def test_avisos_do_csv_aparecem(tmp_path, capsys):
    entrada = tmp_path / "e.csv"
    entrada.write_text("id,texto\nA1,Demorou\nA2\n", encoding="utf-8")
    assert rodar("analisar", entrada, "--offline", "--saida", tmp_path / "s") == 0
    assert "aviso: linha 3 (A2): sem id ou sem texto, ignorada" in capsys.readouterr().err


def test_analise_e_gravada_na_hora_mesmo_se_o_processo_morrer(tmp_path):
    """Sem flush, a análise ficaria no buffer e se perderia numa queda (os._exit simula uma queda)."""
    entrada = tmp_path / "e.csv"
    entrada.write_text("id,texto\nA1,Demorou\nA2,Frio\n", encoding="utf-8")
    raiz = Path(__file__).resolve().parent.parent
    roteiro = f"""
import os, sys
sys.path.insert(0, {str(raiz)!r})
from avaliacoes_ia import __main__ as cli
original = cli.analisar
vistos = []
def cai_na_segunda(texto, cliente):
    vistos.append(texto)
    if len(vistos) == 2:
        os._exit(9)  # morre na hora, sem fechar arquivos nem esvaziar buffers
    return original(texto, cliente)
cli.analisar = cai_na_segunda
cli.main(["analisar", {str(entrada)!r}, "--offline", "--saida", {str(tmp_path / "s")!r}])
"""
    resultado = subprocess.run([sys.executable, "-c", roteiro], capture_output=True, check=False)
    assert resultado.returncode == 9
    assert ids_gravados(tmp_path / "s") == ["A1"]


def test_erro_de_formato_zera_o_contador_de_falhas_da_api(tmp_path, monkeypatch):
    """Erro de formato (ErroAnalise) prova que a API respondeu: não pode contar como queda."""
    entrada = tmp_path / "e.csv"
    entrada.write_text("id,texto\n" + "".join(f"A{i},t{i}\n" for i in range(9)), encoding="utf-8")

    def falhas_mistas(texto, cliente):  # 4 quedas, 1 erro de formato, 4 quedas
        if texto == "t4":
            raise ErroAnalise("formato inválido")
        raise ErroLLM("falha de conexão")

    monkeypatch.setattr(cli, "analisar", falhas_mistas)
    assert rodar("analisar", entrada, "--offline", "--saida", tmp_path / "s") == 2  # e não 3 (parada)


def test_continuar_sem_resultados_anteriores_comeca_normalmente(entrada, tmp_path):
    assert rodar("analisar", entrada, "--offline", "--saida", tmp_path / "nova", "--continuar") == 0
    assert ids_gravados(tmp_path / "nova") == ["A1", "A2", "A3"]


def test_linhas_em_branco_no_jsonl_sao_ignoradas(entrada, tmp_path, capsys):
    saida = tmp_path / "saida"
    rodar("analisar", entrada, "--offline", "--saida", saida)
    arquivo = saida / "analises.jsonl"
    arquivo.write_text("\n\n" + arquivo.read_text(encoding="utf-8") + "\n  \n", encoding="utf-8")
    capsys.readouterr()
    assert rodar("analisar", entrada, "--offline", "--saida", saida, "--continuar") == 0
    assert "0 pendentes" in capsys.readouterr().out


# --- quinta rodada: uso real ao longo do tempo ---------------------------------------------


def test_analise_grava_a_origem(entrada, tmp_path):
    rodar("analisar", entrada, "--offline", "--saida", tmp_path / "s")
    registro = json.loads((tmp_path / "s" / "analises.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert registro["origem"].startswith("offline (regras ")
    assert "| prompt " in registro["origem"]


def test_continuar_refaz_o_que_veio_de_outra_versao(entrada, tmp_path, monkeypatch, capsys):
    """Trocar o modelo ou o prompt e usar --continuar não pode misturar análises de versões diferentes."""
    saida = tmp_path / "s"
    rodar("analisar", entrada, "--offline", "--saida", saida)
    monkeypatch.setattr(analisador, "VERSAO_PROMPT", "outra-versao")
    capsys.readouterr()
    assert rodar("analisar", entrada, "--offline", "--saida", saida, "--continuar") == 0
    assert "outra versão do prompt, serão refeitas: 3" in capsys.readouterr().out
    origens = {json.loads(r)["origem"] for r in (saida / "analises.jsonl").read_text(encoding="utf-8").splitlines()}
    assert any("outra-versao" in o for o in origens)


def test_mudar_as_regras_offline_tambem_muda_a_versao(entrada, tmp_path, monkeypatch, capsys):
    saida = tmp_path / "s"
    rodar("analisar", entrada, "--offline", "--saida", saida)
    monkeypatch.setattr(offline, "GRAVE", [*offline.GRAVE, "novo-termo"])
    capsys.readouterr()
    rodar("analisar", entrada, "--offline", "--saida", saida, "--continuar")
    assert "serão refeitas: 3" in capsys.readouterr().out


def test_registro_antigo_sem_origem_e_refeito(entrada, tmp_path, capsys):
    saida = tmp_path / "s"
    saida.mkdir()
    antigo = {"id": "A1", "texto": "Demorou e veio frio", "analise": VALIDA}
    (saida / "analises.jsonl").write_text(json.dumps(antigo) + "\n", encoding="utf-8")
    rodar("analisar", entrada, "--offline", "--saida", saida, "--continuar")
    assert "0 já analisadas, 3 pendentes" in capsys.readouterr().out


def test_texto_repetido_nao_chama_o_modelo_de_novo(tmp_path, monkeypatch, capsys):
    entrada = tmp_path / "e.csv"
    textos = ["Ótimo", "ótimo ", "Bom", "Ótimo!", "BOM"]
    entrada.write_text("id,texto\n" + "".join(f"R{i},{t}\n" for i, t in enumerate(textos * 8)), encoding="utf-8")
    chamadas = []
    original = cli.analisar
    monkeypatch.setattr(cli, "analisar", lambda t, c: chamadas.append(t) or original(t, c))
    assert rodar("analisar", entrada, "--offline", "--saida", tmp_path / "s") == 0
    assert chamadas == ["Ótimo", "Bom", "Ótimo!"]  # 40 avaliações, 3 textos diferentes
    assert len(ids_gravados(tmp_path / "s")) == 40
    assert "(texto repetido)" in capsys.readouterr().out


def test_texto_repetido_aproveita_analise_de_execucao_anterior(tmp_path, monkeypatch):
    saida = tmp_path / "s"
    primeira = tmp_path / "a.csv"
    primeira.write_text("id,texto\nA1,Demorou muito\n", encoding="utf-8")
    rodar("analisar", primeira, "--offline", "--saida", saida)
    segunda = tmp_path / "b.csv"
    segunda.write_text("id,texto\nA1,Demorou muito\nB7,demorou  MUITO\n", encoding="utf-8")
    chamadas = []
    monkeypatch.setattr(cli, "analisar", lambda t, c: chamadas.append(t))
    assert rodar("analisar", segunda, "--offline", "--saida", saida, "--continuar") == 0
    assert chamadas == []


def test_erro_de_formato_nao_vira_cache(tmp_path, monkeypatch):
    entrada = tmp_path / "e.csv"
    entrada.write_text("id,texto\nA1,Igual\nA2,Igual\n", encoding="utf-8")
    chamadas = []

    def falha(texto, cliente):
        chamadas.append(texto)
        raise ErroAnalise("formato")

    monkeypatch.setattr(cli, "analisar", falha)
    assert rodar("analisar", entrada, "--offline", "--saida", tmp_path / "s") == 2
    assert len(chamadas) == 2  # a falha da primeira não pode ser "reaproveitada" na segunda


def test_offline_com_modelo_e_recusado(entrada, tmp_path, capsys):
    with pytest.raises(SystemExit) as saida:
        rodar("analisar", entrada, "--offline", "--modelo", "claude-opus-5-5", "--saida", tmp_path / "s")
    assert saida.value.code == 2
    assert "só valem com a API" in capsys.readouterr().err


def test_avaliar_salva_resultado_para_comparar(tmp_path, capsys):
    arquivo = tmp_path / "resultado.json"
    assert rodar("avaliar", "--offline", "--salvar", arquivo) == 0
    salvo = json.loads(arquivo.read_text(encoding="utf-8"))
    assert salvo["origem"].startswith("offline")
    assert salvo["metricas"]["total"] == 30
    assert "IC 95%" in capsys.readouterr().out
