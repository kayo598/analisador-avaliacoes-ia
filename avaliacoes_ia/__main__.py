"""Linha de comando.

python -m avaliacoes_ia analisar dados/avaliacoes_exemplo.csv [--offline] [--continuar]
python -m avaliacoes_ia avaliar [--offline]
Sem --offline, usa a Claude API (ANTHROPIC_API_KEY) ou o Google Gemini, que tem plano grátis
(GEMINI_API_KEY, ou --provedor gemini).

Códigos de saída: 0 = tudo certo · 1 = erro de entrada ou de configuração
2 = algumas avaliações falharam · 3 = a API falhou várias vezes seguidas (parou)
130 = interrompido com Ctrl+C
"""

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .analisador import ErroAnalise, analisar, origem_de
from .avaliacao import avaliar, intervalo_wilson
from .entrada import ErroEntrada, Linha, ler_csv
from .llm import PROVEDORES, ErroLLM, criar_cliente as criar_cliente_api
from .modelos import AvaliacaoAnalisada
from .offline import ClienteOffline
from .relatorio import gerar_markdown

GABARITO = Path(__file__).resolve().parent / "gabarito.json"


def erro(mensagem: str) -> None:
    print(mensagem, file=sys.stderr)


def criar_cliente(args):
    if args.offline:
        return ClienteOffline()
    cliente = criar_cliente_api(args.provedor, args.modelo)
    print(f"Usando {cliente.NOME} ({cliente.modelo}).")
    return cliente


def carregar_feitos(arquivo: Path) -> dict[str, AvaliacaoAnalisada]:
    """Lê o que já foi analisado. Uma última linha cortada (queda no meio da gravação) é ignorada."""
    feitos: dict[str, AvaliacaoAnalisada] = {}
    if not arquivo.exists():
        return feitos
    # utf-8-sig: se o arquivo foi aberto e salvo no Bloco de Notas, ele ganha um BOM no início.
    for numero, linha in enumerate(arquivo.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not linha.strip():
            continue
        try:
            item = AvaliacaoAnalisada.model_validate_json(linha)
        except ValueError:
            erro(f"aviso: linha {numero} de {arquivo.name} corrompida, será refeita")
            continue
        feitos[item.id] = item
    return feitos


def preparar_saida(args, linhas: list[Linha], origem: str) -> tuple[Path, dict[str, AvaliacaoAnalisada], list[Linha]]:
    """Cria a pasta, protege resultados anteriores e decide o que falta analisar."""
    args.saida.mkdir(parents=True, exist_ok=True)
    arquivo = args.saida / "analises.jsonl"
    ja_existe = arquivo.exists() and arquivo.stat().st_size > 0
    if ja_existe and not (args.continuar or args.sobrescrever):
        # Essas análises custaram chamadas pagas: não apagar sem o usuário pedir.
        raise ErroEntrada(
            f"Já existem resultados em {arquivo}. Use --continuar para aproveitá-los "
            "ou --sobrescrever para começar do zero."
        )
    feitos = carregar_feitos(arquivo) if args.continuar else {}
    if not args.continuar:
        arquivo.write_text("", encoding="utf-8")

    # Só reaproveita análises do mesmo modelo e da mesma versão do prompt: trocar qualquer
    # um dos dois muda o resultado, e misturar versões deixaria o relatório incoerente.
    outra_versao = [id_ for id_, item in feitos.items() if item.origem != origem]
    for id_ in outra_versao:
        del feitos[id_]
    # E só se o texto for o mesmo: se a avaliação foi editada, a análise antiga não vale.
    mudaram = [linha.id for linha in linhas if linha.id in feitos and feitos[linha.id].texto != linha.texto]
    for id_ in mudaram:
        del feitos[id_]
    pendentes = [linha for linha in linhas if linha.id not in feitos]
    if args.continuar:
        print(f"Continuando: {len(linhas) - len(pendentes)} já analisadas, {len(pendentes)} pendentes.")
        if outra_versao:
            print(f"Feitas com outro modelo ou outra versão do prompt, serão refeitas: {len(outra_versao)}")
        if mudaram:
            print(f"Texto alterado, serão refeitas: {', '.join(mudaram)}")
    return arquivo, feitos, pendentes


def chave_do_texto(texto: str) -> str:
    """Textos iguais a menos de maiúsculas e espaços ("Ótimo" e "ótimo ") têm a mesma análise."""
    return " ".join(texto.casefold().split())


LIMITE_FALHAS_SEGUIDAS = 5  # API fora do ar: para em vez de insistir em cada avaliação


def processar(
    pendentes: list[Linha], cliente, arquivo: Path, feitos: dict[str, AvaliacaoAnalisada], origem: str
) -> tuple[list[str], int]:
    """Analisa e grava uma por uma. Devolve (ids que falharam, código de saída se parou antes)."""
    falhas: list[str] = []
    seguidas = 0  # falhas seguidas da API (rede, sobrecarga): sinal de que ela está fora do ar
    ja_analisados = {chave_do_texto(item.texto): item.analise for item in feitos.values()}
    try:
        with arquivo.open("a", encoding="utf-8") as saida:
            for linha in pendentes:
                repetida = ja_analisados.get(chave_do_texto(linha.texto))
                if repetida is not None:  # texto igual a outro já analisado: não paga a API de novo
                    item = AvaliacaoAnalisada(id=linha.id, texto=linha.texto, analise=repetida, origem=origem)
                    saida.write(item.model_dump_json() + "\n")
                    saida.flush()
                    feitos[linha.id] = item
                    print(f"[{linha.id}] {repetida.sentimento.value:8} urgência {repetida.urgencia} (texto repetido)")
                    continue
                try:
                    analise = analisar(linha.texto, cliente)
                except ErroLLM as e:
                    if e.fatal:
                        erro(f"Erro: {e}. Parando: as próximas avaliações falhariam do mesmo jeito.")
                        return falhas, 1
                    falhas.append(linha.id)
                    erro(f"[{linha.id}] falhou: {e}")
                    seguidas += 1
                    if seguidas >= LIMITE_FALHAS_SEGUIDAS:
                        erro(
                            f"A API falhou em {seguidas} avaliações seguidas e parece fora do ar. "
                            "Parando para não insistir à toa; tente mais tarde com --continuar."
                        )
                        return falhas, 3
                    continue
                except ErroAnalise as e:  # a API respondeu: não é queda
                    falhas.append(linha.id)
                    erro(f"[{linha.id}] falhou: {e}")
                    seguidas = 0
                    continue
                seguidas = 0
                ja_analisados[chave_do_texto(linha.texto)] = analise
                item = AvaliacaoAnalisada(id=linha.id, texto=linha.texto, analise=analise, origem=origem)
                saida.write(item.model_dump_json() + "\n")
                saida.flush()  # gravado na hora: uma queda não perde o que já foi feito
                feitos[linha.id] = item
                print(f"[{linha.id}] {analise.sentimento.value:8} urgência {analise.urgencia}")
    except KeyboardInterrupt:
        erro("\nInterrompido.")
        return falhas, 130
    return falhas, 0


def cmd_analisar(args) -> int:
    linhas, avisos = ler_csv(args.entrada)
    for aviso in avisos:
        erro(f"aviso: {aviso}")
    cliente = criar_cliente(args)
    origem = origem_de(cliente)
    arquivo, feitos, pendentes = preparar_saida(args, linhas, origem)
    falhas, codigo = processar(pendentes, cliente, arquivo, feitos, origem)
    mostrar_uso(cliente)

    resultados = [feitos[linha.id] for linha in linhas if linha.id in feitos]
    (args.saida / "relatorio.md").write_text(gerar_markdown(resultados), encoding="utf-8")
    print(f"\n{len(resultados)}/{len(linhas)} avaliações analisadas. Resultados na pasta {args.saida}.")
    if len(resultados) < len(linhas):
        erro("Para tentar só as que faltam, rode o mesmo comando com --continuar.")
    if codigo:
        return codigo
    return 2 if falhas else 0


def mostrar_uso(cliente) -> None:
    uso = getattr(cliente, "uso", None)
    if uso and any(uso.values()):
        entrada, saida = (f"{uso[k]:,}".replace(",", ".") for k in ("input_tokens", "output_tokens"))
        print(f"Tokens gastos: {entrada} de entrada, {saida} de saída")


def _com_intervalo(acertos: int, total: int) -> str:
    """'83% (IC 95%: 66%–93%)': com 30 casos, a incerteza é grande e precisa aparecer."""
    if not total:
        return "sem casos"
    baixo, alto = intervalo_wilson(acertos, total)
    return f"{acertos / total:.0%} (IC 95%: {baixo:.0%}–{alto:.0%})"


def cmd_avaliar(args) -> int:
    cliente = criar_cliente(args)
    m = avaliar(cliente, args.gabarito)
    print(f"Origem: {origem_de(cliente)}")
    print(f"Avaliações no gabarito: {m.total} (falhas: {m.falhas})")
    print(f"Sentimento   acurácia {_com_intervalo(m.acertos_sentimento, m.total)}")
    print(
        f"Categorias   precisão {m.precisao_categorias:.0%} · revocação {m.revocacao_categorias:.0%}"
        f" · F1 {m.f1_categorias:.2f}"
    )
    print(
        f"Urgentes     encontradas {_com_intervalo(m.urgentes_achadas, m.urgentes_total)}"
        f" · alarmes corretos {m.precisao_urgentes:.0%}"
    )
    mostrar_uso(cliente)
    if m.erros:
        print("\nDiferenças em relação ao gabarito:")
        for e in m.erros:
            print(f"  - {e}")
    if args.salvar:
        registro = {
            "origem": origem_de(cliente),
            "data": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "gabarito": str(args.gabarito),
            "metricas": asdict(m),
            "tokens": getattr(cliente, "uso", None),
        }
        args.salvar.write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nResultado salvo em {args.salvar} (para comparar modelos e versões do prompt).")
    return 0


def tolerar_terminal_sem_acentos() -> None:
    """No Windows, com a saída redirecionada, o terminal usa cp1252: um emoji num id derrubaria
    o programa no meio do lote. Caracteres que não cabem viram "?" em vez de erro."""
    for fluxo in (sys.stdout, sys.stderr):
        if hasattr(fluxo, "reconfigure"):
            fluxo.reconfigure(errors="replace")


def main(argv: list[str] | None = None) -> int:
    tolerar_terminal_sem_acentos()
    p = argparse.ArgumentParser(prog="avaliacoes-ia", description="Analisa avaliações de restaurante com LLM.")
    sub = p.add_subparsers(dest="comando", required=True)

    a = sub.add_parser("analisar", help="analisa um CSV com colunas id e texto")
    a.add_argument("entrada", type=Path)
    a.add_argument("--saida", type=Path, default=Path("saida"), help="pasta dos resultados")
    modo = a.add_mutually_exclusive_group()
    modo.add_argument("--continuar", action="store_true", help="aproveita as análises já feitas na pasta de saída")
    modo.add_argument("--sobrescrever", action="store_true", help="apaga as análises anteriores e começa do zero")

    v = sub.add_parser("avaliar", help="mede o acerto contra o gabarito rotulado à mão")
    v.add_argument("--gabarito", type=Path, default=GABARITO)
    v.add_argument("--salvar", type=Path, default=None, help="grava as métricas em JSON para comparar depois")

    for s in (a, v):
        s.add_argument("--offline", action="store_true", help="usa regras locais em vez da API")
        s.add_argument("--modelo", default=None, help="modelo da API (Claude ou Gemini)")
        s.add_argument(
            "--provedor",
            choices=PROVEDORES,
            type=str.lower,  # "Gemini" e "GEMINI" também valem
            default=None,
            help="claude ou gemini. Sem ele: claude se houver ANTHROPIC_API_KEY, senão gemini",
        )
    args = p.parse_args(argv)
    if args.offline and (args.modelo or args.provedor):
        p.error("--modelo e --provedor só valem com a API; tire o --offline ou eles")

    try:
        return cmd_analisar(args) if args.comando == "analisar" else cmd_avaliar(args)
    except ErroEntrada as e:
        erro(f"Erro: {e}")
        return 1
    except ErroLLM as e:
        erro(f"Erro: {e}")
        return 1
    except OSError as e:  # pasta de saída é um arquivo, sem permissão, disco cheio...
        erro(f"Erro ao ler ou gravar arquivos: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
