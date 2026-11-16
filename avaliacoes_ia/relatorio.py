"""Agrega as análises em um relatório em Markdown."""

from collections import Counter

from .modelos import AvaliacaoAnalisada, Categoria, Sentimento


def _celula(texto: str) -> str:
    """Texto seguro dentro de uma tabela Markdown: "|" e quebras de linha quebrariam a tabela."""
    return " ".join(str(texto).split()).replace("|", "\\|")


def porcentagens(quantidades: list[int]) -> list[int]:
    """Arredonda para inteiros que somam exatamente 100 (método do maior resto).
    Com arredondamento simples, três terços dariam 33% + 33% + 33% = 99%."""
    total = sum(quantidades)
    if not total:
        return [0] * len(quantidades)
    exatas = [100 * q / total for q in quantidades]
    inteiras = [int(e) for e in exatas]
    maiores_restos = sorted(range(len(exatas)), key=lambda i: exatas[i] - inteiras[i], reverse=True)
    for i in maiores_restos[: 100 - sum(inteiras)]:
        inteiras[i] += 1
    return inteiras


def resumir(itens: list[AvaliacaoAnalisada]) -> dict:
    sentimentos = Counter(i.analise.sentimento for i in itens)
    problemas = Counter(c.value for i in itens for c in i.analise.categorias if c is not Categoria.ELOGIO)
    urgentes = sorted((i for i in itens if i.analise.urgencia >= 4), key=lambda i: -i.analise.urgencia)
    return {
        "total": len(itens),
        "sentimentos": {s.value: sentimentos.get(s, 0) for s in Sentimento},
        "problemas": problemas.most_common(),
        "urgentes": urgentes,
    }


def gerar_markdown(itens: list[AvaliacaoAnalisada]) -> str:
    r = resumir(itens)
    pcts = porcentagens(list(r["sentimentos"].values()))
    linhas = ["# Relatório de avaliações", "", f"Avaliações analisadas: **{r['total']}**", "", "## Sentimento", ""]
    linhas += ["| Sentimento | Qtde | % |", "| --- | ---: | ---: |"]
    sentimentos = zip(r["sentimentos"].items(), pcts, strict=True)
    linhas += [f"| {nome} | {qtde} | {pct}% |" for (nome, qtde), pct in sentimentos]

    linhas += ["", "## Problemas mais citados", ""]
    if r["problemas"]:
        linhas += ["| Problema | Citações |", "| --- | ---: |"]
        linhas += [f"| {nome} | {qtde} |" for nome, qtde in r["problemas"]]
    else:
        linhas.append("Nenhum problema citado.")

    linhas += ["", "## Para contatar hoje (urgência 4 ou 5)", ""]
    if r["urgentes"]:
        linhas += ["| ID | Urgência | Resumo |", "| --- | ---: | --- |"]
        linhas += [f"| {_celula(i.id)} | {i.analise.urgencia} | {_celula(i.analise.resumo)} |" for i in r["urgentes"]]
    else:
        linhas.append("Nenhuma avaliação urgente.")
    return "\n".join(linhas) + "\n"
