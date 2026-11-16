"""Mede a qualidade da classificação contra um gabarito rotulado à mão.

- acurácia de sentimento: fração com o sentimento certo
- categorias: precisão (das marcadas, quantas estavam certas), revocação (das certas,
  quantas foram marcadas) e F1 (média harmônica das duas)
- urgentes: das avaliações que pedem contato hoje (urgência 4 ou 5), quantas o sistema achou
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ValidationError

from .analisador import ErroAnalise, analisar
from .entrada import ErroEntrada
from .llm import ClienteLLM, ErroLLM
from .modelos import Categoria, Sentimento


class CasoGabarito(BaseModel):
    """Valida o gabarito: uma categoria digitada errada ("atrazo") deixaria a métrica errada em silêncio."""

    id: str
    texto: str
    sentimento: Sentimento
    categorias: list[Categoria]
    urgente: bool


def carregar_gabarito(caminho: Path) -> list[CasoGabarito]:
    caminho = Path(caminho)
    try:
        dados = json.loads(caminho.read_text(encoding="utf-8"))
    except FileNotFoundError as erro:
        raise ErroEntrada(f"Gabarito não encontrado: {caminho}") from erro
    except (OSError, ValueError) as erro:
        raise ErroEntrada(f"Gabarito ilegível ({caminho}): {erro}") from erro
    if not isinstance(dados, list) or not dados:
        raise ErroEntrada("O gabarito precisa ser uma lista JSON com pelo menos um caso.")
    casos = []
    for n, dado in enumerate(dados, start=1):
        try:
            casos.append(CasoGabarito.model_validate(dado))
        except ValidationError as erro:
            detalhe = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in erro.errors())
            raise ErroEntrada(f"Caso {n} do gabarito inválido: {detalhe}") from erro
    ids = [c.id for c in casos]
    repetidos = sorted({i for i in ids if ids.count(i) > 1})
    if repetidos:
        raise ErroEntrada(f"IDs repetidos no gabarito: {', '.join(repetidos)}")
    return casos


def intervalo_wilson(acertos: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de confiança de 95% para uma proporção (método de Wilson).
    Funciona bem com poucos casos, ao contrário do "p ± 1,96·erro padrão"."""
    if not total:
        return 0.0, 1.0
    p = acertos / total
    centro = (p + z * z / (2 * total)) / (1 + z * z / total)
    margem = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return max(0.0, centro - margem), min(1.0, centro + margem)


def _dividir(a: float, b: float) -> float:
    return a / b if b else 0.0


@dataclass
class Metricas:
    total: int
    falhas: int
    acuracia_sentimento: float
    precisao_categorias: float
    revocacao_categorias: float
    f1_categorias: float
    revocacao_urgentes: float
    precisao_urgentes: float
    acertos_sentimento: int = 0
    urgentes_total: int = 0
    urgentes_achadas: int = 0
    erros: list[str] = field(default_factory=list)


def avaliar(cliente: ClienteLLM, caminho: Path) -> Metricas:
    casos = carregar_gabarito(caminho)
    sentimento_ok = falhas = 0
    vp = fp = fn = 0  # categorias
    u_vp = u_fp = u_fn = 0  # urgentes
    erros = []

    for caso in casos:
        esperadas = {c.value for c in caso.categorias}
        try:
            analise = analisar(caso.texto, cliente)
        except (ErroAnalise, ErroLLM) as erro:
            if isinstance(erro, ErroLLM) and erro.fatal:
                raise
            falhas += 1
            fn += len(esperadas)
            u_fn += caso.urgente
            erros.append(f"{caso.id}: falhou ({erro})")
            continue

        obtidas = {c.value for c in analise.categorias}
        urgente = analise.urgencia >= 4
        sentimento_ok += analise.sentimento is caso.sentimento
        vp += len(obtidas & esperadas)
        fp += len(obtidas - esperadas)
        fn += len(esperadas - obtidas)
        u_vp += urgente and caso.urgente
        u_fp += urgente and not caso.urgente
        u_fn += caso.urgente and not urgente

        diferencas = []
        if analise.sentimento is not caso.sentimento:
            diferencas.append(f"sentimento {analise.sentimento.value} (esperado {caso.sentimento.value})")
        if obtidas != esperadas:
            diferencas.append(f"categorias {sorted(obtidas)} (esperado {sorted(esperadas)})")
        if urgente != caso.urgente:
            diferencas.append(f"urgência {analise.urgencia}")
        if diferencas:
            erros.append(f"{caso.id} {caso.texto[:45]!r}: " + "; ".join(diferencas))

    precisao, revocacao = _dividir(vp, vp + fp), _dividir(vp, vp + fn)
    return Metricas(
        total=len(casos),
        falhas=falhas,
        acuracia_sentimento=_dividir(sentimento_ok, len(casos)),
        precisao_categorias=precisao,
        revocacao_categorias=revocacao,
        f1_categorias=_dividir(2 * precisao * revocacao, precisao + revocacao),
        revocacao_urgentes=_dividir(u_vp, u_vp + u_fn),
        precisao_urgentes=_dividir(u_vp, u_vp + u_fp),
        acertos_sentimento=sentimento_ok,
        urgentes_total=u_vp + u_fn,
        urgentes_achadas=u_vp,
        erros=erros,
    )
