"""Esquema da análise que o LLM precisa devolver.

A validação é rígida no que decide algo (sentimento, categorias, urgência)
e tolerante no que é só forma: "Negativo", "comida fria" e um resumo longo
são aceitos e normalizados, em vez de descartar uma análise correta.
"""

import re
from enum import Enum

from pydantic import BaseModel, Field, field_validator

from .normalizacao import sem_acento

LIMITE_RESUMO = 200


class Sentimento(str, Enum):
    POSITIVO = "positivo"
    NEUTRO = "neutro"
    NEGATIVO = "negativo"


class Categoria(str, Enum):
    ATRASO = "atraso"
    COMIDA_FRIA = "comida_fria"
    PEDIDO_ERRADO = "pedido_errado"
    QUALIDADE = "qualidade"
    EMBALAGEM = "embalagem"
    PRECO = "preco"
    ATENDIMENTO = "atendimento"
    ELOGIO = "elogio"


# Alguns modelos (o Gemini, às vezes) respondem o sentimento em inglês mesmo com o prompt em
# português. É só forma: "negative" e "negativo" dizem a mesma coisa.
SENTIMENTO_EM_INGLES = {"positive": "positivo", "neutral": "neutro", "negative": "negativo"}


def _normalizar(valor):
    """'Comida Fria' -> 'comida_fria', 'Preço' -> 'preco', 'negativo.' -> 'negativo'."""
    if isinstance(valor, str):
        return re.sub(r"\W+", "_", sem_acento(valor).lower()).strip("_")
    return valor


class Analise(BaseModel):
    sentimento: Sentimento
    categorias: list[Categoria] = Field(default_factory=list)
    urgencia: int = Field(ge=1, le=5, description="1 = nada a fazer, 5 = resolver hoje")
    resumo: str = Field(min_length=1, max_length=LIMITE_RESUMO)

    @field_validator("sentimento", mode="before")
    @classmethod
    def _sentimento(cls, valor):
        valor = _normalizar(valor)
        return SENTIMENTO_EM_INGLES.get(valor, valor) if isinstance(valor, str) else valor

    @field_validator("urgencia", mode="before")
    @classmethod
    def _urgencia(cls, valor):
        if isinstance(valor, bool):  # em Python, True vale 1: sem isso, "urgencia": true viraria 1
            raise ValueError("urgência precisa ser um número de 1 a 5")
        return valor

    @field_validator("categorias", mode="before")
    @classmethod
    def _categorias(cls, valor):
        if valor is None:
            return []
        if isinstance(valor, str):
            valor = [valor]
        if isinstance(valor, list):
            # Sem repetição e na ordem. Não usa set/dict: um item que não é texto (ex.: [["atraso"]])
            # quebraria com TypeError; assim ele chega ao pydantic e vira um erro de validação normal.
            unicos: list = []
            for item in map(_normalizar, valor):
                if item not in unicos:
                    unicos.append(item)
            return unicos
        return valor

    @field_validator("resumo", mode="before")
    @classmethod
    def _resumo(cls, valor):
        if isinstance(valor, str):
            valor = " ".join(valor.split())  # tira quebras de linha e espaços duplos
            if len(valor) > LIMITE_RESUMO:
                valor = valor[: LIMITE_RESUMO - 1].rstrip() + "…"
        return valor


class AvaliacaoAnalisada(BaseModel):
    id: str
    texto: str
    analise: Analise
    origem: str | None = None  # modelo + versão do prompt que fizeram a análise
