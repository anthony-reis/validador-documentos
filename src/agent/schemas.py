"""Schemas Pydantic do agente -- saida estruturada obrigatoria (ver
CLAUDE.md: "veredito sem citacao rastreavel e' bug, nao estilo").
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Veredito = Literal["CONFORME", "NAO_CONFORME", "NAO_APLICAVEL", "INDETERMINADO"]


class JulgamentoLLM(BaseModel):
    """Saida bruta do LLM para uma (assercao, contexto) -- antes da
    auto-verificacao deterministica (ver src/agent/nos.py)."""

    veredito: Veredito
    trecho_citado: str = Field(
        description="Trecho literal do CONTEXTO NORMATIVO que sustenta o veredito, "
        "copiado EXATAMENTE como aparece no contexto, sem parafrasear."
    )
    justificativa: str


class AssercoesExtraidas(BaseModel):
    """Saida do no' de extracao de assercoes (ver src/agent/nos.py) --
    heuristica best-effort, NAO e' a fonte dos numeros do TFG (a
    avaliacao da Fase 5 usa um gabarito curado manualmente, ver
    CLAUDE.md > Avaliacao)."""

    assercoes: list[str] = Field(
        description="Afirmacoes verificaveis extraidas do texto (uma por item). "
        "Lista vazia se o trecho nao contiver nenhuma afirmacao relevante para "
        "conformidade regulatoria (ex.: cabecalhos, tabelas de metadados)."
    )


class Citacao(BaseModel):
    norma: str
    artigo: str
    paragrafo: str | None
    titulo_secao: str | None
    trecho_literal: str
    chunk_id: str
    pagina: int
    score_recuperacao: float


class JulgamentoAssercao(BaseModel):
    assercao: str
    veredito: Veredito
    citacao: Citacao | None
    justificativa: str
    motivo_abstencao: str | None = None
    """Preenchido quando a auto-verificacao deterministica rebaixa o
    veredito do LLM para INDETERMINADO (ver CLAUDE.md > regra de
    abstencao) -- explica POR QUE se absteve, distinto da justificativa
    do proprio LLM."""


class RelatorioConformidade(BaseModel):
    documento_analisado: str
    estrategia_recuperacao: Literal["A", "B", "C", "D"]
    modelo_llm: str
    hash_corpus: str
    gerado_em: datetime
    julgamentos: list[JulgamentoAssercao]

    @property
    def contagem_por_veredito(self) -> dict[str, int]:
        contagem: dict[str, int] = {}
        for julgamento in self.julgamentos:
            contagem[julgamento.veredito] = contagem.get(julgamento.veredito, 0) + 1
        return contagem
