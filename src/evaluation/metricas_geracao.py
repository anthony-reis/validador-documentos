"""Metricas de geracao (Fase 5). Ver CLAUDE.md > Avaliacao.

Fidelidade de citacao NAO reprocessa a verificacao de substring aqui --
ela ja acontece dentro do proprio agente a cada julgamento (ver
src/agent/nos.py::julgar_assercao). Essa metrica so' agrega o resultado
dessa verificacao ja' feita: entre os julgamentos que tinham uma
citacao para checar, qual fracao passou. Reprocessar o texto do chunk
aqui de novo seria trabalho duplicado e um segundo lugar pra esse
criterio divergir do que o agente realmente aplica em producao.
"""

from __future__ import annotations

from src.agent.nos import MOTIVO_CITACAO_NAO_ENCONTRADA
from src.agent.schemas import JulgamentoAssercao


def taxa_de_acerto(julgamentos: list[JulgamentoAssercao], veredito_esperado: list[str]) -> float:
    if not julgamentos:
        return 0.0
    acertos = sum(1 for julgamento, esperado in zip(julgamentos, veredito_esperado) if julgamento.veredito == esperado)
    return acertos / len(julgamentos)


def taxa_de_abstencao(julgamentos: list[JulgamentoAssercao]) -> float:
    if not julgamentos:
        return 0.0
    return sum(1 for julgamento in julgamentos if julgamento.veredito == "INDETERMINADO") / len(julgamentos)


def fidelidade_de_citacao(julgamentos: list[JulgamentoAssercao]) -> float:
    """Fracao de julgamentos com citacao (ou seja, a recuperacao trouxe
    algo acima do limiar) cuja citacao passou na verificacao
    deterministica de fidelidade do agente -- nao foi rebaixada por nao
    corresponder literalmente ao chunk recuperado."""
    elegiveis = [julgamento for julgamento in julgamentos if julgamento.citacao is not None]
    if not elegiveis:
        return 0.0
    fieis = sum(1 for julgamento in elegiveis if julgamento.motivo_abstencao != MOTIVO_CITACAO_NAO_ENCONTRADA)
    return fieis / len(elegiveis)
