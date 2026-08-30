"""Seleciona a estrategia de recuperacao por identificador de config
(A/B/C/D, ver CLAUDE.md). Usado pela Fase 5 (avaliacao) para trocar de
estrategia so' via config, sem tocar em codigo -- exigencia do TFG.
"""

from __future__ import annotations

from src import config
from src.retrieval.base import Retriever
from src.retrieval.denso import RetrieverDenso
from src.retrieval.esparso import RetrieverEsparso
from src.retrieval.hibrido import RetrieverHibrido
from src.retrieval.reranqueado import RetrieverReranqueado

_ESTRATEGIAS = {
    "A": RetrieverDenso,
    "B": RetrieverEsparso,
    "C": RetrieverHibrido,
    "D": RetrieverReranqueado,
}


def criar_retriever(estrategia: str) -> Retriever:
    construtor = _ESTRATEGIAS.get(estrategia)
    if construtor is None:
        raise ValueError(
            f"estrategia de recuperacao desconhecida: {estrategia!r} "
            f"(validas: {', '.join(config.RETRIEVAL_STRATEGIES)})"
        )
    return construtor()
