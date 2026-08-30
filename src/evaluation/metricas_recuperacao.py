"""Metricas de recuperacao (Fase 5) -- deterministicas, sem LLM.

Ver CLAUDE.md > Avaliacao: metricas deterministicas contra um ground
truth anotado manualmente sao mais defensaveis numa banca do que "um
LLM avaliando outro LLM". Relevancia e' binaria: um chunk_id esta ou nao
esta na lista de chunks_relevantes do ground truth -- o ground truth nao
anota graus de relevancia, entao nDCG aqui usa ganho binario (0/1), nao
graduado.
"""

from __future__ import annotations

import math


def precision_em_k(recuperados: list[str], relevantes: set[str], k: int) -> float:
    top_k = recuperados[:k]
    if not top_k:
        return 0.0
    acertos = sum(1 for chunk_id in top_k if chunk_id in relevantes)
    return acertos / len(top_k)


def recall_em_k(recuperados: list[str], relevantes: set[str], k: int) -> float:
    if not relevantes:
        return 0.0
    top_k = recuperados[:k]
    acertos = sum(1 for chunk_id in top_k if chunk_id in relevantes)
    return acertos / len(relevantes)


def reciprocal_rank(recuperados: list[str], relevantes: set[str]) -> float:
    for posicao, chunk_id in enumerate(recuperados, start=1):
        if chunk_id in relevantes:
            return 1 / posicao
    return 0.0


def ndcg_em_k(recuperados: list[str], relevantes: set[str], k: int) -> float:
    top_k = recuperados[:k]
    dcg = sum(
        (1.0 if chunk_id in relevantes else 0.0) / math.log2(posicao + 1)
        for posicao, chunk_id in enumerate(top_k, start=1)
    )
    numero_relevantes_no_topo = min(len(relevantes), k)
    idcg = sum(1 / math.log2(posicao + 1) for posicao in range(1, numero_relevantes_no_topo + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg
