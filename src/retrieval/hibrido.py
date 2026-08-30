"""Config C: hibrido (BM25 + denso), fusao por Reciprocal Rank Fusion.

RRF (Cormack, Clarke & Buettcher, 2009) funde listas por RANKING, nao por
score bruto -- BM25 e similaridade de cosseno vivem em escalas
incompativeis, e RRF evita ter que normaliza-las artificialmente para
torna-las comparaveis. Formula: para cada documento d,
score_rrf(d) = soma, para cada lista em que d aparece, de 1/(k_rrf + rank(d)).
"""

from __future__ import annotations

from src import config
from src.retrieval.base import ResultadoRecuperacao
from src.retrieval.denso import RetrieverDenso
from src.retrieval.esparso import RetrieverEsparso


def fundir_rrf(
    listas: list[list[ResultadoRecuperacao]], top_k: int, k_rrf: int = config.RRF_K
) -> list[ResultadoRecuperacao]:
    pontuacao_rrf: dict[str, float] = {}
    resultado_por_id: dict[str, ResultadoRecuperacao] = {}

    for lista in listas:
        for rank, resultado in enumerate(lista, start=1):
            pontuacao_rrf[resultado.chunk_id] = pontuacao_rrf.get(resultado.chunk_id, 0.0) + 1 / (k_rrf + rank)
            resultado_por_id.setdefault(resultado.chunk_id, resultado)

    top_ids = sorted(pontuacao_rrf, key=lambda chunk_id: pontuacao_rrf[chunk_id], reverse=True)[:top_k]
    return [
        ResultadoRecuperacao(
            chunk_id=chunk_id,
            texto=resultado_por_id[chunk_id].texto,
            metadata=resultado_por_id[chunk_id].metadata,
            score=pontuacao_rrf[chunk_id],
        )
        for chunk_id in top_ids
    ]


class RetrieverHibrido:
    def __init__(self, k_rrf: int = config.RRF_K, tamanho_pool: int = config.RETRIEVAL_POOL_SIZE):
        self._denso = RetrieverDenso()
        self._esparso = RetrieverEsparso()
        self.k_rrf = k_rrf
        self.tamanho_pool = tamanho_pool

    def buscar(self, query: str, top_k: int = config.RETRIEVAL_TOP_K) -> list[ResultadoRecuperacao]:
        candidatos_densos = self._denso.buscar(query, top_k=self.tamanho_pool)
        candidatos_esparsos = self._esparso.buscar(query, top_k=self.tamanho_pool)
        return fundir_rrf([candidatos_densos, candidatos_esparsos], top_k=top_k, k_rrf=self.k_rrf)
