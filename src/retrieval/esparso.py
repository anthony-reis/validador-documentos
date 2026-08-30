"""Config B: esparso puro, BM25 (ver CLAUDE.md)."""

from __future__ import annotations

from src import config
from src.indexing import esparso as indice_esparso
from src.retrieval.base import ResultadoRecuperacao


def _converter(r: dict) -> ResultadoRecuperacao:
    return ResultadoRecuperacao(chunk_id=r["chunk_id"], texto=r["texto"], metadata=r["metadata"], score=r["score"])


class RetrieverEsparso:
    def buscar(self, query: str, top_k: int = config.RETRIEVAL_TOP_K) -> list[ResultadoRecuperacao]:
        return [_converter(r) for r in indice_esparso.buscar(query, top_k=top_k)]

    def buscar_lote(self, queries: list[str], top_k: int = config.RETRIEVAL_TOP_K) -> list[list[ResultadoRecuperacao]]:
        """Ver CLAUDE.md > "Melhorias de performance": reaproveita o
        indice BM25 ja cacheado (carregar_indice) para todas as queries,
        em vez de recarrega-lo por assercao."""
        return [
            [_converter(r) for r in resultados] for resultados in indice_esparso.buscar_lote(queries, top_k=top_k)
        ]
