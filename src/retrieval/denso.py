"""Config A: denso puro, por similaridade de cosseno (ver CLAUDE.md)."""

from __future__ import annotations

from src import config
from src.indexing import vetorial
from src.retrieval.base import ResultadoRecuperacao


def _converter(r: dict) -> ResultadoRecuperacao:
    return ResultadoRecuperacao(
        chunk_id=r["chunk_id"],
        texto=r["texto"],
        metadata=r["metadata"],
        # colecao criada com hnsw:space="cosine" (ver vetorial.py):
        # distancia de cosseno -> similaridade = 1 - distancia.
        score=1 - r["distancia"],
    )


class RetrieverDenso:
    def buscar(self, query: str, top_k: int = config.RETRIEVAL_TOP_K) -> list[ResultadoRecuperacao]:
        return [_converter(r) for r in vetorial.buscar(query, top_k=top_k)]

    def buscar_lote(self, queries: list[str], top_k: int = config.RETRIEVAL_TOP_K) -> list[list[ResultadoRecuperacao]]:
        """Ver CLAUDE.md > "Melhorias de performance": uma unica chamada
        de embedding+Chroma para todas as queries, em vez de uma por
        assercao."""
        return [[_converter(r) for r in resultados] for resultados in vetorial.buscar_lote(queries, top_k=top_k)]
