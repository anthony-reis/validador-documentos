"""Config A: denso puro, por similaridade de cosseno (ver CLAUDE.md)."""

from __future__ import annotations

from src import config
from src.indexing import vetorial
from src.retrieval.base import ResultadoRecuperacao


class RetrieverDenso:
    def buscar(self, query: str, top_k: int = config.RETRIEVAL_TOP_K) -> list[ResultadoRecuperacao]:
        resultados = vetorial.buscar(query, top_k=top_k)
        return [
            ResultadoRecuperacao(
                chunk_id=r["chunk_id"],
                texto=r["texto"],
                metadata=r["metadata"],
                # colecao criada com hnsw:space="cosine" (ver vetorial.py):
                # distancia de cosseno -> similaridade = 1 - distancia.
                score=1 - r["distancia"],
            )
            for r in resultados
        ]
