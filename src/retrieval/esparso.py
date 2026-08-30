"""Config B: esparso puro, BM25 (ver CLAUDE.md)."""

from __future__ import annotations

from src import config
from src.indexing import esparso as indice_esparso
from src.retrieval.base import ResultadoRecuperacao


class RetrieverEsparso:
    def buscar(self, query: str, top_k: int = config.RETRIEVAL_TOP_K) -> list[ResultadoRecuperacao]:
        resultados = indice_esparso.buscar(query, top_k=top_k)
        return [
            ResultadoRecuperacao(
                chunk_id=r["chunk_id"], texto=r["texto"], metadata=r["metadata"], score=r["score"]
            )
            for r in resultados
        ]
