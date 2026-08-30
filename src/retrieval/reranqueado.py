"""Config D: hibrido (config C) + rerank por cross-encoder (bge-reranker-v2-m3).

Cross-encoders pontuam o par (query, chunk) diretamente e sao mais
precisos que bi-encoders, mas muito mais caros computacionalmente --
por isso so' rodam sobre o pool ja filtrado pela fusao hibrida, nunca
sobre o corpus inteiro.
"""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

from src import config
from src.retrieval.base import ResultadoRecuperacao
from src.retrieval.hibrido import RetrieverHibrido


@lru_cache(maxsize=1)
def _carregar_reranker() -> CrossEncoder:
    return CrossEncoder(str(config.RERANKER_MODEL_PATH))


class RetrieverReranqueado:
    def __init__(self, tamanho_pool: int = config.RETRIEVAL_POOL_SIZE):
        self._hibrido = RetrieverHibrido(tamanho_pool=tamanho_pool)
        self.tamanho_pool = tamanho_pool

    def buscar(self, query: str, top_k: int = config.RETRIEVAL_TOP_K) -> list[ResultadoRecuperacao]:
        candidatos = self._hibrido.buscar(query, top_k=self.tamanho_pool)
        if not candidatos:
            return []

        pares = [(query, candidato.texto) for candidato in candidatos]
        scores = _carregar_reranker().predict(pares)
        reordenados = sorted(zip(candidatos, scores), key=lambda par: par[1], reverse=True)[:top_k]

        return [
            ResultadoRecuperacao(
                chunk_id=candidato.chunk_id,
                texto=candidato.texto,
                metadata=candidato.metadata,
                score=float(score),
            )
            for candidato, score in reordenados
        ]
