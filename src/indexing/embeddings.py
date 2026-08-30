"""Wrapper fino sobre o modelo local de embeddings (BAAI/bge-m3).

Carregado uma unica vez por processo (lru_cache): o modelo tem ~4GB e
recarrega-lo a cada chamada tornaria indexacao e busca lentas sem
necessidade. `normalize_embeddings=True` faz a similaridade de cosseno
equivaler ao produto escalar -- usado pela config A (denso puro, ver
CLAUDE.md > "quatro configuracoes de recuperacao") e pelo Chroma, que
opera por produto escalar/distancia, nao cosseno bruto.
"""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from src import config


@lru_cache(maxsize=1)
def carregar_modelo() -> SentenceTransformer:
    return SentenceTransformer(str(config.EMBEDDING_MODEL_PATH))


def gerar_embeddings(textos: list[str]) -> list[list[float]]:
    modelo = carregar_modelo()
    return modelo.encode(textos, normalize_embeddings=True, show_progress_bar=False).tolist()
