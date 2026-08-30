"""Indexacao esparsa (BM25) sobre os mesmos chunks do indice denso.

`BM25Retriever` do LangChain roda inteiramente em memoria e nao tem
persistencia nativa -- serializamos via pickle para nao reprocessar o
corpus inteiro a cada consulta (ver CLAUDE.md > config B: "esparso
puro (BM25)").
"""

from __future__ import annotations

import pickle
from pathlib import Path

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src import config
from src.ingestion.chunking import Chunk

CAMINHO_PADRAO = config.BM25_INDEX_DIR / "indice.pkl"


def _documento(chunk: Chunk) -> Document:
    return Document(
        page_content=chunk.texto,
        metadata={
            "chunk_id": chunk.chunk_id,
            "norma": chunk.norma,
            "artigo": chunk.artigo,
            "paragrafo": chunk.paragrafo,
            "titulo_secao": chunk.titulo_secao,
            "pagina": chunk.pagina,
        },
    )


def construir_indice(chunks: list[Chunk]) -> BM25Retriever:
    return BM25Retriever.from_documents([_documento(c) for c in chunks])


def salvar_indice(retriever: BM25Retriever, caminho: Path = CAMINHO_PADRAO) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "wb") as arquivo:
        pickle.dump(retriever, arquivo)


def carregar_indice(caminho: Path = CAMINHO_PADRAO) -> BM25Retriever:
    with open(caminho, "rb") as arquivo:
        return pickle.load(arquivo)


def buscar(query: str, top_k: int = config.RETRIEVAL_TOP_K, caminho: Path = CAMINHO_PADRAO) -> list[dict]:
    """Busca esparsa BM25 (config B, ver CLAUDE.md). Exposta aqui para o
    teste de aceitacao da Fase 2; a Fase 3 reaproveita por baixo da
    interface comum de recuperacao."""
    retriever = carregar_indice(caminho)
    retriever.k = top_k
    documentos = retriever.invoke(query)
    return [{"chunk_id": doc.metadata["chunk_id"], "texto": doc.page_content, "metadata": doc.metadata} for doc in documentos]
