"""Indexacao esparsa (BM25) sobre os mesmos chunks do indice denso.

`BM25Retriever` do LangChain roda inteiramente em memoria e nao tem
persistencia nativa -- serializamos via pickle para nao reprocessar o
corpus inteiro a cada consulta (ver CLAUDE.md > config B: "esparso
puro (BM25)").
"""

from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src import config
from src.ingestion.chunking import Chunk

CAMINHO_PADRAO = config.BM25_INDEX_DIR / "indice.pkl"


def _resolver_caminho(caminho: Path | None) -> Path:
    # Nao usar CAMINHO_PADRAO como valor-padrao de parametro: defaults sao
    # resolvidos na DEFINICAO da funcao, entao um monkeypatch em
    # CAMINHO_PADRAO (comum em testes) nao teria efeito nenhum aqui --
    # bug real encontrado ao escrever os testes da Fase 3.
    return caminho if caminho is not None else CAMINHO_PADRAO


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


def salvar_indice(retriever: BM25Retriever, caminho: Path | None = None) -> None:
    caminho = _resolver_caminho(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "wb") as arquivo:
        pickle.dump(retriever, arquivo)


@lru_cache(maxsize=None)
def _carregar_indice_cacheado(caminho_str: str) -> BM25Retriever:
    with open(caminho_str, "rb") as arquivo:
        return pickle.load(arquivo)


def carregar_indice(caminho: Path | None = None) -> BM25Retriever:
    """Cacheado por caminho (mesmo padrao ja usado para o modelo de
    embeddings e o reranker, ver embeddings.py/reranqueado.py) --
    achado de performance: sem isso, buscar() desserializava o indice
    inteiro do disco A CADA chamada, unico componente sem cache entre
    os quatro modelos/indices carregados pelo agente."""
    return _carregar_indice_cacheado(str(_resolver_caminho(caminho)))


def _pontuar(retriever: BM25Retriever, query: str, top_k: int) -> list[dict]:
    consulta_processada = retriever.preprocess_func(query)
    scores = np.asarray(retriever.vectorizer.get_scores(consulta_processada))
    indices_top_k = np.argsort(scores)[::-1][:top_k]
    return [
        {
            "chunk_id": retriever.docs[i].metadata["chunk_id"],
            "texto": retriever.docs[i].page_content,
            "metadata": retriever.docs[i].metadata,
            "score": float(scores[i]),
        }
        for i in indices_top_k
    ]


def buscar(query: str, top_k: int = config.RETRIEVAL_TOP_K, caminho: Path | None = None) -> list[dict]:
    """Busca esparsa BM25 (config B, ver CLAUDE.md).

    `BM25Retriever.invoke()` do LangChain devolve so' os documentos, sem
    score -- por isso acessamos o vetorizador `rank_bm25.BM25Okapi`
    interno diretamente (`retriever.vectorizer.get_scores`) para expor um
    score real, necessario pela interface comum de recuperacao da Fase 3
    (ver src/retrieval/base.py).
    """
    return _pontuar(carregar_indice(caminho), query, top_k)


def buscar_lote(queries: list[str], top_k: int = config.RETRIEVAL_TOP_K, caminho: Path | None = None) -> list[list[dict]]:
    """Versao em lote de `buscar()`. Diferente da camada densa, o
    `rank_bm25` nao tem uma API vetorizada para multiplas queries de uma
    vez -- o ganho aqui vem inteiramente de `carregar_indice` ja estar
    cacheado (ver acima), nao de vetorizacao entre queries: e' so' um
    loop reaproveitando o MESMO indice carregado uma unica vez, em vez de
    N chamadas a `buscar()` que cada uma desserializaria o indice de novo
    se nao fosse o cache."""
    retriever = carregar_indice(caminho)
    return [_pontuar(retriever, query, top_k) for query in queries]
