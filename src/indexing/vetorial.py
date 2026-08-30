"""Indexacao densa: embeddings dos chunks persistidos no ChromaDB.

Chroma grava a dimensao do embedding na criacao da colecao -- trocar de
modelo de embedding exige apagar CHROMA_PERSIST_DIR e reindexar (ver
CLAUDE.md > Armadilhas conhecidas). Por isso a colecao carrega, em seus
proprios metadados, o nome do modelo e o hash do corpus indexado: serve
de registro de proveniencia para depois checar se um indice existente
ainda corresponde ao corpus/modelo atuais, em vez de deixar o Chroma
falhar so' com um erro de dimensao dificil de diagnosticar.

A colecao usa uma embedding function amarrada ao nosso modelo local
(_FuncaoEmbeddingBGE), nunca a funcao padrao do Chroma -- que baixaria
um modelo pela rede na primeira chamada, violando a restricao offline
(ver CLAUDE.md).

A colecao e' criada com `hnsw:space="cosine"` explicitamente. Sem isso o
Chroma usa distancia L2 ao quadrado por padrao: para vetores ja
normalizados (ver src/indexing/embeddings.py) a ORDEM dos vizinhos mais
proximos seria identica sob L2 ou cosseno (transformacao monotona), mas
o VALOR do score reportado nao seria "similaridade de cosseno" como a
config A pede literalmente (ver CLAUDE.md > "quatro configuracoes de
recuperacao") -- por isso configuramos o espaco de distancia em vez de
confiar so' na equivalencia de ordenacao.
"""

from __future__ import annotations

import hashlib

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from src import config
from src.indexing.embeddings import gerar_embeddings
from src.ingestion.chunking import Chunk

NOME_COLECAO = "normas"


class _FuncaoEmbeddingBGE(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        return gerar_embeddings(list(input))

    def name(self) -> str:
        return "bge-m3-local"


def hash_corpus(chunks: list[Chunk]) -> str:
    """Hash deterministico do conteudo indexado -- registrado a cada
    execucao de experimento para reprodutibilidade (ver CLAUDE.md >
    Avaliacao: "experimento nao reprodutivel nao vale para o TFG")."""
    h = hashlib.sha256()
    for chunk in sorted(chunks, key=lambda c: c.chunk_id):
        h.update(chunk.chunk_id.encode("utf-8"))
        h.update(chunk.texto.encode("utf-8"))
    return h.hexdigest()


def obter_cliente() -> chromadb.ClientAPI:
    # anonymized_telemetry=False: sem isso o Chroma tenta enviar eventos a
    # um endpoint externo (posthog) a cada operacao -- viola a restricao
    # 100% offline (ver CLAUDE.md), mesmo que a tentativa falhe por engano.
    return chromadb.PersistentClient(
        path=str(config.CHROMA_PERSIST_DIR),
        settings=chromadb.Settings(anonymized_telemetry=False),
    )


def _metadados_chunk(chunk: Chunk) -> dict:
    # Chroma nao aceita None em metadados -- campos ausentes viram "".
    return {
        "chunk_id": chunk.chunk_id,
        "norma": chunk.norma,
        "artigo": chunk.artigo,
        "paragrafo": chunk.paragrafo or "",
        "titulo_secao": chunk.titulo_secao or "",
        "pagina": chunk.pagina,
    }


def indexar(chunks: list[Chunk], recriar: bool = True) -> chromadb.Collection:
    cliente = obter_cliente()
    if recriar:
        try:
            cliente.delete_collection(NOME_COLECAO)
        except (ValueError, chromadb.errors.NotFoundError):
            pass

    colecao = cliente.get_or_create_collection(
        NOME_COLECAO,
        embedding_function=_FuncaoEmbeddingBGE(),
        metadata={
            "hnsw:space": "cosine",
            "modelo_embedding": "bge-m3",
            "hash_corpus": hash_corpus(chunks),
        },
    )
    if not chunks:
        return colecao

    textos = [c.texto for c in chunks]
    embeddings = gerar_embeddings(textos)
    colecao.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        documents=textos,
        metadatas=[_metadados_chunk(c) for c in chunks],
    )
    return colecao


def obter_colecao() -> chromadb.Collection:
    return obter_cliente().get_collection(NOME_COLECAO, embedding_function=_FuncaoEmbeddingBGE())


def buscar(query: str, top_k: int = config.RETRIEVAL_TOP_K) -> list[dict]:
    """Busca densa por similaridade de cosseno (config A, ver CLAUDE.md).

    Exposta aqui para o teste de aceitacao da Fase 2; a Fase 3 reaproveita
    isso por baixo da interface comum de recuperacao.
    """
    colecao = obter_colecao()
    embedding_consulta = gerar_embeddings([query])[0]
    resultado = colecao.query(query_embeddings=[embedding_consulta], n_results=top_k)
    return [
        {"chunk_id": id_, "texto": texto, "metadata": metadata, "distancia": distancia}
        for id_, texto, metadata, distancia in zip(
            resultado["ids"][0],
            resultado["documents"][0],
            resultado["metadatas"][0],
            resultado["distances"][0],
        )
    ]
