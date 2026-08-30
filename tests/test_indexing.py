import pytest

from src import config
from src.indexing import esparso, vetorial
from src.ingestion.chunking import Chunk


def _chunk(chunk_id: str, texto: str, artigo: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        texto=texto,
        norma="Norma Teste",
        artigo=artigo,
        paragrafo=None,
        titulo_secao=None,
        pagina=1,
    )


CHUNKS_SINTETICOS = [
    _chunk("t1", "Toda validação de processo deve ser documentada e aprovada pela qualidade.", "10"),
    _chunk("t2", "O treinamento de pessoal deve ser registrado e avaliado periodicamente.", "20"),
    _chunk("t3", "Os equipamentos de fabricação devem passar por qualificação antes do uso.", "30"),
]


@pytest.fixture
def chroma_isolado(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHROMA_PERSIST_DIR", tmp_path / "chroma_teste")


@pytest.fixture
def caminho_bm25_isolado(tmp_path):
    return tmp_path / "bm25_teste" / "indice.pkl"


def test_indexar_e_buscar_denso_retorna_chunk_relevante(chroma_isolado):
    vetorial.indexar(CHUNKS_SINTETICOS, recriar=True)
    resultados = vetorial.buscar("qualificação de equipamentos de produção", top_k=1)
    assert resultados[0]["chunk_id"] == "t3"


def test_indexar_denso_registra_proveniencia_nos_metadados_da_colecao(chroma_isolado):
    vetorial.indexar(CHUNKS_SINTETICOS, recriar=True)
    colecao = vetorial.obter_colecao()
    assert colecao.metadata["hash_corpus"] == vetorial.hash_corpus(CHUNKS_SINTETICOS)
    assert colecao.metadata["modelo_embedding"] == "bge-m3"


def test_hash_corpus_e_deterministico_e_sensivel_ao_conteudo():
    hash_a = vetorial.hash_corpus(CHUNKS_SINTETICOS)
    hash_b = vetorial.hash_corpus(list(reversed(CHUNKS_SINTETICOS)))
    assert hash_a == hash_b  # ordem nao deve importar (sorted por chunk_id)

    alterado = [_chunk("t1", "texto diferente", "10"), *CHUNKS_SINTETICOS[1:]]
    assert vetorial.hash_corpus(alterado) != hash_a


def test_construir_e_buscar_esparso_retorna_chunk_relevante(caminho_bm25_isolado):
    retriever = esparso.construir_indice(CHUNKS_SINTETICOS)
    esparso.salvar_indice(retriever, caminho=caminho_bm25_isolado)
    resultados = esparso.buscar("treinamento de pessoal", top_k=1, caminho=caminho_bm25_isolado)
    assert resultados[0]["chunk_id"] == "t2"


def test_nenhuma_chamada_de_rede_ocorre_durante_indexacao_e_busca(chroma_isolado, monkeypatch):
    """Ver CLAUDE.md > restricao offline: monkeypatch em socket.socket
    levantando excecao durante uma operacao completa de indexacao+busca."""
    import socket

    def _bloqueado(self, *args, **kwargs):
        raise RuntimeError("tentativa de acesso de rede durante operacao offline")

    monkeypatch.setattr(socket.socket, "connect", _bloqueado)

    vetorial.indexar(CHUNKS_SINTETICOS, recriar=True)
    resultados = vetorial.buscar("validação de processo documentada", top_k=1)
    assert resultados[0]["chunk_id"] == "t1"


class TestBuscaNoCorpusRealJaIndexado:
    """Depende de 'python -m src.indexing.build' ja ter sido executado
    contra o corpus real (nao reindexa aqui para o teste rodar rapido)."""

    @pytest.fixture(autouse=True)
    def _requer_indices_reais(self):
        indice_denso = config.CHROMA_PERSIST_DIR / "chroma.sqlite3"
        indice_esparso = esparso.CAMINHO_PADRAO
        if not indice_denso.exists() or not indice_esparso.exists():
            pytest.skip(
                "indices reais nao encontrados -- rode 'python -m src.indexing.build' "
                "com o corpus completo antes deste teste"
            )

    def test_busca_densa_encontra_o_artigo_1_da_rdc_658_para_pergunta_sobre_objetivo(self):
        resultados = vetorial.buscar(
            "qual o objetivo das boas práticas de fabricação de medicamentos?", top_k=3
        )
        assert any(
            r["metadata"]["norma"] == "RDC 658/2022" and r["metadata"]["artigo"] == "1"
            for r in resultados
        )

    def test_busca_esparsa_retorna_resultados_para_termo_normativo_conhecido(self):
        resultados = esparso.buscar("boas práticas de fabricação", top_k=3)
        assert len(resultados) == 3
        assert all(r["metadata"]["norma"] for r in resultados)
