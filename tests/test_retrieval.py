import pytest

from src import config
from src.indexing import esparso as indice_esparso
from src.indexing import vetorial
from src.ingestion.chunking import Chunk
from src.retrieval.base import ResultadoRecuperacao
from src.retrieval.denso import RetrieverDenso
from src.retrieval.esparso import RetrieverEsparso
from src.retrieval.factory import criar_retriever
from src.retrieval.hibrido import RetrieverHibrido, fundir_rrf
from src.retrieval.reranqueado import RetrieverReranqueado


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
def indices_isolados(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHROMA_PERSIST_DIR", tmp_path / "chroma_teste")
    caminho_bm25 = tmp_path / "bm25_teste" / "indice.pkl"
    monkeypatch.setattr(indice_esparso, "CAMINHO_PADRAO", caminho_bm25)

    vetorial.indexar(CHUNKS_SINTETICOS, recriar=True)
    retriever_bm25 = indice_esparso.construir_indice(CHUNKS_SINTETICOS)
    indice_esparso.salvar_indice(retriever_bm25, caminho=caminho_bm25)


def test_retriever_denso_retorna_similaridade_de_cosseno_em_0_1(indices_isolados):
    resultados = RetrieverDenso().buscar("qualificação de equipamentos de produção", top_k=1)
    assert resultados[0].chunk_id == "t3"
    assert 0.0 <= resultados[0].score <= 1.0


def test_retriever_esparso_retorna_chunk_relevante_com_score_bm25(indices_isolados):
    resultados = RetrieverEsparso().buscar("treinamento de pessoal", top_k=1)
    assert resultados[0].chunk_id == "t2"
    assert resultados[0].score > 0


def test_retriever_hibrido_encontra_chunk_relevante_via_fusao(indices_isolados):
    resultados = RetrieverHibrido().buscar("validação de processo documentada", top_k=1)
    assert resultados[0].chunk_id == "t1"


def test_retriever_reranqueado_encontra_chunk_relevante(indices_isolados):
    resultados = RetrieverReranqueado().buscar("qualificação de equipamentos", top_k=1)
    assert resultados[0].chunk_id == "t3"


QUERIES_LOTE = [
    "qualificação de equipamentos de produção",
    "treinamento de pessoal",
    "validação de processo documentada",
]


class TestBuscarLoteEquivaleAChamadasIndividuais:
    """Ver CLAUDE.md > "Melhorias de performance": buscar_lote() precisa
    devolver o mesmo resultado que N chamadas de buscar(), so' que com
    menos round-trips de embedding/Chroma/reranker."""

    def _comparar(self, retriever, queries):
        resultado_lote = retriever.buscar_lote(queries, top_k=2)
        resultado_individual = [retriever.buscar(q, top_k=2) for q in queries]
        assert len(resultado_lote) == len(resultado_individual) == len(queries)
        for lote, individual in zip(resultado_lote, resultado_individual):
            assert [r.chunk_id for r in lote] == [r.chunk_id for r in individual]

    def test_denso(self, indices_isolados):
        self._comparar(RetrieverDenso(), QUERIES_LOTE)

    def test_esparso(self, indices_isolados):
        self._comparar(RetrieverEsparso(), QUERIES_LOTE)

    def test_hibrido(self, indices_isolados):
        self._comparar(RetrieverHibrido(), QUERIES_LOTE)

    def test_reranqueado(self, indices_isolados):
        self._comparar(RetrieverReranqueado(), QUERIES_LOTE)

    def test_lote_vazio_nao_quebra(self, indices_isolados):
        for retriever in (RetrieverDenso(), RetrieverEsparso(), RetrieverHibrido(), RetrieverReranqueado()):
            assert retriever.buscar_lote([], top_k=2) == []


def test_factory_cria_a_estrategia_correta_por_letra(indices_isolados):
    assert isinstance(criar_retriever("A"), RetrieverDenso)
    assert isinstance(criar_retriever("B"), RetrieverEsparso)
    assert isinstance(criar_retriever("C"), RetrieverHibrido)
    assert isinstance(criar_retriever("D"), RetrieverReranqueado)


def test_factory_recusa_estrategia_desconhecida():
    with pytest.raises(ValueError, match="desconhecida"):
        criar_retriever("Z")


class TestFusaoRRF:
    def _resultado(self, chunk_id: str) -> ResultadoRecuperacao:
        return ResultadoRecuperacao(chunk_id=chunk_id, texto=f"texto {chunk_id}", metadata={}, score=0.0)

    def test_documento_no_topo_das_duas_listas_fica_em_primeiro(self):
        lista_a = [self._resultado("x"), self._resultado("y")]
        lista_b = [self._resultado("x"), self._resultado("z")]
        fundido = fundir_rrf([lista_a, lista_b], top_k=3, k_rrf=60)
        assert fundido[0].chunk_id == "x"

    def test_score_rrf_bate_com_a_formula_1_sobre_k_mais_rank(self):
        lista_a = [self._resultado("x")]
        fundido = fundir_rrf([lista_a], top_k=1, k_rrf=60)
        assert fundido[0].score == pytest.approx(1 / (60 + 1))

    def test_documento_ausente_de_uma_lista_ainda_e_considerado(self):
        lista_a = [self._resultado("x"), self._resultado("y")]
        lista_b: list[ResultadoRecuperacao] = []
        fundido = fundir_rrf([lista_a, lista_b], top_k=2, k_rrf=60)
        assert {r.chunk_id for r in fundido} == {"x", "y"}


class TestRecuperacaoNoCorpusRealJaIndexado:
    """Depende de 'python -m src.indexing.build' ja ter sido executado."""

    @pytest.fixture(autouse=True)
    def _requer_indices_reais(self):
        if not (config.CHROMA_PERSIST_DIR / "chroma.sqlite3").exists() or not indice_esparso.CAMINHO_PADRAO.exists():
            pytest.skip("indices reais nao encontrados -- rode 'python -m src.indexing.build' antes")

    @pytest.mark.parametrize("estrategia", ["A", "B", "C", "D"])
    def test_todas_as_estrategias_retornam_algo_para_pergunta_normativa(self, estrategia):
        resultados = criar_retriever(estrategia).buscar(
            "qual o objetivo das boas práticas de fabricação de medicamentos?", top_k=3
        )
        assert len(resultados) == 3

    def test_rerank_d_encontra_o_artigo_1_da_rdc_658_no_topo(self):
        resultados = criar_retriever("D").buscar(
            "qual o objetivo das boas práticas de fabricação de medicamentos?", top_k=3
        )
        assert any(r.metadata["norma"] == "RDC 658/2022" and r.metadata["artigo"] == "1" for r in resultados)
