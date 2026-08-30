import csv

import pytest

from src import config
from src.agent.nos import MOTIVO_CITACAO_NAO_ENCONTRADA
from src.agent.schemas import Citacao, JulgamentoAssercao
from src.evaluation import run as run_recuperacao
from src.evaluation.metricas_geracao import fidelidade_de_citacao, taxa_de_abstencao, taxa_de_acerto
from src.evaluation.metricas_recuperacao import ndcg_em_k, precision_em_k, recall_em_k, reciprocal_rank
from src.indexing import esparso as indice_esparso
from src.indexing import vetorial
from src.ingestion.chunking import Chunk


class TestMetricasRecuperacao:
    RECUPERADOS = ["a", "b", "c", "d", "e"]
    RELEVANTES = {"c", "e"}

    def test_precision_em_k(self):
        assert precision_em_k(self.RECUPERADOS, self.RELEVANTES, 5) == pytest.approx(2 / 5)
        assert precision_em_k(self.RECUPERADOS, self.RELEVANTES, 2) == 0.0

    def test_recall_em_k(self):
        assert recall_em_k(self.RECUPERADOS, self.RELEVANTES, 5) == pytest.approx(1.0)
        assert recall_em_k(self.RECUPERADOS, self.RELEVANTES, 2) == 0.0

    def test_recall_com_ground_truth_vazio_e_zero_nao_erro(self):
        assert recall_em_k(self.RECUPERADOS, set(), 5) == 0.0

    def test_precision_com_lista_vazia_e_zero(self):
        assert precision_em_k([], self.RELEVANTES, 5) == 0.0

    def test_reciprocal_rank_encontra_a_primeira_posicao_relevante(self):
        assert reciprocal_rank(self.RECUPERADOS, self.RELEVANTES) == pytest.approx(1 / 3)

    def test_reciprocal_rank_zero_quando_nada_relevante_aparece(self):
        assert reciprocal_rank(self.RECUPERADOS, {"z"}) == 0.0

    def test_ndcg_em_k_bate_com_calculo_manual(self):
        # DCG = 1/log2(4) [pos 3, "c"] + 1/log2(6) [pos 5, "e"]
        # IDCG = 1/log2(2) + 1/log2(3) (2 relevantes no topo ideal)
        import math

        dcg = 1 / math.log2(4) + 1 / math.log2(6)
        idcg = 1 / math.log2(2) + 1 / math.log2(3)
        assert ndcg_em_k(self.RECUPERADOS, self.RELEVANTES, 5) == pytest.approx(dcg / idcg)

    def test_ndcg_e_um_quando_recuperados_batem_perfeitamente_o_ideal(self):
        assert ndcg_em_k(["c", "e", "a"], self.RELEVANTES, 3) == pytest.approx(1.0)


class TestMetricasGeracao:
    def _julgamento(self, veredito, chunk_id="t1", motivo_abstencao=None):
        citacao = Citacao(
            norma="N",
            artigo="1",
            paragrafo=None,
            titulo_secao=None,
            trecho_literal="trecho",
            chunk_id=chunk_id,
            pagina=1,
            score_recuperacao=0.9,
        )
        return JulgamentoAssercao(
            assercao="a",
            veredito=veredito,
            citacao=citacao,
            justificativa="j",
            motivo_abstencao=motivo_abstencao,
        )

    def test_taxa_de_acerto_compara_posicionalmente_com_o_esperado(self):
        julgamentos = [self._julgamento("CONFORME"), self._julgamento("NAO_CONFORME")]
        assert taxa_de_acerto(julgamentos, ["CONFORME", "CONFORME"]) == pytest.approx(0.5)

    def test_taxa_de_abstencao_conta_apenas_indeterminado(self):
        julgamentos = [self._julgamento("CONFORME"), self._julgamento("INDETERMINADO")]
        assert taxa_de_abstencao(julgamentos) == pytest.approx(0.5)

    def test_fidelidade_de_citacao_exclui_julgamentos_sem_citacao(self):
        sem_citacao = JulgamentoAssercao(
            assercao="a", veredito="INDETERMINADO", citacao=None, justificativa="", motivo_abstencao="sem recuperacao"
        )
        com_citacao_valida = self._julgamento("CONFORME")
        assert fidelidade_de_citacao([sem_citacao, com_citacao_valida]) == pytest.approx(1.0)

    def test_fidelidade_de_citacao_penaliza_citacao_rejeitada(self):
        valida = self._julgamento("CONFORME")
        invalida = self._julgamento("INDETERMINADO", motivo_abstencao=MOTIVO_CITACAO_NAO_ENCONTRADA)
        assert fidelidade_de_citacao([valida, invalida]) == pytest.approx(0.5)

    def test_fidelidade_de_citacao_sem_nenhum_julgamento_elegivel_e_zero(self):
        sem_citacao = JulgamentoAssercao(
            assercao="a", veredito="INDETERMINADO", citacao=None, justificativa="", motivo_abstencao="x"
        )
        assert fidelidade_de_citacao([sem_citacao]) == 0.0


def _chunk(chunk_id: str, texto: str, artigo: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, texto=texto, norma="Norma Teste", artigo=artigo,
        paragrafo=None, titulo_secao=None, pagina=1,
    )


CHUNKS_SINTETICOS = [
    _chunk("t1", "Toda validação de processo deve ser documentada e aprovada pela qualidade.", "10"),
    _chunk("t2", "O treinamento de pessoal deve ser registrado e avaliado periodicamente.", "20"),
    _chunk("t3", "Os equipamentos de fabricação devem passar por qualificação antes do uso.", "30"),
]


class TestRunnerDeRecuperacao:
    @pytest.fixture(autouse=True)
    def _indices_isolados(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "CHROMA_PERSIST_DIR", tmp_path / "chroma_teste")
        caminho_bm25 = tmp_path / "bm25_teste" / "indice.pkl"
        monkeypatch.setattr(indice_esparso, "CAMINHO_PADRAO", caminho_bm25)

        vetorial.indexar(CHUNKS_SINTETICOS, recriar=True)
        retriever_bm25 = indice_esparso.construir_indice(CHUNKS_SINTETICOS)
        indice_esparso.salvar_indice(retriever_bm25, caminho=caminho_bm25)

        self.caminho_ground_truth = tmp_path / "ground_truth.csv"
        with open(self.caminho_ground_truth, "w", newline="", encoding="utf-8") as arquivo:
            escritor = csv.writer(arquivo)
            escritor.writerow(["id", "pergunta", "chunks_relevantes", "norma", "artigo", "categoria_documento"])
            escritor.writerow(["1", "qualificação de equipamentos de produção", "t3", "Norma Teste", "30", "POP"])
            escritor.writerow(["2", "treinamento de pessoal", "t2", "Norma Teste", "20", "POP"])

    def test_avaliar_retorna_metricas_perfeitas_para_perguntas_faceis(self):
        resultado = run_recuperacao.avaliar("A", self.caminho_ground_truth, top_k=1)
        assert resultado["estrategia"] == "A"
        assert resultado["n_perguntas"] == 2
        assert resultado["precision_at_k"] == pytest.approx(1.0)
        assert resultado["recall_at_k"] == pytest.approx(1.0)
        assert resultado["mrr"] == pytest.approx(1.0)
        assert resultado["hash_corpus"] == vetorial.hash_corpus(CHUNKS_SINTETICOS)

    def test_registrar_resultado_escreve_cabecalho_uma_vez_e_acumula_linhas(self, tmp_path):
        caminho_saida = tmp_path / "resultados.csv"
        resultado_a = run_recuperacao.avaliar("A", self.caminho_ground_truth, top_k=1)
        resultado_b = run_recuperacao.avaliar("B", self.caminho_ground_truth, top_k=1)

        run_recuperacao.registrar_resultado(resultado_a, caminho_saida)
        run_recuperacao.registrar_resultado(resultado_b, caminho_saida)

        with open(caminho_saida, newline="", encoding="utf-8") as arquivo:
            linhas = list(csv.DictReader(arquivo))
        assert len(linhas) == 2
        assert {linha["estrategia"] for linha in linhas} == {"A", "B"}

    def test_ground_truth_vazio_leva_a_erro_claro(self, tmp_path):
        caminho_vazio = tmp_path / "vazio.csv"
        with open(caminho_vazio, "w", newline="", encoding="utf-8") as arquivo:
            csv.writer(arquivo).writerow(["id", "pergunta", "chunks_relevantes", "norma", "artigo", "categoria_documento"])

        with pytest.raises(ValueError, match="nenhuma linha"):
            run_recuperacao.avaliar("A", caminho_vazio, top_k=1)
