"""Testes do agente (Fase 4).

A maior parte usa `invocar_estruturado` monkeypatched -- rapidos,
deterministicos, nao dependem de Ollama rodando. Um teste real de ponta
a ponta (marcado, ver TestPipelineRealComOllama) so' roda se
RUN_SLOW_LLM_TESTS=1 estiver no ambiente: cada chamada ao qwen2.5 leva
segundos, entao a suite padrao (`pytest -q`) fica rapida por padrao.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src import config
from src.agent import nos
from src.agent.grafo import compilar_grafo
from src.agent.schemas import AssercoesExtraidas, JulgamentoAssercao, JulgamentoLLM
from src.retrieval.base import ResultadoRecuperacao


class RetrieverFalso:
    def __init__(self, resultados: list[ResultadoRecuperacao]):
        self._resultados = resultados

    def buscar(self, query: str, top_k: int) -> list[ResultadoRecuperacao]:
        return self._resultados[:top_k]


def _resultado(chunk_id: str, texto: str, score: float, artigo: str = "1") -> ResultadoRecuperacao:
    return ResultadoRecuperacao(
        chunk_id=chunk_id,
        texto=texto,
        metadata={"norma": "Norma Teste", "artigo": artigo, "paragrafo": None, "titulo_secao": None, "pagina": 1},
        score=score,
    )


def test_score_abaixo_do_limiar_forca_indeterminado_sem_chamar_llm(monkeypatch):
    chamado = False

    def _llm_nao_deveria_ser_chamado(*args, **kwargs):
        nonlocal chamado
        chamado = True
        raise AssertionError("LLM nao deveria ser chamado quando o score esta abaixo do limiar")

    monkeypatch.setattr(nos, "invocar_estruturado", _llm_nao_deveria_ser_chamado)
    monkeypatch.setattr(config, "RETRIEVAL_SCORE_THRESHOLD", 0.5)

    retriever = RetrieverFalso([_resultado("t1", "texto irrelevante", score=0.1)])
    julgamento = nos.julgar_assercao(retriever, "alguma assercao")

    assert julgamento.veredito == "INDETERMINADO"
    assert julgamento.citacao is None
    assert julgamento.motivo_abstencao is not None
    assert not chamado


def test_sem_nenhum_resultado_de_recuperacao_forca_indeterminado(monkeypatch):
    monkeypatch.setattr(config, "RETRIEVAL_SCORE_THRESHOLD", 0.0)
    retriever = RetrieverFalso([])
    julgamento = nos.julgar_assercao(retriever, "alguma assercao")
    assert julgamento.veredito == "INDETERMINADO"
    assert julgamento.citacao is None


def test_citacao_valida_do_llm_e_repassada_integralmente(monkeypatch):
    monkeypatch.setattr(config, "RETRIEVAL_SCORE_THRESHOLD", 0.0)
    texto_chunk = "Art. 1º Todo lote deve ser aprovado pela Garantia da Qualidade antes da liberação."
    retriever = RetrieverFalso([_resultado("t1", texto_chunk, score=0.9)])

    resposta_llm = JulgamentoLLM(
        veredito="NAO_CONFORME",
        trecho_citado="Todo lote deve ser aprovado pela Garantia da Qualidade antes da liberação.",
        justificativa="A asserção contradiz o texto.",
    )
    monkeypatch.setattr(nos, "invocar_estruturado", lambda *a, **k: resposta_llm)

    julgamento = nos.julgar_assercao(retriever, "o lote foi liberado pela producao")
    assert julgamento.veredito == "NAO_CONFORME"
    assert julgamento.motivo_abstencao is None
    assert julgamento.citacao is not None
    assert julgamento.citacao.chunk_id == "t1"


def test_citacao_inventada_pelo_llm_forca_indeterminado(monkeypatch):
    """Ver CLAUDE.md: 'veredito sem citacao rastreavel e' bug, nao estilo'."""
    monkeypatch.setattr(config, "RETRIEVAL_SCORE_THRESHOLD", 0.0)
    texto_chunk = "Art. 1º Todo lote deve ser aprovado pela Garantia da Qualidade antes da liberação."
    retriever = RetrieverFalso([_resultado("t1", texto_chunk, score=0.9)])

    resposta_llm = JulgamentoLLM(
        veredito="NAO_CONFORME",
        trecho_citado="Este trecho nao existe em lugar nenhum do chunk recuperado.",
        justificativa="justificativa qualquer",
    )
    monkeypatch.setattr(nos, "invocar_estruturado", lambda *a, **k: resposta_llm)

    julgamento = nos.julgar_assercao(retriever, "qualquer assercao")
    assert julgamento.veredito == "INDETERMINADO"
    assert julgamento.motivo_abstencao == "citacao do LLM nao corresponde literalmente ao chunk recuperado"


def test_veredito_indeterminado_do_llm_e_repassado_com_citacao(monkeypatch):
    monkeypatch.setattr(config, "RETRIEVAL_SCORE_THRESHOLD", 0.0)
    retriever = RetrieverFalso([_resultado("t1", "algum texto normativo", score=0.9)])
    resposta_llm = JulgamentoLLM(veredito="INDETERMINADO", trecho_citado="algum texto normativo", justificativa="sem info suficiente")
    monkeypatch.setattr(nos, "invocar_estruturado", lambda *a, **k: resposta_llm)

    julgamento = nos.julgar_assercao(retriever, "qualquer assercao")
    assert julgamento.veredito == "INDETERMINADO"
    assert julgamento.citacao is not None
    assert julgamento.motivo_abstencao is None


def test_extrair_assercoes_agrega_resultados_de_varios_trechos(monkeypatch):
    respostas = iter(
        [
            AssercoesExtraidas(assercoes=["assercao 1", "assercao 2"]),
            AssercoesExtraidas(assercoes=[]),
            AssercoesExtraidas(assercoes=["assercao 3", "  ", ""]),
        ]
    )
    monkeypatch.setattr(nos, "invocar_estruturado", lambda *a, **k: next(respostas))
    monkeypatch.setattr(nos, "dividir_por_tamanho", lambda t: [t])

    assercoes = nos.extrair_assercoes(["pagina 1", "pagina 2", "pagina 3"])
    assert assercoes == ["assercao 1", "assercao 2", "assercao 3"]


def test_grafo_compila_e_roda_ponta_a_ponta_com_nos_falsos(monkeypatch):
    from src.agent import grafo as grafo_mod

    monkeypatch.setattr(grafo_mod, "parse_documento", lambda caminho: ["pagina 1"])
    monkeypatch.setattr(grafo_mod, "extrair_assercoes", lambda paginas: ["assercao a", "assercao b"])
    monkeypatch.setattr(
        grafo_mod,
        "julgar_assercao",
        lambda retriever, assercao: JulgamentoAssercao(
            assercao=assercao, veredito="CONFORME", citacao=None, justificativa="ok"
        ),
    )
    monkeypatch.setattr(grafo_mod, "criar_retriever", lambda estrategia: object())

    grafo = compilar_grafo()
    estado_final = grafo.invoke(
        {
            "caminho_documento": "doc.pdf",
            "estrategia_recuperacao": "D",
            "paginas_texto": [],
            "assercoes": [],
            "julgamentos": [],
        }
    )
    assert len(estado_final["julgamentos"]) == 2
    assert all(j.veredito == "CONFORME" for j in estado_final["julgamentos"])


class TestPipelineRealComOllama:
    """Ponta a ponta contra o Ollama e os indices reais. Lento (minutos)
    -- so' roda com RUN_SLOW_LLM_TESTS=1 no ambiente."""

    @pytest.fixture(autouse=True)
    def _requer_ambiente_completo(self):
        if os.environ.get("RUN_SLOW_LLM_TESTS") != "1":
            pytest.skip("defina RUN_SLOW_LLM_TESTS=1 para rodar o teste real de ponta a ponta (lento)")
        caminho_doc = config.DATA_DOCUMENTOS_TESTE_DIR / "tratamento_de_desvio_ficticio.pdf"
        if not caminho_doc.exists():
            pytest.skip(f"{caminho_doc} nao encontrado")
        if not (config.CHROMA_PERSIST_DIR / "chroma.sqlite3").exists():
            pytest.skip("indice real nao encontrado -- rode 'python -m src.indexing.build' antes")

    def test_analisar_documento_produz_relatorio_com_julgamentos_rastreaveis(self):
        from src.agent.pipeline import analisar_documento

        caminho_doc = config.DATA_DOCUMENTOS_TESTE_DIR / "tratamento_de_desvio_ficticio.pdf"
        relatorio = analisar_documento(caminho_doc, estrategia="D")

        assert len(relatorio.julgamentos) > 0
        for julgamento in relatorio.julgamentos:
            if julgamento.veredito != "INDETERMINADO":
                assert julgamento.citacao is not None
                assert julgamento.citacao.trecho_literal
