"""Testes da interface Streamlit (Fase 6) via `AppTest` -- roda o script
de verdade sem precisar de navegador nem de um servidor Streamlit
rodando. Verificado manualmente tambem com `streamlit run` + inspecao da
pagina antes de escrever estes testes (ver CLAUDE.md)."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from src import config
from src.agent.schemas import Citacao, JulgamentoAssercao

# AppTest.from_file resolve caminho relativo ao arquivo que a CHAMA, nao
# ao cwd -- por isso o caminho absoluto via __file__.
CAMINHO_APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


def test_app_carrega_sem_excecao_e_mostra_aviso_de_supervisao_humana():
    at = AppTest.from_file(CAMINHO_APP)
    at.run(timeout=30)

    assert not at.exception
    assert at.title[0].value == "Validador de Documentos Regulatórios — Indústria Farmacêutica"
    assert len(at.warning) == 1
    assert "supervisão humana" in at.warning[0].value.lower()


def test_seletor_de_estrategia_oferece_as_quatro_configuracoes():
    at = AppTest.from_file(CAMINHO_APP)
    at.run(timeout=30)
    assert not at.exception
    assert len(at.selectbox[0].options) == len(config.RETRIEVAL_STRATEGIES)
    assert at.selectbox[0].value.startswith("D")  # default recomendado


def test_relatorio_renderiza_citacao_e_contagem_por_veredito_sem_rodar_o_pipeline():
    """Popula o estado como se uma analise ja tivesse rodado -- nao chama
    o pipeline de verdade (custa dezenas de minutos, ver CLAUDE.md >
    Fase 4)."""
    julgamentos = [
        JulgamentoAssercao(
            assercao="A amostragem foi conduzida pela produção.",
            veredito="NAO_CONFORME",
            citacao=Citacao(
                norma="RDC 658/2022",
                artigo="238",
                paragrafo=None,
                titulo_secao=None,
                trecho_literal="O pessoal de Controle de Qualidade deve ter acesso às áreas de produção.",
                chunk_id="RDC-658-2022_art238",
                pagina=90,
                score_recuperacao=0.87,
            ),
            justificativa="O contexto indica que a amostragem deve ser feita pelo CQ.",
            motivo_abstencao=None,
        ),
        JulgamentoAssercao(
            assercao="O lote foi identificado com etiqueta.",
            veredito="INDETERMINADO",
            citacao=None,
            justificativa="",
            motivo_abstencao="nenhum trecho normativo recuperado com score acima do limiar configurado",
        ),
    ]

    at = AppTest.from_file(CAMINHO_APP)
    at.session_state["julgamentos"] = julgamentos
    at.session_state["estrategia_usada"] = "D"
    at.session_state["documento_analisado"] = "teste.pdf"
    at.run(timeout=30)

    assert not at.exception
    assert at.subheader[0].value == "Relatório — teste.pdf"
    assert len(at.expander) == 2

    contagem = {m.label: m.value for m in at.metric}
    assert contagem["NAO_CONFORME"] == "1"
    assert contagem["INDETERMINADO"] == "1"
    assert contagem["CONFORME"] == "0"

    expander_nao_conforme = at.expander[0]
    textos_markdown = [m.value for m in expander_nao_conforme.markdown]
    assert any("RDC 658/2022, Art. 238" in texto for texto in textos_markdown)
    assert any("O pessoal de Controle de Qualidade" in texto for texto in textos_markdown)

    expander_indeterminado = at.expander[1]
    assert any("Nenhuma citação" in m.value for m in expander_indeterminado.markdown)
