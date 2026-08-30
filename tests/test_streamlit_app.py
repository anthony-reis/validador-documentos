"""Testes da interface Streamlit (Fase 6, redesenhada) via `AppTest` --
roda o script de verdade sem precisar de navegador. Verificado
manualmente tambem com `streamlit run` + inspecao da pagina antes de
escrever/reescrever estes testes (ver CLAUDE.md)."""

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
    assert at.title[0].value == "🧪 Validador de Documentos Regulatórios"
    assert len(at.warning) == 1
    assert "não substitui revisão humana" in at.warning[0].value.lower()


def test_botao_de_envio_fica_desabilitado_sem_arquivo_anexado():
    at = AppTest.from_file(CAMINHO_APP)
    at.run(timeout=30)
    assert not at.exception
    assert at.button[0].disabled is True


def test_seletor_de_estrategia_oferece_as_quatro_configuracoes():
    at = AppTest.from_file(CAMINHO_APP)
    at.run(timeout=30)
    assert not at.exception
    assert len(at.selectbox[0].options) == len(config.RETRIEVAL_STRATEGIES)
    assert at.selectbox[0].value.startswith("D")  # default recomendado


def _julgamentos_sinteticos() -> list[JulgamentoAssercao]:
    return [
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
            assercao="O sistema de qualidade abrange todo o ciclo de vida.",
            veredito="CONFORME",
            citacao=Citacao(
                norma="RDC 658/2022",
                artigo="6",
                paragrafo=None,
                titulo_secao=None,
                trecho_literal="O SQF deve abranger todas as etapas do ciclo de vida.",
                chunk_id="RDC-658-2022_art6",
                pagina=3,
                score_recuperacao=0.91,
            ),
            justificativa="Confirmado pelo contexto.",
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


def _app_em_modo_relatorio() -> AppTest:
    at = AppTest.from_file(CAMINHO_APP)
    at.session_state["julgamentos"] = _julgamentos_sinteticos()
    at.session_state["estrategia_usada"] = "D"
    at.session_state["documento_analisado"] = "teste.pdf"
    at.run(timeout=30)
    return at


def test_relatorio_renderiza_uma_mensagem_de_chat_por_julgamento():
    at = _app_em_modo_relatorio()
    assert not at.exception
    assert at.subheader[0].value == "Relatório — teste.pdf"
    assert len(at.chat_message) == 3


def test_relatorio_mostra_contagem_por_veredito():
    at = _app_em_modo_relatorio()
    contagem = {m.label: m.value for m in at.metric}
    assert contagem["NAO_CONFORME"] == "1"
    assert contagem["CONFORME"] == "1"
    assert contagem["INDETERMINADO"] == "1"
    assert contagem["NAO_APLICAVEL"] == "0"


def test_filtro_por_veredito_reduz_as_mensagens_exibidas():
    at = _app_em_modo_relatorio()
    filtro = at.pills[0]
    assert set(filtro.options) == set(config.VEREDITOS)
    assert set(filtro.value) == set(config.VEREDITOS)  # tudo selecionado por padrao

    filtro.set_value(["NAO_CONFORME"]).run()
    assert not at.exception
    assert len(at.chat_message) == 1

    textos = [m.value for m in at.chat_message[0].markdown]
    assert any("NAO_CONFORME" in texto for texto in textos)


def test_filtro_sem_nenhum_veredito_selecionado_nao_quebra_e_avisa():
    at = _app_em_modo_relatorio()
    at.pills[0].set_value([]).run()
    assert not at.exception
    assert len(at.chat_message) == 0
    assert any("Nenhum julgamento" in c.value for c in at.caption)
