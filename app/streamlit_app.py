"""Interface Streamlit do validador-docs (Fase 6).

Chama os nos do agente diretamente (src.agent.nos), nao o grafo
compilado (src/agent/grafo.py): `processar_assercoes` la' e' um unico
no' que faz o loop sobre as assercoes internamente, sem nenhum ponto de
checkpoint para a UI observar. Aqui a interface precisa de progresso por
assercao (o custo real medido na Fase 4 e' de dezenas de segundos por
assercao em CPU) -- entao ela orquestra os mesmos nos manualmente, um
por um, atualizando a barra de progresso entre eles.

Uso: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.agent.nos import extrair_assercoes, julgar_assercao, parse_documento  # noqa: E402
from src.agent.schemas import JulgamentoAssercao  # noqa: E402
from src.indexing import vetorial  # noqa: E402
from src.retrieval.factory import criar_retriever  # noqa: E402

st.set_page_config(page_title="Validador de Documentos Regulatórios", page_icon="🧪", layout="wide")

st.title("Validador de Documentos Regulatórios — Indústria Farmacêutica")

# Aviso de supervisão humana obrigatório na interface (ver CLAUDE.md):
# o rascunho do Annex 22 do EudraLex Volume 4 restringe LLMs/IA
# generativa a aplicações não críticas de BPF, com supervisão humana
# obrigatória. O sistema é assistivo, nunca uma aprovação automática.
st.warning(
    "**Ferramenta assistiva — não substitui revisão humana.** Toda saída "
    "deste sistema é um parecer preliminar para apoio à decisão, nunca uma "
    "aprovação ou reprovação automática de conformidade. O rascunho do "
    "Annex 22 do EudraLex Volume 4 (consulta pública encerrada, ainda não "
    "adotado) restringe LLMs a aplicações não críticas de BPF, com "
    "supervisão humana obrigatória. Toda saída requer revisão e assinatura "
    "de um responsável qualificado antes de qualquer decisão regulatória."
)

NOMES_ESTRATEGIA = {
    "A": "A — Denso puro (cosseno)",
    "B": "B — Esparso puro (BM25)",
    "C": "C — Híbrido (RRF)",
    "D": "D — Híbrido + rerank (recomendado)",
}

with st.sidebar:
    st.header("Configuração")
    estrategia = st.selectbox(
        "Estratégia de recuperação",
        options=list(config.RETRIEVAL_STRATEGIES),
        index=list(config.RETRIEVAL_STRATEGIES).index("D"),
        format_func=lambda s: NOMES_ESTRATEGIA[s],
    )

    try:
        colecao = vetorial.obter_colecao()
        st.caption(f"Índice carregado — hash do corpus: `{colecao.metadata['hash_corpus'][:12]}…`")
    except Exception:
        st.error("Índice não encontrado. Rode `python -m src.indexing.build` antes de usar a interface.")

    limitar_paginas = st.number_input(
        "Limitar às N primeiras páginas (0 = sem limite)",
        min_value=0,
        value=0,
        help="Cada asserção extraída custa dezenas de segundos de LLM em CPU "
        "(ver métricas da Fase 4 no README) — útil para um teste rápido "
        "antes de rodar o documento inteiro.",
    )

arquivo = st.file_uploader("Documento a validar (PDF)", type="pdf")

if arquivo is not None and st.button("Analisar documento", type="primary"):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(arquivo.read())
        caminho_tmp = Path(tmp.name)

    try:
        with st.status("Analisando documento…", expanded=True) as status:
            status.write("Extraindo texto do documento…")
            paginas = parse_documento(caminho_tmp)
            if limitar_paginas:
                paginas = paginas[: int(limitar_paginas)]

            status.write(f"Extraindo asserções verificáveis ({len(paginas)} página(s))…")
            assercoes = extrair_assercoes(paginas)
            status.write(f"{len(assercoes)} asserção(ões) extraída(s).")

            retriever = criar_retriever(estrategia)
            julgamentos: list[JulgamentoAssercao] = []
            barra = st.progress(0.0, text="Julgando asserções…")
            for indice, assercao in enumerate(assercoes):
                julgamentos.append(julgar_assercao(retriever, assercao))
                barra.progress((indice + 1) / max(len(assercoes), 1), text=f"Julgando asserções… ({indice + 1}/{len(assercoes)})")

            status.update(label="Análise concluída.", state="complete")

        st.session_state["julgamentos"] = julgamentos
        st.session_state["estrategia_usada"] = estrategia
        st.session_state["documento_analisado"] = arquivo.name
    finally:
        caminho_tmp.unlink(missing_ok=True)

if "julgamentos" in st.session_state:
    julgamentos: list[JulgamentoAssercao] = st.session_state["julgamentos"]

    st.subheader(f"Relatório — {st.session_state['documento_analisado']}")
    st.caption(f"Estratégia de recuperação: {NOMES_ESTRATEGIA[st.session_state['estrategia_usada']]}")

    contagem: dict[str, int] = {}
    for julgamento in julgamentos:
        contagem[julgamento.veredito] = contagem.get(julgamento.veredito, 0) + 1

    colunas = st.columns(len(config.VEREDITOS))
    for coluna, veredito in zip(colunas, config.VEREDITOS):
        coluna.metric(veredito, contagem.get(veredito, 0))

    st.divider()

    cor_por_veredito = {
        "CONFORME": "green",
        "NAO_CONFORME": "red",
        "NAO_APLICAVEL": "gray",
        "INDETERMINADO": "orange",
    }

    for julgamento in julgamentos:
        cor = cor_por_veredito.get(julgamento.veredito, "gray")
        with st.expander(f":{cor}[**{julgamento.veredito}**] — {julgamento.assercao[:100]}"):
            st.write("**Assercão avaliada:**", julgamento.assercao)
            st.write(
                "**Justificativa do modelo:**",
                julgamento.justificativa or "_(nenhuma — o sistema absteve-se antes de chamar o LLM)_",
            )
            if julgamento.motivo_abstencao:
                st.info(f"Motivo da abstenção: {julgamento.motivo_abstencao}")
            if julgamento.citacao:
                citacao = julgamento.citacao
                localizacao = f"{citacao.norma}, Art. {citacao.artigo}"
                if citacao.paragrafo:
                    localizacao += f", {citacao.paragrafo}"
                st.write(f"**Fonte:** {localizacao}")
                st.caption(
                    f"chunk_id: `{citacao.chunk_id}` · página {citacao.pagina} · "
                    f"score de recuperação: {citacao.score_recuperacao:.3f}"
                )
                st.markdown(f"> {citacao.trecho_literal}")
            else:
                st.write("_Nenhuma citação — nenhum trecho normativo recuperado acima do limiar configurado._")
