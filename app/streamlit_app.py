"""Interface Streamlit do validador-docs (Fase 6, redesenhada por
feedback de usabilidade).

Chama os nós do agente (src.agent.nos) diretamente, não o grafo
compilado (src/agent/grafo.py): `processar_assercoes` no grafo é um
único nó com o loop interno, sem checkpoint pra UI observar -- e o
custo real por asserção (dezenas de segundos em CPU, ver métricas da
Fase 4) torna feedback incremental essencial pra usabilidade.

Fluxo em duas etapas:
1. **Ao vivo**: enquanto a análise roda, cada página extraída e cada
   asserção julgada aparece como uma mensagem de chat assim que fica
   pronta, com uma barra de progresso fixa no rodapé da tela (CSS
   simples via `unsafe_allow_html`, sem componente extra). Filtrar
   durante essa transmissão não faz sentido -- a lista ainda não existe
   por inteiro.
2. **Modo relatório**: ao chegar em 100%, a página recarrega
   (`st.rerun()`) para uma visão persistente com filtro por veredito
   (chips via `st.pills`) acima da lista -- pedido explícito do usuário
   depois de ver a primeira versão da interface.

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

st.set_page_config(page_title="Validador de Documentos Regulatórios", page_icon="🧪", layout="centered")

# Visual minimalista: esconde o menu/rodape padrao do Streamlit e reserva
# espaco embaixo pra barra de progresso fixa nao cobrir o ultimo item.
st.markdown(
    "<style>#MainMenu{visibility:hidden;} footer{visibility:hidden;} "
    ".block-container{padding-bottom:5rem;}</style>",
    unsafe_allow_html=True,
)

NOMES_ESTRATEGIA = {
    "A": "A — Denso puro (cosseno)",
    "B": "B — Esparso puro (BM25)",
    "C": "C — Híbrido (RRF)",
    "D": "D — Híbrido + rerank (recomendado)",
}
CORES_VEREDITO = {
    "CONFORME": "green",
    "NAO_CONFORME": "red",
    "NAO_APLICAVEL": "gray",
    "INDETERMINADO": "orange",
}


def _barra_fixa(placeholder: "st.delta_generator.DeltaGenerator", fracao: float, texto: str) -> None:
    """Barra de progresso fixada no rodape da tela via CSS simples --
    fica visivel mesmo com o feed de mensagens crescendo por cima dela."""
    placeholder.markdown(
        f"""
        <div style="position:fixed; bottom:0; left:0; right:0; background:#FFFFFF;
                     border-top:1px solid #E5E7EB; padding:10px 16px; z-index:999;">
          <div style="max-width:730px; margin:0 auto;">
            <div style="background:#E5E7EB; border-radius:8px; height:8px; overflow:hidden;">
              <div style="width:{max(0.0, min(fracao, 1.0)) * 100:.0f}%; background:#2563EB;
                          height:100%; transition:width 0.2s;"></div>
            </div>
            <div style="font-size:13px; color:#6B7280; margin-top:4px;">{texto}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _renderizar_julgamento(julgamento: JulgamentoAssercao) -> None:
    cor = CORES_VEREDITO.get(julgamento.veredito, "gray")
    with st.chat_message("assistant", avatar="🧪"):
        st.markdown(f":{cor}[**{julgamento.veredito}**]  {julgamento.assercao}")
        with st.expander("Detalhes"):
            st.write(
                "**Justificativa:**",
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


st.title("🧪 Validador de Documentos Regulatórios")
st.caption("Indústria farmacêutica · RAG local · RDC 658/2022, ICH Q10, INs 134 e 138/2022")

# Aviso de supervisão humana obrigatório (ver CLAUDE.md > raciocínio do
# Annex 22) -- sempre visível, não é detalhe cosmético.
st.warning(
    "**Ferramenta assistiva — não substitui revisão humana.** Toda saída "
    "deste sistema é um parecer preliminar, nunca uma aprovação ou "
    "reprovação automática de conformidade. Requer revisão e assinatura "
    "de um responsável qualificado antes de qualquer decisão regulatória."
)

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
        st.caption(f"Índice carregado — hash: `{colecao.metadata['hash_corpus'][:12]}…`")
    except Exception:
        st.error("Índice não encontrado. Rode `python -m src.indexing.build` antes de usar a interface.")

    limitar_paginas = st.number_input(
        "Limitar às N primeiras páginas (0 = sem limite)",
        min_value=0,
        value=0,
        help="Cada asserção custa dezenas de segundos de LLM em CPU (ver README) — "
        "útil para um teste rápido antes de rodar o documento inteiro.",
    )

arquivo = st.file_uploader("Documento a validar (PDF)", type="pdf")
enviar = st.button(
    "Enviar para análise",
    type="primary",
    disabled=arquivo is None,
    use_container_width=True,
    help=None if arquivo is not None else "Anexe um PDF para habilitar a análise.",
)

if enviar and arquivo is not None:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(arquivo.read())
        caminho_tmp = Path(tmp.name)

    barra = st.empty()
    _barra_fixa(barra, 0.0, "Lendo documento…")

    with st.chat_message("user", avatar="📎"):
        st.write(f"**{arquivo.name}** enviado para análise — estratégia {estrategia}.")

    try:
        paginas = parse_documento(caminho_tmp)
        if limitar_paginas:
            paginas = paginas[: int(limitar_paginas)]

        # Extracao pagina a pagina (nao de uma vez): cada pagina processada
        # vira uma mensagem de chat imediatamente -- e' a etapa mais
        # "silenciosa" do pipeline, entao feedback incremental aqui importa
        # tanto quanto durante o julgamento.
        assercoes: list[str] = []
        for indice_pagina, pagina in enumerate(paginas):
            _barra_fixa(
                barra,
                (indice_pagina / max(len(paginas), 1)) * 0.3,
                f"Extraindo asserções — página {indice_pagina + 1}/{len(paginas)}…",
            )
            novas = extrair_assercoes([pagina])
            assercoes.extend(novas)
            if novas:
                with st.chat_message("assistant", avatar="📄"):
                    st.write(f"Página {indice_pagina + 1}/{len(paginas)}: {len(novas)} asserção(ões) encontrada(s).")

        if not assercoes:
            _barra_fixa(barra, 1.0, "Concluído — nenhuma asserção verificável encontrada.")
            st.info("Nenhuma asserção verificável foi encontrada neste documento.")
        else:
            retriever = criar_retriever(estrategia)
            julgamentos: list[JulgamentoAssercao] = []
            for indice_assercao, assercao in enumerate(assercoes):
                fracao = 0.3 + 0.7 * (indice_assercao / len(assercoes))
                _barra_fixa(barra, fracao, f"Julgando asserções… {indice_assercao}/{len(assercoes)}")
                julgamento = julgar_assercao(retriever, assercao)
                julgamentos.append(julgamento)
                _renderizar_julgamento(julgamento)

            _barra_fixa(barra, 1.0, f"100% concluído — {len(julgamentos)} asserção(ões) julgada(s).")

            st.session_state["julgamentos"] = julgamentos
            st.session_state["estrategia_usada"] = estrategia
            st.session_state["documento_analisado"] = arquivo.name
            st.rerun()
    finally:
        caminho_tmp.unlink(missing_ok=True)

elif "julgamentos" in st.session_state:
    julgamentos: list[JulgamentoAssercao] = st.session_state["julgamentos"]

    st.subheader(f"Relatório — {st.session_state['documento_analisado']}")
    st.caption(f"Estratégia: {NOMES_ESTRATEGIA[st.session_state['estrategia_usada']]}")

    contagem: dict[str, int] = {}
    for julgamento in julgamentos:
        contagem[julgamento.veredito] = contagem.get(julgamento.veredito, 0) + 1

    colunas = st.columns(len(config.VEREDITOS))
    for coluna, veredito in zip(colunas, config.VEREDITOS):
        coluna.metric(veredito, contagem.get(veredito, 0))

    st.write("")
    filtro = st.pills(
        "Filtrar por veredito",
        options=list(config.VEREDITOS),
        selection_mode="multi",
        default=list(config.VEREDITOS),
    )

    st.divider()

    julgamentos_filtrados = [j for j in julgamentos if j.veredito in (filtro or [])]
    if not julgamentos_filtrados:
        st.caption("Nenhum julgamento para os filtros selecionados.")
    for julgamento in julgamentos_filtrados:
        _renderizar_julgamento(julgamento)
