"""Interface Streamlit do validador-docs (Fase 6, redesenhada por
feedback de usabilidade).

Chama os nós do agente (src.agent.nos) diretamente, não o grafo
compilado (src/agent/grafo.py): `processar_assercoes` no grafo é um
único nó com o loop interno, sem checkpoint pra UI observar -- e o
custo real por asserção (dezenas de segundos em CPU, ver métricas da
Fase 4) torna feedback incremental essencial pra usabilidade.

Fluxo em duas etapas:
1. **Ao vivo**: extração e julgamento rodam INTERCALADOS por página (não
   duas fases separadas) -- assim que uma página é extraída, suas
   asserções já são julgadas antes de seguir para a próxima página.
   Cada página extraída e cada asserção julgada aparece como uma
   mensagem de chat assim que fica pronta, com uma barra de progresso
   fixa no rodapé da tela (CSS simples via `unsafe_allow_html`, sem
   componente extra). Ver CLAUDE.md > "Melhorias de performance": antes
   disso a extração de TODAS as páginas terminava por completo antes de
   qualquer julgamento começar, deixando a tela parada muito tempo antes
   do primeiro resultado. Filtrar durante essa transmissão não faz
   sentido -- a lista ainda não existe por inteiro.
2. **Modo relatório**: ao chegar em 100%, a página recarrega
   (`st.rerun()`) para uma visão persistente com filtro por veredito
   (chips via `st.pills`) acima da lista -- pedido explícito do usuário
   depois de ver a primeira versão da interface.

3. **Feedback em paralelo**: a análise roda numa thread de segundo plano
   (`src/agent/analise_job.py`) e a tela ao vivo é um `st.fragment` que
   se atualiza sozinho -- assim dá para dar feedback nas asserções já
   julgadas sem cancelar a análise (um clique em widget reexecuta o
   script inteiro e abortaria a execução em andamento).

Uso: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.agent import chat  # noqa: E402
from src.agent.analise_job import AnaliseJob  # noqa: E402
from src.agent.schemas import JulgamentoAssercao  # noqa: E402
from src.feedback import store as feedback_store  # noqa: E402
from src.indexing import vetorial  # noqa: E402

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


def _formulario_feedback(julgamento: JulgamentoAssercao, chave: str, documento: str, estrategia: str) -> None:
    """Feedback humano sobre um julgamento, salvo em SQLite local (ver
    src/feedback/store.py). Só aparece no modo relatório: durante a
    transmissão ao vivo a página ainda vai recarregar e o formulário
    seria descartado."""
    anterior = feedback_store.obter(documento, estrategia, julgamento.assercao)
    st.markdown("**Seu feedback**")
    with st.form(f"fb_{chave}"):
        opcoes = ["Correto", "Incorreto"]
        indice_inicial = 1 if anterior and not anterior.veredito_correto else 0
        avaliacao = st.radio("O veredito do agente está…", opcoes, index=indice_inicial, horizontal=True)
        outros = [v for v in config.VEREDITOS if v != julgamento.veredito]
        veredito_certo = st.selectbox(
            "Se incorreto, qual seria o veredito certo?",
            outros,
            index=outros.index(anterior.veredito_certo) if anterior and anterior.veredito_certo in outros else 0,
        )
        citacao_sustenta = st.checkbox(
            "A citação sustenta o veredito",
            value=True if anterior is None or anterior.citacao_sustenta is None else anterior.citacao_sustenta,
            disabled=julgamento.citacao is None,
        )
        justificativa = st.text_area(
            "Justificativa (obrigatória se incorreto)", value=anterior.justificativa if anterior else ""
        )
        enviado = st.form_submit_button("Salvar feedback")
    if anterior:
        st.caption(f"✔ Revisado em {anterior.criado_em:%d/%m/%Y %H:%M}. Salvar novamente substitui o feedback.")
    if not enviado:
        return

    correto = avaliacao == "Correto"
    citacao = julgamento.citacao
    try:
        # Lido aqui (e não a cada render) porque abre a coleção Chroma.
        hash_corpus = vetorial.obter_colecao().metadata.get("hash_corpus", "")
        feedback_store.salvar(
            feedback_store.FeedbackAssercao(
                documento=documento,
                estrategia=estrategia,
                assercao=julgamento.assercao,
                veredito_agente=julgamento.veredito,
                veredito_correto=correto,
                veredito_certo=None if correto else veredito_certo,
                citacao_sustenta=citacao_sustenta if citacao else None,
                justificativa=justificativa,
                modelo_llm=config.OLLAMA_MODEL,
                hash_corpus=hash_corpus,
                norma=citacao.norma if citacao else None,
                artigo=citacao.artigo if citacao else None,
                chunk_id=citacao.chunk_id if citacao else None,
                score_recuperacao=citacao.score_recuperacao if citacao else None,
                trecho_literal=citacao.trecho_literal if citacao else None,
                motivo_abstencao=julgamento.motivo_abstencao,
            )
        )
    except ValidationError as erro:
        st.error(erro.errors()[0]["msg"].removeprefix("Value error, "))
        return
    st.success("Feedback salvo localmente.")


def _renderizar_julgamento(
    julgamento: JulgamentoAssercao, chave: str | None = None, documento: str = "", estrategia: str = ""
) -> None:
    """`chave` só é passada no modo relatório; sem ela, não há formulário
    de feedback (transmissão ao vivo)."""
    cor = CORES_VEREDITO.get(julgamento.veredito, "gray")
    with st.chat_message("assistant", avatar="🧪"):
        revisado = chave is not None and feedback_store.obter(documento, estrategia, julgamento.assercao)
        st.markdown(f":{cor}[**{julgamento.veredito}**]  {julgamento.assercao}" + ("  ✔ revisado" if revisado else ""))
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
            if chave is not None:
                st.divider()
                _formulario_feedback(julgamento, chave, documento, estrategia)


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
    disabled=arquivo is None or st.session_state.get("job") is not None and not st.session_state["job"].finalizado_na_ui,
    use_container_width=True,
    help=None if arquivo is not None else "Anexe um PDF para habilitar a análise.",
)

job: AnaliseJob | None = st.session_state.get("job")
analise_em_andamento = job is not None and not job.finalizado_na_ui

if enviar and arquivo is not None and not analise_em_andamento:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(arquivo.read())
    job = AnaliseJob(
        documento=arquivo.name,
        estrategia=estrategia,
        caminho_pdf=Path(tmp.name),
        limitar_paginas=int(limitar_paginas),
    )
    job.iniciar()
    st.session_state["job"] = job
    # Um relatório antigo não deve aparecer nem confundir o feedback.
    for chave in ("julgamentos", "indice_chat", "chat_historico"):
        st.session_state.pop(chave, None)
    analise_em_andamento = True


@st.fragment(run_every=3)
def _painel_ao_vivo(job: AnaliseJob) -> None:
    """A análise roda em thread (ver src/agent/analise_job.py); este
    fragmento só lê o estado a cada 3s e reexecuta sozinho -- interagir
    com o formulário de feedback reexecuta apenas o fragmento, nunca
    cancela a análise."""
    estado = job.instantaneo()

    with st.chat_message("user", avatar="📎"):
        st.write(f"**{job.documento}** em análise — estratégia {job.estrategia}.")
    for mensagem in estado["mensagens_pagina"]:
        with st.chat_message("assistant", avatar="📄"):
            st.write(mensagem)
    for indice, julgamento in enumerate(estado["julgamentos"]):
        _renderizar_julgamento(julgamento, chave=str(indice), documento=job.documento, estrategia=job.estrategia)

    _barra_fixa(st.empty(), estado["fracao"], estado["status"])

    if not estado["concluido"]:
        if st.button("Cancelar análise", key="cancelar_analise"):
            job.cancelar()
        return

    # Terminou: promove o resultado para o modo relatório e recarrega a
    # página inteira.
    job.finalizado_na_ui = True
    if estado["erro"]:
        st.session_state.pop("job", None)
        st.session_state["erro_analise"] = estado["erro"]
    elif not estado["julgamentos"]:
        st.session_state.pop("job", None)
        st.session_state["aviso_analise"] = "Nenhuma asserção verificável foi encontrada neste documento."
    else:
        st.session_state["julgamentos"] = estado["julgamentos"]
        st.session_state["estrategia_usada"] = job.estrategia
        st.session_state["documento_analisado"] = job.documento
        st.session_state["indice_chat"] = job.indice_chat
        st.session_state["chat_historico"] = []
    st.rerun()


if "erro_analise" in st.session_state:
    st.error(f"A análise falhou: {st.session_state.pop('erro_analise')}")
if "aviso_analise" in st.session_state:
    st.info(st.session_state.pop("aviso_analise"))

if analise_em_andamento:
    _painel_ao_vivo(job)

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
        _renderizar_julgamento(
            julgamento,
            chave=str(julgamentos.index(julgamento)),
            documento=st.session_state["documento_analisado"],
            estrategia=st.session_state["estrategia_usada"],
        )

    st.divider()
    st.subheader("Converse sobre este documento")
    st.caption(
        "Assistente exploratório — respostas baseadas no texto do documento enviado e no "
        "relatório acima. Não é uma nova avaliação de conformidade."
    )

    st.session_state.setdefault("chat_historico", [])
    for turno in st.session_state["chat_historico"]:
        with st.chat_message(turno["role"]):
            st.write(turno["content"])

    pergunta = st.chat_input("Pergunte sobre o documento ou o relatório…")
    if pergunta:
        st.session_state["chat_historico"].append({"role": "user", "content": pergunta})
        with st.chat_message("user"):
            st.write(pergunta)
        with st.chat_message("assistant"):
            resposta = st.write_stream(
                chat.responder_stream(
                    pergunta,
                    st.session_state["chat_historico"][:-1],
                    st.session_state["indice_chat"],
                    julgamentos,
                )
            )
        st.session_state["chat_historico"].append({"role": "assistant", "content": resposta})
