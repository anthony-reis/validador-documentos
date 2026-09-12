"""Chat exploratorio sobre o documento analisado e o relatorio de
conformidade ja gerado (feature adicional, pos-Fase 7 -- ver CLAUDE.md).

Diferente do julgamento formal (src/agent/nos.py), este chat NAO gera
vereditos nem citacao com auto-verificacao deterministica: e' um
assistente de navegacao/consulta sobre o que ja foi extraido e julgado,
por isso usa saida em texto livre (streaming), nao saida estruturada.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from src.agent.llm import stream_chat
from src.agent.prompts import SISTEMA_CHAT, prompt_chat
from src.agent.schemas import JulgamentoAssercao
from src.indexing.embeddings import gerar_embeddings
from src.ingestion.texto_utils import dividir_por_tamanho

TOP_K_TRECHOS = 5


@dataclass
class IndiceDocumentoChat:
    trechos: list[str]
    embeddings: np.ndarray  # (N, dim), normalizados -- produto escalar = cosseno


def construir_indice(paginas: list[str]) -> IndiceDocumentoChat:
    """Chunking generico por tamanho (nao hierarquico como o das normas
    em src/ingestion/chunking.py) -- um documento arbitrario enviado
    pelo usuario nao segue Art./paragrafo nem secao numerada. Embeddings
    computados uma unica vez por documento e mantidos em memoria da
    sessao, nunca persistidos: e' um documento efemero do usuario, nao
    faz parte da base de conhecimento versionada em data/normas/."""
    trechos = [trecho for pagina in paginas for trecho in dividir_por_tamanho(pagina) if trecho.strip()]
    if not trechos:
        return IndiceDocumentoChat(trechos=[], embeddings=np.zeros((0, 0)))
    vetores = np.array(gerar_embeddings(trechos))
    return IndiceDocumentoChat(trechos=trechos, embeddings=vetores)


def buscar_trechos(indice: IndiceDocumentoChat, pergunta: str, top_k: int = TOP_K_TRECHOS) -> list[str]:
    if not indice.trechos:
        return []
    vetor_pergunta = np.array(gerar_embeddings([pergunta])[0])
    scores = indice.embeddings @ vetor_pergunta
    melhores = np.argsort(scores)[::-1][:top_k]
    return [indice.trechos[i] for i in melhores]


def formatar_relatorio(julgamentos: list[JulgamentoAssercao]) -> str:
    """Resumo compacto (veredito + assercao + justificativa) de TODOS os
    julgamentos, nao so' dos relevantes -- simplificacao deliberada:
    documentos reais tem dezenas de assercoes, nao milhares (ver
    metricas da Fase 4 em CLAUDE.md), entao cabem inteiras na janela de
    contexto do Ollama (4096 tokens) sem precisar de uma segunda camada
    de recuperacao so' para o relatorio."""
    if not julgamentos:
        return "(nenhuma assercao foi extraida deste documento)"
    linhas = [
        f"- [{j.veredito}] {j.assercao}" + (f" -- {j.justificativa}" if j.justificativa else "")
        for j in julgamentos
    ]
    return "\n".join(linhas)


def responder_stream(
    pergunta: str,
    historico: list[dict[str, str]],
    indice: IndiceDocumentoChat,
    julgamentos: list[JulgamentoAssercao],
) -> Iterator[str]:
    """Gera a resposta em streaming. `historico` e' a lista de turnos
    anteriores desta conversa (dicts {"role": "user"|"assistant",
    "content": ...}, mesmo formato usado por st.chat_message). Os
    TRECHOS/RESUMO so' entram na mensagem de sistema da pergunta ATUAL
    -- nao sao reinjetados para cada turno anterior do historico, para
    nao estourar a janela de contexto em conversas longas. Limitacao
    aceita e documentada aqui, nao resolvida com uma abstracao maior
    sem necessidade comprovada (mesmo espirito das simplificacoes
    deliberadas ja registradas em CLAUDE.md)."""
    trechos = buscar_trechos(indice, pergunta)
    resumo = formatar_relatorio(julgamentos)

    mensagens: list[tuple[str, str]] = [("system", SISTEMA_CHAT)]
    for turno in historico:
        papel_lc = "human" if turno["role"] == "user" else "ai"
        mensagens.append((papel_lc, turno["content"]))
    mensagens.append(("human", prompt_chat(trechos, resumo, pergunta)))

    yield from stream_chat(mensagens)
