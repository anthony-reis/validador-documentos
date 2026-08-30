"""Nos do grafo LangGraph (ver CLAUDE.md > "Arquitetura do agente")."""

from __future__ import annotations

from pathlib import Path

from src import config
from src.agent.llm import invocar_estruturado
from src.agent.prompts import (
    SISTEMA_EXTRACAO_ASSERCOES,
    SISTEMA_JULGAMENTO,
    prompt_extracao,
    prompt_julgamento,
)
from src.agent.schemas import AssercoesExtraidas, Citacao, JulgamentoAssercao, JulgamentoLLM
from src.ingestion.loader import carregar_pdf
from src.ingestion.texto_utils import dividir_por_tamanho
from src.retrieval.base import Retriever, ResultadoRecuperacao


def parse_documento(caminho: Path) -> list[str]:
    """Texto do documento sob analise, uma string por pagina nao-vazia."""
    return [pagina.texto for pagina in carregar_pdf(caminho) if pagina.texto.strip()]


def extrair_assercoes(paginas_texto: list[str]) -> list[str]:
    """Extracao de assercoes verificaveis via LLM -- heuristica
    best-effort para uso interativo (Fase 6). NAO e' a fonte dos numeros
    do TFG: a avaliacao da Fase 5 usa um gabarito curado manualmente
    (ver CLAUDE.md > Avaliacao), justamente para isolar a qualidade da
    recuperacao/geracao da qualidade desta extracao heuristica.
    """
    assercoes: list[str] = []
    for pagina in paginas_texto:
        for trecho in dividir_por_tamanho(pagina):
            if not trecho.strip():
                continue
            resultado = invocar_estruturado(
                SISTEMA_EXTRACAO_ASSERCOES, prompt_extracao(trecho), AssercoesExtraidas
            )
            assercoes.extend(a.strip() for a in resultado.assercoes if a.strip())
    return assercoes


def _normalizar(texto: str) -> str:
    return " ".join(texto.split())


def _citacao_de(resultado: ResultadoRecuperacao, trecho_citado: str) -> Citacao:
    # Metadados de origem densa usam "" para campo ausente (Chroma nao
    # aceita None); metadados de origem esparsa usam None diretamente
    # (ver src/indexing/vetorial.py vs. src/indexing/esparso.py) -- os
    # dois casos convergem aqui para None.
    return Citacao(
        norma=resultado.metadata["norma"],
        artigo=resultado.metadata["artigo"],
        paragrafo=resultado.metadata.get("paragrafo") or None,
        titulo_secao=resultado.metadata.get("titulo_secao") or None,
        trecho_literal=trecho_citado,
        chunk_id=resultado.chunk_id,
        pagina=resultado.metadata["pagina"],
        score_recuperacao=resultado.score,
    )


def julgar_assercao(retriever: Retriever, assercao: str) -> JulgamentoAssercao:
    resultados = retriever.buscar(assercao, top_k=config.RETRIEVAL_TOP_K)

    if not resultados or resultados[0].score < config.RETRIEVAL_SCORE_THRESHOLD:
        # Corte deterministico (ver CLAUDE.md > regra de abstencao): sem
        # trecho normativo com score acima do limiar, o veredito e'
        # INDETERMINADO sem excecao. O LLM nem chega a ser chamado -- nao
        # faz sentido pedir um julgamento quando ja sabemos que nao ha
        # base para ele.
        return JulgamentoAssercao(
            assercao=assercao,
            veredito="INDETERMINADO",
            citacao=None,
            justificativa="",
            motivo_abstencao="nenhum trecho normativo recuperado com score acima do limiar configurado",
        )

    melhor = resultados[0]
    julgamento_llm = invocar_estruturado(
        SISTEMA_JULGAMENTO, prompt_julgamento(melhor.texto, assercao), JulgamentoLLM
    )
    citacao = _citacao_de(melhor, julgamento_llm.trecho_citado)

    if julgamento_llm.veredito == "INDETERMINADO":
        return JulgamentoAssercao(
            assercao=assercao,
            veredito="INDETERMINADO",
            citacao=citacao,
            justificativa=julgamento_llm.justificativa,
            motivo_abstencao=None,
        )

    # Auto-verificacao deterministica (decisao da Fase 4, ver CLAUDE.md):
    # a citacao alegada pelo LLM precisa ser um trecho literal do chunk
    # REALMENTE recuperado. Se nao for, o LLM alucinou a citacao --
    # forcamos INDETERMINADO em vez de confiar num veredito sem
    # rastreabilidade real. Normalizamos espacos/quebras de linha antes
    # de comparar: o LLM pode reproduzir o texto com espacamento
    # ligeiramente diferente do extraido do PDF sem que isso signifique
    # que a citacao e' inventada.
    if _normalizar(julgamento_llm.trecho_citado) not in _normalizar(melhor.texto):
        return JulgamentoAssercao(
            assercao=assercao,
            veredito="INDETERMINADO",
            citacao=citacao,
            justificativa=julgamento_llm.justificativa,
            motivo_abstencao="citacao do LLM nao corresponde literalmente ao chunk recuperado",
        )

    return JulgamentoAssercao(
        assercao=assercao,
        veredito=julgamento_llm.veredito,
        citacao=citacao,
        justificativa=julgamento_llm.justificativa,
        motivo_abstencao=None,
    )
