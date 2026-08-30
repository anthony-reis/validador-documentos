"""Ponto de entrada do agente: PDF sob analise -> RelatorioConformidade."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src import config
from src.agent.grafo import compilar_grafo
from src.agent.schemas import RelatorioConformidade
from src.indexing import vetorial


def analisar_documento(caminho_pdf: Path, estrategia: str = "D") -> RelatorioConformidade:
    grafo = compilar_grafo()
    estado_final = grafo.invoke(
        {
            "caminho_documento": str(caminho_pdf),
            "estrategia_recuperacao": estrategia,
            "paginas_texto": [],
            "assercoes": [],
            "julgamentos": [],
        }
    )

    # Le' o hash do corpus gravado na propria colecao Chroma (ver
    # src/indexing/vetorial.py), nao recalcula a partir de data/normas/:
    # reporta o hash do que FOI DE FATO indexado e usado na busca, que
    # pode divergir do corpus em disco se ele mudou sem reindexacao.
    hash_corpus_indexado = vetorial.obter_colecao().metadata["hash_corpus"]

    return RelatorioConformidade(
        documento_analisado=Path(caminho_pdf).name,
        estrategia_recuperacao=estrategia,
        modelo_llm=config.OLLAMA_MODEL,
        hash_corpus=hash_corpus_indexado,
        gerado_em=datetime.now(timezone.utc),
        julgamentos=estado_final["julgamentos"],
    )
