"""Orquestra loader -> limpeza -> chunking para um PDF de norma."""

from __future__ import annotations

from pathlib import Path

from src.ingestion.chunking import Chunk, gerar_chunks
from src.ingestion.cleaning import limpar_texto
from src.ingestion.loader import carregar_pdf


def ingerir_pdf(caminho: Path, norma: str) -> list[Chunk]:
    paginas = carregar_pdf(caminho)
    texto_paginas = [limpar_texto(pagina.texto) for pagina in paginas]
    return gerar_chunks(texto_paginas, norma=norma)
