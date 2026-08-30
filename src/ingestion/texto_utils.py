"""Utilitarios compartilhados pelas estrategias de chunking."""

from __future__ import annotations

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config


def dividir_por_tamanho(texto: str) -> list[str]:
    """Divisao secundaria por tamanho, so acionada quando um bloco ja
    isolado pela estrutura do documento (artigo/paragrafo ou secao
    numerada) ainda excede CHUNK_SIZE. Ver CLAUDE.md > "Chunking"."""
    if len(texto) <= config.CHUNK_SIZE:
        return [texto]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(texto)


def slug(texto: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", texto).strip("-")
