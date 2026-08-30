"""Orquestra loader -> limpeza -> chunking para um PDF de norma.

O formato estrutural do documento (`artigos` vs `secoes_numeradas`) e'
passado explicitamente pelo chamador, nunca inferido automaticamente: um
palpite errado aplicaria o chunker errado silenciosamente e corromperia
a base de conhecimento sem nenhum aviso (ver CLAUDE.md > "sem mocks
silenciosos").
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from src.ingestion.chunking import Chunk, gerar_chunks
from src.ingestion.chunking_numerado import gerar_chunks_numerados
from src.ingestion.chunking_perguntas_respostas import gerar_chunks_perguntas_respostas
from src.ingestion.cleaning import limpar_texto
from src.ingestion.loader import carregar_linhas_estilizadas, carregar_pdf

FormatoDocumento = Literal["artigos", "secoes_numeradas", "perguntas_respostas"]


def ingerir_pdf(caminho: Path, norma: str, formato: FormatoDocumento = "artigos") -> list[Chunk]:
    if formato == "artigos":
        paginas = carregar_pdf(caminho)
        texto_paginas = [limpar_texto(pagina.texto) for pagina in paginas]
        return gerar_chunks(texto_paginas, norma=norma)
    if formato == "secoes_numeradas":
        linhas = carregar_linhas_estilizadas(caminho)
        return gerar_chunks_numerados(linhas, norma=norma)
    if formato == "perguntas_respostas":
        linhas = carregar_linhas_estilizadas(caminho)
        return gerar_chunks_perguntas_respostas(linhas, norma=norma)
    raise ValueError(f"formato de documento desconhecido: {formato!r}")
