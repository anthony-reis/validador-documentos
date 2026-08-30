"""Extracao de texto de PDFs normativos.

PyMuPDF e' o extrator primario (ver CLAUDE.md > stack fixa). Paginas com
pouco ou nenhum texto (comum em digitalizacoes da ANVISA sem camada de
texto) acionam o fallback pdfplumber; se ainda assim vier vazio, a pagina
fica vazia e cabe a uma fase futura decidir sobre OCR (ver CLAUDE.md >
Armadilhas conhecidas) -- nao mascaramos isso aqui.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pdfplumber
import pymupdf

# Abaixo deste numero de caracteres, consideramos a extracao do PyMuPDF
# suspeita (pagina provavelmente escaneada) e tentamos o fallback.
_MIN_CHARS_PYMUPDF = 20


@dataclass(frozen=True)
class Pagina:
    numero: int  # 1-indexado, como aparece no PDF
    texto: str


@dataclass(frozen=True)
class LinhaEstilizada:
    pagina: int
    texto: str
    negrito: bool


# Bit de negrito nas flags de span do PyMuPDF (ver PDF text extraction docs).
_FLAG_NEGRITO = 1 << 4


def carregar_linhas_estilizadas(caminho: Path) -> list[LinhaEstilizada]:
    """Extrai linhas com informacao de negrito, necessaria para o chunking
    de documentos organizados em secoes numeradas (ver chunking_numerado.py):
    la, negrito x regular e' o unico sinal confiavel para distinguir um
    cabecalho real de uma entrada de sumario com o mesmo formato textual.
    """
    linhas: list[LinhaEstilizada] = []
    doc = pymupdf.open(caminho)
    try:
        for i, pagina in enumerate(doc):
            info = pagina.get_text("dict")
            for bloco in info["blocks"]:
                for linha in bloco.get("lines", []):
                    texto = "".join(span["text"] for span in linha["spans"]).strip()
                    if not texto:
                        continue
                    negrito = bool(linha["spans"][0]["flags"] & _FLAG_NEGRITO)
                    linhas.append(LinhaEstilizada(pagina=i + 1, texto=texto, negrito=negrito))
    finally:
        doc.close()
    return linhas


def carregar_pdf(caminho: Path) -> list[Pagina]:
    paginas: list[Pagina] = []
    doc = pymupdf.open(caminho)
    try:
        for i, pagina in enumerate(doc):
            texto = pagina.get_text()
            if len(texto.strip()) < _MIN_CHARS_PYMUPDF:
                texto = _extrair_com_pdfplumber(caminho, i) or texto
            paginas.append(Pagina(numero=i + 1, texto=texto))
    finally:
        doc.close()
    return paginas


def _extrair_com_pdfplumber(caminho: Path, indice_pagina: int) -> str | None:
    with pdfplumber.open(caminho) as pdf:
        if indice_pagina >= len(pdf.pages):
            return None
        return pdf.pages[indice_pagina].extract_text()
