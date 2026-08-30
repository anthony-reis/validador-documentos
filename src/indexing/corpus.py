"""Manifesto do corpus normativo.

Mapeia cada PDF em data/normas/ para sua norma (rotulo citavel) e formato
estrutural (ver src/ingestion/pipeline.py). Deliberadamente explicito, e
nao descoberto por convencao de nome de arquivo: ver CLAUDE.md > "sem
mocks silenciosos" -- um mapeamento errado aplicaria o chunker errado a
uma norma sem nenhum aviso.
"""

from __future__ import annotations

from dataclasses import dataclass

from src import config
from src.ingestion.chunking import Chunk
from src.ingestion.pipeline import FormatoDocumento, ingerir_pdf


@dataclass(frozen=True)
class EntradaCorpus:
    arquivo: str
    norma: str
    formato: FormatoDocumento


MANIFESTO: tuple[EntradaCorpus, ...] = (
    EntradaCorpus("RDC_658_2022.pdf", "RDC 658/2022", "artigos"),
    EntradaCorpus("IN_134_2022.pdf", "IN 134/2022", "artigos"),
    EntradaCorpus("IN_138_2022.pdf", "IN 138/2022", "artigos"),
    EntradaCorpus("ICH_Q10.pdf", "ICH Q10", "secoes_numeradas"),
)


def ingerir_corpus() -> list[Chunk]:
    chunks: list[Chunk] = []
    for entrada in MANIFESTO:
        caminho = config.DATA_NORMAS_DIR / entrada.arquivo
        if not caminho.exists():
            raise FileNotFoundError(
                f"{entrada.arquivo} esta no manifesto do corpus (src/indexing/corpus.py) "
                f"mas nao existe em {config.DATA_NORMAS_DIR} -- baixe o arquivo antes de indexar."
            )
        chunks.extend(ingerir_pdf(caminho, norma=entrada.norma, formato=entrada.formato))
    return chunks
