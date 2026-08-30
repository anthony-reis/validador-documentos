"""Interface comum das quatro configuracoes de recuperacao (A/B/C/D, ver
CLAUDE.md > "quatro configuracoes de recuperacao").

Cada estrategia devolve uma lista de ResultadoRecuperacao, da mais
relevante para a menos relevante. O `score` so' e' comparavel DENTRO da
mesma estrategia -- BM25 (config B) e similaridade de cosseno (config A)
vivem em escalas incompativeis entre si; por isso a fusao hibrida
(config C) usa Reciprocal Rank Fusion sobre o RANK de cada lista, nunca
o score bruto (ver src/retrieval/hibrido.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ResultadoRecuperacao:
    chunk_id: str
    texto: str
    metadata: dict
    score: float


class Retriever(Protocol):
    def buscar(self, query: str, top_k: int) -> list[ResultadoRecuperacao]: ...
