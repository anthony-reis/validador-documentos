"""Ponto de entrada da indexacao: corpus -> chunks -> Chroma + BM25.

Uso: python -m src.indexing.build
"""

from __future__ import annotations

from src.indexing.corpus import ingerir_corpus
from src.indexing.esparso import construir_indice, salvar_indice
from src.indexing.vetorial import hash_corpus, indexar


def main() -> None:
    chunks = ingerir_corpus()
    print(f"[indexacao] {len(chunks)} chunks extraidos do corpus.")

    por_norma: dict[str, int] = {}
    for chunk in chunks:
        por_norma[chunk.norma] = por_norma.get(chunk.norma, 0) + 1
    for norma, total in sorted(por_norma.items()):
        print(f"  - {norma}: {total} chunks")

    print(f"[indexacao] hash do corpus: {hash_corpus(chunks)}")

    print("[indexacao] gravando indice denso (Chroma)...")
    indexar(chunks, recriar=True)

    print("[indexacao] construindo indice esparso (BM25)...")
    retriever = construir_indice(chunks)
    salvar_indice(retriever)

    print("[indexacao] concluido.")


if __name__ == "__main__":
    main()
