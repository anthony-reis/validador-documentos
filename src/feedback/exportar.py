"""Exporta o feedback humano no formato de
`experiments/gabarito_geracao_template.csv` (id, assercao,
veredito_esperado, norma, artigo).

Decisão metodológica (CLAUDE.md > Avaliação): a saída vai para um
arquivo SEPARADO, nunca direto para o gabarito oficial. O usuário revisa
e mescla manualmente -- assim o gabarito continua sendo anotação
humana curada, e feedbacks dados sobre asserções que o próprio agente
extraiu não entram no gabarito sem passar por essa decisão.

Uso: python -m src.feedback.exportar [--saida caminho.csv]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src import config
from src.feedback import store

CABECALHO = ["id", "assercao", "veredito_esperado", "norma", "artigo"]


def exportar(saida: Path, caminho_db: Path | None = None) -> int:
    feedbacks = store.listar(caminho_db)
    saida.parent.mkdir(parents=True, exist_ok=True)
    with open(saida, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(CABECALHO)
        for indice, f in enumerate(feedbacks):
            # Se o revisor concordou, o veredito do agente é o esperado.
            esperado = f.veredito_agente if f.veredito_correto else f.veredito_certo
            escritor.writerow([f"fb-{indice}", f.assercao, esperado, f.norma or "", f.artigo or ""])
    return len(feedbacks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--saida", type=Path, default=config.PROJECT_ROOT / "experiments" / "gabarito_feedback.csv")
    argumentos = parser.parse_args()
    print(f"{exportar(argumentos.saida)} feedback(s) exportado(s) para {argumentos.saida}")
