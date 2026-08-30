"""Runner de avaliacao de geracao (Fase 5).

Uso: python -m src.evaluation.run_geracao --gabarito experiments/gabarito_geracao_template.csv --estrategia D

Mais lento que a avaliacao de recuperacao: cada linha do gabarito chama
o LLM (custo real observado na Fase 4: ~20-25s por julgamento em CPU,
ver CLAUDE.md). Nao roda automaticamente com a suite de testes.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

from src import config
from src.agent.nos import julgar_assercao
from src.evaluation.metricas_geracao import fidelidade_de_citacao, taxa_de_abstencao, taxa_de_acerto
from src.indexing import vetorial
from src.retrieval.factory import criar_retriever

COLUNAS_RESULTADO = [
    "timestamp",
    "estrategia",
    "n_assercoes",
    "taxa_de_acerto",
    "taxa_de_abstencao",
    "fidelidade_de_citacao",
    "hash_corpus",
    "modelo_llm",
]


def _ler_gabarito(caminho: Path) -> list[dict]:
    with open(caminho, newline="", encoding="utf-8") as arquivo:
        linhas = list(csv.DictReader(arquivo))
    if not linhas:
        raise ValueError(f"{caminho} nao tem nenhuma linha de gabarito anotada.")
    return linhas


def avaliar(gabarito_path: Path, estrategia: str) -> dict:
    linhas = _ler_gabarito(gabarito_path)
    retriever = criar_retriever(estrategia)

    julgamentos = [julgar_assercao(retriever, linha["assercao"]) for linha in linhas]
    veredito_esperado = [linha["veredito_esperado"] for linha in linhas]

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "estrategia": estrategia,
        "n_assercoes": len(linhas),
        "taxa_de_acerto": taxa_de_acerto(julgamentos, veredito_esperado),
        "taxa_de_abstencao": taxa_de_abstencao(julgamentos),
        "fidelidade_de_citacao": fidelidade_de_citacao(julgamentos),
        "hash_corpus": vetorial.obter_colecao().metadata["hash_corpus"],
        "modelo_llm": config.OLLAMA_MODEL,
    }


def registrar_resultado(resultado: dict, caminho_saida: Path) -> None:
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    existe = caminho_saida.exists()
    with open(caminho_saida, "a", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=COLUNAS_RESULTADO)
        if not existe:
            escritor.writeheader()
        escritor.writerow(resultado)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gabarito", required=True, type=Path)
    parser.add_argument("--estrategia", required=True, choices=config.RETRIEVAL_STRATEGIES)
    parser.add_argument("--saida", type=Path, default=Path("experiments/resultados_geracao.csv"))
    args = parser.parse_args()

    resultado = avaliar(args.gabarito, args.estrategia)
    caminho_saida = config.PROJECT_ROOT / args.saida
    registrar_resultado(resultado, caminho_saida)

    print(
        f"[avaliacao-geracao] estrategia={resultado['estrategia']} "
        f"taxa_de_acerto={resultado['taxa_de_acerto']:.3f} "
        f"taxa_de_abstencao={resultado['taxa_de_abstencao']:.3f} "
        f"fidelidade_de_citacao={resultado['fidelidade_de_citacao']:.3f}"
    )
    print(f"[avaliacao-geracao] resultado gravado em {caminho_saida}")


if __name__ == "__main__":
    main()
