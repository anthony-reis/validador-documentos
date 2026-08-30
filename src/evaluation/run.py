"""Runner de avaliacao de recuperacao -- experimento central do TFG.

Uso: python -m src.evaluation.run --config experiments/A.yaml
(idem B/C/D.yaml)

Registra hash do corpus, versoes de modelo, parametros e timestamp a
cada execucao (ver CLAUDE.md > Avaliacao: "experimento nao reprodutivel
nao vale para o TFG"). As quatro configs escrevem no MESMO CSV de saida
por padrao -- trivial comparar A/B/C/D depois.
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src import config
from src.evaluation.metricas_recuperacao import ndcg_em_k, precision_em_k, recall_em_k, reciprocal_rank
from src.indexing import vetorial
from src.retrieval.factory import criar_retriever

COLUNAS_RESULTADO = [
    "timestamp",
    "estrategia",
    "top_k",
    "n_perguntas",
    "precision_at_k",
    "recall_at_k",
    "mrr",
    "ndcg_at_k",
    "tempo_medio_consulta_s",
    "hash_corpus",
    "modelo_embedding",
    "reranker",
    "rrf_k",
    "retrieval_pool_size",
]


def _ler_ground_truth(caminho: Path) -> list[dict]:
    with open(caminho, newline="", encoding="utf-8") as arquivo:
        linhas = list(csv.DictReader(arquivo))
    if not linhas:
        raise ValueError(f"{caminho} nao tem nenhuma linha de ground truth anotada.")
    return linhas


def avaliar(estrategia: str, ground_truth_path: Path, top_k: int) -> dict:
    perguntas = _ler_ground_truth(ground_truth_path)
    retriever = criar_retriever(estrategia)

    precisoes, recalls, rrs, ndcgs, tempos = [], [], [], [], []
    for linha in perguntas:
        relevantes = {c.strip() for c in linha["chunks_relevantes"].split(";") if c.strip()}

        inicio = time.perf_counter()
        resultados = retriever.buscar(linha["pergunta"], top_k=top_k)
        tempos.append(time.perf_counter() - inicio)

        recuperados = [resultado.chunk_id for resultado in resultados]
        precisoes.append(precision_em_k(recuperados, relevantes, top_k))
        recalls.append(recall_em_k(recuperados, relevantes, top_k))
        rrs.append(reciprocal_rank(recuperados, relevantes))
        ndcgs.append(ndcg_em_k(recuperados, relevantes, top_k))

    n = len(perguntas)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "estrategia": estrategia,
        "top_k": top_k,
        "n_perguntas": n,
        "precision_at_k": sum(precisoes) / n,
        "recall_at_k": sum(recalls) / n,
        "mrr": sum(rrs) / n,
        "ndcg_at_k": sum(ndcgs) / n,
        "tempo_medio_consulta_s": sum(tempos) / n,
        "hash_corpus": vetorial.obter_colecao().metadata["hash_corpus"],
        "modelo_embedding": "bge-m3",
        "reranker": "bge-reranker-v2-m3" if estrategia == "D" else "",
        "rrf_k": config.RRF_K if estrategia in ("C", "D") else "",
        "retrieval_pool_size": config.RETRIEVAL_POOL_SIZE if estrategia in ("C", "D") else "",
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
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as arquivo:
        cfg = yaml.safe_load(arquivo)

    resultado = avaliar(
        estrategia=cfg["estrategia"],
        ground_truth_path=config.PROJECT_ROOT / cfg["ground_truth"],
        top_k=int(cfg.get("top_k", config.RETRIEVAL_TOP_K)),
    )
    caminho_saida = config.PROJECT_ROOT / cfg.get("saida", "experiments/resultados.csv")
    registrar_resultado(resultado, caminho_saida)

    print(
        f"[avaliacao] estrategia={resultado['estrategia']} "
        f"precision@{resultado['top_k']}={resultado['precision_at_k']:.3f} "
        f"recall@{resultado['top_k']}={resultado['recall_at_k']:.3f} "
        f"mrr={resultado['mrr']:.3f} "
        f"ndcg@{resultado['top_k']}={resultado['ndcg_at_k']:.3f} "
        f"tempo_medio={resultado['tempo_medio_consulta_s'] * 1000:.1f}ms"
    )
    print(f"[avaliacao] resultado gravado em {caminho_saida}")


if __name__ == "__main__":
    main()
