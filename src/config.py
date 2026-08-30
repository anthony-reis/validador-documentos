"""Configuracao central do projeto. Nada de parametro hardcoded fora daqui.

Todo valor vem do ambiente (.env), com um default explicito. Isso permite
variar chunk size, overlap, top-k e limiar de score entre experimentos sem
tocar em codigo -- exigencia do TFG (ver CLAUDE.md, secao "Chunking").
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Garante a restricao offline mesmo se o .env nao tiver sido carregado
# (ex.: em CI). Ver CLAUDE.md: "Restricao inegociavel: 100% offline".
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# O Chroma so' desativa o telemetry client (posthog.disabled=True) quando
# anonymized_telemetry=False e' passado nas Settings (ver
# src/indexing/vetorial.py) -- nenhuma chamada de rede chega a ocorrer
# (confirmado com socket.socket.connect monkeypatched). O log abaixo e'
# so' silenciado porque a versao instalada do posthog mudou a assinatura
# de capture() e o chromadb ainda chama com a assinatura antiga, gerando
# um TypeError local que so' polui o log -- nao indica vazamento de rede.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _path(env_var: str, default: str) -> Path:
    return PROJECT_ROOT / os.getenv(env_var, default)


# --- Caminhos ---
DATA_NORMAS_DIR = _path("DATA_NORMAS_DIR", "data/normas")
DATA_DOCUMENTOS_TESTE_DIR = _path("DATA_DOCUMENTOS_TESTE_DIR", "data/documentos_teste")
MODELS_DIR = _path("MODELS_DIR", "models")
CHROMA_PERSIST_DIR = _path("CHROMA_PERSIST_DIR", "chroma_db")
BM25_INDEX_DIR = _path("BM25_INDEX_DIR", "bm25_index")

# --- Modelos locais ---
EMBEDDING_MODEL_PATH = _path("EMBEDDING_MODEL_PATH", "models/bge-m3")
RERANKER_MODEL_PATH = _path("RERANKER_MODEL_PATH", "models/bge-reranker-v2-m3")

# --- LLM via Ollama ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M")

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# --- Recuperacao ---
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
RETRIEVAL_SCORE_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.0"))
# Candidatos buscados em CADA retriever (denso/esparso) antes da fusao
# hibrida (config C) e do rerank (config D) -- maior que RETRIEVAL_TOP_K
# para dar chance de sobreposicao entre as duas listas.
RETRIEVAL_POOL_SIZE = int(os.getenv("RETRIEVAL_POOL_SIZE", "20"))
# Constante k do Reciprocal Rank Fusion (Cormack et al., 2009). 60 e' o
# valor de referencia da literatura original; parametrizavel para os
# experimentos reportarem o efeito de varia-lo.
RRF_K = int(os.getenv("RRF_K", "60"))
# Tamanho do sub-lote ao buscar/reranquear varias assercoes de uma vez
# (ver CLAUDE.md > "Melhorias de performance" -- buscar_lote). Um lote
# unico gigante perderia feedback incremental na UI; sub-lotes pequenos
# demais perdem o ganho de agrupar as chamadas de embedding/Chroma/
# reranker.
RETRIEVAL_BATCH_SIZE = int(os.getenv("RETRIEVAL_BATCH_SIZE", "20"))

# Vereditos possiveis do agente (ver CLAUDE.md: regra de abstencao).
VEREDITOS = ("CONFORME", "NAO_CONFORME", "NAO_APLICAVEL", "INDETERMINADO")

# As quatro configuracoes de recuperacao do experimento central do TFG.
RETRIEVAL_STRATEGIES = ("A", "B", "C", "D")
