"""Verificação final de restrição offline (Fase 7).

Ver CLAUDE.md > "Restrição inegociável: 100% offline" e
PROMPT-CLAUDE-CODE.md > Parte 3 > "Verificação final de offline":
"Escreva e execute um teste que bloqueia toda a rede (monkeypatch em
socket.socket) e roda uma consulta ponta a ponta. Se qualquer parte
tentar acessar a internet, quero ver a falha."

Loopback (127.0.0.1/::1) NÃO conta como "rede" para essa restrição: é
assim que o próprio Ollama local é consultado (HTTP sobre 127.0.0.1).
Bloquear indiscriminadamente TODA conexão de socket, incluindo
loopback, impediria o próprio requisito de funcionar com um LLM local
via Ollama -- por isso o bloqueio aqui é seletivo: qualquer destino que
NÃO seja loopback derruba o teste com uma mensagem clara.
"""

from __future__ import annotations

import os
import socket

import pytest

from src import config

_ENDERECOS_LOCAIS = {"127.0.0.1", "::1", "localhost"}


def _bloquear_rede_externa(monkeypatch: pytest.MonkeyPatch) -> None:
    connect_original = socket.socket.connect

    def connect_vigiado(self, endereco, *args, **kwargs):
        host = endereco[0] if isinstance(endereco, tuple) else endereco
        if host not in _ENDERECOS_LOCAIS:
            raise RuntimeError(
                f"Tentativa de conexão de rede para host EXTERNO bloqueada: {host!r}. "
                "O sistema deve rodar 100% offline em produção (ver CLAUDE.md)."
            )
        return connect_original(self, endereco, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect_vigiado)


def test_variaveis_de_ambiente_offline_estao_ativas():
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


class TestRecuperacaoPontaAPontaSemRedeExterna:
    """As quatro estratégias (A/B/C/D) não chamam o LLM -- só embeddings
    e reranker locais. Roda sempre, não depende de Ollama."""

    @pytest.fixture(autouse=True)
    def _indice_isolado_offline(self, tmp_path, monkeypatch):
        if not config.EMBEDDING_MODEL_PATH.exists() or not config.RERANKER_MODEL_PATH.exists():
            pytest.skip("modelos locais (bge-m3 / bge-reranker-v2-m3) ausentes -- ver README > setup")

        from src.indexing import esparso as indice_esparso
        from src.indexing import vetorial
        from src.ingestion.chunking import Chunk

        monkeypatch.setattr(config, "CHROMA_PERSIST_DIR", tmp_path / "chroma_offline")
        caminho_bm25 = tmp_path / "bm25_offline" / "indice.pkl"
        monkeypatch.setattr(indice_esparso, "CAMINHO_PADRAO", caminho_bm25)

        chunks = [
            Chunk(
                chunk_id="t1",
                texto="Art. 1º Todo lote deve ser aprovado pela Garantia da Qualidade antes da liberação.",
                norma="Norma Teste",
                artigo="1",
                paragrafo=None,
                titulo_secao=None,
                pagina=1,
            )
        ]
        vetorial.indexar(chunks, recriar=True)
        indice_esparso.salvar_indice(indice_esparso.construir_indice(chunks), caminho=caminho_bm25)

        _bloquear_rede_externa(monkeypatch)

    @pytest.mark.parametrize("estrategia", ["A", "B", "C", "D"])
    def test_estrategia_funciona_com_rede_externa_bloqueada(self, estrategia):
        from src.retrieval.factory import criar_retriever

        retriever = criar_retriever(estrategia)
        resultados = retriever.buscar("aprovação de lote pela garantia da qualidade", top_k=1)
        assert resultados
        assert resultados[0].chunk_id == "t1"


class TestJulgamentoPontaAPontaSemRedeExterna:
    """Cobre o LLM via Ollama (loopback permitido, externo bloqueado).
    Lento (chama o LLM de verdade) -- so' roda com RUN_SLOW_LLM_TESTS=1,
    mesmo criterio ja usado em tests/test_agent.py."""

    @pytest.fixture(autouse=True)
    def _requer_ambiente_completo(self, tmp_path, monkeypatch):
        if os.environ.get("RUN_SLOW_LLM_TESTS") != "1":
            pytest.skip("defina RUN_SLOW_LLM_TESTS=1 para rodar o teste real de ponta a ponta (lento)")
        if not config.EMBEDDING_MODEL_PATH.exists():
            pytest.skip("modelo bge-m3 ausente -- ver README > setup")

        from src.indexing import esparso as indice_esparso
        from src.indexing import vetorial
        from src.ingestion.chunking import Chunk

        monkeypatch.setattr(config, "CHROMA_PERSIST_DIR", tmp_path / "chroma_offline_llm")
        caminho_bm25 = tmp_path / "bm25_offline_llm" / "indice.pkl"
        monkeypatch.setattr(indice_esparso, "CAMINHO_PADRAO", caminho_bm25)

        chunks = [
            Chunk(
                chunk_id="t1",
                texto="Art. 1º Todo lote deve ser aprovado pela Garantia da Qualidade antes da liberação.",
                norma="Norma Teste",
                artigo="1",
                paragrafo=None,
                titulo_secao=None,
                pagina=1,
            )
        ]
        vetorial.indexar(chunks, recriar=True)
        indice_esparso.salvar_indice(indice_esparso.construir_indice(chunks), caminho=caminho_bm25)

        _bloquear_rede_externa(monkeypatch)

    def test_julgamento_completo_funciona_com_rede_externa_bloqueada(self):
        """Se o LLM (Ollama local), o embedding ou o reranker tentarem
        alcançar QUALQUER host que não seja loopback, este teste falha
        com RuntimeError -- essa é a prova que a especificação pede."""
        from src.agent.nos import julgar_assercao
        from src.retrieval.factory import criar_retriever

        retriever = criar_retriever("D")
        julgamento = julgar_assercao(retriever, "o lote foi liberado pela produção sem aprovação da qualidade")

        assert julgamento.veredito in ("CONFORME", "NAO_CONFORME", "NAO_APLICAVEL", "INDETERMINADO")
        if julgamento.veredito != "INDETERMINADO":
            assert julgamento.citacao is not None
