"""Testes do chat exploratorio (src/agent/chat.py).

Todos monkeypatcham `gerar_embeddings`/`stream_chat` -- rapidos,
deterministicos, nao dependem do modelo de embedding nem do Ollama
rodando (mesmo padrao de tests/test_agent.py)."""

from __future__ import annotations

import numpy as np

from src.agent import chat
from src.agent.schemas import JulgamentoAssercao


def _vetor(*valores: float) -> list[float]:
    return list(valores)


def test_construir_indice_ignora_paginas_vazias_e_gera_um_embedding_por_trecho(monkeypatch):
    monkeypatch.setattr(chat, "dividir_por_tamanho", lambda texto: [texto])
    chamadas = []

    def _gerar_embeddings_falso(textos):
        chamadas.append(textos)
        return [_vetor(1.0, 0.0) for _ in textos]

    monkeypatch.setattr(chat, "gerar_embeddings", _gerar_embeddings_falso)

    indice = chat.construir_indice(["pagina 1", "   ", "pagina 2"])

    assert indice.trechos == ["pagina 1", "pagina 2"]
    assert indice.embeddings.shape == (2, 2)
    assert chamadas == [["pagina 1", "pagina 2"]]


def test_construir_indice_com_documento_vazio_nao_chama_embeddings(monkeypatch):
    def _nao_deveria_ser_chamado(textos):
        raise AssertionError("gerar_embeddings nao deveria ser chamado para documento vazio")

    monkeypatch.setattr(chat, "gerar_embeddings", _nao_deveria_ser_chamado)

    indice = chat.construir_indice(["   ", ""])
    assert indice.trechos == []


def test_buscar_trechos_retorna_os_mais_similares_em_ordem(monkeypatch):
    indice = chat.IndiceDocumentoChat(
        trechos=["sobre limpeza", "sobre treinamento", "sobre embalagem"],
        embeddings=np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]),
    )
    monkeypatch.setattr(chat, "gerar_embeddings", lambda textos: [_vetor(1.0, 0.0)])

    resultado = chat.buscar_trechos(indice, "pergunta sobre limpeza", top_k=2)

    assert resultado == ["sobre limpeza", "sobre embalagem"]


def test_buscar_trechos_em_indice_vazio_nao_chama_embeddings(monkeypatch):
    def _nao_deveria_ser_chamado(textos):
        raise AssertionError("gerar_embeddings nao deveria ser chamado para indice vazio")

    monkeypatch.setattr(chat, "gerar_embeddings", _nao_deveria_ser_chamado)

    indice = chat.IndiceDocumentoChat(trechos=[], embeddings=np.zeros((0, 0)))
    assert chat.buscar_trechos(indice, "qualquer pergunta") == []


def test_formatar_relatorio_lista_um_item_por_julgamento():
    julgamentos = [
        JulgamentoAssercao(assercao="a1", veredito="CONFORME", citacao=None, justificativa="ok"),
        JulgamentoAssercao(assercao="a2", veredito="INDETERMINADO", citacao=None, justificativa=""),
    ]
    resumo = chat.formatar_relatorio(julgamentos)
    assert "- [CONFORME] a1 -- ok" in resumo
    assert "- [INDETERMINADO] a2" in resumo


def test_formatar_relatorio_sem_julgamentos_retorna_texto_explicativo():
    assert "nenhuma assercao" in chat.formatar_relatorio([]).lower()


def test_responder_stream_monta_mensagens_com_historico_trechos_e_resumo(monkeypatch):
    monkeypatch.setattr(chat, "buscar_trechos", lambda indice, pergunta, top_k=5: ["trecho relevante"])
    monkeypatch.setattr(chat, "formatar_relatorio", lambda julgamentos: "resumo do relatorio")

    mensagens_capturadas = []

    def _stream_chat_falso(mensagens):
        mensagens_capturadas.extend(mensagens)
        yield "ola"
        yield " mundo"

    monkeypatch.setattr(chat, "stream_chat", _stream_chat_falso)

    historico = [
        {"role": "user", "content": "pergunta anterior"},
        {"role": "assistant", "content": "resposta anterior"},
    ]
    indice = chat.IndiceDocumentoChat(trechos=[], embeddings=np.zeros((0, 0)))
    resposta = list(chat.responder_stream("nova pergunta", historico, indice, []))

    assert resposta == ["ola", " mundo"]
    assert mensagens_capturadas[0][0] == "system"
    assert mensagens_capturadas[1] == ("human", "pergunta anterior")
    assert mensagens_capturadas[2] == ("ai", "resposta anterior")
    papel_final, conteudo_final = mensagens_capturadas[3]
    assert papel_final == "human"
    assert "trecho relevante" in conteudo_final
    assert "resumo do relatorio" in conteudo_final
    assert "nova pergunta" in conteudo_final
