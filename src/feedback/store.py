"""Armazenamento local (SQLite) do feedback humano por asserção.

SQLite (biblioteca padrão) em vez de JSONL: o revisor pode corrigir um
feedback já dado (upsert) e o exportador/calibrador precisam consultar
por filtros -- ambos incômodos em arquivo append-only. Continua 100%
offline e sem dependência nova.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator

from src import config

Veredito = Literal["CONFORME", "NAO_CONFORME", "NAO_APLICAVEL", "INDETERMINADO"]


class FeedbackAssercao(BaseModel):
    documento: str
    estrategia: Literal["A", "B", "C", "D"]
    assercao: str
    veredito_agente: Veredito
    veredito_correto: bool
    # Só preenchido quando o revisor discorda do agente.
    veredito_certo: Veredito | None = None
    citacao_sustenta: bool | None = None
    justificativa: str = ""
    # Contexto do julgamento, congelado no momento do feedback: reindexar
    # o corpus depois muda hash/chunks, e o feedback precisa dizer sobre
    # qual versão foi dado.
    modelo_llm: str = ""
    hash_corpus: str = ""
    norma: str | None = None
    artigo: str | None = None
    chunk_id: str | None = None
    score_recuperacao: float | None = None
    trecho_literal: str | None = None
    motivo_abstencao: str | None = None
    criado_em: datetime | None = None

    @model_validator(mode="after")
    def _exige_coerencia(self) -> "FeedbackAssercao":
        # Discordar sem dizer o veredito certo e sem justificar gera um
        # rótulo inutilizável para gabarito/treino.
        if not self.veredito_correto:
            if self.veredito_certo is None:
                raise ValueError("veredito_certo é obrigatório quando o veredito do agente está incorreto")
            if self.veredito_certo == self.veredito_agente:
                raise ValueError("veredito_certo não pode ser igual ao veredito do agente")
            if not self.justificativa.strip():
                raise ValueError("justificativa é obrigatória quando o veredito do agente está incorreto")
        return self


_COLUNAS = list(FeedbackAssercao.model_fields)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    documento TEXT NOT NULL,
    estrategia TEXT NOT NULL,
    assercao TEXT NOT NULL,
    veredito_agente TEXT NOT NULL,
    veredito_correto INTEGER NOT NULL,
    veredito_certo TEXT,
    citacao_sustenta INTEGER,
    justificativa TEXT NOT NULL,
    modelo_llm TEXT NOT NULL,
    hash_corpus TEXT NOT NULL,
    norma TEXT,
    artigo TEXT,
    chunk_id TEXT,
    score_recuperacao REAL,
    trecho_literal TEXT,
    motivo_abstencao TEXT,
    criado_em TEXT NOT NULL,
    PRIMARY KEY (documento, estrategia, assercao)
)
"""


def _resolver_caminho(caminho: Path | None) -> Path:
    # Resolvido a cada chamada (não como default do parâmetro): um
    # monkeypatch em config.FEEDBACK_DB_PATH nos testes precisa valer.
    return Path(caminho) if caminho is not None else config.FEEDBACK_DB_PATH


def _conectar(caminho: Path | None) -> sqlite3.Connection:
    destino = _resolver_caminho(caminho)
    destino.parent.mkdir(parents=True, exist_ok=True)
    conexao = sqlite3.connect(destino)
    conexao.row_factory = sqlite3.Row
    conexao.execute(_SCHEMA)
    return conexao


def _para_feedback(linha: sqlite3.Row) -> FeedbackAssercao:
    dados = dict(linha)
    dados["veredito_correto"] = bool(dados["veredito_correto"])
    if dados["citacao_sustenta"] is not None:
        dados["citacao_sustenta"] = bool(dados["citacao_sustenta"])
    return FeedbackAssercao(**dados)


def salvar(feedback: FeedbackAssercao, caminho: Path | None = None) -> None:
    """Insere ou substitui (upsert) o feedback da tripla
    (documento, estratégia, asserção)."""
    dados = feedback.model_dump()
    dados["criado_em"] = (feedback.criado_em or datetime.now()).isoformat(timespec="seconds")
    dados["veredito_correto"] = int(feedback.veredito_correto)
    if feedback.citacao_sustenta is not None:
        dados["citacao_sustenta"] = int(feedback.citacao_sustenta)
    marcadores = ", ".join(f":{c}" for c in _COLUNAS)
    with closing(_conectar(caminho)) as conexao, conexao:
        conexao.execute(
            f"INSERT OR REPLACE INTO feedback ({', '.join(_COLUNAS)}) VALUES ({marcadores})", dados
        )


def obter(documento: str, estrategia: str, assercao: str, caminho: Path | None = None) -> FeedbackAssercao | None:
    with closing(_conectar(caminho)) as conexao:
        linha = conexao.execute(
            "SELECT * FROM feedback WHERE documento=? AND estrategia=? AND assercao=?",
            (documento, estrategia, assercao),
        ).fetchone()
    return _para_feedback(linha) if linha else None


def listar(caminho: Path | None = None) -> list[FeedbackAssercao]:
    with closing(_conectar(caminho)) as conexao:
        linhas = conexao.execute("SELECT * FROM feedback ORDER BY criado_em").fetchall()
    return [_para_feedback(linha) for linha in linhas]
