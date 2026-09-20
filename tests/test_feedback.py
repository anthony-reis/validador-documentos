"""Testes do feedback humano por asserção (store, exportação e UI)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from pydantic import ValidationError
from streamlit.testing.v1 import AppTest

from src import config
from src.feedback import exportar, store
from tests.test_streamlit_app import CAMINHO_APP, _julgamentos_sinteticos


def _fb(**kw) -> store.FeedbackAssercao:
    base = dict(
        documento="d.pdf", estrategia="D", assercao="a1", veredito_agente="CONFORME", veredito_correto=True,
        norma="RDC 658/2022", artigo="6",
    )
    return store.FeedbackAssercao(**{**base, **kw})


def test_salvar_e_obter_e_upsert(tmp_path):
    db = tmp_path / "fb.db"
    store.salvar(_fb(), db)
    assert store.obter("d.pdf", "D", "a1", db).veredito_correto is True
    store.salvar(_fb(veredito_correto=False, veredito_certo="NAO_CONFORME", justificativa="porque"), db)
    assert len(store.listar(db)) == 1
    assert store.obter("d.pdf", "D", "a1", db).veredito_certo == "NAO_CONFORME"
    assert store.obter("d.pdf", "D", "outra", db) is None


def test_incorreto_exige_veredito_certo_diferente_e_justificativa():
    with pytest.raises(ValidationError):
        _fb(veredito_correto=False, justificativa="x")
    with pytest.raises(ValidationError):
        _fb(veredito_correto=False, veredito_certo="CONFORME", justificativa="x")
    with pytest.raises(ValidationError):
        _fb(veredito_correto=False, veredito_certo="NAO_CONFORME", justificativa="  ")


def test_exportar_usa_veredito_do_agente_se_correto_senao_o_corrigido(tmp_path):
    db = tmp_path / "fb.db"
    store.salvar(_fb(assercao="a1"), db)
    store.salvar(_fb(assercao="a2", veredito_correto=False, veredito_certo="INDETERMINADO", justificativa="j"), db)
    saida = tmp_path / "out.csv"
    assert exportar.exportar(saida, db) == 2
    linhas = list(csv.DictReader(open(saida, encoding="utf-8")))
    assert list(linhas[0]) == exportar.CABECALHO
    assert [l["veredito_esperado"] for l in linhas] == ["CONFORME", "INDETERMINADO"]


def _botao_salvar(at: AppTest):
    return next(b for b in at.button if b.label == "Salvar feedback")


def _app(tmp_path, monkeypatch) -> AppTest:
    monkeypatch.setattr(config, "FEEDBACK_DB_PATH", tmp_path / "fb.db")
    at = AppTest.from_file(CAMINHO_APP)
    at.session_state["julgamentos"] = _julgamentos_sinteticos()
    at.session_state["estrategia_usada"] = "D"
    at.session_state["documento_analisado"] = "teste.pdf"
    at.run(timeout=60)
    return at


def test_ui_salva_feedback_incorreto_e_marca_revisado(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch)
    assert not at.exception
    at.radio[0].set_value("Incorreto")
    at.text_area[0].set_value("A norma exige o contrário.")
    _botao_salvar(at).click()
    at.run(timeout=60)
    assert not at.exception
    salvo = store.listar(tmp_path / "fb.db")
    assert len(salvo) == 1
    assert salvo[0].veredito_correto is False and salvo[0].justificativa == "A norma exige o contrário."
    assert salvo[0].chunk_id == "RDC-658-2022_art238"


def test_ui_rejeita_incorreto_sem_justificativa(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch)
    at.radio[0].set_value("Incorreto")
    _botao_salvar(at).click()
    at.run(timeout=60)
    assert not at.exception
    assert store.listar(tmp_path / "fb.db") == []
    assert any("justificativa" in e.value for e in at.error)
