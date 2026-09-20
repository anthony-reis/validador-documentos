"""Testes da análise em segundo plano (sem LLM/índice reais: as funções
pesadas são substituídas por funções determinísticas de teste, e isso é
declarado aqui -- o pipeline real é coberto por tests/test_agente*.py)."""

from __future__ import annotations

import time

from src.agent import analise_job
from src.agent.analise_job import AnaliseJob
from src.agent.schemas import JulgamentoAssercao


def _esperar(job: AnaliseJob, timeout: float = 10) -> None:
    limite = time.time() + timeout
    while not job.instantaneo()["concluido"] and time.time() < limite:
        time.sleep(0.02)
    assert job.instantaneo()["concluido"]


def _preparar(monkeypatch, paginas, atraso=0.0):
    monkeypatch.setattr(analise_job, "parse_documento", lambda caminho: paginas)
    monkeypatch.setattr(analise_job, "extrair_assercoes", lambda ps: [f"{ps[0]}-a{i}" for i in range(2)])

    class R:
        def buscar_lote(self, qs, top_k):
            return [[] for _ in qs]

    monkeypatch.setattr(analise_job, "criar_retriever", lambda e: R())

    def julgar(resultados, assercao):
        time.sleep(atraso)
        return JulgamentoAssercao(assercao=assercao, veredito="INDETERMINADO", citacao=None, justificativa="")

    monkeypatch.setattr(analise_job, "julgar_assercao_com_resultados", julgar)
    monkeypatch.setattr(analise_job.chat, "construir_indice", lambda ps: "indice")


def test_job_julga_todas_as_assercoes_em_segundo_plano(tmp_path, monkeypatch):
    _preparar(monkeypatch, ["p1", "p2"])
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"x")
    job = AnaliseJob("x.pdf", "D", pdf)
    job.iniciar()
    _esperar(job)
    estado = job.instantaneo()
    assert estado["erro"] is None
    assert [j.assercao for j in estado["julgamentos"]] == ["p1-a0", "p1-a1", "p2-a0", "p2-a1"]
    assert job.indice_chat == "indice"
    assert not pdf.exists()  # arquivo temporário removido ao fim


def test_job_pode_ser_cancelado_e_mantem_o_que_ja_foi_julgado(tmp_path, monkeypatch):
    _preparar(monkeypatch, ["p1", "p2", "p3"], atraso=0.2)
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"x")
    job = AnaliseJob("x.pdf", "D", pdf)
    job.iniciar()
    time.sleep(0.3)
    job.cancelar()
    _esperar(job)
    n = len(job.instantaneo()["julgamentos"])
    assert 1 <= n < 6


def test_job_registra_erro_em_vez_de_morrer_calado(tmp_path, monkeypatch):
    _preparar(monkeypatch, ["p1"])

    def quebra(caminho):
        raise RuntimeError("pdf ruim")

    monkeypatch.setattr(analise_job, "parse_documento", quebra)
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"x")
    job = AnaliseJob("x.pdf", "D", pdf)
    job.iniciar()
    _esperar(job)
    assert "pdf ruim" in job.instantaneo()["erro"]
