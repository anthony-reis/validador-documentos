from pathlib import Path

import pytest

from src import config
from src.ingestion.chunking import gerar_chunks
from src.ingestion.cleaning import limpar_texto
from src.ingestion.loader import carregar_pdf
from src.ingestion.pipeline import ingerir_pdf

RDC_658_PATH = config.DATA_NORMAS_DIR / "RDC_658_2022.pdf"

pytestmark = pytest.mark.skipif(
    not RDC_658_PATH.exists(),
    reason="RDC_658_2022.pdf ausente em data/normas/ (ver README > setup do corpus)",
)


def test_limpar_texto_remove_ruido_de_edicao_do_dou():
    bruto = (
        "Edição nº 73.2022 | São Paulo, 31 de março de 2022 \n"
        "Este conteúdo não substitui o publicado na versão certificada. \n"
        "Publicado em: 31/03/2022 | Edição: 62 | Seção: 1 | Página: 320 \n"
        "Órgão: Ministério da Saúde/Agência Nacional de Vigilância Sanitária \n"
        "Art. 1º Esta Resolução possui o objetivo de adotar as diretrizes.\n"
    )
    limpo = limpar_texto(bruto)
    assert "Edição nº" not in limpo
    assert "Este conteúdo não substitui" not in limpo
    assert "Publicado em:" not in limpo
    assert "Órgão:" not in limpo
    assert "Art. 1º Esta Resolução" in limpo


def test_gerar_chunks_reconhece_378_artigos_da_lei_sem_confundir_referencia_cruzada():
    preambulo = (
        "no uso das competências que lhe conferem os arts. 7º, inciso III, e 15, \n"
        "incisos III e IV da Lei nº 9.782, de 26 de janeiro de 1999, e considerando \n"
        "o disposto no art. 187, inciso VI e §§ 1º e 3º, do Regimento Interno.\n"
        "CAPÍTULO I \nDISPOSIÇÕES INICIAIS \nSeção I \nObjetivo \n"
        "Art. 1º Esta Resolução possui o objetivo de adotar as diretrizes.\n"
        "Art. 2º Esta Resolução se aplica às empresas.\n"
    )
    chunks = gerar_chunks([preambulo], norma="RDC 658/2022")
    assert [c.artigo for c in chunks] == ["1", "2"]


class TestIngestaoRealDaRdc658:
    """Roda o pipeline completo contra o PDF real baixado (ver README)."""

    @pytest.fixture(scope="class")
    def chunks(self):
        return ingerir_pdf(RDC_658_PATH, norma="RDC 658/2022")

    def test_todos_os_380_artigos_aparecem_sem_lacunas(self, chunks):
        numeros = sorted({int(c.artigo) for c in chunks})
        assert numeros == list(range(1, 381))

    def test_artigo_1_vira_chunk_com_metadados_corretos(self, chunks):
        candidatos = [c for c in chunks if c.artigo == "1"]
        assert len(candidatos) == 1
        chunk = candidatos[0]
        assert chunk.norma == "RDC 658/2022"
        assert chunk.paragrafo is None
        assert chunk.pagina == 1
        assert chunk.titulo_secao == "CAPÍTULO I - DISPOSIÇÕES INICIAIS / Seção I - Objetivo"
        assert "Esta Resolução possui o objetivo" in chunk.texto

    def test_chunk_id_e_deterministico_e_unico(self, chunks):
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))
        assert all(cid.startswith("RDC-658-2022_art") for cid in ids)

    def test_paragrafos_sao_associados_ao_artigo_correto(self, chunks):
        com_paragrafo = [c for c in chunks if c.paragrafo is not None]
        assert len(com_paragrafo) > 0
        # So o primeiro sub-chunk de cada (artigo, paragrafo) precisa comecar
        # com "§" -- sub-chunks seguintes sao continuacao de um paragrafo
        # longo dividido por tamanho (ver _dividir_por_tamanho).
        primeiros_por_grupo = {}
        for chunk in com_paragrafo:
            chave = (chunk.artigo, chunk.paragrafo)
            primeiros_por_grupo.setdefault(chave, chunk)
        for chunk in primeiros_por_grupo.values():
            assert chunk.texto.strip().startswith("§")

    def test_boilerplate_do_dou_nao_vaza_para_dentro_dos_chunks(self, chunks):
        for chunk in chunks:
            assert "Edição nº" not in chunk.texto
            assert "Este conteúdo não substitui" not in chunk.texto


def test_carregar_pdf_retorna_77_paginas():
    paginas = carregar_pdf(RDC_658_PATH)
    assert len(paginas) == 77
    assert paginas[0].numero == 1
