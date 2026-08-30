from pathlib import Path

import pytest

from src import config
from src.ingestion.chunking_perguntas_respostas import gerar_chunks_perguntas_respostas
from src.ingestion.loader import LinhaEstilizada, carregar_linhas_estilizadas
from src.ingestion.pipeline import ingerir_pdf

PR_166_PATH = config.DATA_NORMAS_DIR / "PR_RDC_166_2017.pdf"
GUIA_62_PATH = config.DATA_NORMAS_DIR / "GUIA_ANVISA_62_2023_GERENCIAMENTO_RISCOS.pdf"


def _linha(pagina: int, texto: str, negrito: bool) -> LinhaEstilizada:
    return LinhaEstilizada(pagina=pagina, texto=texto, negrito=negrito)


def test_ignora_entrada_de_sumario_nao_negrito():
    linhas = [
        _linha(1, "1.", False),
        _linha(1, "INTRODUÇÃO .......................................... 4", False),
        _linha(2, "1. Introdução", True),
        _linha(2, "Texto real do corpo.", False),
    ]
    chunks = gerar_chunks_perguntas_respostas(linhas, norma="Teste")
    assert len(chunks) == 1
    assert chunks[0].artigo == "1"
    assert "Texto real do corpo" in chunks[0].texto


def test_numero_e_titulo_juntos_no_mesmo_negrito_sao_reconhecidos():
    linhas = [
        _linha(1, "3.1.2. Pergunta completa em uma unica linha em negrito?", True),
        _linha(1, "Resposta em texto normal.", False),
    ]
    chunks = gerar_chunks_perguntas_respostas(linhas, norma="Teste")
    assert len(chunks) == 1
    assert chunks[0].artigo == "3.1.2"
    assert chunks[0].texto.startswith("Pergunta completa")
    assert "Resposta em texto normal" in chunks[0].texto


def test_marcador_sem_ponto_final_tambem_e_reconhecido():
    """Bug real: o mesmo documento mistura '3.1.2.' (com ponto) e
    '3.1.1 ' (sem ponto) para marcadores."""
    linhas = [
        _linha(1, "3.1.1 Pergunta sem ponto apos o numero?", True),
        _linha(1, "Resposta.", False),
    ]
    chunks = gerar_chunks_perguntas_respostas(linhas, norma="Teste")
    assert len(chunks) == 1
    assert chunks[0].artigo == "3.1.1"


def test_marcadores_consecutivos_sem_texto_entre_eles_nao_se_engolem():
    """Bug real encontrado validando contra o PDF: '3.4.5. PRECISÃO'
    imediatamente seguido por '3.4.5.1 <pergunta>' (sem texto normal
    entre eles) fazia o segundo marcador ser engolido como continuacao
    do titulo do primeiro."""
    linhas = [
        _linha(1, "3.4.5. PRECISÃO", True),
        _linha(1, "3.4.5.1 Primeira pergunta sobre precisão?", True),
        _linha(1, "Resposta da primeira pergunta.", False),
        _linha(1, "3.4.5.2. Segunda pergunta sobre precisão?", True),
        _linha(1, "Resposta da segunda pergunta.", False),
    ]
    chunks = gerar_chunks_perguntas_respostas(linhas, norma="Teste")
    numeros = [c.artigo for c in chunks]
    assert numeros == ["3.4.5", "3.4.5.1", "3.4.5.2"]
    assert chunks[0].texto == "PRECISÃO"
    assert "Primeira pergunta" in chunks[1].texto
    assert "Segunda pergunta" in chunks[2].texto


def test_titulo_secao_reflete_a_hierarquia_de_ancestrais():
    linhas = [
        _linha(1, "3. PERGUNTAS E RESPOSTAS", True),
        _linha(1, "3.1.", True),
        _linha(1, "CAPÍTULO I – DAS DISPOSIÇÕES INICIAIS", True),
        _linha(1, "3.1.1 Pergunta do capitulo I?", True),
        _linha(1, "Resposta.", False),
    ]
    chunks = gerar_chunks_perguntas_respostas(linhas, norma="Teste")
    pergunta = [c for c in chunks if c.artigo == "3.1.1"][0]
    assert pergunta.titulo_secao == "PERGUNTAS E RESPOSTAS / CAPÍTULO I – DAS DISPOSIÇÕES INICIAIS"


@pytest.mark.parametrize(
    "caminho, norma",
    [
        (PR_166_PATH, "Perguntas e Respostas RDC 166/2017"),
        (GUIA_62_PATH, "Guia ANVISA 62/2023"),
    ],
)
class TestIngestaoRealDeDocumentosPerguntasERespostas:
    """Roda contra os PDFs reais baixados (ver README > setup do corpus)."""

    def test_documento_existe_e_gera_chunks_com_conteudo(self, caminho: Path, norma: str):
        if not caminho.exists():
            pytest.skip(f"{caminho.name} ausente em data/normas/")
        chunks = ingerir_pdf(caminho, norma=norma, formato="perguntas_respostas")
        assert len(chunks) > 50
        assert all(chunk.texto.strip() for chunk in chunks)
        assert all(chunk.norma == norma for chunk in chunks)

    def test_chunk_ids_sao_unicos(self, caminho: Path, norma: str):
        if not caminho.exists():
            pytest.skip(f"{caminho.name} ausente em data/normas/")
        chunks = ingerir_pdf(caminho, norma=norma, formato="perguntas_respostas")
        ids = [chunk.chunk_id for chunk in chunks]
        assert len(ids) == len(set(ids))


def test_secao_1_do_guia_anvisa_62_tem_conteudo_correto():
    if not GUIA_62_PATH.exists():
        pytest.skip("GUIA_ANVISA_62_2023 ausente em data/normas/")
    linhas = carregar_linhas_estilizadas(GUIA_62_PATH)
    chunks = gerar_chunks_perguntas_respostas(linhas, norma="Guia ANVISA 62/2023")
    secao_1 = [c for c in chunks if c.artigo == "1"][0]
    assert "gerenciamento de riscos" in secao_1.texto.lower()
