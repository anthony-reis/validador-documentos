import pytest

from src import config
from src.ingestion.chunking_numerado import gerar_chunks_numerados
from src.ingestion.loader import LinhaEstilizada, carregar_linhas_estilizadas
from src.ingestion.pipeline import ingerir_pdf

ICH_Q10_PATH = config.DATA_NORMAS_DIR / "ICH_Q10.pdf"

pytestmark = pytest.mark.skipif(
    not ICH_Q10_PATH.exists(),
    reason="ICH_Q10.pdf ausente em data/normas/ (ver README > setup do corpus)",
)


def _linha(pagina: int, texto: str, negrito: bool) -> LinhaEstilizada:
    return LinhaEstilizada(pagina=pagina, texto=texto, negrito=negrito)


def test_ignora_entrada_de_sumario_com_pontilhado():
    linhas = [
        _linha(1, "1.1", False),
        _linha(1, "Introdução .......................................... 1", False),
        _linha(2, "1.1", True),
        _linha(2, "Introdução", True),
        _linha(2, "Texto real do corpo da secao.", False),
    ]
    chunks = gerar_chunks_numerados(linhas, norma="ICH Q10")
    assert len(chunks) == 1
    assert chunks[0].artigo == "1.1"
    assert "Texto real do corpo" in chunks[0].texto


def test_nao_engole_subsecao_seguinte_sem_corpo_entre_marcadores():
    linhas = [
        _linha(1, "1.", True),
        _linha(1, "TÍTULO DO CAPÍTULO", True),
        _linha(1, "1.1", True),
        _linha(1, "Introdução", True),
        _linha(1, "Corpo da introducao.", False),
    ]
    chunks = gerar_chunks_numerados(linhas, norma="ICH Q10")
    numeros = [c.artigo for c in chunks]
    assert numeros == ["1", "1.1"]
    assert chunks[0].titulo_secao == "TÍTULO DO CAPÍTULO"
    assert chunks[1].titulo_secao == "Introdução"


def test_marcador_sem_titulo_em_negrito_usa_o_proprio_numero_como_rotulo():
    linhas = [
        _linha(1, "Anexo 2", True),
        _linha(1, "Título do anexo em fonte regular, não negrito.", False),
        _linha(1, "Corpo do anexo.", False),
    ]
    chunks = gerar_chunks_numerados(linhas, norma="ICH Q10")
    assert len(chunks) == 1
    assert chunks[0].artigo == "Anexo 2"


class TestIngestaoRealDoIchQ10:
    @pytest.fixture(scope="class")
    def chunks(self):
        return ingerir_pdf(ICH_Q10_PATH, norma="ICH Q10", formato="secoes_numeradas")

    def test_todas_as_41_secoes_do_sumario_sao_encontradas(self, chunks):
        esperadas = {
            "1", "1.1", "1.2", "1.3", "1.4", "1.5", "1.5.1", "1.5.2", "1.5.3", "1.5.4",
            "1.6", "1.6.1", "1.6.2", "1.7",
            "2", "2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7", "2.8",
            "3", "3.1", "3.1.1", "3.1.2", "3.1.3", "3.1.4", "3.2", "3.2.1", "3.2.2", "3.2.3", "3.2.4",
            "4", "4.1", "4.2", "4.3",
            "5", "Anexo 1", "Anexo 2",
        }
        encontradas = {c.artigo for c in chunks}
        assert encontradas == esperadas

    def test_secao_introducao_tem_metadados_corretos(self, chunks):
        candidatos = [c for c in chunks if c.artigo == "1.1"]
        assert len(candidatos) >= 1
        chunk = candidatos[0]
        assert chunk.norma == "ICH Q10"
        assert chunk.paragrafo is None
        assert chunk.pagina == 5
        assert "Este documento estabelece" in chunk.texto

    def test_chunk_id_e_deterministico_e_unico(self, chunks):
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))
        assert all(cid.startswith("ICH-Q10_sec") for cid in ids)


def test_carregar_linhas_estilizadas_marca_cabecalhos_reais_como_negrito():
    linhas = carregar_linhas_estilizadas(ICH_Q10_PATH)
    negritos = [l for l in linhas if l.negrito and l.texto.strip() == "1.1"]
    assert len(negritos) >= 1
