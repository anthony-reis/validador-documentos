"""Chunking hierarquico sobre a estrutura juridica de normas brasileiras.

Decisao de pesquisa (ver CLAUDE.md > "Chunking"): um chunk que corta um
artigo no meio destrói a rastreabilidade da citação. Por isso a divisão
primária segue a estrutura do texto legal (Art. -> paragrafo), não um
tamanho fixo de caracteres. O limite de tamanho (CHUNK_SIZE/CHUNK_OVERLAP)
so entra como divisão secundária, dentro de um artigo/paragrafo já
isolado, quando o bloco continua grande demais para o modelo de
embedding.

Os marcadores estruturais (CAPÍTULO, Seção, Art., paragrafo, inciso) só
são reconhecidos no início de linha e com a grafia oficial ("Art." com A
maiúsculo). Isso é deliberado: o preâmbulo de resoluções da ANVISA cita
artigos de OUTRAS normas em prosa, com grafia minúscula ("art. 187,
inciso VI") -- exigir maiúscula e início de linha evita que essas
referências cruzadas sejam confundidas com a estrutura do próprio
documento. Validado contra a RDC 658/2022: os 380 artigos encontrados
formam uma sequência 1..380 sem lacunas nem duplicatas.

**Bug real encontrado e corrigido com a RDC 166/2017**: essa norma tem
TRES grafias diferentes para o numero do artigo dentro do MESMO PDF --
"Art. 9°" (sinal de grau, U+00B0, nao "º" indicador ordinal U+00BA;
visualmente identicos, Unicode diferente), "Art. 10." (numero + ponto,
convencao padrao para artigos >= 10) e "Art. 11" em diante, SEM nenhuma
pontuacao apos o numero. Com o regex antigo exigindo "º" ou ".", so' 10
dos 71 artigos eram reconhecidos (o resto do documento virava um unico
bloco preso ao Art. 10). O caractere apos o numero agora e' OPCIONAL --
o que de fato distingue um artigo real de uma referencia cruzada em
prosa continua sendo maiuscula + inicio de linha (ver acima), entao
tornar a pontuacao opcional nao reintroduz esse problema.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.ingestion.texto_utils import dividir_por_tamanho, slug

_ARTIGO_RE = re.compile(r"^[ \t]*Art\.\s*(\d+)(?:[º°]|\.)?\s*", re.MULTILINE)
_PARAGRAFO_RE = re.compile(r"^[ \t]*§\s*(\d+[º°]?|único)\s*", re.MULTILINE)
_CAPITULO_RE = re.compile(r"^[ \t]*(CAPÍTULO\s+[IVXLCDM]+)\s*\n\s*(.+)$", re.MULTILINE)
_SECAO_RE = re.compile(r"^[ \t]*(Seção\s+[IVXLCDM]+)\s*\n\s*(.+)$", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    texto: str
    norma: str
    artigo: str
    paragrafo: str | None
    titulo_secao: str | None
    pagina: int


def _mapa_paginas(paginas_texto: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Concatena paginas e devolve os offsets [inicio, fim) de cada uma.

    Necessario para responder "em que pagina comeca este chunk?" depois
    de já termos concatenado tudo em uma unica string para os regex de
    estrutura rodarem sem interrupcao entre paginas.
    """
    texto_completo = ""
    faixas: list[tuple[int, int]] = []
    for pagina in paginas_texto:
        inicio = len(texto_completo)
        texto_completo += pagina
        faixas.append((inicio, len(texto_completo)))
    return texto_completo, faixas


def _pagina_do_offset(offset: int, faixas: list[tuple[int, int]]) -> int:
    for numero_menos_um, (inicio, fim) in enumerate(faixas):
        if inicio <= offset < fim:
            return numero_menos_um + 1
    return len(faixas)  # offset apos o fim do documento: atribui a ultima pagina


def _titulo_secao_por_offset(texto: str) -> list[tuple[int, str]]:
    """Lista (offset, "CAPÍTULO X - Título / Seção Y - Subtítulo") em ordem."""
    marcos: list[tuple[int, str]] = []
    capitulo_atual: str | None = None
    eventos = sorted(
        [(m.start(), "capitulo", f"{m.group(1)} - {m.group(2).strip()}") for m in _CAPITULO_RE.finditer(texto)]
        + [(m.start(), "secao", f"{m.group(1)} - {m.group(2).strip()}") for m in _SECAO_RE.finditer(texto)]
    )
    for offset, tipo, rotulo in eventos:
        if tipo == "capitulo":
            capitulo_atual = rotulo
            marcos.append((offset, capitulo_atual))
        else:
            combinado = f"{capitulo_atual} / {rotulo}" if capitulo_atual else rotulo
            marcos.append((offset, combinado))
    return marcos


def _titulo_secao_em(offset: int, marcos: list[tuple[int, str]]) -> str | None:
    atual = None
    for marco_offset, rotulo in marcos:
        if marco_offset > offset:
            break
        atual = rotulo
    return atual


def gerar_chunks(texto_paginas: list[str], norma: str) -> list[Chunk]:
    """Gera chunks hierarquicos a partir do texto (ja limpo) de cada pagina.

    `norma` identifica a fonte nos metadados (ex.: "RDC 658/2022") e no
    prefixo do `chunk_id` -- passado explicitamente pelo chamador porque
    o nome de arquivo/cabecalho varia entre normas (nacional x ICH) e nao
    ha um jeito confiavel de inferi-lo sem acoplar este modulo a formato
    de arquivo especifico.
    """
    texto, faixas_paginas = _mapa_paginas(texto_paginas)
    marcos_secao = _titulo_secao_por_offset(texto)

    artigos = list(_ARTIGO_RE.finditer(texto))
    chunks: list[Chunk] = []

    for i, artigo_match in enumerate(artigos):
        numero_artigo = artigo_match.group(1)
        inicio_artigo = artigo_match.start()
        fim_artigo = artigos[i + 1].start() if i + 1 < len(artigos) else len(texto)
        bloco_artigo = texto[inicio_artigo:fim_artigo]
        titulo_secao = _titulo_secao_em(inicio_artigo, marcos_secao)

        paragrafos = list(_PARAGRAFO_RE.finditer(bloco_artigo))
        if not paragrafos:
            sub_blocos = [(None, bloco_artigo, inicio_artigo)]
        else:
            sub_blocos = [(None, bloco_artigo[: paragrafos[0].start()], inicio_artigo)]
            for j, paragrafo_match in enumerate(paragrafos):
                fim_paragrafo = paragrafos[j + 1].start() if j + 1 < len(paragrafos) else len(bloco_artigo)
                texto_paragrafo = bloco_artigo[paragrafo_match.start() : fim_paragrafo]
                offset_paragrafo = inicio_artigo + paragrafo_match.start()
                sub_blocos.append((paragrafo_match.group(1), texto_paragrafo, offset_paragrafo))

        for numero_paragrafo, texto_bloco, offset_bloco in sub_blocos:
            texto_bloco = texto_bloco.strip()
            if not texto_bloco:
                continue
            pagina = _pagina_do_offset(offset_bloco, faixas_paginas)
            partes = dividir_por_tamanho(texto_bloco)
            for k, parte in enumerate(partes):
                sufixo_paragrafo = f"_p{numero_paragrafo}" if numero_paragrafo else ""
                sufixo_subchunk = f"_{k}" if len(partes) > 1 else ""
                chunk_id = f"{slug(norma)}_art{numero_artigo}{sufixo_paragrafo}{sufixo_subchunk}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        texto=parte,
                        norma=norma,
                        artigo=numero_artigo,
                        paragrafo=numero_paragrafo,
                        titulo_secao=titulo_secao,
                        pagina=pagina,
                    )
                )

    return chunks
