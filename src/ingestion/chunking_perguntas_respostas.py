"""Chunking hierarquico para documentos de Perguntas e Respostas da
ANVISA (ex.: "Perguntas e Respostas - RDC 166/2017 e Guia 10/2017").

Terceira familia de chunking do projeto (ver CLAUDE.md > "Duas familias
de chunking"): nem Art./paragrafo (normas brasileiras) nem secoes
numeradas "puras" como o ICH (numero sozinho numa linha em negrito,
titulo em linhas negrito SEPARADAS). Aqui o numero e o texto da
pergunta/titulo aparecem JUNTOS no MESMO trecho em negrito -- ex.:
"3.1.2. Em casos que não se tratam de registro de IFA: [...]?" -- com a
resposta em texto regular ate' o proximo marcador.

O sumario deste documento NAO usa negrito (diferente do ICH Q10, onde
titulos de capitulo aparecem em negrito tanto no sumario quanto no
corpo) -- entao negrito sozinho ja' basta para distinguir um cabecalho
real de uma entrada de sumario, sem precisar do filtro de pontilhado
usado em chunking_numerado.py.

Cada pergunta numerada vira um chunk citavel; marcadores de nivel mais
raso sem corpo proprio (capitulos, ex. "3.1. CAPÍTULO I...") ainda geram
um chunk minimo (so' o titulo), preservando a hierarquia como
titulo_secao para as perguntas aninhadas embaixo.

**Limitacoes conhecidas** (documento validado contra o PDF real da
"Perguntas e Respostas - RDC 166/2017 e Guia 10/2017", 139 marcadores
encontrados): (1) o `titulo_secao` fica impreciso para os ultimos
marcadores de nivel raso do documento (a partir de "3.9" ate' "3.11" --
ANEXO III, GUIA, OUTRAS DUVIDAS), provavelmente por outra variacao de
formatacao ainda nao mapeada nessa cauda especifica do PDF -- o
CONTEUDO retornado desses chunks continua correto, so' o rotulo de
contexto hierarquico e' que fica desatualizado. (2) Um numero pequeno de
falsos positivos (ex. "5.5.3.1.4", uma citacao de capitulo da
Farmacopeia dentro de uma resposta) vira marcador espurio por
coincidir com o mesmo padrao numerico em negrito -- gera um chunk extra
de baixa relevancia, nao corrompe o restante. Ambos aceitos como
limitacao de corpus documentada (mesmo espirito do defeito de numeracao
do ICH Q10 na Fase 1), nao bloqueiam o uso pratico do documento.
"""

from __future__ import annotations

import re

from src.ingestion.chunking import Chunk
from src.ingestion.loader import LinhaEstilizada
from src.ingestion.texto_utils import dividir_por_tamanho, slug

# O ponto final apos o numero e' OPCIONAL: o mesmo documento mistura
# "3.1.2." (com ponto) e "3.1.1 " (sem ponto) para marcadores -- mesma
# inconsistencia real ja encontrada na RDC 166/2017 principal (ver
# chunking.py). Espaco depois do numero e' obrigatorio para nao casar um
# numero em prosa por acidente.
_MARCADOR_RE = re.compile(r"^(\d+(?:\.\d+){0,4})\.?\s+(.+)$", re.DOTALL)
# Usado so' para checar se uma linha ISOLADA comeca um NOVO marcador --
# ver nota abaixo sobre o bug de marcadores consecutivos sem texto entre
# eles (ex.: "3.4.5. PRECISÃO" seguido imediatamente por "3.4.5.1 ...").
# "(?:\s|$)" (nao so' "\s"): uma linha que e' SO' o numero, sem titulo na
# mesma linha (padrao tipo ICH, ex. "3.1." sozinho seguido do titulo em
# outra linha negrito), nao tem nenhum espaco apos o numero pra casar --
# sem o "$" essa linha nao era reconhecida como inicio de marcador e
# ficava grudada como continuacao do marcador anterior (bug real
# encontrado escrevendo os testes).
_INICIO_MARCADOR_RE = re.compile(r"^\d+(?:\.\d+){0,4}\.?(?:\s|$)")


def gerar_chunks_perguntas_respostas(linhas: list[LinhaEstilizada], norma: str) -> list[Chunk]:
    secoes: list[dict] = []
    secao_atual: dict | None = None
    titulo_por_nivel: dict[int, str] = {}

    i = 0
    total = len(linhas)
    while i < total:
        linha = linhas[i]

        if linha.negrito:
            # Agrupa linhas em negrito consecutivas em um "run" -- mas
            # PARA se uma linha subsequente ja' comeca um NOVO marcador
            # (ex.: "3.4.5. PRECISÃO" imediatamente seguido, sem texto
            # normal entre eles, por "3.4.5.1. <pergunta>"). Sem essa
            # checagem, o marcador seguinte era engolido inteiro como
            # continuacao do titulo do marcador atual -- bug real
            # encontrado validando contra o PDF (3.4.5.1 desaparecia).
            j = i + 1
            partes: list[str] = [linhas[i].texto.strip()]
            while j < total and linhas[j].negrito and not _INICIO_MARCADOR_RE.match(linhas[j].texto.strip()):
                partes.append(linhas[j].texto.strip())
                j += 1
            texto_run = " ".join(partes).strip()
            m = _MARCADOR_RE.match(texto_run)

            if m:
                numero = m.group(1)
                pergunta_ou_titulo = m.group(2).strip()
                nivel = numero.count(".") + 1

                titulo_por_nivel[nivel] = pergunta_ou_titulo
                for n in list(titulo_por_nivel):
                    if n > nivel:
                        del titulo_por_nivel[n]
                ancestrais = [titulo_por_nivel[n] for n in sorted(titulo_por_nivel) if n < nivel]
                titulo_secao = " / ".join(ancestrais) if ancestrais else None

                secao_atual = {
                    "numero": numero,
                    "pergunta": pergunta_ou_titulo,
                    "titulo_secao": titulo_secao,
                    "pagina": linha.pagina,
                    "corpo": [],
                }
                secoes.append(secao_atual)
                i = j
                continue

            # Negrito que nao e' um marcador numerado (enfase dentro de
            # uma resposta) -- vira conteudo normal da secao atual.
            if secao_atual is not None:
                secao_atual["corpo"].append(texto_run)
            i = j
            continue

        texto = linha.texto.strip()
        if secao_atual is not None and texto and not texto.isdigit():
            secao_atual["corpo"].append(texto)
        i += 1

    chunks: list[Chunk] = []
    for secao in secoes:
        corpo = "\n".join(secao["corpo"]).strip()
        texto_completo = f"{secao['pergunta']}\n{corpo}".strip() if corpo else secao["pergunta"]
        partes = dividir_por_tamanho(texto_completo)
        for k, parte in enumerate(partes):
            sufixo = f"_{k}" if len(partes) > 1 else ""
            chunk_id = f"{slug(norma)}_q{secao['numero']}{sufixo}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    texto=parte,
                    norma=norma,
                    artigo=secao["numero"],
                    paragrafo=None,
                    titulo_secao=secao["titulo_secao"],
                    pagina=secao["pagina"],
                )
            )
    return chunks
