"""Chunking hierarquico para documentos organizados em secoes numeradas
(ex.: guias do ICH), sem a estrutura Art./paragrafo das normas brasileiras.

Diferente de chunking.py (onde regex sobre texto puro basta, porque a
grafia "Art." e' inconfundivel), aqui regex sobre texto puro NAO basta:
um sumario ("1.1 Introdução .......... 1") tem exatamente o mesmo formato
"numero + titulo" de um cabecalho real no corpo do documento. O sinal
que de fato distingue os dois e' tipografico -- cabecalhos reais estao
em negrito; a maioria das entradas do sumario nao esta, EXCETO os
titulos de capitulo de nivel 1 ("1.", "2." etc.), que sao negrito nos
dois lugares. Por isso o filtro final usa uma segunda pista: entradas de
sumario carregam pontilhado de preenchimento (dot leaders, "....") antes
do numero de pagina, que um titulo real nunca tem.

Validado contra data/normas/ICH_Q10.pdf (ver testes): as 41 secoes do
sumario sao encontradas (1..1.7, 2..2.8, 3..3.2.4, 4..4.3, 5, Anexo 1,
Anexo 2). Nota de limitacao: o PDF de origem tem um defeito de
numeracao proprio a partir da secao 1.5.4 (o corpo do texto imprime
"1.5.4", "1.6.1", "1.6.2", "1.6", "1.7" onde o sumario indicaria "1.6",
"1.6.1", "1.6.2", "1.7", "1.8" -- o proprio PDF traz o numero correto
entre parenteses no titulo, ex.: "Facilitadores: ... (1.6)"). O chunker
extrai o numero exatamente como impresso no corpo, sem tentar corrigi-lo
via o parenteses -- alterar dados de origem silenciosamente seria pior
do que preservar um defeito conhecido e documentado (ver CLAUDE.md:
"nenhum numero é inventado"). Registrar como limitação do corpus no TFG.
"""

from __future__ import annotations

import re

from src.ingestion.chunking import Chunk
from src.ingestion.loader import LinhaEstilizada
from src.ingestion.texto_utils import dividir_por_tamanho, slug

_NUMERO_SECAO_RE = re.compile(r"^(?:\d+(?:\.\d+){0,3}\.?|Anexo\s+\d+)$")
_DOT_LEADER_RE = re.compile(r"\.{2,}")

# Cabecalho de pagina repetido em todo documento ICH (titulo do guia em
# italico, reimpresso a cada pagina) -- ruido, nao conteudo de secao.
_LINHAS_RUIDO = {"Sistema de Qualidade Farmacêutica"}


def gerar_chunks_numerados(linhas: list[LinhaEstilizada], norma: str) -> list[Chunk]:
    secoes: list[dict] = []
    secao_atual: dict | None = None
    i = 0
    total = len(linhas)

    while i < total:
        linha = linhas[i]
        texto = linha.texto.strip()

        if linha.negrito and _NUMERO_SECAO_RE.match(texto):
            numero = texto.rstrip(".")
            titulo_partes: list[str] = []
            j = i + 1
            while j < total and linhas[j].negrito and not _NUMERO_SECAO_RE.match(linhas[j].texto.strip()):
                titulo_partes.append(linhas[j].texto.strip())
                j += 1
            if titulo_partes:
                titulo = " ".join(titulo_partes).strip()
                valido = bool(titulo) and not _DOT_LEADER_RE.search(titulo)
            else:
                # Sem linha em negrito apos o marcador (ex.: "Anexo 2", cujo
                # titulo no PDF esta em fonte regular, nao em negrito) --
                # aceitamos mesmo assim, usando o proprio numero como rotulo.
                titulo = numero
                valido = True

            if valido:
                secao_atual = {"numero": numero, "titulo": titulo, "pagina": linha.pagina, "corpo": []}
                secoes.append(secao_atual)
                i = j
                continue
            # Marcador sem titulo real, ou titulo com pontilhado (entrada
            # de sumario): nao abre secao, apenas segue a varredura.
        elif secao_atual is not None and texto not in _LINHAS_RUIDO and not texto.isdigit():
            secao_atual["corpo"].append(texto)

        i += 1

    chunks: list[Chunk] = []
    for secao in secoes:
        corpo = "\n".join(secao["corpo"]).strip()
        texto_completo = f"{secao['titulo']}\n{corpo}".strip() if corpo else secao["titulo"]
        partes = dividir_por_tamanho(texto_completo)
        for k, parte in enumerate(partes):
            sufixo = f"_{k}" if len(partes) > 1 else ""
            chunk_id = f"{slug(norma)}_sec{secao['numero']}{sufixo}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    texto=parte,
                    norma=norma,
                    artigo=secao["numero"],
                    paragrafo=None,
                    titulo_secao=secao["titulo"],
                    pagina=secao["pagina"],
                )
            )
    return chunks
