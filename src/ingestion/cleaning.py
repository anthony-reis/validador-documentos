"""Limpeza de ruido tipografico de publicacoes reproduzidas do DOU.

PDFs de normas brasileiras (mesmo quando reproduzidos por terceiros a
partir do Diario Oficial da Uniao) repetem cabecalho/rodape de edicao em
toda pagina -- ex.: "Edicao no 73.2022 | Sao Paulo, 31 de marco de 2022"
e "Este conteudo nao substitui o publicado na versao certificada.".
Sem remover essas linhas, elas se intercalam com o texto juridico e
quebram os regex de estrutura (Art./paragrafo/inciso) na fase de
chunking -- por exemplo um "Art." pode terminar em uma pagina e o
cabecalho da pagina seguinte aparecer no meio do proprio artigo.
"""

from __future__ import annotations

import re

_PADROES_RUIDO = (
    re.compile(r"^\s*Edição\s*n[ºo]\s*\d.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(
        r"^\s*Este conteúdo não substitui o publicado na versão certificada\.?\s*$",
        re.MULTILINE | re.IGNORECASE,
    ),
    re.compile(r"^\s*Publicado em:.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*Órgão:.*$", re.MULTILINE | re.IGNORECASE),
)


def limpar_texto(texto: str) -> str:
    limpo = texto
    for padrao in _PADROES_RUIDO:
        limpo = padrao.sub("", limpo)
    return limpo
