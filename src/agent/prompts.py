"""Prompts do agente. Cada um documenta por que tem essa forma -- ver
CLAUDE.md > "Como trabalhar neste projeto": decisao de prompt leva
comentario de motivo.
"""

SISTEMA_EXTRACAO_ASSERCOES = """\
Voce le trechos de documentos da industria farmaceutica (POPs, \
relatorios, protocolos) e extrai afirmacoes factuais verificaveis -- \
frases que descrevem o que foi feito, por quem, quando ou como, e que \
poderiam ser avaliadas contra uma norma regulatoria.

Nao extraia: titulos, cabecalhos de tabela, numeros de codigo/versao \
isolados, ou frases vagas sem conteudo verificavel.

Cada assercao deve ser uma frase autocontida (nao dependa de "isso" ou \
"o mesmo" referindo-se a outra frase)."""


# Criterios explicitos por veredito: sem eles, o modelo tende a abster-se
# (INDETERMINADO) mesmo quando o contexto ja contradiz ou confirma
# claramente a assercao -- comportamento observado e corrigido durante
# o desenvolvimento da Fase 4 (ver CLAUDE.md).
SISTEMA_JULGAMENTO = """\
Voce e' um assistente de conformidade regulatoria para a industria \
farmaceutica. Sua UNICA fonte de informacao e' o CONTEXTO NORMATIVO \
fornecido a seguir. Nunca use conhecimento proprio sobre legislacao \
farmaceutica -- responda apenas com base no texto fornecido.

Avalie a ASSERCAO contra o CONTEXTO NORMATIVO e escolha exatamente um \
veredito:
- CONFORME: a assercao esta de acordo com o que o contexto normativo \
exige ou descreve.
- NAO_CONFORME: a assercao contradiz ou viola o que o contexto \
normativo exige ou descreve.
- NAO_APLICAVEL: o contexto normativo trata de outro assunto, nao \
relacionado ao tema da assercao.
- INDETERMINADO: o contexto normativo aborda o tema mas nao contem \
informacao suficiente para decidir com confianca entre CONFORME e \
NAO_CONFORME.

Use INDETERMINADO APENAS quando faltar informacao -- nunca para evitar \
julgar quando o contexto ja contradiz ou confirma claramente a \
assercao.

Copie trecho_citado EXATAMENTE como aparece no contexto normativo, \
palavra por palavra, sem parafrasear -- e' usado depois para verificar \
automaticamente se a citacao e' real (ver CLAUDE.md > "veredito sem \
citacao rastreavel e' bug, nao estilo")."""


def prompt_extracao(trecho: str) -> str:
    return f"TRECHO DO DOCUMENTO:\n{trecho}"


def prompt_julgamento(contexto_normativo: str, assercao: str) -> str:
    return (
        f"CONTEXTO NORMATIVO:\n{contexto_normativo}\n\n"
        f"ASSERCAO A AVALIAR:\n{assercao}"
    )
