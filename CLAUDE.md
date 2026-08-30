# CLAUDE.md — validador-docs

Memória de decisões do projeto. Releia isto no início de toda sessão nova
antes de tocar em código. Se uma instrução aqui parecer errada ou pior que
uma alternativa, diga isso ao usuário — não decida sozinho e não a ignore
silenciosamente.

## O que é este projeto

Agente local de validação de conformidade documental para a indústria
farmacêutica, baseado em RAG (Retrieval-Augmented Generation). É o artefato
prático de um Trabalho Final de Graduação (TFG) em Engenharia da Computação.
O sistema lê um documento sob análise (ex.: um POP) e emite um parecer sobre
sua conformidade frente à base normativa indexada (RDC 658/2022 da ANVISA,
ICH Q10 e correlatas).

O código não é "só um chatbot": ele precisa produzir **números reportáveis**
(métricas de recuperação e geração) e **decisões de projeto justificáveis**
por escrito para banca acadêmica.

## Restrição inegociável: 100% offline em tempo de execução

O sistema roda numa máquina isolada, sem internet durante a execução real.

- **Proibido**: qualquer chamada de rede a serviço externo — OpenAI,
  Anthropic, Cohere, Pinecone, Weaviate Cloud, HuggingFace Inference API, etc.
- Todo modelo (embedding, reranker, LLM) carrega de caminho **local**
  (`models/` ou via Ollama local). Nunca via download em tempo de execução.
- `.env` deve conter `HF_HUB_OFFLINE=1` e `TRANSFORMERS_OFFLINE=1`, e o
  código precisa funcionar corretamente com essas variáveis ativas.
- Ao final de cada fase que toca em modelos/rede, escrever um teste que
  **falha se houver acesso de rede** (monkeypatch em `socket.socket`
  levantando exceção durante uma consulta ponta a ponta).
- Downloads de modelos e corpus (`hf download`, `ollama pull`, PDFs da
  ANVISA/ICH) são etapas manuais feitas **antes**, com internet — nunca
  responsabilidade do código do agente.

## Stack fixa — não trocar sem perguntar e justificar

| Camada | Ferramenta |
|---|---|
| Linguagem | Python 3.11 |
| Orquestração | LangChain + LangGraph (LangGraph = máquina de estados do agente) |
| Extração de PDF | PyMuPDF (primário), pdfplumber (fallback para tabelas) |
| Metadados | pandas |
| Embeddings | `BAAI/bge-m3` local via sentence-transformers |
| Banco vetorial | ChromaDB persistente (`chroma_db/`) |
| Recuperação esparsa | BM25 (`rank_bm25` via `BM25Retriever` do LangChain) |
| Reranker | `BAAI/bge-reranker-v2-m3` local (CrossEncoder) |
| LLM | Ollama, modelo configurável via `.env` — default `qwen2.5:7b-instruct` (fixar tag exata, incl. quantização, para reprodutibilidade) |
| Interface | Streamlit |
| Testes | pytest |

Alternativa registrada para memória apertada: `intfloat/multilingual-e5-base`
no lugar de `bge-m3` — só trocar se a máquina não aguentar, e registrar como
limitação metodológica no texto do TFG.

## Arquitetura do agente (LangGraph)

Não é um pipeline linear de Q&A. É um agente com estados:

```
[ingestão] → [parse do documento sob análise] → [extrair asserções verificáveis]
    → para cada requisito normativo aplicável:
        [recuperação híbrida: BM25 + denso]
            → [rerank cross-encoder → top-k final]
            → [julgamento com LLM + citação obrigatória]
            → [auto-verificação: a citação sustenta o veredito? se não → ABSTER]
    → [relatório de conformidade estruturado]
```

### Regras de comportamento do agente (não negociáveis sem discussão explícita)

- Veredito é sempre um de: `CONFORME`, `NAO_CONFORME`, `NAO_APLICAVEL`,
  `INDETERMINADO`.
- **`INDETERMINADO` é resposta válida e desejável.** Se a recuperação não
  trouxer trecho normativo com score acima do limiar configurado, o veredito
  é `INDETERMINADO`, sem exceção. Um sistema regulatório que chuta é pior
  que um que se abstém.
- Todo veredito carrega: artigo/seção da norma, trecho literal recuperado,
  `chunk_id` e score de recuperação. **Veredito sem citação rastreável é
  bug, não estilo.**
- O LLM nunca responde de conhecimento paramétrico. O prompt de sistema
  instrui explicitamente: responder apenas a partir do contexto fornecido.
- **Motivo regulatório**: o rascunho do Annex 22 do EudraLex Volume 4
  (consulta pública encerrada em out/2025, ainda não adotado) restringe
  LLMs/IA generativa a aplicações não críticas de BPF, com supervisão
  humana obrigatória. Por isso o sistema é explicitamente **assistivo**: a
  saída é um parecer para revisão humana, nunca uma aprovação automática —
  isso precisa estar visível na interface do Streamlit.

## As quatro configurações de recuperação (experimento central do TFG)

Estratégias intercambiáveis por config, todas sobre o mesmo índice:

| ID | Configuração |
|---|---|
| `A` | Denso puro (similaridade de cosseno) |
| `B` | Esparso puro (BM25) |
| `C` | Híbrido (BM25 + denso, fusão RRF) |
| `D` | Híbrido + rerank cross-encoder |

Precisa ser trivial rodar `python -m src.evaluation.run --config experiments/A.yaml`
(idem B/C/D) e obter um CSV comparável. A tabela A→D é o resultado central do TFG.

## Chunking — decisão de pesquisa, não detalhe de implementação

Normas brasileiras são estruturadas por artigo, parágrafo e inciso. Um chunk
que corta um artigo no meio destrói a rastreabilidade da citação.
Implementar **chunking hierárquico com regex sobre a estrutura jurídica**
(`Art. Nº`, `§ Nº`, incisos romanos), preservando em metadados: `norma`,
`artigo`, `paragrafo`, `titulo_secao`, `pagina`, `chunk_id`.
Tamanho e overlap de chunk devem ser **parametrizáveis** (nunca hardcoded) —
são variáveis do experimento.

## Avaliação — duas camadas, ambas offline

1. **Recuperação** (determinística, sem LLM): `precision@k`, `recall@k`,
   `MRR`, `nDCG@k` contra ground truth anotado manualmente
   (`experiments/ground_truth_template.csv`: `id, pergunta, chunks_relevantes,
   norma, artigo, categoria_documento`).
2. **Geração**: taxa de acerto do veredito vs. gabarito, taxa de abstenção,
   fidelidade da citação (correspondência de string do trecho citado dentro
   do chunk recuperado — **não** verificação por LLM).

Se usar RAGAS, o juiz precisa ser o Ollama local; se instável/lento, preferir
as métricas determinísticas acima — mais defensáveis na banca do que "um LLM
avaliando outro LLM". Toda execução registra: hash do corpus, versões dos
modelos, parâmetros e timestamp — experimento não reprodutível não vale para
o TFG. **Nenhum número de desempenho entra em código, docstring ou README
sem ter saído de uma execução real.**

## Estrutura de pastas

```
validador-docs/  (raiz deste repo)
├── CLAUDE.md
├── README.md
├── .env.example
├── pyproject.toml
├── data/
│   ├── normas/                # base de conhecimento (RDC 658, ICH Q10, INs)
│   └── documentos_teste/      # documentos a validar (POPs, protocolos etc.)
├── models/                    # pesos locais, git-ignored
├── chroma_db/                 # índice persistente, git-ignored
├── src/
│   ├── config.py              # tudo configurável, nada hardcoded
│   ├── ingestion/              # loaders, limpeza, chunking, metadados
│   ├── indexing/                # embeddings, escrita no Chroma, índice BM25
│   ├── retrieval/                # esparso, denso, híbrido, reranker
│   ├── agent/                    # grafo LangGraph, nós, prompts, schemas
│   └── evaluation/                # métricas, runner de experimentos
├── app/streamlit_app.py
├── experiments/                # configs YAML + resultados CSV
└── tests/
```

## Como trabalhar neste projeto

- **Fases pequenas.** Ao fim de cada uma: rodar os testes, commit com
  mensagem descritiva, e **parar para revisão do usuário**. Não emendar fases.
- **Explicar as escolhas.** É o usuário quem defende isso na banca. Toda
  decisão (limiar, estratégia de fusão, formato de prompt) leva um comentário
  curto de *por quê* no código, e um aviso no chat.
- **Nenhum número inventado** — nada de acurácia/precisão chutada em código,
  docstring ou README.
- **Sem mocks silenciosos.** Se algo não funcionar offline, avisar; nunca
  contornar com stub que finge o comportamento.
- Se a especificação (aqui ou no `PROMPT-CLAUDE-CODE.md`) estiver errada ou
  pior que uma alternativa, dizer isso — não seguir instrução ruim por inércia.

## Fases do projeto

**Progresso atual: Fase 4 concluída** (agente LangGraph completo,
citação obrigatória, regra de abstenção, testado ponta a ponta contra
documento real). Próxima: Fase 5 (avaliação: métricas, runner,
ground truth).

- **Fase 0** — `CLAUDE.md` (este arquivo), `pyproject.toml`, `.env.example`,
  `config.py`, esqueleto de pastas, `pytest` rodando vazio.
- **Fase 1** — Ingestão: leitura de PDFs, limpeza, chunking hierárquico,
  extração de metadados.
- **Fase 2** — Indexação: embeddings locais, persistência no Chroma, índice BM25.
- **Fase 3** — Recuperação A/B/C/D atrás de uma interface comum.
- **Fase 4** — Agente LangGraph com saída estruturada (Pydantic), citação
  obrigatória e regra de abstenção.
- **Fase 5** — Avaliação: métricas, runner, template de ground truth.
- **Fase 6** — Streamlit: upload, seleção de estratégia, relatório com
  citações clicáveis, aviso de supervisão humana.
- **Fase 7** — Empacotamento offline: script de verificação sem rede,
  README de reprodução.

## Pré-requisitos de ambiente (fora do escopo do código do agente)

Estas etapas são manuais, feitas com internet, **antes** de rodar o sistema:

1. Python 3.11 (venv dedicado) — evitar 3.13+ por incompatibilidade de
   dependências.
2. Ollama instalado e rodando (`ollama serve`), com o modelo puxado
   (`ollama pull qwen2.5:7b-instruct` ou tag equivalente fixada).
3. Corpus normativo baixado em `data/normas/` (ver `REFERENCIAS-E-CORPUS.md`
   na raiz para as fontes: RDC 658/2022, ICH Q10, INs 134/2022 e 138/2022,
   Perguntas & Respostas de BPF).
4. Modelos de embedding/reranker baixados localmente:
   `hf download BAAI/bge-m3 --local-dir models/bge-m3` e
   `hf download BAAI/bge-reranker-v2-m3 --local-dir models/bge-reranker-v2-m3`.

Status atual (30/08/2026): ambiente completo -- Python 3.11, Ollama com
`qwen2.5:7b-instruct-q4_K_M` puxado, `models/bge-m3` e
`models/bge-reranker-v2-m3` baixados, corpus normativo completo em
`data/normas/` versionado no repositório (ver nota abaixo). Índices
denso (Chroma) e esparso (BM25) construídos e testados. Ver `README.md`
para o passo a passo de setup.

### Indexação (Fase 2)

`src/indexing/corpus.py` mantém o manifesto explícito arquivo → (norma,
formato) — mesmo princípio da Fase 1: nunca inferir automaticamente.
`src/indexing/vetorial.py` (Chroma) e `src/indexing/esparso.py` (BM25)
expõem `buscar()` já usados nos testes de aceitação; a Fase 3 reaproveita
essas funções por baixo da interface comum A/B/C/D. `python -m
src.indexing.build` reconstrói os dois índices a partir do zero.

**Armadilha real encontrada**: o Chroma tenta enviar telemetria
(posthog) mesmo com `anonymized_telemetry=False` — a chamada falha
localmente por incompatibilidade de versão com o `posthog` instalado
(erro de assinatura, não de rede). Confirmado com
`socket.socket.connect` bloqueado que nenhuma rede é acionada; o log de
erro é apenas silenciado em `config.py` (não afeta o comportamento).
Ver teste `test_nenhuma_chamada_de_rede_ocorre_durante_indexacao_e_busca`
em `tests/test_indexing.py`.

**Armadilha de dependências**: instalar `huggingface_hub[cli]` sem
pinar a versão puxa a 1.x, que quebra `transformers`/`tokenizers`
(exigem `<1.0`). Fixado em `pyproject.toml`.

### Recuperação A/B/C/D (Fase 3)

`src/retrieval/base.py` define a interface comum (`Retriever.buscar(query,
top_k) -> list[ResultadoRecuperacao]`); `src/retrieval/factory.py` seleciona
a estratégia por letra. Score só é comparável **dentro** da mesma
estratégia — cosseno (A) e BM25 (B) vivem em escalas incompatíveis, por
isso a fusão híbrida (C, `hibrido.py`) usa **Reciprocal Rank Fusion**
sobre o *rank* de cada lista, não o score bruto. A config D
(`reranqueado.py`) roda o cross-encoder só sobre o pool já filtrado por C
(`RETRIEVAL_POOL_SIZE`, default 20), nunca sobre o corpus inteiro —
cross-encoders são precisos mas caros demais para escala.

**Correção feita nesta fase**: o Chroma usa distância L2 por padrão: a
coleção agora é criada com `hnsw:space="cosine"` explícito (ver
`src/indexing/vetorial.py`) para o score da config A ser literalmente
similaridade de cosseno, como a especificação pede — com vetores
normalizados a *ordem* dos vizinhos seria idêntica sob L2, mas o *valor*
do score não seria intuitivo. Reindexar do zero foi necessário.

**Bug real encontrado nos testes**: `caminho: Path = CAMINHO_PADRAO`
como valor-padrão de parâmetro é resolvido na definição da função, não a
cada chamada — um `monkeypatch` em `CAMINHO_PADRAO` (comum em testes)
não tinha nenhum efeito, e `RetrieverEsparso` silenciosamente caía de
volta no índice BM25 real em vez do isolado do teste. Corrigido trocando
o default por `None` + resolução dentro do corpo da função
(`src/indexing/esparso.py::_resolver_caminho`).

### Agente LangGraph (Fase 4)

Grafo linear (`src/agent/grafo.py`): `parse → extrair_assercoes →
processar_assercoes → fim`. **Simplificação deliberada**: o loop "para
cada requisito normativo aplicável" do diagrama da especificação é uma
iteração Python dentro do nó `processar_assercoes`, não um fan-out
dinâmico via `Send` do LangGraph — os julgamentos são independentes
entre si e não há necessidade real de paralelismo, então `Send`
adicionaria complexidade sem mudar o comportamento observável.

**Auto-verificação é determinística, não uma segunda chamada de LLM**
(decisão confirmada com o usuário antes de implementar): o
`trecho_citado` que o LLM alega precisa ser substring literal (após
normalizar espaços) do chunk realmente recuperado — se não for, o
veredito é forçado para `INDETERMINADO` com `motivo_abstencao`
explícito. Mais defensável na banca que "um LLM verificando outro LLM"
(mesmo raciocínio já usado na Fase 2 para a telemetria).

**Corte determinístico antes de chamar o LLM**: se o melhor score de
recuperação está abaixo de `RETRIEVAL_SCORE_THRESHOLD`, o veredito já é
`INDETERMINADO` sem sequer invocar o LLM — evita pressionar o modelo a
"inventar" um julgamento sem contexto adequado.

**Descobertas empíricas sobre `ChatOllama.with_structured_output`**
(testado antes de escrever o código de produção): o método padrão
(`function_calling`) retorna `None` silenciosamente com
`qwen2.5:7b-instruct-q4_K_M` — o modelo não produz uma tool call que o
parser reconheça. `method="json_schema"` (suporte nativo do Ollama a
saída restrita por schema) funciona de forma confiável. O campo de
veredito precisa ser `Literal[...]`, não `str` — com `str` o modelo
despeja texto livre no campo. O prompt de julgamento precisa de
critérios de decisão explícitos por veredito: sem eles, o modelo se
abstém (`INDETERMINADO`) mesmo quando o contexto já contradiz ou
confirma claramente a asserção.

**Extração de asserções é heurística best-effort via LLM, não fonte dos
números do TFG** — a Fase 5 usa um gabarito curado manualmente
justamente para isolar a qualidade de recuperação/geração da qualidade
desta extração. Validação real (documento fictício de 4 páginas, 62
asserções extraídas): distribuição `INDETERMINADO 44, NAO_CONFORME 11,
CONFORME 3, NAO_APLICAVEL 4`. A taxa alta de abstenção é esperada e
defensável: muitas "asserções" extraídas são fatos narrativos (datas,
números de lote) sem uma exigência normativa correspondente para
confirmar/negar — sinal de que o design "abster > chutar" está
funcionando, não uma falha. A verificação de fidelidade de citação
disparou de fato 2 vezes nessa execução real (não só em teste sintético).

**Custo real de execução**: ~25 min para 62 julgamentos + extração,
CPU-only (sem GPU), qwen2.5:7b-instruct-q4_K_M — relevante para
dimensionar os experimentos da Fase 5.

### Duas famílias de chunking (decisão da Fase 1)

Nem todo documento do corpus segue a estrutura `Art./§/inciso` das normas
brasileiras: o ICH Q10 é um manual internacional organizado em seções
numeradas (`1.`, `1.1`, `3.1.2`...). Por isso existem **duas estratégias de
chunking intercambiáveis**, escolhidas explicitamente pelo chamador via
`formato=` (nunca inferidas automaticamente — um palpite errado corromperia
a base silenciosamente):

- `formato="artigos"` (`src/ingestion/chunking.py`) — para RDC/IN brasileiras,
  via regex sobre `Art.`/`§`/inciso.
- `formato="secoes_numeradas"` (`src/ingestion/chunking_numerado.py`) — para
  documentos como o ICH Q10. Aqui regex sobre texto puro não basta: o
  sumário do documento tem o mesmo formato textual "número + título" de um
  cabeçalho real. O sinal confiável é tipográfico (negrito, via
  `PyMuPDF get_text("dict")`), com um filtro adicional por pontilhado de
  preenchimento (dot leaders) para descartar entradas de sumário que também
  estejam em negrito.

**Limitação de corpus documentada**: o PDF do ICH Q10 usado tem um defeito
de numeração própria a partir da seção 1.5.4 (números "andam" uma posição —
o próprio documento anota a correção entre parênteses no título, ex.:
"Facilitadores: ... (1.6)"). O chunker extrai o número exatamente como
impresso, sem corrigi-lo — alterar dado de origem silenciosamente seria
pior do que preservar e documentar um defeito conhecido do corpus. Citar
isso como limitação na metodologia do TFG.

### Nota sobre o corpus normativo e o git

Ao contrário de `models/` e `chroma_db/` (artefatos derivados, git-ignored),
`data/normas/` **é versionado no git**: é a base de conhecimento que o
sistema indexa, então precisa acompanhar o repositório para reprodutibilidade
do experimento. `data/documentos_teste/` continua git-ignored — são
documentos de terceiros a validar, potencialmente sensíveis, não parte da
base de conhecimento.

O link oficial do `antigo.anvisa.gov.br` para a RDC 658/2022 citado em
`REFERENCIAS-E-CORPUS.md` está fora do ar (redireciona para página genérica
do portal gov.br). O PDF em `data/normas/RDC_658_2022.pdf` foi obtido via
mirror do Sindusfarma, com conteúdo conferido contra o cabeçalho oficial do
Diário Oficial da União (Edição 62, Seção 1, Página 320, publicado em
31/03/2022). Registrar essa proveniência na metodologia do TFG.

## Armadilhas conhecidas

- `bge-m3` é pesado (~2 GB); alternativa menor é `multilingual-e5-base`, mas
  só trocar se necessário e registrar como limitação metodológica.
- Chroma grava a dimensão do embedding na criação da coleção — trocar de
  modelo de embedding exige apagar `chroma_db/` e reindexar.
- PDFs da ANVISA vindos de scanner podem não ter camada de texto; se
  PyMuPDF retornar vazio, será necessário OCR (Tesseract, `por`).
- Ollama precisa estar rodando antes de subir o Streamlit.
- Fixar a tag exata do modelo Ollama no `.env` (ex.: `qwen2.5:7b-instruct-q4_K_M`),
  não `qwen2.5` — quantização muda o resultado e afasta a reprodutibilidade.
