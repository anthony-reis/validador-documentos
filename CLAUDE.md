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

**Progresso atual: Fase 1 concluída** (esqueleto + ingestão/chunking da
RDC 658/2022 funcionando e testado). Próxima: Fase 2 (indexação).

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

Status atual (30/08/2026): Python 3.11 e Ollama instalados via Homebrew;
`data/normas/RDC_658_2022.pdf` versionado no repositório (ver nota abaixo).
Ainda faltam: `ollama pull` do modelo, download de `bge-m3`/`bge-reranker-v2-m3`,
e o restante do corpus (ICH Q10, INs 134/2022 e 138/2022, Perguntas &
Respostas de BPF). Ver `README.md` para o passo a passo de setup.

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
