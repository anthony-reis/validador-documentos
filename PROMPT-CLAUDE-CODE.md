
---

## PARTE 1 — O que fazer ANTES de abrir o Claude Code

Faça isso na sua máquina. São passos que o Claude Code faria mal ou lentamente.

```bash
# 1. Pasta do projeto
mkdir -p ~/projetos/validador-docs && cd ~/projetos/validador-docs
git init

# 2. Ambiente Python (3.11 ou 3.12; evite 3.13 por causa de dependências)
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Ollama (runtime do LLM local)
#    Instale de https://ollama.com/download e então:
ollama pull qwen2.5:7b-instruct     # baseline; troque depois se quiser
ollama list                          # confirme que baixou

# 4. Corpus — baixe os PDFs ANTES (veja REFERENCIAS-E-CORPUS.md)
mkdir -p data/normas data/documentos_teste

# 5. Modelos de embedding e reranker (baixe COM internet, use SEM)
pip install -U "huggingface_hub[cli]"
hf download BAAI/bge-m3 --local-dir models/bge-m3
hf download BAAI/bge-reranker-v2-m3 --local-dir models/bge-reranker-v2-m3
```

Depois abra o Claude Code na raiz do projeto (`claude`) e cole o prompt da Parte 2.

---

## PARTE 2 — O PROMPT (cole inteiro no Claude Code)

> Cole tudo daqui até o fim da Parte 2. Ele é longo de propósito: quanto mais
> restrições explícitas, menos o Claude Code inventa arquitetura por conta própria.

---

Você vai me ajudar a construir, do zero, um **agente local de validação de
documentos regulatórios da indústria farmacêutica**, baseado em RAG. É o
artefato prático do meu Trabalho Final de Graduação em Engenharia da Computação.

Antes de escrever qualquer código: leia esta especificação inteira, então
**me apresente um plano** com a estrutura de pastas e a ordem de implementação.
Não comece a codar até eu aprovar o plano.

### Contexto acadêmico

O sistema precisa sustentar quatro objetivos específicos do TFG:

1. Caracterizar técnicas de RAG e PLN aplicáveis a contextos de alta regulação
   (recuperação esparsa, densa e híbrida).
2. Sistematizar requisitos normativos (RDC 658/2022 da ANVISA e ICH Q10) sobre
   controle documental, mapeando categorias de documentos e critérios de
   conformidade.
3. Desenvolver o pipeline de RAG: pré-processamento, indexação semântica,
   recuperação por similaridade em banco vetorial, geração contextualizada.
4. Avaliar o desempenho com experimentos controlados e métricas de acurácia e
   confiabilidade.

Isso significa que o código não é só um chatbot: ele precisa **produzir números
que eu possa colocar em tabelas** e **registrar decisões de projeto** que eu
possa justificar por escrito.

### Restrição inegociável: 100% offline

O sistema roda em máquina isolada, **sem internet em tempo de execução**.
Portanto:

- **Proibido**: qualquer chamada a API externa (OpenAI, Anthropic, Cohere,
  Pinecone, Weaviate Cloud, HuggingFace Inference API).
- Todos os modelos carregam de caminhos locais em `models/`.
- Configure `HF_HUB_OFFLINE=1` e `TRANSFORMERS_OFFLINE=1` no `.env` e garanta
  que o código funcione com essas variáveis ativas.
- No fim de cada fase, **escreva um teste que falha se houver acesso de rede**
  (ex.: monkeypatch em `socket.socket` levantando exceção durante uma query
  completa).

### Stack obrigatória

| Camada | Ferramenta | Observação |
|---|---|---|
| Linguagem | Python 3.11 | |
| Orquestração | LangChain + LangGraph | LangGraph para a máquina de estados do agente |
| Extração PDF | PyMuPDF (primário), pdfplumber (fallback p/ tabelas) | |
| Metadados | pandas | |
| Embeddings | `BAAI/bge-m3` local via sentence-transformers | multilíngue, forte em PT-BR |
| Banco vetorial | ChromaDB persistente (`chroma_db/`) | |
| Esparso | BM25 (`rank_bm25` via LangChain `BM25Retriever`) | |
| Reranker | `BAAI/bge-reranker-v2-m3` local (CrossEncoder) | |
| LLM | Ollama, modelo configurável via `.env` (default `qwen2.5:7b-instruct`) | |
| Interface | Streamlit | |
| Testes | pytest | |

Não substitua nada dessa tabela sem me perguntar antes e explicar o motivo.

### Arquitetura do agente (LangGraph)

Não é um pipeline linear de Q&A. É um agente com estados:

```
[ingestão]  →  [parse do documento sob análise]
                        ↓
              [extrair asserções verificáveis]
                        ↓
              ┌──── para cada requisito normativo aplicável ────┐
              │  [recuperação híbrida: BM25 + denso]            │
              │              ↓                                  │
              │  [rerank cross-encoder → top-k final]           │
              │              ↓                                  │
              │  [julgamento com LLM + citação obrigatória]     │
              │              ↓                                  │
              │  [auto-verificação: a citação sustenta o        │
              │   veredito? se não → ABSTER]                    │
              └────────────────────────────────────────────────┘
                        ↓
              [relatório de conformidade estruturado]
```

Regras de comportamento do agente:

- Cada veredito é um de: `CONFORME`, `NAO_CONFORME`, `NAO_APLICAVEL`,
  `INDETERMINADO`.
- **`INDETERMINADO` é resposta válida e desejável.** Um sistema regulatório que
  chuta é pior que um que se abstém. Se a recuperação não trouxer trecho
  normativo com score acima do limiar, o veredito é `INDETERMINADO` — sem
  exceção.
- Todo veredito carrega: artigo/seção da norma, trecho literal recuperado,
  identificador do chunk e score de recuperação. **Veredito sem citação
  rastreável é bug, não estilo.**
- O LLM nunca responde de conhecimento paramétrico. O prompt do sistema deve
  instruir explicitamente: responda **apenas** a partir do contexto fornecido.

Motivo regulatório disso (registre como comentário no código): o rascunho do
**Annex 22 do EudraLex Volume 4** (consulta pública encerrada em out/2025,
ainda não adotado) restringe LLMs e IA generativa a aplicações **não críticas**
de BPF, com supervisão humana obrigatória. Então o sistema é explicitamente
**assistivo**: a saída é um parecer para revisão humana, nunca uma aprovação
automática. Isso deve estar visível na interface.

### Estrutura de pastas esperada

```
validador-docs/
├── CLAUDE.md                  # você vai criar (ver "Primeira tarefa")
├── .env.example
├── pyproject.toml
├── data/
│   ├── normas/                # RDC 658/2022, ICH Q10, INs (base de conhecimento)
│   └── documentos_teste/      # POPs, protocolos etc. (documentos a validar)
├── models/                    # pesos locais (git-ignored)
├── chroma_db/                 # índice persistente (git-ignored)
├── src/
│   ├── config.py              # tudo configurável, nada hardcoded
│   ├── ingestion/             # loaders, limpeza, chunking, metadados
│   ├── indexing/              # embeddings, escrita no Chroma, índice BM25
│   ├── retrieval/             # esparso, denso, híbrido, reranker
│   ├── agent/                 # grafo LangGraph, nós, prompts, schemas
│   └── evaluation/            # métricas, runner de experimentos
├── app/streamlit_app.py
├── experiments/               # configs YAML + resultados CSV
└── tests/
```

### Chunking — trate como decisão de pesquisa, não detalhe

Documentos normativos brasileiros são estruturados por artigo, parágrafo e
inciso. Um chunk que corta um artigo no meio destrói a rastreabilidade.

Implemente **chunking hierárquico com regex sobre a estrutura jurídica**
(`Art. Nº`, `§ Nº`, incisos romanos), preservando em metadados: `norma`,
`artigo`, `paragrafo`, `titulo_secao`, `pagina`, `chunk_id`.

Mantenha o tamanho e o overlap **parametrizáveis** — vou variar isso nos
experimentos e reportar o efeito. Não fixe valores no código.

### Recuperação progressiva (isto é o experimento principal do TFG)

Implemente as quatro configurações como estratégias **intercambiáveis por
config**, todas rodando sobre o mesmo índice:

| ID | Configuração |
|---|---|
| `A` | Denso puro (similaridade de cosseno) |
| `B` | Esparso puro (BM25) |
| `C` | Híbrido (BM25 + denso, fusão RRF) |
| `D` | Híbrido + rerank cross-encoder |

Precisa ser trivial rodar `python -m src.evaluation.run --config experiments/A.yaml`
e obter um CSV comparável. A tabela A→D é o resultado central do meu TFG.

### Avaliação

Duas camadas, ambas offline:

**1. Recuperação (determinística, sem LLM):** `precision@k`, `recall@k`, `MRR`,
`nDCG@k` contra um ground truth que eu vou anotar manualmente. Gere para mim um
template CSV (`experiments/ground_truth_template.csv`) com colunas
`id, pergunta, chunks_relevantes, norma, artigo, categoria_documento`.

**2. Geração:** taxa de acerto do veredito contra gabarito, **taxa de abstenção**
e **fidelidade da citação** (o trecho citado realmente existe no chunk
recuperado? — verificação por correspondência de string, não por LLM).

Se usar RAGAS, o LLM juiz precisa ser o Ollama local. Se isso ficar instável ou
lento, prefira as métricas determinísticas acima — elas são mais defensáveis
numa banca do que "um LLM avaliou outro LLM".

Registre em cada execução: hash do corpus, versões dos modelos, parâmetros e
timestamp. Experimento não reprodutível não vale para o TFG.

### Como quero que você trabalhe

- **Fases pequenas.** Ao fim de cada uma: rode os testes, faça commit com
  mensagem descritiva e **pare para eu revisar**. Não emende fases.
- **Explique as escolhas.** Sou eu que vou defender isso na banca. Quando
  decidir algo (limiar, estratégia de fusão, formato de prompt), escreva um
  comentário curto dizendo *por quê*, e me avise no chat.
- **Não invente números.** Nenhum valor de acurácia entra no código, em
  docstring ou em README sem ter saído de uma execução real.
- **Sem mocks silenciosos.** Se algo não funcionar offline, me diga; não
  contorne com stub que simula o comportamento.
- Se em algum ponto a especificação acima estiver errada ou for pior que uma
  alternativa, **me diga**. Não siga uma instrução ruim só porque eu escrevi.

### Fases

- **Fase 0** — `CLAUDE.md`, `pyproject.toml`, `.env.example`, `config.py`,
  esqueleto de pastas, `pytest` rodando vazio.
- **Fase 1** — Ingestão: leitura de PDFs, limpeza, chunking hierárquico,
  extração de metadados. Teste: um artigo da RDC 658 vira um chunk com
  `artigo` correto nos metadados.
- **Fase 2** — Indexação: embeddings locais, persistência no Chroma, índice
  BM25. Teste: busca por um termo normativo retorna o artigo esperado.
- **Fase 3** — Recuperação A/B/C/D atrás de uma interface comum.
- **Fase 4** — Agente LangGraph com saída estruturada (Pydantic), citação
  obrigatória e regra de abstenção.
- **Fase 5** — Avaliação: métricas, runner, template de ground truth.
- **Fase 6** — Streamlit: upload, seleção de estratégia, relatório com citações
  clicáveis, aviso de supervisão humana.
- **Fase 7** — Empacotamento offline: script de verificação de que tudo roda
  sem rede; README de reprodução.

### Primeira tarefa

Crie o `CLAUDE.md` na raiz consolidando: restrição offline, stack fixa, as
quatro configurações de recuperação, a regra de abstenção e a exigência de
citação rastreável. É o arquivo que você vai reler a cada sessão — escreva-o
para o "você" de daqui a duas semanas.

Depois disso, apresente o plano da Fase 0 e aguarde minha aprovação.

---

## PARTE 3 — Prompts de follow-up

Use conforme avança. Um por vez.

**Ao terminar cada fase:**
```
Rode os testes, mostre o resultado e faça commit. Depois me explique em até
5 linhas quais decisões você tomou nesta fase que eu preciso saber para
defender na banca.
```

**Se ele começar a improvisar arquitetura:**
```
Pare. Releia o CLAUDE.md. Você está desviando de [X]. Volte à especificação
ou me convença de que a alternativa é melhor — mas não decida sozinho.
```

**Para o capítulo de metodologia:**
```
Gere docs/decisoes-tecnicas.md listando cada escolha de projeto (chunking,
embedding, limiar, fusão, prompt), com justificativa e a alternativa
descartada. Sem números de desempenho — só decisões e razões.
```

**Para rodar os experimentos:**
```
Execute as configurações A, B, C e D sobre o mesmo ground truth e gere uma
tabela comparativa em experiments/resultados.csv com precision@5, recall@5,
MRR, nDCG@5 e tempo médio por consulta. Não interprete os números — só reporte.
```

**Verificação final de offline:**
```
Escreva e execute um teste que bloqueia toda a rede (monkeypatch em
socket.socket) e roda uma consulta ponta a ponta. Se qualquer parte tentar
acessar a internet, quero ver a falha.
```

---

## PARTE 4 — Armadilhas conhecidas

- **`bge-m3` é pesado** (~2 GB). Se a máquina apertar, alternativa multilíngue
  menor é `intfloat/multilingual-e5-base` — mas troque só se precisar, e
  registre a troca como limitação metodológica.
- **Chroma guarda a dimensão do embedding na criação da coleção.** Trocou de
  modelo de embedding? Apague `chroma_db/` e reindexe, senão dá erro de
  dimensão (mesmo problema que o artigo do Medium descreve com o Pinecone).
- **PDFs da ANVISA vindos de scanner** podem não ter camada de texto. Se
  `PyMuPDF` retornar vazio, você precisará de OCR (Tesseract, `por`) — e isso
  vira mais uma dependência offline para baixar antes.
- **Ollama precisa estar rodando** (`ollama serve`) antes de subir o Streamlit.
- **Quantização muda o resultado.** Fixe a tag exata do modelo no `.env`
  (`qwen2.5:7b-instruct-q4_K_M`, não `qwen2.5`), ou seus experimentos não
  reproduzem.
