# validador-docs

Agente local de validação de conformidade documental (RAG) para a
indústria farmacêutica — artefato prático de um Trabalho Final de
Graduação em Engenharia da Computação. Lê um documento sob análise (ex.:
um POP) e emite um parecer de conformidade frente à base normativa
indexada (RDC 658/2022, ICH Q10, IN 134/2022, IN 138/2022), sempre com
citação rastreável e sujeito a revisão humana — nunca uma aprovação
automática (ver `CLAUDE.md` para o raciocínio completo por trás de cada
decisão de projeto).

**Status atual**: Fase 2 de 7 concluída (ingestão + indexação). Ver
`CLAUDE.md` > "Fases do projeto" para o roteiro completo.

## Requisitos

- macOS com [Homebrew](https://brew.sh/)
- ~10 GB livres em disco (modelos de embedding/reranker + LLM)
- Internet **apenas durante o setup** — depois disso o sistema roda 100%
  offline (restrição de projeto, ver `CLAUDE.md`)

## Setup — do zero até rodar

### 1. Runtime (Python 3.11 e Ollama)

```bash
brew install python@3.11 ollama
```

### 2. Ambiente virtual e dependências do projeto

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

### 3. Modelo de linguagem (Ollama)

```bash
brew services start ollama   # ou: ollama serve
ollama pull qwen2.5:7b-instruct-q4_K_M
```

> A tag exata (`-q4_K_M`) importa: quantização diferente muda os
> resultados e quebra a reprodutibilidade dos experimentos. Não use só
> `qwen2.5`.

### 4. Modelos de embedding e reranker (baixados uma vez, usados offline depois)

```bash
pip install "huggingface_hub[cli]>=0.23.2,<1.0"   # já incluso no pip install -e ".[dev]" acima
hf download BAAI/bge-m3 --local-dir models/bge-m3                       # ~4.3 GB
hf download BAAI/bge-reranker-v2-m3 --local-dir models/bge-reranker-v2-m3  # ~2.1 GB
```

> Não instale `huggingface_hub` sem o pin de versão: a série 1.x quebra
> compatibilidade com `transformers`/`tokenizers` usados aqui.

### 5. Corpus normativo

O corpus normativo (`RDC_658_2022.pdf`, `IN_134_2022.pdf`,
`IN_138_2022.pdf`, `ICH_Q10.pdf`) já está versionado em `data/normas/`
neste repositório — não precisa baixar de novo. Para adicionar uma nova
norma ao corpus, veja `REFERENCIAS-E-CORPUS.md` para as fontes oficiais
e registre o arquivo em `src/indexing/corpus.py` (manifesto explícito
arquivo → norma/formato).

### 6. Variáveis de ambiente

```bash
cp .env.example .env
```

Os defaults do `.env.example` já funcionam com os caminhos acima; ajuste
só se você mudou algum local de instalação.

### 7. Confirmar que tudo está no lugar

```bash
python -m pytest -q
```

Todos os testes devem passar. Testes que dependem de arquivos ainda não
baixados são pulados automaticamente (`SKIPPED`), não falham.

## Como indexar o corpus

```bash
source .venv/bin/activate
python -m src.indexing.build
```

Isso lê todo `data/normas/`, gera os chunks (hierárquicos por
artigo/parágrafo para normas brasileiras, por seção numerada para o ICH
Q10), grava o índice denso em `chroma_db/` (ChromaDB + embeddings
`bge-m3`) e o índice esparso em `bm25_index/` (BM25). Saída esperada:

```
[indexacao] 821 chunks extraidos do corpus.
  - ICH Q10: 65 chunks
  - IN 134/2022: 60 chunks
  - IN 138/2022: 162 chunks
  - RDC 658/2022: 534 chunks
[indexacao] hash do corpus: <hash sha256>
...
[indexacao] concluido.
```

Reindexe sempre que o corpus (`data/normas/`) mudar. Se você trocar o
modelo de embedding, apague `chroma_db/` inteiro antes de reindexar (o
Chroma grava a dimensão do vetor na criação da coleção e não migra
sozinho).

## Estrutura do projeto

Ver `CLAUDE.md` > "Estrutura de pastas" para o mapa completo e o
raciocínio por trás de cada módulo.

## Documentação de decisões de projeto

Todo o histórico de decisões técnicas (por que chunking hierárquico, por
que negrito em vez de regex para o ICH Q10, por que essas quatro
configurações de recuperação, etc.) está em `CLAUDE.md` — é o documento
de referência para escrever a metodologia do TFG e para retomar o
projeto em uma sessão nova.
