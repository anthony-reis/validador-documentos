# validador-docs

Agente local de validação de conformidade documental (RAG) para a
indústria farmacêutica — artefato prático de um Trabalho Final de
Graduação em Engenharia da Computação. Lê um documento sob análise (ex.:
um POP) e emite um parecer de conformidade frente à base normativa
indexada (RDC 658/2022, ICH Q10, IN 134/2022, IN 138/2022), sempre com
citação rastreável e sujeito a revisão humana — nunca uma aprovação
automática (ver `CLAUDE.md` para o raciocínio completo por trás de cada
decisão de projeto).

**Status atual**: Fase 4 de 7 concluída (ingestão + indexação +
recuperação A/B/C/D + agente LangGraph). Ver `CLAUDE.md` > "Fases do
projeto" para o roteiro completo.

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

## Como testar uma busca (A/B/C/D)

Com o índice já construído (`python -m src.indexing.build`):

```python
from src.retrieval.factory import criar_retriever

retriever = criar_retriever("D")  # "A" denso | "B" esparso | "C" hibrido | "D" hibrido+rerank
for resultado in retriever.buscar("qual o objetivo das boas práticas de fabricação?", top_k=3):
    print(resultado.metadata["norma"], resultado.metadata["artigo"], resultado.score)
```

## Como analisar um documento com o agente

Com o índice já construído e o Ollama rodando:

```python
from pathlib import Path
from src.agent.pipeline import analisar_documento

relatorio = analisar_documento(Path("data/documentos_teste/seu_arquivo.pdf"), estrategia="D")
print(relatorio.contagem_por_veredito)
for julgamento in relatorio.julgamentos:
    print(julgamento.veredito, julgamento.assercao)
    if julgamento.citacao:
        print("  ->", julgamento.citacao.norma, julgamento.citacao.artigo)
```

> Em CPU (sem GPU), espere ~20-25 minutos para um documento de poucas
> páginas com `qwen2.5:7b-instruct-q4_K_M` — cada asserção extraída gera
> pelo menos uma chamada ao LLM. O teste de ponta a ponta real
> (`tests/test_agent.py::TestPipelineRealComOllama`) só roda com
> `RUN_SLOW_LLM_TESTS=1` por causa desse custo.

Todo veredito diferente de `INDETERMINADO` carrega uma citação rastreável
(`norma`, `artigo`, `trecho_literal`, `chunk_id`, `score`). O sistema é
assistivo — a saída é um parecer para revisão humana, nunca uma
aprovação automática (ver `CLAUDE.md`).

### Decisões de projeto do agente (Fase 4)

Cinco decisões que sustentam o comportamento do agente — raciocínio
completo em `CLAUDE.md` > "Agente LangGraph (Fase 4)":

1. **Auto-verificação é determinística (substring), não uma segunda
   chamada de LLM.** O `trecho_citado` alegado pelo LLM precisa existir
   literalmente (após normalizar espaços) no chunk realmente recuperado;
   se não existir, o sistema força `INDETERMINADO` com
   `motivo_abstencao` explícito. Mais defensável na banca do que "um LLM
   verificando outro LLM".
2. **`ChatOllama.with_structured_output` com o método padrão
   (`function_calling`) falha silenciosamente** (retorna `None`) com
   `qwen2.5:7b-instruct-q4_K_M` — descoberto empiricamente antes de
   escrever o código de produção. `method="json_schema"` (suporte
   nativo do Ollama a saída restrita por schema) funciona de forma
   confiável.
3. **O prompt de julgamento precisa de critérios de decisão explícitos
   por veredito.** Sem eles, o modelo se abstinha (`INDETERMINADO`)
   mesmo quando o contexto já contradizia ou confirmava claramente a
   asserção — corrigido e validado com casos de teste reais.
4. **Corte determinístico antes de chamar o LLM**: se o melhor score de
   recuperação está abaixo de `RETRIEVAL_SCORE_THRESHOLD`, o veredito já
   é `INDETERMINADO` sem sequer invocar o LLM — evita pressionar o
   modelo a "inventar" um julgamento sem contexto adequado.
5. **A extração de asserções do documento é heurística best-effort via
   LLM, não a fonte dos números do TFG.** A Fase 5 avalia recuperação e
   geração contra um gabarito curado manualmente, justamente para isolar
   essas métricas da qualidade desta extração automática.

### Métricas observadas (execução real, não sintética)

Ponta a ponta contra `data/documentos_teste/tratamento_de_desvio_ficticio.pdf`
(4 páginas, documento fictício de teste), estratégia de recuperação `D`:

| Métrica | Valor |
|---|---|
| Asserções extraídas e julgadas | 62 |
| `INDETERMINADO` | 44 (71%) |
| `NAO_CONFORME` | 11 (18%) |
| `CONFORME` | 3 (5%) |
| `NAO_APLICAVEL` | 4 (6%) |
| Verificação de fidelidade de citação disparada (citação rejeitada por não ser substring literal do chunk) | 2 ocorrências |
| Tempo total (CPU, sem GPU, `qwen2.5:7b-instruct-q4_K_M`) | ~25 min |

A taxa alta de `INDETERMINADO` é esperada e defensável, não uma falha:
muitas asserções extraídas são fatos narrativos do incidente (datas,
números de lote) sem uma exigência normativa correspondente para
confirmar ou negar — o design "abster > chutar" (ver `CLAUDE.md` > regra
de abstenção) está se comportando como pretendido.

> Estes números vêm de uma única execução exploratória sobre um
> documento de teste, não do experimento controlado A→B→C→D da Fase 5
> — não usar como resultado de desempenho do TFG.

## Estrutura do projeto

Ver `CLAUDE.md` > "Estrutura de pastas" para o mapa completo e o
raciocínio por trás de cada módulo.

## Documentação de decisões de projeto

Todo o histórico de decisões técnicas (por que chunking hierárquico, por
que negrito em vez de regex para o ICH Q10, por que essas quatro
configurações de recuperação, etc.) está em `CLAUDE.md` — é o documento
de referência para escrever a metodologia do TFG e para retomar o
projeto em uma sessão nova.
