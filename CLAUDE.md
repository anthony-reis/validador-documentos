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

**Progresso atual: as 7 fases do roteiro estão concluídas, mais uma
rodada de melhorias de performance e expansão de corpus (ver seções
"Melhorias de performance" e "Expansão do corpus" abaixo).** Pendências
que continuam do usuário, não do código: anotar o ground truth real
(Fase 5) e rodar o experimento comparativo A→D com dados de verdade.

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

### Avaliação (Fase 5)

**Decisão metodológica importante**: o ground truth (`experiments/
ground_truth_template.csv` e `gabarito_geracao_template.csv`) foi
deixado **vazio** (só uma linha de exemplo, claramente marcada para
apagar) — eu não anotei dados reais nele. Se a mesma sessão que
implementa o retriever também fabricasse o gabarito que o avalia, a
avaliação ficaria circular e indefensável na banca. O usuário anota
manualmente (ver `REFERENCIAS-E-CORPUS.md`: o documento de Perguntas &
Respostas de BPF da ANVISA é a fonte sugerida, "quase pronta" para
ground truth).

`src/evaluation/metricas_recuperacao.py`: `precision@k`, `recall@k`,
`MRR`, `nDCG@k` com relevância **binária** (o ground truth não anota
graus de relevância — só "relevante" ou não). Puramente determinístico,
sem LLM.

`src/evaluation/metricas_geracao.py`: taxa de acerto (veredito vs.
gabarito), taxa de abstenção, fidelidade de citação. A fidelidade de
citação **não reprocessa a verificação de substring** — ela já acontece
dentro do agente a cada julgamento (`src/agent/nos.py::julgar_assercao`);
essa métrica só agrega o resultado já calculado ali (via
`MOTIVO_CITACAO_NAO_ENCONTRADA`, constante compartilhada entre os dois
módulos para não duplicar a string e arriscar divergência).

`src/evaluation/run.py` (`python -m src.evaluation.run --config
experiments/{A,B,C,D}.yaml`): as quatro configs escrevem no mesmo
`experiments/resultados.csv` por padrão — trivial montar a tabela
comparativa A→D pedida pela especificação. Cada linha registra hash do
corpus (lido da própria coleção Chroma, não recalculado — reflete o que
foi de fato indexado), modelo de embedding, reranker, `RRF_K`,
`RETRIEVAL_POOL_SIZE` e timestamp.

`src/evaluation/run_geracao.py`: runner separado e mais lento (usa o
agente completo, uma chamada de LLM por linha do gabarito) — não faz
sentido acoplar ao runner rápido de recuperação.

### Interface Streamlit (Fase 6)

`app/streamlit_app.py` chama os nós do agente (`src/agent/nos.py`)
**diretamente**, não o grafo compilado (`src/agent/grafo.py`):
`processar_assercoes` no grafo é um único nó que faz o loop sobre as
assercões internamente, sem nenhum ponto de checkpoint para a UI
observar — e o custo real por asserção (dezenas de segundos em CPU, ver
métricas da Fase 4) torna progresso visível essencial para a
usabilidade. A UI orquestra os mesmos nós manualmente para atualizar a
barra de progresso entre um julgamento e outro.

Aviso de supervisão humana obrigatório e sempre visível no topo da
página (`st.warning`), citando o raciocínio do Annex 22 — exigência
explícita da especificação, não um detalhe cosmético.

**Testado com `streamlit.testing.v1.AppTest`** (não apenas leitura de
código): verifiquei manualmente primeiro que o servidor sobe (`streamlit
run` + `curl` retornando 200), depois formalizei em
`tests/test_streamlit_app.py` — carrega sem exceção, aviso de supervisão
presente, seletor com as 4 estratégias, e o relatório renderiza citação/
score/contagem por veredito corretamente a partir de um estado
pré-populado (sem rodar o pipeline completo, que levaria dezenas de
minutos).

Opção de **limitar a análise às N primeiras páginas** adicionada na
sidebar — não estava na especificação original, mas é uma acomodação
pragmática ao custo real de execução em CPU (permite testar a interface
sem esperar 25+ minutos a cada iteração).

### Verificação offline (Fase 7)

`tests/test_offline_completo.py` implementa a verificação final pedida
pela especificação ("escreva e execute um teste que bloqueia toda a
rede... roda uma consulta ponta a ponta"). **Decisão importante**: o
bloqueio é seletivo, não total — loopback (`127.0.0.1`/`::1`) é
permitido porque é assim que o próprio Ollama local é consultado (HTTP
sobre loopback); bloquear indiscriminadamente todo socket impediria o
próprio requisito de ter um LLM local funcionando. Só conexões para
hosts que não sejam loopback derrubam o teste.

Duas camadas: uma rápida (A/B/C/D sem LLM, roda sempre) e uma completa
com o LLM real via Ollama (`RUN_SLOW_LLM_TESTS=1`, pelo mesmo motivo de
custo já documentado na Fase 4). **Executei as duas de verdade** nesta
sessão, não só escrevi o código: a rápida confirmou as quatro
estratégias funcionando com rede externa bloqueada; a completa rodou um
julgamento real do agente (embedding → busca → Ollama →
auto-verificação) com a mesma restrição, em ~30s. Também validei
separadamente que o bloqueio genuinamente rejeita um host externo real
(`8.8.8.8`) com `RuntimeError` — não é um mecanismo que passaria por
não estar de fato monitorando nada.

### Script de instalação automática (`instalar.py`)

Adicionado para permitir que um colega clone o repositório e rode um
único comando (`python3 instalar.py`) para ter tudo pronto -- Python
3.11, Ollama, o modelo de LLM, os modelos de embedding/reranker, o
corpus indexado e a interface aberta no navegador.

**Decisões**:
- Só usa a biblioteca padrão do Python (nenhuma dependência externa) --
  precisa funcionar ANTES de qualquer coisa estar instalada.
- Idempotente: cada etapa verifica se já está pronta antes de refazer
  (`.venv` existe? modelo já em `ollama list`? `models/bge-m3/config.json`
  já existe?). Rodar de novo depois de uma etapa falhar retoma do ponto
  certo, não repete tudo.
- Alvo é macOS com Homebrew — mesma restrição já documentada no
  `README.md` > Requisitos; o projeto nunca foi validado em outro SO.

**Bug real encontrado e corrigido durante o teste**: com o stdout
redirecionado para um arquivo (não um terminal), o `print()` do script
fica bufferizado pelo Python -- só a saída dos subprocessos
(brew/pip/ollama/hf, que escrevem direto no descritor de arquivo)
aparecia em tempo real; as mensagens de progresso do próprio script só
apareceriam quando o processo terminasse (nunca, já que a última etapa
sobe um servidor que fica rodando). Corrigido com
`sys.stdout.reconfigure(line_buffering=True)` logo no início do script.
Testado de verdade (não só lido no código): rodei o script duas vezes
nesta máquina redirecionando para arquivo, confirmando que todas as
etapas foram puladas corretamente (Python/Ollama já instalados, `.venv`
existente, modelo Ollama já baixado, ambos os modelos de embedding já
baixados) e que a barra de progresso chegou em 100% com a Streamlit
subindo em seguida.

### Fechamento do projeto

Todas as 7 fases do roteiro (`PROMPT-CLAUDE-CODE.md`) estão
implementadas e testadas. O que resta é trabalho do usuário, não do
código: anotar o ground truth real (Fase 5) e rodar
`python -m src.evaluation.run --config experiments/{A,B,C,D}.yaml` para
obter os números reais do experimento central do TFG. Ver `README.md` >
"Limitações conhecidas" para a lista consolidada de ressalvas
metodológicas a citar na redação.

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

## Melhorias de performance (pós-Fase 7)

O usuário reportou que a análise de documento na Streamlit demorava
muito e que a extração de TODAS as páginas terminava por completo antes
de qualquer julgamento começar a aparecer. Investigação (agentes
Explore + Plan) confirmou a causa e mapeou oportunidades de
paralelismo/batching; plano completo em
`~/.claude/plans/temos-todo-o-contexto-magical-piglet.md`.

**Achado empírico mais importante, medido três vezes de formas
diferentes nesta máquina (Apple M4, 10 núcleos, 16GB, Ollama com
aceleração Metal, `OLLAMA_NUM_PARALLEL` não configurado)**: chamadas
concorrentes ao LLM (extração ou julgamento) **não reduzem o tempo
total** — o Ollama serializa gerações para o mesmo modelo por padrão.
Uma primeira medição sugeriu 66% de ganho, mas era viés de aquecimento
do modelo (a chamada "sequencial" incluía o custo de carregar o modelo
frio); refeita com aquecimento controlado e ordem invertida, o
resultado caiu para zero/negativo. **Isso mudou a priorização**: as
melhorias implementadas foram as que não dependem de paralelismo
incerto do Ollama, não a concorrência de chamadas de LLM em si (essa
ficou fora de escopo por enquanto — precisaria reconfigurar o serviço
Ollama compartilhado do usuário e testar `OLLAMA_NUM_PARALLEL`, uma
mudança de sistema que exige confirmação explícita antes de fazer).

**Implementado**:
1. **Cache do índice BM25** (`src/indexing/esparso.py`): `buscar()`
   desserializava o índice inteiro do disco a cada chamada — único
   componente sem cache entre os quatro modelos/índices do agente
   (embedding, reranker e LLM já eram `lru_cache`). Achado durante o
   mapeamento, não relatado pelo usuário.
2. **Filtro de relevância na extração** (`src/agent/prompts.py`):
   critério mais rígido para `SISTEMA_EXTRACAO_ASSERCOES` — só extrai
   uma afirmação se for plausível imaginar uma citação normativa
   específica que a confirme ou negue, com exemplos negativos concretos
   (datas/lotes/leituras pontuais sem exigência associada). **Essa foi
   a melhoria de maior impacto real**: no documento fictício de teste,
   reduziu de 62 para 29 asserções extraídas, cortando o tempo total de
   ~25min para ~12min14s — quase inteiramente explicado pela redução no
   número de chamadas ao LLM (66→33), não por batching (o custo por
   chamada de LLM individual ficou praticamente idêntico, ~22s antes e
   depois — batching acelera embedding/Chroma/reranker/disco, uma
   fração pequena comparada ao custo de geração do LLM).
3. **Interleaving por página** (`app/streamlit_app.py`): um único loop
   por página julga as asserções daquela página antes de seguir para a
   próxima, em vez de extrair TODAS as páginas primeiro. Resolve
   literalmente a reclamação do usuário (tempo até o primeiro resultado
   visível cai de "100% da extração" para "1 página") — reordenação
   pura, não muda o tempo total sozinha.
4. **`julgar_assercao` dividido** em wrapper fino + núcleo reutilizável
   `julgar_assercao_com_resultados(resultados, assercao)`
   (`src/agent/nos.py`) — refatoração pura, comportamento e testes
   inalterados, mas permite reaproveitar a lógica de julgamento com
   resultados de recuperação já buscados em lote.
5. **Recuperação em lote** (`buscar_lote`, método aditivo em toda a
   camada de recuperação — não mexe no `Retriever.buscar()` nem nos
   testes fake existentes): `src/indexing/vetorial.py` faz uma única
   chamada de embedding + uma única chamada `collection.query()` com
   múltiplos `query_embeddings` (Chroma já suporta isso nativamente,
   só não era usado) em vez de um par de chamadas por assercão.
   `src/retrieval/reranqueado.py::buscar_lote` é o maior ganho desta
   camada: uma única chamada `CrossEncoder.predict()` sobre os pares de
   TODAS as asserções do lote, em vez de uma chamada por asserção.
   `src/retrieval/hibrido.py` roda denso+esparso em paralelo via
   `ThreadPoolExecutor` (são independentes entre si; isso sim ajuda,
   diferente da concorrência de LLM, porque usa embeddings/BM25, não o
   Ollama). Novo `RETRIEVAL_BATCH_SIZE` em `src/config.py`.
   `app/streamlit_app.py` usa `buscar_lote` por página (com fallback via
   `hasattr` para retrievers que não implementem o método).

**Ocorrência rara e não relacionada, registrada por transparência**: ao
rodar `tests/test_retrieval.py` repetidas vezes após adicionar
`ThreadPoolExecutor` em `hibrido.py`, uma vez (1 em 6 execuções) apareceu
`libc++abi: terminating due to uncaught exception ... recursive_mutex
lock failed` no encerramento do processo, DEPOIS do pytest já ter
reportado sucesso (exit code 0). Não reproduziu em 5 execuções
subsequentes — características de uma race condition conhecida entre
threading e bibliotecas nativas de ML (torch/hnswlib) no encerramento
do interpretador em macOS, não um bug de lógica.

## Expansão do corpus (pós-Fase 7)

Usuário reportou taxa alta de `INDETERMINADO` e pediu para investigar
se era falta de cobertura na base de conhecimento. Evidência real
encontrada antes de expandir (não decidido por suposição): uma citação
de julgamento real foi parar em "ICH Q10 4.3" por falta de uma norma
sobre gestão de risco no corpus. **Peguei o pedido original do usuário
(reestruturar tudo em `base-conhecimento/`, baixar ~13 documentos
incluindo Farmacopeia e traduções livres de fontes não oficiais como
Scribd) e negociei um escopo reduzido**: manter `data/normas/` (já
conectado ao manifesto/pipeline existente, não criar árvore paralela),
adicionar só documentos com evidência real de lacuna, verificar cada um
por conteúdo antes de aceitar (mesmo rigor da RDC 658 original), e evitar
traduções não-oficiais sempre que uma fonte oficial existir.

**Documentos adicionados**:
- `RDC_166_2017.pdf` (validação de métodos analíticos) — mirror
  fitoterapiabrasil.com.br, conteúdo conferido contra o cabeçalho oficial
  (Diretoria Colegiada, Art. 1º–71).
- `PR_RDC_166_2017.pdf` (Perguntas e Respostas RDC 166/2017 e Guia
  10/2017) — baixado direto do gov.br/anvisa oficial.
- `GUIA_ANVISA_62_2023_GERENCIAMENTO_RISCOS.pdf` — **usado no lugar do
  ICH Q9(R1)**: não existe tradução oficial em português do ICH Q9(R1)
  (confirmado por busca), e o usuário pediu tudo em pt-BR. Em vez de
  usar uma "tradução livre" de proveniência incerta (risco real: um erro
  de tradução faria o agente citar uma exigência que não existe),
  encontrei que a ANVISA publica seu **próprio guia oficial** (Guia nº
  62/2023, 19/07/2023) cobrindo o mesmo conteúdo do ICH Q9(R1)
  (estrutura do sumário quase idêntica: introdução, escopo, princípios,
  processo geral, metodologia, integração, definições, anexo de
  ferramentas) — oficial, em português, sem o risco de tradução
  não-verificada. Resolve o pedido de pt-BR e a preocupação de rigor ao
  mesmo tempo.

**Terceira família de chunking** (`src/ingestion/chunking_perguntas_respostas.py`):
nem Art./§ (normas brasileiras) nem seções numeradas "puras" do ICH Q10
(número sozinho numa linha em negrito, título em linhas negrito
SEPARADAS) -- em documentos de Perguntas e Respostas o número e o texto
da pergunta/título aparecem JUNTOS no MESMO trecho em negrito (ex.:
"3.1.2. Em casos que não se tratam de registro de IFA: [...]?"). Essa
mesma família também funcionou para o Guia ANVISA 62/2023 (que segue o
padrão do ICH Q9(R1), também "número+título juntos") — não foi
necessária uma quarta estratégia.

**Três bugs reais encontrados e corrigidos durante a validação**:
1. `src/ingestion/chunking.py` (afeta TAMBÉM RDC 658/IN 134/138 já
   indexadas, retroativo): a RDC 166/2017 usa "°" (sinal de grau,
   U+00B0) em vez de "º" (indicador ordinal, U+00BA) para os artigos
   1-9, e NENHUMA pontuação para os artigos 11+ (só o artigo 10 usa
   ponto) -- três grafias diferentes no MESMO PDF. Com o regex exigindo
   "º" ou ".", só 1 dos 71 artigos era reconhecido. Corrigido tornando o
   caractere final opcional.
2. Marcadores consecutivos sem texto normal entre eles (ex.: "3.4.5.
   PRECISÃO" imediatamente seguido por "3.4.5.1 <pergunta>") faziam o
   segundo ser engolido como continuação do título do primeiro -- mesma
   classe de bug já corrigida no chunker do ICH Q10 na Fase 1, mas essa
   correção não cobria o caso aqui porque marcador+título vêm juntos, não
   separados.
3. Marcadores em que o número está sozinho numa linha (sem título na
   mesma linha, ex. "3.1." seguido de "ANEXO I" na linha seguinte --
   mistura do padrão ICH dentro de um documento que majoritariamente usa
   o padrão "número+título juntos") não eram reconhecidos como início de
   novo marcador porque a checagem exigia espaço em branco após o
   número, que não existe no fim de uma linha isolada. Corrigido
   aceitando também fim-de-string.

**Limitação de corpus aceita e documentada** (não corrigida, mesmo
espírito do defeito de numeração do ICH Q10 na Fase 1): o `titulo_secao`
fica impreciso para ~3 marcadores na cauda do documento de Perguntas e
Respostas (seções ANEXO III/GUIA/OUTRAS DÚVIDAS), provavelmente por um
fragmento de resposta com número em negrito sendo confundido com um
marcador real. O CONTEÚDO desses chunks (pergunta+resposta) continua
correto — só o rótulo de contexto hierárquico fica desatualizado.

Corpus final: 1248 chunks (RDC 658: 534, P&R RDC 166: 213, IN 138: 162,
RDC 166: 122, Guia ANVISA 62: 92, ICH Q10: 65, IN 134: 60). Retesting
manual confirmou relevância alta (score > 0.94) para consultas sobre
validação analítica e gestão de risco -- exatamente os tópicos que
motivaram a expansão. 93 testes passando.

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
