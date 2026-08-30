# Referências e corpus — TFG II

## A. Corpus normativo (`data/normas/`)

Estes são os documentos que viram a **base de conhecimento indexada**. São a
autoridade contra a qual o agente julga conformidade.

### A.1 — Norma brasileira vigente ⚠️ ATENÇÃO

| Documento | Situação | Link |
|---|---|---|
| **RDC nº 658, de 30/03/2022** — Diretrizes Gerais de BPF de Medicamentos | **VIGENTE**. Revogou a RDC 301/2019. Em vigor desde 02/05/2022 | `http://antigo.anvisa.gov.br/documents/10181/6415119/RDC_658_2022_.pdf/aff5cdd7-4ad1-40e8-8751-87df566e6424` |
| **RDC nº 972, de 22/04/2025** | Altera a RDC 658/2022 (art. 372, controle em linha) | Busque em AnvisaLegis |
| **RDC nº 301, de 21/08/2019** | **REVOGADA** — use só como histórico no TFG | `https://anvisalegis.datalegis.net/` (busca por RDC 301/2019) |

Portal oficial de busca: `https://anvisalegis.datalegis.net/`
Se o link `antigo.anvisa.gov.br` cair, busque "RDC 658 2022" no AnvisaLegis.

**Instruções Normativas complementares relevantes ao seu escopo:**

- **IN nº 134/2022** — BPF complementares a **sistemas computadorizados**.
  Essa é a mais importante para você: é a norma que o *seu próprio sistema*
  teria que atender se fosse implantado numa planta. Rende um parágrafo forte
  na discussão.
- **IN nº 138/2022** — BPF complementares a **estudos de qualificação e
  validação**. Substituiu a IN 47/2019.

**Perguntas & Respostas de BPF (ANVISA, 23/02/2023)** — documento com 674
perguntas e respostas sobre a RDC 658/2022 e suas INs. Para você isso é ouro:
é uma fonte quase pronta de **ground truth** para o conjunto de avaliação, com
perguntas reais e respostas com respaldo normativo. Procure em
`gov.br/anvisa` → Centrais de Conteúdo → Publicações → Perguntas e Respostas.

### A.2 — Normas internacionais

| Documento | Link |
|---|---|
| **ICH Q10 — Pharmaceutical Quality System** (Step 4, jun/2008) | `https://database.ich.org/sites/default/files/Q10%20Guideline.pdf` |
| ICH Q9(R1) — Quality Risk Management | `https://database.ich.org/` |
| **EudraLex Vol. 4, Annex 22 — Artificial Intelligence** (rascunho, consulta pública 07/07 a 07/10/2025) | `https://www.gmp-compliance.org/files/guidemgr/mp_vol4_chap4_annex22_consultation_guideline_en.pdf` |

**Sobre o Annex 22:** continua em rascunho. A consulta pública encerrou em
07/10/2025 com cerca de 1.300 comentários; a EMA realizou workshop
multissetorial em 30/06 e 01/07/2026 e o plano de trabalho do Inspectors
Working Group aponta o 4º trimestre de 2026 como alvo para entregar o texto
final à Comissão Europeia. Nada foi adotado até agora — cite explicitamente
como *rascunho*, o que você já faz corretamente no artigo. Verifique o status
antes da defesa.

---

## B. Referências científicas (fundamentação teórica)

### B.1 — Já citadas no seu TFG

| Autor / Ano | Título | Link |
|---|---|---|
| Vaswani et al., 2017 | Attention Is All You Need | `https://arxiv.org/abs/1706.03762` |
| Devlin et al., 2019 | BERT | `https://arxiv.org/abs/1810.04805` |
| Brown et al., 2020 | Language Models are Few-Shot Learners (GPT-3) | `https://arxiv.org/abs/2005.14165` |
| Lewis et al., 2020 | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | `https://arxiv.org/abs/2005.11401` |
| Finardi et al., 2024 | The Chronicles of RAG: The Retriever, the Chunk and the Generator | `https://arxiv.org/abs/2401.07883` |
| Hasan et al., 2025 | Engineering RAG Systems for Real-World Applications | `https://arxiv.org/abs/2506.20869` |
| Arslan, Ghanem, Munawar & Cruz, 2024 | A Survey on RAG with LLMs — *Procedia Computer Science*, v. 246, p. 3781–3790 | DOI `10.1016/j.procs.2024.09.178` |
| Şakar & Emekci, 2025 | Maximizing RAG Efficiency: A Comparative Analysis of RAG Methods | DOI `10.1017/nlp.2024.53` |

Correção de referência: no seu artigo, `arslan2024` aparece sem entrada
formatada na lista final. É artigo do KES 2024, publicado na *Procedia
Computer Science*, não preprint arXiv.

### B.2 — Lacunas que valem preencher

| Referência | Por que importa para você |
|---|---|
| **Kim & Min, 2024 — From RAG to QA-RAG: Integrating Generative AI for Pharmaceutical Regulatory Compliance Process** — `https://arxiv.org/abs/2402.01717` | **Prioridade máxima.** É exatamente o seu problema: chatbot RAG para conformidade regulatória farmacêutica, com recuperação de duplo caminho, testado sobre documentos regulatórios reais. Publicado também no ACM SAC 2025. Sua seção 2.4 hoje cita o Arslan dizendo que o caso de uso "existe"; o Kim & Min é o trabalho que de fato o executou. Sem ele, a banca pode perguntar por que você não comparou. |
| Khan et al., 2024 — Developing RAG-based LLM Systems from PDFs: an Experience Report — `https://arxiv.org/abs/2410.15944` | Relato de engenharia sobre RAG a partir de PDFs — sustenta suas decisões de extração e chunking. |
| Robertson & Zaragoza, 2009 — The Probabilistic Relevance Framework: BM25 and Beyond | Você usa BM25 como configuração B do experimento e hoje não tem referência canônica para ele. |
| Karpukhin et al., 2020 — Dense Passage Retrieval — `https://arxiv.org/abs/2004.04906` | Referência canônica para bi-encoders. Você descreve recuperação densa sem citar a origem. |
| Nogueira et al., 2020 — Document Ranking with a Pretrained Sequence-to-Sequence Model (monoT5) — `https://arxiv.org/abs/2003.06713` | Você menciona monoT5 no cronograma e no texto sem citar a fonte. |
| Es et al., 2023 — RAGAS: Automated Evaluation of RAG — `https://arxiv.org/abs/2309.15473` | Seu cronograma cita RAGAS; falta a referência. |

---

## C. Pendências no artigo

1. **RDC 301/2019 → RDC 658/2022 (+ RDC 972/2025)** em todo o texto. Ocorre
   na introdução, nos objetivos 2, na fundamentação 2.4 e 2.5, na metodologia
   3.1 e 3.2, e no corpus 3.3. Sugestão de tratamento: uma nota de rodapé
   explicando que a 301/2019 foi revogada e que o trabalho adota a norma
   vigente, citando a 301 apenas como marco histórico. Isso transforma um erro
   em demonstração de rigor.

2. **Resumo, abstract e palavras-chave ainda em *lorem ipsum*.**

3. **Folha de rosto com "Nome do Primeiro Autor / Nome do Segundo Autor"** —
   placeholder não substituído.

4. **Citações em formato de chave BibTeX vazando no texto**: `arslan2024`,
   `lewis2020`, `vaswani2017`, `devlin2019`, `brown2020`, `huyen2025`,
   `finardi2024`, `hasan2025`, `sakar2025` aparecem cruas em vez do formato
   ABNT (AUTOR, ano). Provável `\citeauthor` sem entrada no `.bib`, ou chave
   digitada fora do comando de citação. Note que algumas referências aparecem
   nos dois formatos no mesmo parágrafo — a 2.2 tem `lewis2020` e
   `(Lewis et al., 2020)` a poucas linhas de distância.

5. **`huyen2025` é citado quatro vezes e não está na lista de referências.**
   Presumo *AI Engineering* (Chip Huyen, O'Reilly) — confirme e inclua.

6. **Falta um trecho na frase da seção 2.4** onde se lê "não devem ser
   utilizados isoladamente" — confira se a redação do Annex 22 está fielmente
   parafraseada, já que ela é a base da sua justificativa metodológica.

7. **Cronograma diz "Período: Julho a Novembro de 2026"** e a data do artigo é
   17/08/2026. Confira se a coluna de julho já está toda concluída.
