# State-of-the-Art Research: Code Intelligence & Codebase RAG (2024–2026)

Compiled: 2026-03-22. Focus on findings directly applicable to HyperRetrieval architecture.

---

## 1. Code Embedding Models

### Current Rankings (early 2026)

The landscape has shifted toward large LLM-backbone embedders trained on code-specific datasets.

**Qwen3-Embedding-8B** (released June 2025, [blog](https://qwenlm.github.io/blog/qwen3-embedding/), [paper arxiv:2506.05176](https://arxiv.org/pdf/2506.05176)):
- Ranks #1 on MTEB Multilingual leaderboard (score 70.58 as of June 2025)
- MTEB Code task score: **80.68** — highest reported across all publicly evaluated models
- 8B parameters, 32K context, up to 4096-dim embeddings
- Three-stage training: contrastive pre-training on weak supervision → supervised fine-tuning on labeled data → model merging
- **Relevance to HyperRetrieval:** This is what you're running. The MTEB Code score confirms it is the correct choice for symbol-level retrieval at this scale.

**Nomic Embed Code** (7B, released March 2025, [announcement](https://www.nomic.ai/news/introducing-state-of-the-art-nomic-embed-code)):
- Built on CoRNStack dataset (ICLR 2025 paper)
- Outperforms voyage-code-3 and OpenAI text-embedding-3-large on CodeSearchNet
- Open weights, GGUF available — practical for self-hosted deployment
- CoRNStack: 21M `<query, positive, negative>` triplets across multiple languages

**CodeRankEmbed** (nomic-ai, 137M parameters, 521MB, [HuggingFace](https://huggingface.co/nomic-ai/CodeRankEmbed)):
- Bi-encoder fine-tuned on CoRNStack with InfoNCE loss
- Excellent throughput/quality tradeoff for high-QPS systems
- Much faster inference than 7B+ models; suitable as a first-stage retriever in cascade

**voyage-code-3** (Voyage AI, December 2024, [blog](https://blog.voyageai.com/2024/12/04/voyage-code-3/)):
- Outperformed OpenAI text-embedding-3-large by **+13.80% average** across code retrieval datasets
- Available at 1024, 256 dimensions with quantization — 1/3 storage of competitors at comparable accuracy
- Evaluated head-to-head vs: OpenAI-v3-large/small, CodeSage-large, CodeRankEmbed, Jina-v2-code, voyage-code-2
- Now superseded by Nomic Embed Code 7B on CodeSearchNet

**BGE-M3** (BAAI):
- General-purpose multilingual model, MTEB overall 63.0
- Not competitive on code-specific benchmarks vs specialized models; useful only when multilingual + code is required simultaneously

**OpenAI text-embedding-3-large**:
- Solid baseline but consistently 13–17% behind voyage-code-3 on code retrieval tasks

**GraphCodeBERT / UniXcoder**:
- Pre-2024 models; now clearly inferior to LLM-backbone embedders on all benchmarks; not recommended for new deployments

### Key Benchmarks

**CoIR** ([ACL 2025](https://aclanthology.org/2025.acl-long.1072/), [GitHub](https://github.com/CoIR-team/coir)):
- 10 code datasets, 8 retrieval task types, 7 domains, 2M documents
- Now surpasses CodeSearchNet in monthly downloads; the de facto standard for code retrieval evaluation
- Compatible with MTEB/BEIR schema — results are cross-comparable
- First-place on CoIR leaderboard (Feb 2025): Salesforce model; E5-Mistral consistent top performer among evaluated models at paper time

**RepoQA** ([ICML 2024 Long-Context Workshop](https://evalplus.github.io/repoqa.html)):
- "Searching Needle Function" (SNF): 500 tasks, 5 languages × 10 repos × 10 needle functions
- Tests long-context code understanding, not embedding retrieval directly — complementary benchmark

**CodeSearchNet**: Still widely used; 6 languages; Nomic Embed Code 7B currently top-performing open model.

### Takeaway for HyperRetrieval
Qwen3-Embedding-8B is the correct production choice as of early 2026. If you need a lightweight cascade (fast first-stage + expensive re-rank), CodeRankEmbed (137M) → Qwen3-8B rerank is the pattern to follow.

---

## 2. Graph-Based Code Retrieval

### GraphCoder (ASE 2024, [paper](https://arxiv.org/abs/2406.07003))
- Builds a **Code Context Graph (CCG)** from control-flow + data-dependence + control-dependence edges between statements
- Coarse-to-fine retrieval: first retrieves at file/module level, then zooms to statement level
- Demonstrated improvements on repository-level code completion

### CodeRAG with Bigraph (arxiv:2504.10046, 2025)
- Bipartite graph: code entities on one side, natural language descriptions on the other
- **Pass@1 on repo-level generation: +35.57 points** (18.57 → 54.41) over baseline
- Key insight: the bipartite structure allows cross-modal retrieval (query in NL → code node via shared graph edges)

### Code Graph Model (arxiv:2505.16901, June 2025)
- Proposes treating the entire codebase as a typed property graph
- Nodes: functions, classes, modules, variables; Edges: calls, imports, inherits, defines
- Graph message-passing used for re-ranking retrieved candidates

### Community Detection for Code Graphs
- Louvain (used in HyperRetrieval) remains practical but has known resolution-limit issues for large dense graphs
- **Leiden algorithm** (Traag et al. 2019, but gaining adoption in 2024–2025) fixes Louvain's disconnected community problem — produces strictly better-connected communities
- Deep learning methods for community detection peaked in publications 2023–2024 ([survey, Frontiers AI 2025](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1572645/full)) but are computationally prohibitive for symbol graphs of 94K nodes
- **Practical recommendation:** Switch Louvain → Leiden for cluster summarization. Same interface (python-igraph / networkx-leiden), provably better output.

### Industry Systems
- **Sourcegraph Cody**: "search-first" architecture — pre-indexes entire repo into vector embeddings, feeds ~100K lines of related code per response; recently added agentic context gathering
- **Cursor**: Dynamic context loading, streams ≤8K lines; relies on fast BM25 + semantic search, not pre-built graphs
- **GitHub Copilot Workspace**: Hierarchical planning over file-level summaries, uses import graph to scope changes — not public architecture detail beyond blog posts
- **Augment Code**: Explicit graph of symbol references + embeddings, claims best monorepo performance at scale

---

## 3. Multi-Hop Retrieval: Fusion Strategies

### State of Practice (2025)

**Reciprocal Rank Fusion (RRF)** is the consensus standard for combining ranked lists from vector + BM25 + graph traversal:
- Formula: `RRF(d) = Σ 1/(k + rank_i(d))` where k=60 is standard
- OpenSearch 2.19 added native RRF in Neural Search plugin (2025)
- Outperforms simple score normalization + weighted sum in most empirical comparisons
- No hyperparameters to tune per-corpus — robust default

**Graph-Augmented Hybrid Retrieval with Multi-Stage Re-ranking (GAHR-MSR):**
- Stage 1: RRF over BM25 + dense vector pools
- Stage 2: Graph-aware re-ranking using structural proximity (callers/callees/imports)
- Stage 3: ColBERT late-interaction fine-grained re-scoring
- GraphRAG systems showed **77.6% improvement in MRR** over pure vector retrieval in knowledge graph settings

**Learned re-ranking**: Cross-encoders (e.g., ColBERT, MonoT5) outperform score-fusion on precision but add 50–200ms latency per query. Voyage AI published "The Case Against LLMs as Rerankers" (Oct 2025) arguing that LLM-based re-rankers are not cost-effective vs. dedicated cross-encoders.

### Ordering Strategy (what works)
1. BM25 for exact symbol names / identifiers (high precision on known queries)
2. Dense vector for semantic / paraphrase queries
3. Graph traversal for structural proximity (callers, dependencies)
4. RRF to merge lists → top-50 candidates
5. Cross-encoder rerank → top-10 for context window

---

## 4. Evolutionary Coupling / Co-Change Analysis

### ESEM 2024 Paper (most directly relevant)
**"Enhancing Change Impact Prediction by Integrating Evolutionary Coupling with Software Change Relationships"** ([ACM DL](https://dl.acm.org/doi/10.1145/3674805.3686668)):
- Key finding: pure co-occurrence association rules fail when entities **rarely co-change** (cold-start problem)
- Solution: integrate 12 types of structural relationships (imports, clones, call-graph edges) with co-change history
- Results on 6 open-source systems: top-5 precision, recall, F1, MAP all significantly improved
- This is the 2024 SOTA for change impact prediction

### ML-Based Approaches
- No dominant ML model for co-change prediction has emerged yet as of early 2026
- Graph Neural Networks on commit graphs are an active research direction but no production-ready system published
- Practical state-of-the-art remains: **weighted co-occurrence + structural coupling** (what HyperRetrieval does), augmented with the ESEM 2024 fusion of structural edges

### Recommendation for HyperRetrieval
Current co-change index (7,363 modules, 111,005 pairs) already captures co-occurrence. The gap is the lack of **structural coupling fallback** for modules with zero co-change history. Adding call-graph and import-graph edges as synthetic co-change pairs with lower weight (e.g., weight=0.1 vs observed co-change weight=1.0) would address cold-start.

---

## 5. Symbol-Level vs Chunk-Level Indexing

### cAST Paper (arxiv:2506.15655, June 2025)
**"cAST: Enhancing Code Retrieval-Augmented Generation with Structural Chunking via Abstract Syntax Tree"**:
- AST-based chunking vs. sliding-window chunking on RepoEval and SWE-bench
- **+2.7 to +5.5 points** improvement using AST-aligned chunks
- Key metric: Intersection over Union (IoU) — AST chunks have higher IoU because they don't split function bodies mid-statement

### Granularity Comparison (synthesis from multiple 2024–2025 papers)

| Granularity | Pros | Cons | Best For |
|---|---|---|---|
| Sliding-window (256–512 tokens) | Simple, uniform | Splits at arbitrary points, context bleed | General text RAG |
| File-level | Minimal indexing cost | Too coarse for large files | First-stage coarse retrieval |
| **Function/symbol-level** | Semantic unit, no bleed | Misses cross-function context | Symbol lookup, call tracing |
| AST subtree | Preserves structure | Complex to implement | Class + method together |
| Multi-granularity (hierarchical) | Best recall | Complex index | Production systems |

- **Hierarchical Code Graph Summarization showed 82% relative improvement** in retrieval precision in a 2025 study
- Production recommendation: index at symbol level (what HyperRetrieval does) AND maintain file-level summaries as a coarser retrieval layer — confirmed as optimal by multiple 2024–2025 findings
- An "Exploratory Study of Code Retrieval Techniques in Coding Agents" (Preprints.org, Oct 2025) found that content search + file-name search + structure search (LSP-level) at multiple granularities outperforms any single granularity

---

## 6. Incremental / Streaming Index Updates

### Current Practice

No academic paper establishes a canonical algorithm for incremental embedding index updates as of early 2026. The engineering consensus from production systems is:

1. **Git diff-based detection**: `git diff --name-only <last_indexed_commit>..HEAD` to identify changed files
2. **File-level re-embedding**: Re-embed only changed files; delete old vectors by file ID, insert new ones
3. **Symbol-level delta**: For AST-aware systems, diff at the function level — only re-embed functions whose body hash changed
4. **LanceDB specifics**: Supports `delete(filter)` + `add()` without full reindex. Automatic versioning with zero-copy reads allows point-in-time query during update.
5. **Milvus specifics**: 2025 Rust rewrite delivered 4× faster writes; supports upsert by primary key

### Key Engineering Pattern (from Beacon/Claude Code plugin, 2025)
- On startup: compare current git HEAD to last-indexed commit SHA
- Compute diff → re-embed only modified files
- After any edit: re-embed that file immediately (sub-second for single-file)
- Full re-index only on: model change, schema change, or >50% of files modified

### Dimension Lock-In Warning
Embedding dimension is fixed at index build time. Switching models (e.g., Qwen3-8B → future model) requires full re-index of all 94K symbols. Plan schema migrations accordingly.

---

## 7. ReAct Agent Loops for Code

### SWE-bench Verified Results (2025)

| Agent | Backbone | Resolution Rate | Cost/Issue | Avg API Calls |
|---|---|---|---|---|
| OpenHands + inference scaling | Claude 3.7 Sonnet | **66.4%** (5 attempts) | — | — |
| OpenHands (single trajectory) | Claude 3.7 Sonnet | 60.6% | — | — |
| Claude Opus 4.5 + Live-SWE-agent | Claude Opus 4.5 | **79.2%** | — | — |
| Agentless | Qwen3-32B | 48% | $0.34 | **83.1 API calls** |
| Mini-SWE-Agent | — | >74% | — | — |
| SWE-agent (NeurIPS 2024) | various | ~23–30% baseline | — | — |

### Key Architectural Findings

**Tool minimalism works**: Mini-SWE-Agent (100 lines Python, single bash tool) scores >74% on SWE-bench Verified. CodeScout (2025) uses only 1 tool vs 3–5 in prior methods. Adding more tools does not reliably improve performance with frontier models.

**Hierarchical localization** (Agentless pattern): Locate → generate candidates → rank by test pass rate. Effective but expensive (83 API calls average).

**Inference-time scaling** (OpenHands Nov 2025): Run N trajectories in parallel, use a trained critic model to select the best. +5.8 points absolute over single trajectory. This is the current frontier.

**Optimal tool call budget**: The SWE-Effi paper (arxiv:2509.09853) shows effectiveness degrades sharply beyond ~30 LLM calls for most scaffolds. HyperRetrieval's MAX_TOOL_CALLS=12 is aggressive but defensible for a retrieval assistant (not a full repair agent).

**ReAct loop structure that works**:
1. Localize (2–3 calls): module search → symbol search → body fetch
2. Understand (1–2 calls): trace callers/callees
3. Answer or generate patch
Total: 4–7 calls for well-scoped questions. Aligns with HyperRetrieval's "~5k tokens / 4–6 calls" guideline.

---

## 8. LLM Context Utilization in Code RAG

### Lost-in-the-Middle Problem
- Established finding (2023, confirmed in 2024–2025): LLMs attend poorly to information in the middle of long contexts
- Models with million-token windows still exhibit this; it is architectural, not just a context-length limitation
- For code RAG: place the most relevant symbol body **first or last** in the assembled context, not buried in the middle

### Reranking: Key 2025 Finding
**Voyage AI "The Case Against LLMs as Rerankers"** (Oct 2025):
- LLM-based reranking (using the generation LLM to score candidates) is not cost-effective
- Dedicated cross-encoder rerankers (ColBERT, MonoT5, BGE reranker) achieve comparable or better NDCG at 10–50× lower cost
- Recommendation: use a lightweight cross-encoder as stage-2 reranker, not the main LLM

### Prompt Compression
**LLMLingua / LongLLMLingua** (Microsoft Research):
- Up to **20× compression** with only 1.5% performance loss on reasoning tasks
- LongLLMLingua specifically addresses lost-in-middle: compresses middle tokens more aggressively, preserving start/end
- +21.4% accuracy boost on RAG tasks in controlled experiments
- Practical for HyperRetrieval: compress retrieved function bodies before assembling context when total tokens > 8K

**Multi-scale Positional Encoding (Ms-PoE)** (ICLR 2025):
- Plug-and-play — no fine-tuning required
- Improves middle-position accuracy by 20–40% vs baseline
- Can be applied to any transformer at inference time via position ID remapping

### 2025 Consensus Architecture for Code RAG
```
Query
 ├─ BM25 (exact symbol/identifier match)         → 20 candidates
 ├─ Dense vector (Qwen3-8B or Nomic-7B)           → 20 candidates
 └─ Graph traversal (callers/callees/co-change)   → 10 candidates
        ↓ RRF merge → 50 candidates
        ↓ Cross-encoder rerank → 10 candidates
        ↓ LLMLingua compression (if >8K tokens)
        ↓ Assemble: most relevant FIRST in context
```

---

## Summary: Gaps in Current HyperRetrieval vs SOTA

| Area | Current State | SOTA Gap | Priority |
|---|---|---|---|
| Embedding model | Qwen3-8B ✓ | None — already SOTA | — |
| Clustering | Louvain | Leiden (strictly better communities) | Medium |
| Co-change cold-start | Co-occurrence only | Add structural edges as synthetic pairs | High |
| Retrieval fusion | Vector only | BM25 + vector + graph RRF | High |
| Cross-encoder rerank | None | ColBERT/BGE reranker (stage 2) | Medium |
| Context assembly | Unknown order | Most relevant first | Low |
| Prompt compression | None | LLMLingua for >8K contexts | Low |
| Incremental updates | Full re-index | Git-diff + symbol-hash delta | Medium |
| Indexing granularity | Symbol-level ✓ | Add file-level coarse layer | Low |

---

## Sources

- [Qwen3 Embedding paper](https://arxiv.org/pdf/2506.05176) (June 2025)
- [Qwen3 Embedding blog](https://qwenlm.github.io/blog/qwen3-embedding/)
- [voyage-code-3 blog](https://blog.voyageai.com/2024/12/04/voyage-code-3/) (Dec 2024)
- [Voyage code retrieval evaluation methodology](https://blog.voyageai.com/2024/12/04/code-retrieval-eval/)
- [Nomic Embed Code announcement](https://www.nomic.ai/news/introducing-state-of-the-art-nomic-embed-code) (Mar 2025)
- [CoIR benchmark — ACL 2025](https://aclanthology.org/2025.acl-long.1072/)
- [CoIR GitHub](https://github.com/CoIR-team/coir)
- [GraphCoder — ASE 2024](https://arxiv.org/abs/2406.07003)
- [CodeRAG Bigraph](https://arxiv.org/html/2504.10046v1/) (2025)
- [Code Graph Model arxiv:2505.16901](https://arxiv.org/pdf/2505.16901) (2025)
- [Community detection deep learning survey — Frontiers AI 2025](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1572645/full)
- [ESEM 2024 — Evolutionary Coupling + Structural Relationships](https://dl.acm.org/doi/10.1145/3674805.3686668)
- [cAST paper arxiv:2506.15655](https://arxiv.org/html/2506.15655v1/) (June 2025)
- [RAG for code — survey arxiv:2510.04905](https://arxiv.org/html/2510.04905v1/)
- [OpenHands SOTA SWE-bench blog](https://openhands.dev/blog/sota-on-swe-bench-verified-with-inference-time-scaling-and-critic-model) (Nov 2025)
- [SWE-Effi — resource constraints analysis](https://arxiv.org/html/2509.09853v2)
- [Mini-SWE-Agent GitHub](https://github.com/SWE-agent/mini-swe-agent)
- [Voyage AI: Case Against LLMs as Rerankers](https://blog.voyageai.com/2025/10/22/the-case-against-llms-as-rerankers/) (Oct 2025)
- [LongLLMLingua — LlamaIndex blog](https://www.llamaindex.ai/blog/longllmlingua-bye-bye-to-middle-loss-and-save-on-your-rag-costs-via-prompt-compression-54b559b9ddf7)
- [RAG review 2025 — RAGFlow](https://ragflow.io/blog/rag-review-2025-from-rag-to-context)
- [RepoQA benchmark](https://evalplus.github.io/repoqa.html)
- [6 best code embedding models — Modal](https://modal.com/blog/6-best-code-embedding-models-compared)
- [Sourcegraph Cody anatomy](https://sourcegraph.com/blog/anatomy-of-a-coding-assistant)
- [RRF in hybrid search — advanced guide](https://glaforge.dev/posts/2026/02/10/advanced-rag-understanding-reciprocal-rank-fusion-in-hybrid-search/)
