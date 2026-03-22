# HyperRetrieval — Benchmarking Report
**Date:** 2026-03-23
**Scope:** Build pipeline + Serve layer vs. global SOTA
**Basis:** Academic papers (ACL/ICML/ASE/ESEM/ICLR 2024–2026), industry engineering blogs, MTEB/CoIR leaderboards

---

## Executive Summary

HyperRetrieval's core design is **well-aligned with proven patterns** for production codebase intelligence. The symbol-level indexing, instruction-tuned embeddings, and graph-augmented retrieval are all first-principles correct. The gaps are incremental, not architectural.

**Overall maturity: 7.5 / 10**

---

## Part 1 — Build Pipeline

### Stage 1: Symbol Extraction (`01_extract.py`)

| Dimension | HyperRetrieval | SOTA |
|---|---|---|
| Granularity | Function / type / class | ✅ Symbol-level is the proven winner |
| AST fidelity | Regex + brace-matching (Haskell/Rust) | ⚠️ Full AST parsers (Tree-sitter) outperform regex by +2.7–5.5 pts on RepoEval |
| Python | Python `ast` module (correct) | ✅ |
| Body truncation | 8000 chars / 200 lines | ⚠️ Hard-truncation mid-function distorts embeddings; hash-based skip is better |
| Call graph | Within-module + fully-qualified | ⚠️ Dynamic dispatch not resolved (acceptable for static typed langs) |
| Co-change collection | All-pairs within commit (≤40 files) | ✅ Standard approach |

**Gap:** Tree-sitter (Rust-native, 40+ languages) would give exact function boundary detection for Haskell and Rust, replacing the brace-counting heuristic. The `cAST` paper (June 2025, arxiv:2506.15655) demonstrated +5.5 pts on SWE-bench using AST-accurate boundaries vs sliding-window. HyperRetrieval is closer to AST-accurate than sliding-window but not fully there for Haskell.

---

### Stage 2: Graph Clustering (`02_build_graph.py`)

| Dimension | HyperRetrieval | SOTA |
|---|---|---|
| Algorithm | Louvain (resolution=1.2) | ⚠️ **Leiden algorithm** supersedes Louvain as of 2024 |
| Graph level | Module-level for clustering | ✅ Correct — node-level Louvain on 94K nodes is intractable |
| Isolated modules | Own singleton cluster | ✅ |
| Edge types | Import + co-change | ⚠️ ESEM 2024: adding call-graph edges to clustering input improves community quality |

**Louvain vs. Leiden:** Louvain has a proven theoretical defect — it can produce disconnected communities (nodes in the same cluster with no path between them). The Leiden algorithm (Traag et al.) provably eliminates this, runs in O(n log n), and uses the same API (`python-igraph`). For a 4,809-module graph it's a drop-in fix.

---

### Stage 3: Embedding (`03_embed.py`)

| Dimension | HyperRetrieval | SOTA |
|---|---|---|
| Model | Qwen3-Embedding-8B | ✅ **#1 MTEB Code (80.68)** as of March 2026 — correct choice |
| Instruction prefix | Yes ("Represent this code module for...") | ✅ Required for Qwen3; omitting drops MTEB score ~4 pts |
| Dimension | 4096d | ✅ Full fidelity; 1024d truncated Matryoshka is only ~0.5 pts worse for 2× speed |
| Batch size | 32 on RTX 5090 fp16 | ✅ Optimal for 8B model |
| Text synthesis | lang + service + module + sig + cluster + docstring + ghost_deps + commits | ✅ Rich context; matches multi-field best practice |
| Checkpoint | `.npy` before LanceDB write | ✅ Correct crash safety |

**Closest competitor:** Nomic Embed Code 7B (Mar 2025) on CoRNStack — marginally competitive, not clearly superior. Qwen3-8B is the right choice. No action needed.

---

### Stage 4: LLM Summarization (`04_summarize.py`)

| Dimension | HyperRetrieval | SOTA |
|---|---|---|
| Sampling | Stratified across modules | ✅ Avoids bias toward large modules |
| Retry | Exponential backoff (2^n, max 5) | ✅ |
| Resume | Hash-based skip on re-run | ✅ |
| Summary schema | name + purpose + contracts + data_flows + ghost_deps + risk_flags + cross_service_links | ✅ More structured than most systems |
| Per-cluster context | Node subset from that cluster only | ⚠️ Cross-cluster callers/callees not included — inter-cluster contracts may be incomplete |

**Gap:** Cross-cluster call edges are not fed into the LLM summarizer. A cluster that exports a critical API to 3 other clusters will not mention those callers in its summary unless they happen to fall in the stratified sample. Adding "top-5 external callers" as context would improve `cross_service_links` accuracy.

---

### Stage 6: Co-Change Index (`06_build_cochange.py`)

| Dimension | HyperRetrieval | SOTA |
|---|---|---|
| Weighting | Raw co-occurrence count | ✅ Simple, interpretable |
| Memory | Streaming `ijson` | ✅ O(1) memory on 1.1GB file |
| Mega-commit filter | Skip >40-file commits | ✅ Standard noise filter |
| Cold-start | No fallback for zero-history modules | ❌ **Gap** |
| Integration type | Co-change only | ⚠️ ESEM 2024: hybrid with structural edges improves F1 |

**Cold-start gap:** 94,244 symbols, but co-change index has 111,005 pairs across 7,363 modules. Many new modules have zero co-change history. ESEM 2024 showed that adding import-graph and call-graph as synthetic co-change pairs (at lower weight, e.g. 0.1 vs observed 1.0) plugs this gap. This is especially relevant for `get_blast_radius` on recently-added modules.

---

### Stage 7: Doc Chunking (`07_chunk_docs.py`)

| Dimension | HyperRetrieval | SOTA |
|---|---|---|
| Split strategy | H2/H3 headings, max 3000 chars | ✅ Semantic boundaries beat fixed-size for markdown |
| Max chunk size | 3000 chars | ⚠️ 3000 chars ≈ 750 tokens — fine for Qwen3's 8192 token limit |
| Auto-tagging | Regex domain patterns | ✅ Lightweight, no false negatives for known domains |
| Same embedding model | Yes (Qwen3) | ✅ Consistent embedding space |

No significant gap here.

---

## Part 2 — Serve Layer

### Retrieval Strategy (`retrieval_engine.py`)

#### Current pipeline order (per query):
```
1. stratified_vector_search(queries)     ← dense semantic
2. cross_service_keyword_search(query)   ← exact token match
3. module_graph_expand(seed_modules)     ← BFS on import graph
4. cochange_path_traverse(seed)          ← BFS on co-change graph
5. score combination (additive weights)  ← custom scoring
```

#### SOTA pipeline (GAHR-MSR pattern, 2024–2025):
```
1. BM25 / TF-IDF                         ← exact match, 20 candidates
2. Dense vector                           ← semantic, 20 candidates
3. RRF merge (k=60)                       ← no hyperparameters
4. Graph traversal (callers/callees)      ← structural expansion
5. Cross-encoder rerank                   ← precision at top-10
```

| Component | HyperRetrieval | SOTA |
|---|---|---|
| Dense retrieval | ✅ Stratified, multi-query | ✅ |
| Exact/keyword | ✅ keyword_search | ⚠️ Custom regex vs. proper BM25 — miss phrase matching, IDF weighting |
| Fusion method | Additive weighted scores | ⚠️ RRF (k=60) is parameter-free and outperforms fixed weights in ablations |
| Graph expansion | ✅ Both import + co-change BFS | ✅ |
| Reranking | ❌ None | ⚠️ Cross-encoder (ColBERT/BGE-reranker) adds +8–15% precision on top-10 |
| Multi-query | ✅ Generates query variants | ✅ |
| Service budget | ✅ Per-service allocation | ✅ Prevents 1 large service from monopolizing results |

**Biggest gap:** No BM25 + no reranker. Exact symbol name queries (`TxnSplitDetail`, `createPaymentLink`) rely on keyword_search which uses simple word-presence matching without IDF weighting. BM25 (e.g., `rank_bm25` library, 200 lines to add) would score rare identifiers much higher than common words.

---

### Tool Interface (`tools.py` / MCP)

| Dimension | HyperRetrieval | SOTA |
|---|---|---|
| Tool count | 8 tools | ✅ Research shows >10 tools degrades LLM performance |
| Tool granularity | Coarse (module) → fine (symbol) | ✅ Matches optimal ReAct pattern |
| Recommended call sequence | search_modules → get_module → get_function_body | ✅ Correct coarse-to-fine |
| Max tool calls | 30 (with extend) | ✅ SWE-Effi 2025: degradation starts at ~30 LLM calls |
| Loop detection | ✅ (seen_calls tracking) | ✅ |
| Intermediate reasoning | ✅ (emitted between rounds) | ✅ |

Tool design is well-tuned. No significant gaps.

---

### `search_modules` — Scoring Cap Analysis

Current: top 20 modules by score, scores are semantic similarity + keyword hits.

**Problem:** 20 is too many for LLM context efficiency; quality drops as rank increases. The 20th result is often noise.

**SOTA finding (RAG precision curve):** Precision@K for code retrieval peaks at K=5–8 for well-formed queries. Beyond K=10, marginal utility approaches zero while token cost scales linearly.

**Recommendation:** Cap at 10, apply density-based score threshold (drop modules below `score < 0.6 × top_score`). This is a pure serve-layer change, no re-index.

---

### Context Assembly & LLM Utilization

| Dimension | HyperRetrieval | SOTA |
|---|---|---|
| Context ordering | Appended in retrieval order | ⚠️ Lost-in-the-Middle: put highest-relevance results first |
| Token budget | 16K tool rounds / 65K final | ✅ Generous; Kimi supports 128K |
| Compression | None | ⚠️ LLMLingua-2 (20× compression, 1.5% accuracy loss) for large contexts |
| History truncation | Full conversation history | ⚠️ Long sessions grow unbounded; sliding window or summarization needed |

---

## Part 3 — Comparative Positioning

### vs. Sourcegraph Cody
- Cody: BM25 + semantic, no pre-built graph, ~100K lines per response, no symbol-level embeddings
- HyperRetrieval: symbol-level graph + co-change + structured summaries + MCP tool interface
- **HyperRetrieval advantage:** Graph topology awareness, co-change coupling, cluster summaries, blast-radius analysis
- **Cody advantage:** Real-time (no pre-build), multi-language Tree-sitter parsing

### vs. Cursor
- Cursor: Dynamic BM25 + semantic on-the-fly, no persistent graph, repo-map (ctags-level)
- **HyperRetrieval advantage:** Deep pre-computed graph, cluster summaries, cross-service coupling
- **Cursor advantage:** Zero setup, always current, handles any language

### vs. Augment Code
- Augment: Maintains explicit symbol-reference graph, near-real-time, cloud-hosted
- Most similar architecture to HyperRetrieval; Augment has real-time delta updates, HyperRetrieval requires full rebuild
- **HyperRetrieval advantage:** Co-change index, domain-specific graph (Juspay payments)
- **Augment advantage:** Always-current, Tree-sitter exact boundaries

### vs. GitHub Copilot Workspace
- Copilot: Retrieval-augmented, no persistent graph, heavy embedding cache
- **HyperRetrieval advantage:** Richer structural context (call graph, co-change, cluster summaries)

---

## Part 4 — Prioritized Gap Table

| # | Gap | Layer | Effort | Impact | Paper / Source |
|---|---|---|---|---|---|
| 1 | **BM25 + RRF fusion** | Serve | Medium | 🔴 High | RRF consensus 2024; exact symbol queries failing |
| 2 | **Leiden → replace Louvain** | Build | Low | 🟡 Medium | Traag et al. 2024; disconnected community fix |
| 3 | **Synthetic co-change edges from call/import graph** | Build | Medium | 🟡 Medium | ESEM 2024; cold-start coverage for new modules |
| 4 | **Cross-encoder reranker** | Serve | Medium | 🟡 Medium | Voyage AI Oct 2025; +8–15% precision @10 |
| 5 | **search_modules cap 20 → 10 + density threshold** | Serve | Low | 🟡 Medium | RAG precision curve finding |
| 6 | **Context ordering: highest-relevance first** | Serve | Low | 🟡 Medium | Lost-in-the-Middle (archived finding, confirmed 2025) |
| 7 | **Tree-sitter for Haskell/Rust body extraction** | Build | High | 🟡 Medium | cAST, June 2025; +5.5 pts on boundary accuracy |
| 8 | **Cross-cluster callers in summarization context** | Build | Low | 🟢 Low | Internal design gap |
| 9 | **Incremental delta updates (hash-based re-embed)** | Build | High | 🟢 Low | Engineering consensus; only matters at high rebuild frequency |
| 10 | **LLMLingua compression for large contexts** | Serve | Low | 🟢 Low | Microsoft Research; only needed for >8K final answers |

---

## Verdict

HyperRetrieval is **production-quality** for a v1 system. The embedding model choice (Qwen3-8B) is globally optimal. The graph + co-change dual-indexing is more sophisticated than most commercial tools. The MCP tool structure and ReAct loop design align with best-practice findings.

The primary engineering debt is in **retrieval fusion** (no BM25, no reranker, additive scoring vs. RRF) — this is the layer most likely to produce noisy results for exact-identifier queries. The **Leiden clustering fix** is a low-effort correctness improvement. Everything else is refinement.

---

*Sources: MTEB leaderboard (Mar 2026), CoIR (ACL 2025), cAST (arxiv:2506.15655), ESEM 2024, GraphCoder (ASE 2024), SWE-Effi (arxiv:2509.09853), OpenHands blog (Nov 2025), Voyage AI (Oct 2025), LLMLingua (Microsoft Research), Leiden algorithm (Traag et al.), Frontiers AI survey (2025)*
