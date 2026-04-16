"""
retrieval_engine.py — Generic codebase intelligence retrieval engine

Extracted from demo_server_v6.py. Works for ANY codebase that has been
processed through the pipeline (graph_with_summaries.json + vectors.lance).

Config-driven: point it at your artifact_dir and a config.yaml and it works.
No Chainlit dependency. Safe to import from MCP servers, CLI tools, or CI scripts.

GPU sharing: set EMBED_SERVER_URL=http://localhost:8001 and the embedder
runs as a separate service (embed_server.py). Both Chainlit and MCP server
then share one GPU load via HTTP — no OOM.
"""
import json, math, os, pathlib, re, time, threading, urllib.request, urllib.error
from collections import defaultdict
try:
    from rank_bm25 import BM25Okapi as _BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25Okapi = None
    _BM25_AVAILABLE = False

# ── LLM config (from env or config.yaml) ─────────────────────────────────────
LLM_API_KEY  = os.environ.get("LLM_API_KEY",  "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "")
LLM_MODEL    = os.environ.get("LLM_MODEL",    "kimi-latest")
MAX_TOOL_CALLS = 12

# ── Embed server URL (set to delegate GPU model to embed_server.py) ───────────
# When set, _encode_query calls this instead of loading the model in-process.
# Allows Chainlit + MCP server to share one GPU load.
EMBED_SERVER_URL = os.environ.get("EMBED_SERVER_URL", "")  # e.g. http://localhost:8001

EMBED_INSTRUCTION = (
    "Instruct: Represent this code module for finding semantically similar "
    "components across microservices. Query: "
)

# ── Default artifact dir ───────────────────────────────────────────────────────
_DEFAULT_ARTIFACT_DIR = pathlib.Path(
    os.environ.get("ARTIFACT_DIR",
                   str(pathlib.Path(__file__).parent / "demo_artifact"))
)

# ── Global state ───────────────────────────────────────────────────────────────
embedder:      object = None   # SentenceTransformer (loaded in-process, or None if using embed server)
_llm_client:   object = None   # openai.OpenAI (sync)
G:             object = None   # NetworkX node-level graph
MG:            object = None   # NetworkX module-level import graph
lance_tbl:     object = None   # LanceDB code vector table
doc_lance_tbl: object = None   # LanceDB doc vector table
reranker:      object = None   # CrossEncoder — disabled (kept for reference)

cluster_summaries:   dict  = {}
cochange_index:      dict  = {}
ownership_index:     dict  = {}   # module → [{"email","name","commits"}, ...]
granger_index:       dict  = {}   # "A→B" → {"source","target","best_lag","p_value","f_statistic"}
community_index:     dict  = {}   # community_id → {"size","services","label","cross_service"}
module_to_community: dict  = {}   # module_cc_key → community_id
activity_index:      dict  = {}   # module_name → {"activity_score", "activity_50", "activity_200"}
file_to_nodes:       dict  = {}   # module_name → [node_id, ...]
filepath_to_module:  dict  = {}   # relative_file_path → module_name
_cochange_loaded_at: float = 0.0
_mg_to_cc:           dict  = {}   # MG dot-name → cochange ::key
_ownership_name_map: dict  = {}   # MG dot-name → ownership ::key
_cc_to_mg:           dict  = {}   # cochange ::key → MG dot-name
_filepath_suffix_idx: dict = {}   # normalized_basename → [(normalized_full_path, module)]
_stem_to_modules:    dict  = {}   # stem → [module_name, ...]

body_store:   dict = {}
call_graph:   dict = {}
log_patterns: dict = {}
doc_chunks:   list = []
doc_by_id:    dict = {}
gw_integrity: dict = {}

# BM25 index (built at startup from symbol names + module names)
_bm25:      object = None   # BM25Okapi instance
_bm25_ids:  list   = []     # parallel list of node IDs
_bm25_svcs: list   = []     # parallel list of service names
_bm25_data: list   = []     # parallel list of node dicts (for result construction)

# Service profiles from config (name → {role, traffic_weight, region})
SERVICE_PROFILES: dict = {}

# ── Retrieval tuning defaults (override via config.yaml) ────────────────────
KNOWN_SERVICES: list = [
    "euler-api-gateway", "euler-api-txns", "UCS", "euler-db",
    "euler-api-order", "graphh", "euler-api-pre-txn",
    "euler-api-customer", "basilisk-v3", "euler-drainer",
    "token_issuer_portal_backend", "haskell-sequelize",
]
_KW_ALLOWLIST: set = {"upi", "pix", "emi", "ucs", "cvv", "pan", "otp", "kyc", "bnpl", "nfc", "qr"}
_idf: dict = {}  # built during initialize(); word → IDF score

# Known payment gateway names — used to generate better query variants.
# We deliberately don't hardcode gateway→service here since routing varies per deployment.
_KNOWN_GATEWAYS: frozenset = frozenset({
    "payu", "razorpay", "stripe", "adyen", "paypal", "braintree", "checkout",
    "worldpay", "cybersource", "nuvei", "bluesnap", "rapyd", "iatapay",
    "itaubank", "bambora", "tsys", "shift4", "globepay", "helcim", "gocardless",
    "mollie", "multisafepay", "nexinets", "noon", "nmi", "paybox", "payme",
    "payone", "square", "stax", "trustpay", "airwallex", "authorizedotnet",
    "hdfc", "icici", "axis", "kotak", "yesbank",
})

# Path segments that indicate test/harness code — deprioritised in search results
_TEST_PATH_SEGMENTS: frozenset = frozenset({
    "test", "tests", "spec", "specs", "harness", "mock", "mocks",
    "scenario", "scenarios", "fixture", "fixtures", "ucs-connector-tests",
    "examples", "example",
    # UCS Hyperswitch connector-integration is Rust scaffolding not used in production
    # payment-related test files — deprioritise so core business logic surfaces first
    "connector-integration",
})

_L2_THRESHOLD = 1.18
_RERANK_BATCH = 64


# ════════════════════════════════════════════════════════════════════════════
# CONFIG LOADING  (makes the engine generic / deployment-agnostic)
# ════════════════════════════════════════════════════════════════════════════

def load_config(config_path: pathlib.Path | str) -> dict:
    """
    Load a config.yaml that overrides defaults.  Example config.yaml:

        llm:
          api_key: sk-...
          base_url: https://api.openai.com/v1
          model: gpt-4o

        embed:
          server_url: http://localhost:8001        # use embed_server.py
          instruction: "Represent this code: "    # custom instruction prefix

        services:
          - my-api-gateway
          - my-worker-service

        kw_allowlist: [api, rpc, grpc, sdk]

        personas:
          domain_expert:
            label: "🏛️ Domain Expert"
            system_prompt: "You are an expert in this codebase..."
            framework: "Trace the data flow end to end..."
    """
    global LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, EMBED_SERVER_URL, EMBED_INSTRUCTION
    global KNOWN_SERVICES, _KW_ALLOWLIST

    try:
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    except ImportError:
        # Fall back to json if PyYAML not installed
        with open(config_path) as f:
            cfg = json.load(f)
    except FileNotFoundError:
        return {}

    llm = cfg.get("llm", {})
    if llm.get("api_key"):
        LLM_API_KEY  = llm["api_key"]
    if llm.get("base_url"):
        LLM_BASE_URL = llm["base_url"]
    if llm.get("model"):
        LLM_MODEL    = llm["model"]

    embed = cfg.get("embed", {})
    if embed.get("server_url"):
        EMBED_SERVER_URL  = embed["server_url"]
    if embed.get("instruction"):
        EMBED_INSTRUCTION = embed["instruction"]

    if cfg.get("services"):
        KNOWN_SERVICES = cfg["services"]
    if cfg.get("kw_allowlist"):
        _KW_ALLOWLIST = set(cfg["kw_allowlist"])

    # Service profiles (new)
    if cfg.get("service_profiles"):
        SERVICE_PROFILES.update(cfg["service_profiles"])
        # Rebuild KNOWN_SERVICES order by traffic_weight descending
        KNOWN_SERVICES[:] = sorted(
            SERVICE_PROFILES.keys(),
            key=lambda s: SERVICE_PROFILES[s].get("traffic_weight", 0.5),
            reverse=True,
        )

    # LLM API key from config (workspace config is not in git — safe to store there)
    if cfg.get("llm_api_key") and not LLM_API_KEY:
        globals()["LLM_API_KEY"] = cfg["llm_api_key"]

    # Persona overrides: call tools.apply_persona_config(cfg) after this returns
    # if you want config-driven persona customisation (tools.py owns the persona dicts).

    return cfg


# ════════════════════════════════════════════════════════════════════════════
# PERSONA DICTS  (empty — populated by tools.py at import time)
# ════════════════════════════════════════════════════════════════════════════
# tools.py sets these when it is imported. If tools.py is not used (e.g. in a
# unit test or CLI that only needs retrieval primitives), they stay empty and
# callers that depend on them should import tools directly.

PERSONA_SYSTEM_PROMPTS: dict = {}
PERSONA_FRAMEWORKS:     dict = {}
PERSONA_LABELS:         dict = {}


# ════════════════════════════════════════════════════════════════════════════
# INITIALIZATION
# ════════════════════════════════════════════════════════════════════════════

def initialize(
    artifact_dir: pathlib.Path | None = None,
    load_embedder: bool = True,
    config_path: pathlib.Path | str | None = None,
) -> None:
    """
    Load all data stores. Idempotent — skips if already loaded at requested level.

    artifact_dir:  path to demo_artifact/ (contains graph_with_summaries.json, vectors.lance).
                   Defaults to the directory next to this file.
    load_embedder: False skips the ~6GB Qwen3 model → keyword-only search.
                   Ignored when EMBED_SERVER_URL is set (model runs as separate service).
    config_path:   optional path to config.yaml.
    """
    global embedder, _llm_client, G, MG, lance_tbl, cluster_summaries
    global body_store, call_graph, log_patterns, doc_chunks, doc_by_id, gw_integrity
    global doc_lance_tbl, _cochange_loaded_at, _idf

    if config_path:
        load_config(config_path)

    # When using embed server, we never load the model in-process
    using_embed_server = bool(EMBED_SERVER_URL)
    if using_embed_server:
        load_embedder = False

    if G is not None and (not load_embedder or embedder is not None or using_embed_server):
        return  # Already at desired level

    if artifact_dir is None:
        artifact_dir = _DEFAULT_ARTIFACT_DIR
    artifact_dir = pathlib.Path(artifact_dir)

    GRAPH_PATH     = str(artifact_dir / "graph_with_summaries.json")
    LANCE_PATH     = str(artifact_dir / "vectors.lance")
    EMBED_MODEL    = os.environ.get("EMBED_MODEL",
                                str(artifact_dir.parent / "models" / "qwen3-embed-8b"))
    BODY_STORE_P   = artifact_dir.parent / "output" / "body_store.json"
    CALL_GRAPH_P   = artifact_dir.parent / "output" / "call_graph.json"
    LOG_PATTERNS_P = artifact_dir.parent / "output" / "log_patterns.json"
    DOC_CHUNKS_P   = artifact_dir.parent / "output" / "doc_chunks.json"
    GW_INTEGRITY_P = artifact_dir.parent / "output" / "gateway_integrity_config.json"
    cochange_path  = artifact_dir / "cochange_index.json"

    import networkx as nx, lancedb

    if G is None:
        print("Loading graph...")
        with open(GRAPH_PATH) as _f:
            graph_data = json.load(_f)
        G = nx.node_link_graph(graph_data["networkx"])
        cluster_summaries.update(graph_data.get("cluster_summaries", {}))

        cluster_attrs = {
            n["id"]: {
                "cluster_name":    n.get("cluster_name", ""),
                "cluster_purpose": n.get("cluster_purpose", ""),
                "type":            n.get("type", ""),
                "ghost_deps":      n.get("ghost_deps", []),
                "file":            n.get("file", ""),
            }
            for n in graph_data["nodes"]
            if n.get("cluster_name") or n.get("type") or n.get("file")
        }
        nx.set_node_attributes(G, cluster_attrs)
        print(f"  {G.number_of_nodes():,} nodes  {G.number_of_edges():,} edges  {len(cluster_summaries)} summaries")

        # Build IDF index for keyword search
        _doc_freq: dict[str, int] = defaultdict(int)
        _n_nodes = 0
        for _nid, _nd in G.nodes(data=True):
            if _nd.get("kind") == "phantom":
                continue
            _n_nodes += 1
            _tokens = set(re.split(r"\W+", (_nd.get("name", "") + " " + _nd.get("module", "")).lower()))
            for _tok in _tokens:
                if _tok:
                    _doc_freq[_tok] += 1
        _idf = {w: math.log(_n_nodes / df) for w, df in _doc_freq.items()}
        print(f"  IDF index: {len(_idf):,} terms from {_n_nodes:,} nodes")

        print("Building module-level traversal graph...")
        _mg = nx.DiGraph()
        raw_edges  = graph_data.get("edges", [])
        mod_to_svc = {n.get("module", ""): n.get("service", "") for n in graph_data["nodes"]}
        for e in raw_edges:
            src, dst, kind = e.get("from",""), e.get("to",""), e.get("kind","")
            if not src or not dst or kind != "import":
                continue
            if src in mod_to_svc and dst in mod_to_svc:
                _mg.add_node(src, service=mod_to_svc.get(src,""))
                _mg.add_node(dst, service=mod_to_svc.get(dst,""))
                if _mg.has_edge(src, dst):
                    _mg[src][dst]["weight"] += 1
                else:
                    _mg.add_edge(src, dst, kind="import", weight=1)
        # Assign to module-level MG
        globals()["MG"] = _mg

        cs_edges = sum(1 for u,v in _mg.edges()
                       if _mg.nodes[u].get("service") != _mg.nodes[v].get("service"))
        print(f"  {_mg.number_of_nodes():,} modules  {_mg.number_of_edges():,} import edges  {cs_edges:,} cross-service")

        print("Loading vector index...")
        USE_QUANTIZED = os.environ.get("USE_QUANTIZED", "").lower() in ("1", "true", "yes")
        QUANT_PATH = str(artifact_dir / "vectors_quantized.npz")

        if USE_QUANTIZED and os.path.exists(QUANT_PATH):
            try:
                from quantized_loader import QuantizedSearchTable
                # Load metadata from LanceDB for field lookups, then use quantized vectors
                _meta_df = None
                try:
                    _meta_db = lancedb.connect(LANCE_PATH)
                    _meta_tbl = _meta_db.open_table("chunks")
                    _meta_df = _meta_tbl.to_pandas().drop(columns=["vector"], errors="ignore")
                    print(f"  Metadata loaded from LanceDB ({len(_meta_df):,} rows)")
                except Exception:
                    print("  WARNING: Could not load metadata from LanceDB — search results will lack field data")
                lance_tbl = QuantizedSearchTable(QUANT_PATH, metadata_df=_meta_df)
                print(f"  Using QUANTIZED vectors ({len(lance_tbl):,} @ {lance_tbl.dim}d, {lance_tbl.bits}-bit)")
            except Exception as e:
                lance_tbl = None
                print(f"  Quantized load failed ({e}) — falling back to keyword-only mode")
        else:
            try:
                db = lancedb.connect(LANCE_PATH)
                lance_tbl = db.open_table("chunks")
                print(f"  {len(lance_tbl):,} vectors @ 4096d")
            except (ValueError, FileNotFoundError, OSError) as e:
                lance_tbl = None
                print(f"  vectors.lance: not available ({e}) — keyword-only mode")

        try:
            doc_db = lancedb.connect(str(artifact_dir.parent / "output"))
            _tnames = doc_db.list_tables()
            _tnames = _tnames.tables if hasattr(_tnames, "tables") else list(_tnames)
            if "docs" in _tnames:
                doc_lance_tbl = doc_db.open_table("docs")
                print(f"  Doc vectors: {len(doc_lance_tbl):,} chunks @ 4096d")
            else:
                print("  docs.lance: not built yet — run 07_chunk_docs.py")
        except Exception as e:
            print(f"  docs.lance: {e}")

        print("Building module + filepath indexes...")
        for nid, d in G.nodes(data=True):
            mod = d.get("module", "")
            if mod:
                file_to_nodes.setdefault(mod, []).append(nid)
                # Build stem→modules index for fast stem lookups
                for sep in (".", "::"):
                    stem = mod.rsplit(sep, 1)[-1]
                    if stem:
                        _stem_to_modules.setdefault(stem, []).append(mod)
            f = d.get("file", "")
            m = d.get("module", "") or nid
            if f and m:
                filepath_to_module[f] = m
        # Build suffix index for fast file→module resolution
        for known_path, mod in filepath_to_module.items():
            norm = known_path.replace("\\", "/").lstrip("/")
            basename = norm.rsplit("/", 1)[-1]
            _filepath_suffix_idx.setdefault(basename, []).append((norm, mod))

        # v6 stores (all optional — degrade gracefully)
        for path, store, name in [
            (BODY_STORE_P,   body_store,   "body store"),
            (CALL_GRAPH_P,   call_graph,   "call graph"),
            (LOG_PATTERNS_P, log_patterns, "log patterns"),
        ]:
            if path.exists():
                print(f"Loading {name}...")
                store.update(json.loads(path.read_text()))
                print(f"  {len(store):,} entries")
            else:
                print(f"  {name}: not found — run 01_extract_v2.py")

        if DOC_CHUNKS_P.exists():
            print("Loading doc chunks...")
            doc_chunks.extend(json.loads(DOC_CHUNKS_P.read_text()))
            doc_by_id.update({c["id"]: c for c in doc_chunks})
            print(f"  {len(doc_chunks):,} chunks")

        if GW_INTEGRITY_P.exists():
            print("Loading gateway integrity config...")
            gw_integrity.update(json.loads(GW_INTEGRITY_P.read_text()))
            print(f"  {len(gw_integrity):,} gateway configs")

        if cochange_path.exists():
            print("Loading co-change index...")
            with open(str(cochange_path)) as _f:
                ci = json.load(_f)
            cochange_index.update(ci.get("edges", {}))
            _cochange_loaded_at = cochange_path.stat().st_mtime
            meta = ci.get("meta", {})
            print(f"  {meta.get('total_modules',0):,} modules  {meta.get('total_pairs',0):,} pairs")

            # Load cross-repo co-change index (additive — merges into same dict)
            cross_cochange_path = artifact_dir / "cross_cochange_index.json"
            if cross_cochange_path.exists():
                print("Loading cross-repo co-change index...")
                with open(str(cross_cochange_path)) as _f:
                    xci = json.load(_f)
                xedges = xci.get("edges", {})
                # Merge: for existing modules, append new cross-repo partners
                for mod, partners in xedges.items():
                    if mod in cochange_index:
                        existing = {p["module"] for p in cochange_index[mod]}
                        for p in partners:
                            if p["module"] not in existing:
                                cochange_index[mod].append(p)
                    else:
                        cochange_index[mod] = partners
                xmeta = xci.get("meta", {})
                print(f"  +{xmeta.get('total_modules',0):,} cross-repo modules  "
                      f"{xmeta.get('total_pairs',0):,} pairs  "
                      f"{xmeta.get('repo_pairs',0)} repo pairs")

            _build_cochange_name_map()
        else:
            print("  co-change index: not built yet")

        ownership_path = artifact_dir / "ownership_index.json"
        if ownership_path.exists():
            print("Loading ownership index...")
            with open(str(ownership_path)) as _f:
                oi = json.load(_f)
            ownership_index.update(oi.get("modules", {}))
            meta_oi = oi.get("meta", {})
            print(f"  {meta_oi.get('total_modules',0):,} modules  "
                  f"{meta_oi.get('total_unique_authors',0):,} authors")
            # Build MG→ownership name map (same strategy as cochange)
            _ownership_name_map.clear()
            for own_key in ownership_index:
                parts = own_key.split("::")
                mg_name = None
                for i, part in enumerate(parts):
                    if part and part[0].isupper():
                        mg_name = ".".join(parts[i:])
                        break
                if mg_name is None and len(parts) >= 2:
                    mg_name = ".".join(parts[1:])
                if mg_name and mg_name not in _ownership_name_map:
                    _ownership_name_map[mg_name] = own_key

        granger_path = artifact_dir / "granger_index.json"
        if granger_path.exists():
            print("Loading Granger causality index...")
            with open(str(granger_path)) as _f:
                gi = json.load(_f)
            granger_index.update(gi.get("causal_pairs", {}))
            meta_gi = gi.get("metadata", {})
            print(f"  {meta_gi.get('significant_results', 0):,} causal pairs  "
                  f"(p<{meta_gi.get('p_threshold', 0.05)})")

        community_path = artifact_dir / "community_index.json"
        if community_path.exists():
            print("Loading community index...")
            with open(str(community_path)) as _f:
                ci_data = json.load(_f)
            community_index.update(ci_data.get("communities", {}))
            module_to_community.update(ci_data.get("module_to_community", {}))
            meta_ci = ci_data.get("meta", {})
            print(f"  {meta_ci.get('n_communities', 0)} communities  "
                  f"{meta_ci.get('cross_service_communities', 0)} cross-service  "
                  f"modularity={meta_ci.get('modularity', 0)}")

        # Load activity index (from 10_build_activity.py)
        activity_path = artifact_dir / "activity_index.json"
        if activity_path.exists():
            print("Loading activity index...")
            with open(str(activity_path)) as _f:
                activity_index.update(json.load(_f))
            print(f"  {len(activity_index)} modules with activity data")

        # Inject synthetic co-change edges from call_graph (cold-start fix)
        if call_graph and cochange_index is not None:
            _inject_synthetic_cochange()

        # Build BM25 index from all symbol names + module names
        if _BM25_AVAILABLE:
            _build_bm25_index()

        # Hot-reload watcher for co-change (used when builder is still running)
        _cochange_stop = threading.Event()
        def _cochange_watcher():
            global _cochange_loaded_at
            while not _cochange_stop.is_set():
                _cochange_stop.wait(30)
                try:
                    mtime = cochange_path.stat().st_mtime
                    if mtime > _cochange_loaded_at and cochange_path.stat().st_size > 1000:
                        with open(str(cochange_path)) as _f:
                            ci = json.load(_f)
                        cochange_index.clear()
                        cochange_index.update(ci.get("edges", {}))
                        _cochange_loaded_at = mtime
                        _build_cochange_name_map()
                        meta = ci.get("meta", {})
                        print(f"[hot-reload] co-change: {meta.get('total_modules',0):,} modules")
                except Exception as _e:
                    print(f"[hot-reload] error: {_e}")
        threading.Thread(target=_cochange_watcher, daemon=True).start()

    # Embedder: load in-process only if not using embed server
    if load_embedder and embedder is None and not using_embed_server:
        import torch
        from sentence_transformers import SentenceTransformer
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Loading embedding model on {device}...")
        embedder = SentenceTransformer(
            EMBED_MODEL, device=device, trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.float16} if device == "cuda" else {},
        )
        print(f"  Loaded")
    elif using_embed_server:
        print(f"  Embed server: {EMBED_SERVER_URL} (no local GPU load)")
    else:
        print("  Embedder: skipped (keyword-only mode)")

    if _llm_client is None:
        from openai import OpenAI
        _llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    status = (
        f"embedder={'server' if using_embed_server else ('yes' if embedder else 'keyword-only')}, "
        f"bodies={'yes' if body_store else 'no'}, "
        f"cochange={'yes' if cochange_index else 'no'}"
    )
    print(f"✓ Engine ready ({status})")


def _build_cochange_name_map():
    """Build bidirectional mapping between MG dot-names and cochange ::keys.

    Cochange keys look like: ``repo::path::Module::Parts``
    MG nodes look like:      ``Module.Parts``

    Haskell module name segments start with uppercase; repo/path segments
    are lowercase (or contain hyphens/digits).  We extract the module name
    by finding the first uppercase segment in the cochange key.
    """
    _mg_to_cc.clear()
    _cc_to_mg.clear()
    mg_nodes = set(MG.nodes()) if MG is not None else set()
    for cc_key in cochange_index:
        parts = cc_key.split("::")
        mg_name = None
        # Strategy 1 (Haskell): find first uppercase segment → module name start
        for i, part in enumerate(parts):
            if part and part[0].isupper():
                mg_name = ".".join(parts[i:])
                break
        # Strategy 2 (Python/other): first segment is repo name, rest is module path
        if mg_name is None and len(parts) >= 2:
            mg_name = ".".join(parts[1:])
        if mg_name:
            _cc_to_mg[cc_key] = mg_name
            # Only map mg→cc if the MG node actually exists (avoid ambiguity)
            if mg_name in mg_nodes:
                _mg_to_cc[mg_name] = cc_key
            elif mg_name not in _mg_to_cc:
                # Store even without MG match — useful for traversal results
                _mg_to_cc[mg_name] = cc_key
    print(f"  Co-change name map: {len(_mg_to_cc):,} mg→cc, {len(_cc_to_mg):,} cc→mg")


# ════════════════════════════════════════════════════════════════════════════
# EMBEDDING  (local model OR embed server — transparent to callers)
# ════════════════════════════════════════════════════════════════════════════

def _encode_queries_batch(queries: list) -> list:
    """Encode multiple queries in one HTTP call to embed server, or locally."""
    if not queries:
        return []
    if EMBED_SERVER_URL:
        try:
            payload = json.dumps({"texts": queries, "instruction": EMBED_INSTRUCTION}).encode()
            req = urllib.request.Request(
                EMBED_SERVER_URL.rstrip("/") + "/embed",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
            return resp["embeddings"]
        except Exception as e:
            print(f"[embed_server] batch error: {e} — falling back to keyword search")
            return [[] for _ in queries]
    if embedder is not None:
        texts = [EMBED_INSTRUCTION + q for q in queries]
        vecs = embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return [v.tolist() for v in vecs]
    return [[] for _ in queries]


def _encode_query(query: str) -> list:
    """Encode a single query."""
    vecs = _encode_queries_batch([query])
    return vecs[0] if vecs else []


def can_embed() -> bool:
    """True when vector search is available (either embed server or local model)."""
    return bool(EMBED_SERVER_URL) or embedder is not None


# ════════════════════════════════════════════════════════════════════════════
# RETRIEVAL
# ════════════════════════════════════════════════════════════════════════════

def _is_test_path(fpath: str) -> bool:
    """Return True if the file lives in a test/harness directory."""
    parts = set(pathlib.PurePosixPath(fpath.replace("\\", "/")).parts)
    return bool(parts & _TEST_PATH_SEGMENTS)


def stratified_vector_search(queries: list, k_total: int = 250,
                              service_weights: dict = None) -> dict:
    """
    Multi-query vector search with dynamic per-service budget allocation.
    Returns {} when embedding is unavailable (keyword-only mode).
    """
    if not can_embed() or lance_tbl is None:
        return {}

    # Encode all queries in ONE HTTP call, then search LanceDB per vector
    vecs = _encode_queries_batch(queries)
    all_hits: dict = {}
    for qvec in vecs:
        if not qvec:
            continue
        for h in lance_tbl.search(qvec).limit(k_total).to_list():
            nid = h.get("id") or h.get("name", "")
            if nid and (nid not in all_hits or
                        h.get("_distance", 1) < all_hits[nid].get("_distance", 1)):
                all_hits[nid] = h

    # Down-weight test/harness paths so production code surfaces first
    for h in all_hits.values():
        if _is_test_path(h.get("file", "")):
            h["_distance"] = h.get("_distance", 1.0) * 1.5

    if service_weights:
        total_w = sum(service_weights.values()) or 1.0
        budgets = {svc: max(3, int(k_total * w / total_w)) for svc, w in service_weights.items()}
        default_budget = max(8, k_total // max(len(KNOWN_SERVICES), 1))
    else:
        budgets = {}
        default_budget = 12

    by_service: dict = defaultdict(list)
    for h in sorted(all_hits.values(), key=lambda x: x.get("_distance", 1.0)):
        svc = h.get("service", "unknown")
        cap = budgets.get(svc, default_budget)
        if len(by_service[svc]) < cap:
            by_service[svc].append(h)

    return dict(by_service)


def cross_service_keyword_search(query: str, max_per_service: int = 15) -> dict:
    if G is None:
        return {}
    words = [
        w.lower() for w in re.split(r"\W+", query)
        if len(w) >= 4 or w.lower() in _KW_ALLOWLIST
    ]
    if not words:
        return {}
    # IDF for unseen words: maximally discriminating
    _max_idf = math.log(G.number_of_nodes()) if G.number_of_nodes() > 0 else 1.0
    prod_results: dict = defaultdict(list)
    test_results: dict = defaultdict(list)
    seen: set = set()
    for nid, d in G.nodes(data=True):
        if d.get("kind") == "phantom":
            continue
        haystack = (d.get("name", "") + " " + d.get("module", "")).lower()
        score = sum(_idf.get(w, _max_idf) for w in words if w in haystack)
        if score > 0 and nid not in seen:
            svc = d.get("service", "unknown")
            seen.add(nid)
            node = {**d, "id": nid, "_kw_score": score}
            bucket = test_results if _is_test_path(d.get("file", "")) else prod_results
            bucket[svc].append(node)

    # Sort by match count descending — nodes matching more query words rank first
    for bucket in (prod_results, test_results):
        for svc in bucket:
            bucket[svc].sort(key=lambda n: (-n.get("_kw_score", 0), len(n.get("name", "") + n.get("module", ""))))

    # Merge: fill each service slot with production results first, test code as overflow
    results: dict = defaultdict(list)
    all_svcs = set(prod_results) | set(test_results)
    for svc in all_svcs:
        combined = prod_results[svc][:max_per_service]
        remaining = max_per_service - len(combined)
        if remaining > 0:
            combined += test_results[svc][:remaining]
        results[svc] = combined
    return dict(results)


def doc_vector_search(query_vec: list, top_k: int = 20) -> list:
    if doc_lance_tbl is None or not query_vec:
        return []
    try:
        return doc_lance_tbl.search(query_vec).limit(top_k).to_list()
    except Exception as e:
        print(f"[doc_vector_search] error: {e}")
        return []


def module_graph_expand(seed_modules: list, depth: int = 2) -> dict:
    if MG is None or not seed_modules:
        return {}
    visited = {}
    queue = [(m, 0) for m in seed_modules if m in MG]
    for m in seed_modules:
        if m in MG:
            visited[m] = {"service": MG.nodes[m].get("service",""), "hop": 0, "direction": "seed"}
    while queue:
        current, hop = queue.pop(0)
        if hop >= depth:
            continue
        for nb in list(MG.successors(current)) + list(MG.predecessors(current)):
            if nb not in visited:
                direction = "imports" if nb in MG.successors(current) else "imported_by"
                visited[nb] = {"service": MG.nodes[nb].get("service",""), "hop": hop+1, "direction": direction}
                queue.append((nb, hop+1))
    return {m: d for m, d in visited.items() if d["hop"] > 0}


def _resolve_cc(name: str) -> str:
    """Resolve a module name to its cochange_index key.
    Accepts both MG dot-format and native cochange ::format."""
    if name in cochange_index:
        return name
    return _mg_to_cc.get(name, name)


def _resolve_mg(name: str) -> str:
    """Resolve a cochange key back to MG dot-format."""
    if "." in name and "::" not in name:
        return name  # already dot-format
    return _cc_to_mg.get(name, name)


def cochange_path_traverse(seed_modules: list, max_hops: int = 4,
                            top_k: int = 15, min_weight: int = None) -> list:
    if not cochange_index:
        return []
    # Auto-scale min_weight: small indexes need lower thresholds
    if min_weight is None:
        total_modules = len(cochange_index)
        if total_modules < 50:
            min_weight = 2
        elif total_modules < 500:
            min_weight = 3
        else:
            min_weight = 5
    visited = {}
    for m in seed_modules:
        cc_key = _resolve_cc(m)
        if cc_key in cochange_index:
            visited[cc_key] = (999, 0)
    queue = [(k, 999, 0) for k in visited]
    while queue:
        current, _, hop = queue.pop(0)
        if hop >= max_hops:
            continue
        for p in cochange_index.get(current, [])[:top_k]:
            pm, pw = p["module"], p["weight"]
            if pw < min_weight:
                break
            # Normalize neighbor key for consistent lookup
            cc_pm = _resolve_cc(pm)
            if cc_pm not in visited:
                visited[cc_pm] = (pw, hop+1)
                queue.append((cc_pm, pw, hop+1))
    # Return results in MG dot-format for downstream compatibility
    result = [{"module": _resolve_mg(m), "weight": w, "hop": h}
              for m, (w, h) in visited.items() if h > 0]
    result.sort(key=lambda x: (x["hop"], -x["weight"]))
    return result


def get_entry_points(query_words: list) -> list:
    if G is None:
        return []
    entry_keywords = {"route","server","api","handler","endpoint","app","main","controller","middleware","router"}
    results = []
    for nid, d in G.nodes(data=True):
        if d.get("kind") == "phantom":
            continue
        mod  = d.get("module","").lower()
        name = d.get("name","").lower()
        if (any(k in mod or k in name for k in entry_keywords)
                and any(w in name or w in mod for w in query_words)):
            results.append({**d, "id": nid, "_entry": True})
            if len(results) >= 8:
                break
    return results


def get_cluster_context_for_services(services: list) -> dict:
    if G is None:
        return {}
    svc_clusters: dict = defaultdict(set)
    for nid, d in G.nodes(data=True):
        svc = d.get("service","")
        if svc in services:
            cid = str(d.get("cluster",-1))
            if cid != "-1" and cid in cluster_summaries:
                svc_clusters[svc].add(cid)
    return {
        svc: [{"cluster_id": cid, **cluster_summaries[cid]} for cid in cids]
        for svc, cids in svc_clusters.items()
    }


# ════════════════════════════════════════════════════════════════════════════
# FILE → MODULE RESOLUTION  (for PR blast-radius analysis)
# ════════════════════════════════════════════════════════════════════════════

def resolve_files_to_modules(file_paths: list) -> dict:
    """
    Map git-diff file paths to module names.

    Tries in order:
    1. Exact suffix match against filepath_to_module keys
    2. Partial suffix match (shrinking window of path components)
    3. Stem match against MG node names (basename without extension)

    Returns {file_path: [module_name, ...]}. Unmatched → [].
    """
    result = {}
    for fp in file_paths:
        fp_norm = fp.replace("\\", "/").lstrip("/")
        found = []

        # 1 + 2: suffix match via pre-built index (O(candidates) not O(all_paths))
        basename = fp_norm.rsplit("/", 1)[-1]
        candidates = _filepath_suffix_idx.get(basename, [])
        if candidates:
            fp_parts = fp_norm.split("/")
            for n in range(len(fp_parts), 0, -1):
                suffix = "/".join(fp_parts[-n:])
                for known_norm, mod in candidates:
                    if known_norm == suffix or known_norm.endswith("/" + suffix) or suffix.endswith("/" + known_norm):
                        if mod not in found:
                            found.append(mod)
                if found:
                    break

        # 3: stem match via pre-built index, fallback to MG scan if index empty
        if not found:
            stem = pathlib.Path(fp_norm).stem
            stem_candidates = _stem_to_modules.get(stem, [])
            if stem_candidates:
                for mod in stem_candidates[:3]:
                    if mod not in found:
                        found.append(mod)
            elif MG is not None:
                for mod in MG.nodes():
                    if mod.split(".")[-1] == stem or mod.split("::")[-1] == stem:
                        if mod not in found:
                            found.append(mod)
                        if len(found) >= 3:
                            break

        # 4: direct path→module conversion (zero-config fallback)
        # Convert file path to dot-notation: serve/retrieval_engine.py → serve.retrieval_engine
        if not found:
            p = pathlib.PurePosixPath(fp_norm)
            if p.suffix in (".py", ".hs", ".rs", ".js", ".ts", ".tsx", ".jsx", ".go", ".java", ".groovy"):
                dot_mod = str(p.with_suffix("")).replace("/", ".")
                found.append(dot_mod)

        result[fp] = found
    return result


# ════════════════════════════════════════════════════════════════════════════
# BLAST RADIUS  (composite: import graph + co-change)
# ════════════════════════════════════════════════════════════════════════════

def get_blast_radius(module_names: list, max_hops: int = 2) -> dict:
    """
    Returns:
    {
      "seed_modules":       [str, ...],
      "import_neighbors":   [{"module", "service", "hop", "direction"}, ...],
      "cochange_neighbors": [{"module", "weight", "hop"}, ...],
      "affected_services":  [str, ...],
    }
    """
    seed = [m for m in module_names if m]
    affected_services: set = set()
    import_neighbors = []
    cochange_neighbors = []

    if MG is not None:
        for m in seed:
            if m in MG.nodes:
                svc = MG.nodes[m].get("service","")
                if svc:
                    affected_services.add(svc)
        for mod, info in module_graph_expand(seed, depth=max_hops).items():
            import_neighbors.append({
                "module": mod, "service": info.get("service",""),
                "hop": info["hop"], "direction": info["direction"],
            })
            if info.get("service"):
                affected_services.add(info["service"])

    if cochange_index:
        cochange_neighbors = cochange_path_traverse(seed, max_hops=max_hops)
        for item in cochange_neighbors:
            if MG is not None and item["module"] in MG.nodes:
                svc = MG.nodes[item["module"]].get("service","")
                if svc:
                    affected_services.add(svc)

    import_neighbors.sort(key=lambda x: (x["hop"], x["module"]))

    # ── Tiered impact analysis ──────────────────────────────────────────
    # Combines import + co-change + Granger into unified scored tiers:
    #   will_break  — direct structural dependency (hop 1)
    #   may_break   — transitive dependency OR strong Granger causal signal
    #   review      — co-change only (no static link) or weak signals
    tiered: dict = {}  # module → {tier, confidence, signals}
    import_set = set()

    for nb in import_neighbors:
        mod = nb["module"]
        import_set.add(mod)
        hop = nb["hop"]
        static_score = 1.0 / hop  # 1.0 for hop 1, 0.5 for hop 2, etc.

        # Check for Granger causal evidence
        granger_score = 0.0
        granger_info = None
        if granger_index:
            for s in seed:
                cc_s = _resolve_cc(s)
                cc_t = _resolve_cc(mod)
                for key in (f"{cc_s}→{cc_t}", f"{cc_t}→{cc_s}"):
                    if key in granger_index:
                        g = granger_index[key]
                        gs = 1.0 - min(g["p_value"] * 20, 1.0)  # p=0 → 1.0, p=0.05 → 0.0
                        if gs > granger_score:
                            granger_score = gs
                            granger_info = {"lag": g["best_lag"], "p_value": g["p_value"]}

        # Activity boost: recently changed modules are more likely to be affected
        act_score = activity_index.get(mod, {}).get("activity_score", 0.0)

        confidence = round(0.4 * static_score + 0.25 * granger_score + 0.2 * act_score + 0.15, 3)
        tier = "will_break" if hop == 1 else "may_break"

        tiered[mod] = {
            "module": mod, "tier": tier, "confidence": confidence,
            "service": nb.get("service", ""),
            "signals": {"static_hop": hop, "direction": nb["direction"]},
        }
        if act_score > 0:
            tiered[mod]["signals"]["activity_score"] = round(act_score, 3)
        if granger_info:
            tiered[mod]["signals"]["granger"] = granger_info

    for nb in cochange_neighbors:
        mod = nb["module"]
        cc_weight = nb.get("weight", 0)
        max_w = max((n.get("weight", 1) for n in cochange_neighbors), default=1)
        cc_score = min(cc_weight / max(max_w, 1), 1.0)

        granger_score = 0.0
        granger_info = None
        if granger_index:
            for s in seed:
                cc_s = _resolve_cc(s)
                cc_t = _resolve_cc(mod)
                for key in (f"{cc_s}→{cc_t}", f"{cc_t}→{cc_s}"):
                    if key in granger_index:
                        g = granger_index[key]
                        gs = 1.0 - min(g["p_value"] * 20, 1.0)
                        if gs > granger_score:
                            granger_score = gs
                            granger_info = {"lag": g["best_lag"], "p_value": g["p_value"]}

        if mod in tiered:
            # Already in import graph — boost confidence with co-change + activity evidence
            existing = tiered[mod]
            act_score = activity_index.get(mod, {}).get("activity_score", 0.0)
            cc_boost = 0.12 * cc_score + 0.08 * granger_score + 0.1 * act_score
            existing["confidence"] = round(min(1.0, existing["confidence"] + cc_boost), 3)
            existing["signals"]["cochange_weight"] = cc_weight
            if act_score > 0:
                existing["signals"]["activity_score"] = round(act_score, 3)
            if granger_info and "granger" not in existing["signals"]:
                existing["signals"]["granger"] = granger_info
        else:
            # Co-change only — no static dependency
            act_score = activity_index.get(mod, {}).get("activity_score", 0.0)
            confidence = round(0.3 * cc_score + 0.3 * granger_score + 0.3 * act_score + 0.1, 3)
            tier = "may_break" if granger_score > 0.5 or act_score > 0.5 else "review"
            svc = ""
            if MG is not None and mod in MG.nodes:
                svc = MG.nodes[mod].get("service", "")
            tiered[mod] = {
                "module": mod, "tier": tier, "confidence": confidence,
                "service": svc,
                "signals": {"cochange_weight": cc_weight},
            }
            if act_score > 0:
                tiered[mod]["signals"]["activity_score"] = round(act_score, 3)
            if granger_info:
                tiered[mod]["signals"]["granger"] = granger_info

    tiered_list = sorted(tiered.values(), key=lambda x: -x["confidence"])
    # ── End tiered impact ───────────────────────────────────────────────

    # ── Community context (if available) ────────────────────────────────
    community_context = None
    if module_to_community:
        seed_communities = set()
        for s in seed:
            cc_key = _resolve_cc(s)
            comm = module_to_community.get(cc_key)
            if comm is not None:
                seed_communities.add(str(comm))
        if seed_communities:
            affected_communities = set(seed_communities)
            for item in tiered_list[:20]:  # check top impacted modules
                cc_key = _resolve_cc(item["module"])
                comm = module_to_community.get(cc_key)
                if comm is not None:
                    affected_communities.add(str(comm))
            community_context = {
                "seed_communities": sorted(seed_communities),
                "affected_communities": sorted(affected_communities),
                "cross_community": len(affected_communities) > len(seed_communities),
                "details": {
                    cid: {
                        "label": community_index.get(cid, {}).get("label", ""),
                        "size": community_index.get(cid, {}).get("size", 0),
                        "cross_service": community_index.get(cid, {}).get("cross_service", False),
                    }
                    for cid in affected_communities if cid in community_index
                },
            }

    result = {
        "seed_modules":       seed,
        "import_neighbors":   import_neighbors,
        "cochange_neighbors": cochange_neighbors,
        "affected_services":  sorted(affected_services),
        "tiered_impact":      tiered_list,
    }
    if community_context:
        result["community_context"] = community_context
    return result


def predict_missing_changes(changed_modules: list, min_weight: int = 5,
                             top_k: int = 20) -> dict:
    """Predict modules likely missing from a changeset based on co-change history.

    Given a set of changed modules (e.g. from a PR), finds co-change neighbors
    that are NOT in the changeset but historically change together with the
    changed modules. Higher confidence = more often changed together.

    Returns:
    {
      "changed": [str, ...],
      "predictions": [{"module", "reason", "weight", "confidence", "service"}, ...],
      "coverage_score": float  (0-1, how well the changeset covers expected changes)
    }
    """
    changed_set = set(m for m in changed_modules if m)
    if not changed_set or not cochange_index:
        return {"changed": list(changed_set), "predictions": [], "coverage_score": 1.0}

    # Gather co-change evidence for each changed module
    candidate_evidence: dict = {}  # module → {"total_weight", "sources", "max_single"}

    for mod in changed_set:
        cc_key = _resolve_cc(mod)
        neighbors = cochange_index.get(cc_key, [])
        for nb in neighbors:
            nb_mod = _resolve_mg(nb["module"])
            w = nb["weight"]
            if w < min_weight or nb_mod in changed_set:
                continue
            if nb_mod not in candidate_evidence:
                candidate_evidence[nb_mod] = {
                    "total_weight": 0, "max_single": 0, "sources": []
                }
            candidate_evidence[nb_mod]["total_weight"] += w
            candidate_evidence[nb_mod]["max_single"] = max(
                candidate_evidence[nb_mod]["max_single"], w)
            candidate_evidence[nb_mod]["sources"].append(
                {"from": mod, "weight": w})

    # Score and rank predictions
    predictions = []
    for mod, ev in candidate_evidence.items():
        # Confidence: how many changed modules point to this candidate
        source_count = len(ev["sources"])
        # Normalize confidence: multiple sources + high weight = high confidence
        confidence = min(1.0, (ev["total_weight"] / 50) * (source_count / len(changed_set)))

        # Granger causality boost: if any changed module causally predicts this candidate
        causal_info = None
        if granger_index:
            best_causal = None
            for src in ev["sources"]:
                cc_src = _resolve_cc(src["from"])
                cc_tgt = _resolve_cc(mod)
                key = f"{cc_src}→{cc_tgt}"
                if key in granger_index:
                    g = granger_index[key]
                    if best_causal is None or g["p_value"] < best_causal["p_value"]:
                        best_causal = g
            if best_causal:
                causal_info = {
                    "direction": f"{best_causal['source'].split('::')[-1]}→{best_causal['target'].split('::')[-1]}",
                    "lag": best_causal["best_lag"],
                    "p_value": best_causal["p_value"],
                    "strength": "strong" if best_causal["p_value"] < 0.01 else "moderate",
                }
                # Boost confidence for causal relationships
                if best_causal["p_value"] < 0.01:
                    confidence = min(1.0, confidence * 1.3)  # 30% boost for strong causal
                else:
                    confidence = min(1.0, confidence * 1.15)  # 15% boost for moderate

        svc = ""
        if MG is not None and mod in MG.nodes:
            svc = MG.nodes[mod].get("service", "")

        # Build human-readable reason
        top_source = max(ev["sources"], key=lambda s: s["weight"])
        if source_count == 1:
            reason = f"co-changes with {top_source['from']} (w={top_source['weight']})"
        else:
            reason = (f"co-changes with {source_count} changed modules "
                      f"(strongest: {top_source['from']}, w={top_source['weight']})")
        if causal_info:
            reason += f" [causal: {causal_info['direction']}, lag={causal_info['lag']}]"

        pred = {
            "module": mod,
            "reason": reason,
            "weight": ev["total_weight"],
            "confidence": round(confidence, 3),
            "service": svc,
            "source_count": source_count,
        }
        if causal_info:
            pred["causal"] = causal_info

        predictions.append(pred)

    predictions.sort(key=lambda x: (-x["confidence"], -x["weight"]))
    predictions = predictions[:top_k]

    # Coverage score: what fraction of expected changes are actually in the changeset
    total_expected = len(changed_set) + len([p for p in predictions if p["confidence"] > 0.3])
    coverage = len(changed_set) / total_expected if total_expected > 0 else 1.0

    return {
        "changed": sorted(changed_set),
        "predictions": predictions,
        "coverage_score": round(coverage, 3),
    }


def suggest_reviewers(changed_modules: list, top_k: int = 5) -> dict:
    """Suggest PR reviewers based on module ownership from git history.

    For each changed module + its blast radius neighbors, looks up who has
    changed those modules most frequently. Returns ranked reviewers with
    context about which modules they own.

    Returns:
    {
      "reviewers": [{"email", "name", "score", "modules": [str], "commits"}, ...],
      "coverage": {"module": [top_author_email, ...], ...},
      "source": "ownership_index" | "unavailable"
    }
    """
    if not ownership_index:
        return {"reviewers": [], "coverage": {}, "source": "unavailable"}

    changed_set = set(m for m in changed_modules if m)
    if not changed_set:
        return {"reviewers": [], "coverage": {}, "source": "ownership_index"}

    # Get blast radius to include affected neighbors
    blast = get_blast_radius(list(changed_set), max_hops=1)
    all_modules = set(changed_set)
    for n in blast.get("import_neighbors", []):
        all_modules.add(n["module"])
    for n in blast.get("cochange_neighbors", []):
        all_modules.add(n["module"])

    # Aggregate author scores across all affected modules
    author_scores = {}  # email -> {"score", "modules", "commits", "name"}

    for mod in all_modules:
        # Try direct lookup, then ownership name map, then last segment, then cochange key
        own_key = _ownership_name_map.get(mod, "")
        last_seg = mod.rsplit(".", 1)[-1] if "." in mod else mod
        own_key_last = _ownership_name_map.get(last_seg, "")
        cc_key = _resolve_cc(mod)
        authors = (ownership_index.get(mod) or ownership_index.get(own_key)
                   or ownership_index.get(own_key_last)
                   or ownership_index.get(cc_key) or [])

        for author in authors:
            email = author["email"]
            # Weight: direct changes scored higher than blast radius neighbors
            weight = 2.0 if mod in changed_set else 1.0
            score = author["commits"] * weight

            if email not in author_scores:
                author_scores[email] = {
                    "email": email,
                    "name": author["name"],
                    "score": 0,
                    "commits": 0,
                    "modules": [],
                }
            author_scores[email]["score"] += score
            author_scores[email]["commits"] += author["commits"]
            if mod not in author_scores[email]["modules"]:
                author_scores[email]["modules"].append(mod)

    # Rank and return top-k reviewers
    reviewers = sorted(author_scores.values(), key=lambda x: -x["score"])[:top_k]

    # Per-module coverage: who's the top author for each changed module
    coverage = {}
    for mod in changed_set:
        cc_key = _resolve_cc(mod)
        authors = ownership_index.get(mod) or ownership_index.get(cc_key) or []
        coverage[mod] = [a["email"] for a in authors[:3]]

    return {
        "reviewers": reviewers,
        "coverage": coverage,
        "source": "ownership_index",
    }


# ════════════════════════════════════════════════════════════════════════════
# Change Risk Scoring  (composite risk score from all signals)
# ════════════════════════════════════════════════════════════════════════════

def score_change_risk(modules: list, rules: dict | None = None) -> dict:
    """Compute a composite risk score (0-100) for a set of changed modules.

    Combines blast radius, coverage gap, reviewer concentration, and service
    spread into a single actionable number with per-component breakdown.

    Args:
        modules: list of changed module names (any format accepted)
        rules:   optional dict with 'risk_weights' key to override defaults

    Returns:
        dict with risk_score (0-100), risk_level, components, recommendation
    """
    changed = [m for m in modules if m]
    if not changed:
        return {
            "risk_score": 0, "risk_level": "LOW",
            "components": {}, "recommendation": "No modules provided.",
        }

    # ── Configurable weights ──
    defaults = {"blast_radius": 0.35, "coverage_gap": 0.30,
                "reviewer_risk": 0.20, "service_spread": 0.15}
    weights = defaults.copy()
    if rules and "risk_weights" in rules:
        for k, v in rules["risk_weights"].items():
            if k in weights:
                weights[k] = float(v)
        # Re-normalize so weights sum to 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

    # ── 1. Blast Radius Score ──
    blast = get_blast_radius(changed, max_hops=2)
    n_services = len(blast.get("affected_services", []))
    n_cochange = len(blast.get("cochange_neighbors", []))
    n_import   = len(blast.get("import_neighbors", []))

    if n_services == 0:
        blast_score = 0
    elif n_services <= 2:
        blast_score = 20
    elif n_services <= 5:
        blast_score = 50
    elif n_services <= 10:
        blast_score = 75
    else:
        blast_score = 100

    blast_component = {
        "score": blast_score,
        "services_affected": n_services,
        "import_neighbors": n_import,
        "cochange_neighbors": n_cochange,
        "detail": f"{n_services} services in blast radius",
    }

    # ── 2. Coverage Gap Score ──
    pmc = predict_missing_changes(changed)
    coverage_score = pmc.get("coverage_score", 1.0)
    n_missing = len(pmc.get("predictions", []))
    gap_score = round((1.0 - coverage_score) * 100)

    coverage_component = {
        "score": gap_score,
        "missing_changes": n_missing,
        "coverage_score": coverage_score,
        "detail": f"{round((1 - coverage_score) * 100)}% of typical co-changes not included",
    }

    # ── 3. Reviewer Risk Score ──
    rev = suggest_reviewers(changed, top_k=5)
    reviewers_list = rev.get("reviewers", [])

    if rev.get("source") == "unavailable" or not reviewers_list:
        reviewer_score = 50  # unknown = moderate risk
        reviewer_detail = "No ownership data available"
        top_dominance = None
    else:
        total_score = sum(r["score"] for r in reviewers_list)
        top_dominance = reviewers_list[0]["score"] / total_score if total_score > 0 else 0
        if top_dominance > 0.8:
            reviewer_score = 100  # bus factor = 1
        elif top_dominance > 0.6:
            reviewer_score = 70
        elif top_dominance > 0.4:
            reviewer_score = 40
        elif len(reviewers_list) >= 3:
            reviewer_score = 20
        else:
            reviewer_score = 35
        reviewer_detail = (f"Top reviewer has {round(top_dominance * 100)}% of expertise"
                          if top_dominance else "Distributed ownership")

    reviewer_component = {
        "score": reviewer_score,
        "top_reviewer_dominance": round(top_dominance, 2) if top_dominance is not None else None,
        "available_reviewers": len(reviewers_list),
        "detail": reviewer_detail,
    }

    # ── 4. Service Spread Score ──
    unique_services = set()
    for mod in changed:
        # Extract service from module path if possible
        cc = _resolve_cc(mod)
        if cc and "::" in cc:
            unique_services.add(cc.split("::")[0])
    # Also count from blast radius
    for svc in blast.get("affected_services", []):
        unique_services.add(svc)

    n_unique = len(unique_services)
    if n_unique <= 1:
        spread_score = 0
    elif n_unique == 2:
        spread_score = 30
    elif n_unique == 3:
        spread_score = 50
    elif n_unique <= 5:
        spread_score = 80
    else:
        spread_score = 100

    spread_component = {
        "score": spread_score,
        "unique_services": n_unique,
        "services": sorted(unique_services),
        "detail": f"Changes span {n_unique} service{'s' if n_unique != 1 else ''}",
    }

    # ── Composite Score ──
    composite = round(
        blast_score * weights["blast_radius"]
        + gap_score * weights["coverage_gap"]
        + reviewer_score * weights["reviewer_risk"]
        + spread_score * weights["service_spread"]
    )

    if composite <= 30:
        level = "LOW"
    elif composite <= 60:
        level = "MEDIUM"
    elif composite <= 80:
        level = "HIGH"
    else:
        level = "CRITICAL"

    # ── Recommendation ──
    parts = []
    if n_services > 3:
        parts.append(f"{n_services} services affected")
    if n_missing > 5:
        parts.append(f"{n_missing} predicted co-changes missing")
    if reviewer_score >= 70 and reviewers_list:
        parts.append(f"high reviewer concentration on {reviewers_list[0]['name']}")
    if n_unique > 3:
        parts.append(f"spans {n_unique} services")

    if not parts:
        if level == "LOW":
            recommendation = "Low risk change. Proceed normally."
        else:
            recommendation = f"{level} risk. Review component scores for details."
    else:
        recommendation = f"{level} risk. {'; '.join(parts)}."
        if level in ("HIGH", "CRITICAL") and reviewers_list:
            top_names = [r["name"] for r in reviewers_list[:3]]
            recommendation += f" Suggested reviewers: {', '.join(top_names)}."

    return {
        "risk_score": composite,
        "risk_level": level,
        "components": {
            "blast_radius": blast_component,
            "coverage_gap": coverage_component,
            "reviewer_risk": reviewer_component,
            "service_spread": spread_component,
        },
        "recommendation": recommendation,
    }


# ════════════════════════════════════════════════════════════════════════════
# BM25  (exact-match complement to dense vector search)
# ════════════════════════════════════════════════════════════════════════════

def _tokenize_for_bm25(text: str) -> list:
    return [t for t in re.split(r'[^a-zA-Z0-9]+', text.lower()) if len(t) >= 2]


def _build_bm25_index():
    """Build BM25 index from all graph nodes. Called once during initialize()."""
    global _bm25, _bm25_ids, _bm25_svcs, _bm25_data
    if G is None or not _BM25_AVAILABLE:
        return
    print("Building BM25 index...")
    corpus, ids, svcs, data = [], [], [], []
    for nid, d in G.nodes(data=True):
        if d.get("kind") == "phantom":
            continue
        text = " ".join([
            d.get("name", ""),
            d.get("module", ""),
            d.get("type", "")[:100],
            d.get("cluster_name", ""),
        ])
        tokens = _tokenize_for_bm25(text)
        if not tokens:
            continue
        corpus.append(tokens)
        ids.append(nid)
        svcs.append(d.get("service", "unknown"))
        data.append({**d, "id": nid})
    if corpus:
        _bm25      = _BM25Okapi(corpus)
        _bm25_ids  = ids
        _bm25_svcs = svcs
        _bm25_data = data
    print(f"  BM25 index: {len(ids):,} symbols")


def bm25_search(query: str, top_k: int = 60) -> dict:
    """BM25 search. Returns {service: [node_dict, ...]}."""
    if _bm25 is None:
        return {}
    tokens = _tokenize_for_bm25(query)
    if not tokens:
        return {}
    scores = _bm25.get_scores(tokens)
    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
    results: dict = defaultdict(list)
    for idx in ranked:
        if scores[idx] <= 0:
            break
        node = {**_bm25_data[idx], "_bm25_score": float(scores[idx])}
        if not _is_test_path(node.get("file", "")):
            results[_bm25_svcs[idx]].append(node)
    return dict(results)


# ════════════════════════════════════════════════════════════════════════════
# RRF FUSION
# ════════════════════════════════════════════════════════════════════════════

def rrf_merge(*result_dicts, k: int = 60) -> dict:
    """
    Reciprocal Rank Fusion: RRF(d) = Σ 1/(k + rank_i(d)).
    Accepts any number of {service: [node, ...]} dicts.
    Returns merged {service: [node, ...]} sorted by RRF score.
    """
    rrf_scores: dict = defaultdict(float)
    node_by_id: dict = {}

    for result_dict in result_dicts:
        flat = [n for nodes in result_dict.values() for n in nodes]
        if not flat:
            continue
        if "_distance" in flat[0]:
            flat.sort(key=lambda x: x.get("_distance", 1.0))
        elif "_bm25_score" in flat[0]:
            flat.sort(key=lambda x: -x.get("_bm25_score", 0))
        elif "_cochange_score" in flat[0]:
            flat.sort(key=lambda x: -x.get("_cochange_score", 0))
        elif "_rrf_score" in flat[0]:
            flat.sort(key=lambda x: -x.get("_rrf_score", 0))
        else:
            flat.sort(key=lambda x: -x.get("_kw_score", 0))

        for rank, node in enumerate(flat, start=1):
            nid = node.get("id") or node.get("name", "")
            if not nid:
                continue
            rrf_scores[nid] += 1.0 / (k + rank)
            if nid not in node_by_id:
                node_by_id[nid] = node

    merged: dict = defaultdict(list)
    for nid, score in sorted(rrf_scores.items(), key=lambda x: -x[1]):
        node = {**node_by_id[nid], "_rrf_score": score}
        merged[node.get("service", "unknown")].append(node)
    return dict(merged)


# ════════════════════════════════════════════════════════════════════════════
# SYNTHETIC CO-CHANGE  (cold-start fix for new modules)
# ════════════════════════════════════════════════════════════════════════════

def _inject_synthetic_cochange():
    """
    Augment cochange_index with pairs derived from call_graph for modules
    that have zero observed co-change history (cold-start fix).
    Synthetic weight = 0.3 — well below real evidence (typically 3–50).
    """
    if not call_graph or MG is None:
        return
    SYNTHETIC_WEIGHT = 0.3
    # Build short-name → module map from MG
    shortname_to_mod: dict = defaultdict(list)
    for mn in MG.nodes():
        short = mn.split(".")[-1].split("::")[-1]
        shortname_to_mod[short].append(mn)

    injected = 0
    for nid, info in call_graph.items():
        parts = nid.split(".")
        caller_mod = ".".join(parts[:-1]) if len(parts) > 1 else nid
        if caller_mod not in MG.nodes:
            continue
        caller_cc = _resolve_cc(caller_mod)
        has_history = bool(cochange_index.get(caller_cc))
        for callee_name in info.get("callees", []):
            for callee_mod in shortname_to_mod.get(callee_name, []):
                if callee_mod == caller_mod:
                    continue
                callee_cc = _resolve_cc(callee_mod)
                callee_has_history = bool(cochange_index.get(callee_cc))
                if has_history and callee_has_history:
                    continue   # both have real data — no need to synthesise
                existing = {p["module"] for p in cochange_index.get(caller_cc, [])}
                if callee_cc not in existing:
                    cochange_index.setdefault(caller_cc, []).append(
                        {"module": callee_cc, "weight": SYNTHETIC_WEIGHT, "synthetic": True}
                    )
                    injected += 1
                existing2 = {p["module"] for p in cochange_index.get(callee_cc, [])}
                if caller_cc not in existing2:
                    cochange_index.setdefault(callee_cc, []).append(
                        {"module": caller_cc, "weight": SYNTHETIC_WEIGHT, "synthetic": True}
                    )
                    injected += 1
    if injected:
        print(f"  Synthetic co-change: +{injected} pairs injected from call graph")


# ════════════════════════════════════════════════════════════════════════════
# UNIFIED SEARCH  (vector + BM25 via RRF, service-weight-aware)
# ════════════════════════════════════════════════════════════════════════════

def _cochange_expand(seed_results: dict, top_seed: int = 5,
                     max_neighbors: int = 15,
                     nodes_per_module: int = 5) -> dict:
    """Build a co-change result dict from seed search results.

    1. Extract top modules from seed results (by rank)
    2. Find co-change neighbors (1-hop, weight >= 5)
    3. Return {service: [node_dict with _cochange_score, ...]} for RRF

    Caps at ``nodes_per_module`` nodes per neighbor module to avoid
    flooding results from large modules (e.g. Gateway.Common has 300+ symbols).
    """
    if not cochange_index or G is None:
        return {}

    # Extract top modules from seed results
    flat = [(n.get("_rrf_score", 0) or n.get("_distance", 1), n)
            for nodes in seed_results.values() for n in nodes]
    if flat and "_rrf_score" in flat[0][1]:
        flat.sort(key=lambda x: -x[0])
    else:
        flat.sort(key=lambda x: x[0])

    seed_modules = []
    seen_mods = set()
    for _, node in flat[:top_seed * 3]:
        mod = node.get("module", "")
        if mod and mod not in seen_mods:
            seen_mods.add(mod)
            seed_modules.append(mod)
            if len(seed_modules) >= top_seed:
                break

    if not seed_modules:
        return {}

    neighbors = cochange_path_traverse(seed_modules, max_hops=1,
                                       top_k=10, min_weight=5)
    if not neighbors:
        return {}

    # Convert neighbor modules to nodes, capped per module
    results: dict = defaultdict(list)
    for nb in neighbors[:max_neighbors]:
        mod_name = nb["module"]
        weight = nb["weight"]
        node_ids = file_to_nodes.get(mod_name, [])
        mod_count = 0
        for nid in node_ids:
            if mod_count >= nodes_per_module:
                break
            if nid not in G.nodes:
                continue
            d = G.nodes[nid]
            node = {
                "id": nid,
                "name": d.get("name", nid),
                "module": mod_name,
                "service": d.get("service", ""),
                "type": d.get("type", ""),
                "file": d.get("file", ""),
                "_cochange_score": float(weight),
            }
            results[node["service"] or "unknown"].append(node)
            mod_count += 1
    return dict(results)


def unified_search(queries: list, k_total: int = 250) -> dict:
    """
    Primary search entry point. Fuses dense vector + BM25 + co-change via RRF.
    Service budget allocated proportional to traffic_weight from SERVICE_PROFILES.
    Falls back gracefully: vector → BM25 → keyword.
    """
    svc_weights = {
        svc: prof.get("traffic_weight", 0.5)
        for svc, prof in SERVICE_PROFILES.items()
    } if SERVICE_PROFILES else {}

    vec_results  = stratified_vector_search(queries, k_total=k_total,
                                             service_weights=svc_weights or None)
    bm25_results = bm25_search(queries[0] if queries else "", top_k=60)

    if vec_results or bm25_results:
        sources = [r for r in (vec_results, bm25_results) if r]
        merged = rrf_merge(*sources)
    else:
        merged = cross_service_keyword_search(queries[0] if queries else "")

    # Co-change expansion: add evolutionary coupling signal to RRF
    if cochange_index and merged:
        cochange_results = _cochange_expand(merged)
        if cochange_results:
            merged = rrf_merge(merged, cochange_results)

    # Apply per-service cap based on traffic_weight
    if SERVICE_PROFILES:
        capped: dict = {}
        for svc, nodes in merged.items():
            weight = SERVICE_PROFILES.get(svc, {}).get("traffic_weight", 0.5)
            cap = max(3, int(25 * weight))
            capped[svc] = nodes[:cap]
        return capped

    return merged
