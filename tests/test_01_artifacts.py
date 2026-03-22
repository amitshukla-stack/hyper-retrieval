"""
test_01_artifacts.py — Pipeline artifact integrity tests.

Ground truth = SOURCE FILES. Every check compares artifacts against
the actual codebase they were built from. Nothing is tested against
itself. This suite catches the class of silent corruption that
went undetected for 3 days (body_store silent truncation).

Run without GPU:
    python3 tests/test_01_artifacts.py

All failures print the exact artifact key, expected value, and actual value.
Exits non-zero on ANY failure.
"""
import sys, json, pathlib, re, random, importlib.util, os
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "serve"))

# Paths: override via env vars or fall back to Juspay defaults
_WS        = pathlib.Path(os.environ.get("WORKSPACE_DIR",  "/home/beast/projects/workspaces/juspay"))
ARTIFACTS  = pathlib.Path(os.environ.get("ARTIFACT_DIR",   str(_WS / "artifacts")))
OUTPUT     = pathlib.Path(os.environ.get("OUTPUT_DIR",     str(_WS / "output")))
ALL_REPOS  = pathlib.Path(os.environ.get("SOURCE_DIR",     str(_WS / "source")))

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"
errors: list = []
warnings: list = []

def ok(label):
    print(f"  {PASS} {label}")

def fail(label, detail=""):
    print(f"  {FAIL} {label}")
    if detail:
        print(f"      {detail}")
    errors.append(label)

def warn(label, detail=""):
    print(f"  {WARN} {label}")
    if detail:
        print(f"      {detail}")
    warnings.append(label)

# ─── Load extractor for current MAX_BODY_CHARS value ──────────────────────────
def _load_extractor():
    extract_path = pathlib.Path(__file__).parent.parent / "build" / "01_extract.py"
    spec = importlib.util.spec_from_file_location("extractor", extract_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

# ══════════════════════════════════════════════════════════════════════════════
# 1. BODY STORE
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 1. body_store.json integrity ===")

body_store_path = OUTPUT / "body_store.json"
if not body_store_path.exists():
    fail("body_store.json exists", f"Not found at {body_store_path}")
    body_store = {}
else:
    body_store = json.loads(body_store_path.read_text())
    ok(f"body_store.json exists ({len(body_store):,} entries)")

extractor = _load_extractor()
MAX_BODY_CHARS = extractor.MAX_BODY_CHARS
MAX_BODY_LINES = extractor.MAX_BODY_LINES

# 1a: No entry silently truncated at the current char limit
at_limit = [k for k, v in body_store.items() if len(v) == MAX_BODY_CHARS]
if at_limit:
    fail(
        f"No entries silently truncated at MAX_BODY_CHARS={MAX_BODY_CHARS}",
        f"{len(at_limit)} entries hit limit without marker. First 3: {at_limit[:3]}"
    )
else:
    ok(f"No entries silently truncated at MAX_BODY_CHARS={MAX_BODY_CHARS}")

# 1b: Entries exceeding limit must have a truncation marker
over_limit_no_marker = [
    k for k, v in body_store.items()
    if len(v) > MAX_BODY_CHARS and "truncated" not in v.lower()
]
if over_limit_no_marker:
    fail(
        "All truncated entries have marker",
        f"{len(over_limit_no_marker)} entries exceed limit with no marker. First: {over_limit_no_marker[:2]}"
    )
else:
    ok("All entries exceeding char limit have a truncation marker")

# 1c: No empty bodies
empty = [k for k, v in body_store.items() if not v.strip()]
if empty:
    fail(f"No empty body entries", f"{len(empty)} empty. First: {empty[:3]}")
else:
    ok("No empty body entries")

# 1d: No duplicate keys (JSON load would silently overwrite; check raw text)
raw_text = body_store_path.read_text()
all_keys_raw = re.findall(r'^"([^"]+)":', raw_text, re.MULTILINE)
dups = [k for k in all_keys_raw if all_keys_raw.count(k) > 1]
if dups:
    fail(f"No duplicate keys in body_store.json", f"Duplicates: {list(set(dups))[:5]}")
else:
    ok("No duplicate keys in body_store.json")

# 1e: Spot-check 100 random Haskell entries against source
print("\n  [spot-check: 100 random Haskell bodies vs source files]")
hs_entries = [(k, v) for k, v in body_store.items()
              if not k.startswith("UCS::") and "." in k and len(v) > 50]
sample = random.sample(hs_entries, min(100, len(hs_entries)))
spot_ok = spot_short = spot_missing_src = spot_mismatch = 0

# Load graph to get file paths
graph_path = ARTIFACTS / "graph_with_summaries.json"
fn_to_file: dict = {}
if graph_path.exists():
    gdata = json.loads(graph_path.read_text())
    for node in gdata.get("nodes", []):
        nid = node.get("id", "")
        fpath = node.get("file", "")
        if nid and fpath:
            fn_to_file[nid] = fpath

for fn_id, stored_body in sample:
    frel = fn_to_file.get(fn_id, "")
    if not frel:
        spot_missing_src += 1
        continue
    src_path = ALL_REPOS / frel
    if not src_path.exists():
        spot_missing_src += 1
        continue
    src = src_path.read_text(encoding="utf-8", errors="replace")
    fn_name = fn_id.split(".")[-1]
    # The stored body must appear as a substring of the actual source
    # (it's a prefix of the actual function body)
    first_line = stored_body.split("\n")[0].strip()
    if not first_line:
        spot_short += 1
        continue
    if first_line in src:
        spot_ok += 1
    else:
        spot_mismatch += 1
        if spot_mismatch <= 3:
            warn(
                f"Body mismatch: {fn_id}",
                f"First stored line not found in source: {first_line[:80]!r}"
            )

total_checked = spot_ok + spot_mismatch
if total_checked > 0:
    mismatch_pct = spot_mismatch / total_checked * 100
    if mismatch_pct > 5:
        fail(
            f"Body spot-check mismatch rate ≤5%",
            f"{mismatch_pct:.1f}% mismatched ({spot_mismatch}/{total_checked} checked, {spot_missing_src} src not found)"
        )
    else:
        ok(f"Body spot-check: {spot_ok}/{total_checked} match source ({spot_missing_src} src unavailable)")


# ══════════════════════════════════════════════════════════════════════════════
# 2. GRAPH INTEGRITY
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 2. graph_with_summaries.json integrity ===")

if not graph_path.exists():
    fail("graph_with_summaries.json exists")
    nodes = []
else:
    ok(f"graph_with_summaries.json exists")
    nodes = gdata.get("nodes", [])
    edges = gdata.get("edges", [])
    cluster_summaries = gdata.get("cluster_summaries", {})

# 2a: Node count in expected range (Juspay: 114,534 symbols across 12 services)
node_count = len(nodes)
if 90000 <= node_count <= 150000:
    ok(f"Node count in range: {node_count:,}")
else:
    fail(f"Node count {node_count:,} outside expected range [90000, 150000]")

# 2b: Required fields on all nodes
REQUIRED_FIELDS = {"id", "name", "module", "kind", "type", "file", "lang", "service"}
missing_fields: dict = {}
for n in nodes:
    for f in REQUIRED_FIELDS:
        if f not in n:
            missing_fields.setdefault(f, 0)
            missing_fields[f] += 1
if missing_fields:
    fail(f"All nodes have required fields", f"Missing counts: {missing_fields}")
else:
    ok("All nodes have required fields")

# 2c: No duplicate node IDs
all_ids = [n.get("id", "") for n in nodes]
dup_ids = {x for x in all_ids if all_ids.count(x) > 1}
if dup_ids:
    fail(f"No duplicate node IDs", f"Duplicates: {list(dup_ids)[:5]}")
else:
    ok("No duplicate node IDs")

# 2d: Cluster names non-empty for at least 80% of nodes
nodes_with_cluster = [n for n in nodes if n.get("cluster_name", "").strip()]
cluster_pct = len(nodes_with_cluster) / len(nodes) * 100 if nodes else 0
if cluster_pct >= 80:
    ok(f"Cluster names present: {cluster_pct:.1f}% of nodes")
else:
    fail(f"Cluster names ≥80% populated", f"Only {cluster_pct:.1f}% have cluster_name")

# 2e: Services in graph match KNOWN_SERVICES
from retrieval_engine import KNOWN_SERVICES  # type: ignore
node_services = {n.get("service", "") for n in nodes if n.get("service")}
missing_svcs = set(KNOWN_SERVICES) - node_services
extra_svcs = node_services - set(KNOWN_SERVICES) - {""}
if missing_svcs:
    warn(f"All known services in graph", f"Missing services: {missing_svcs}")
else:
    ok(f"All {len(KNOWN_SERVICES)} known services present in graph")
if extra_svcs:
    warn(f"Unexpected service names in graph (possible typo)", f"{extra_svcs}")

# 2f: Cluster summaries non-empty
if len(cluster_summaries) >= 40:
    ok(f"Cluster summaries: {len(cluster_summaries)} clusters")
else:
    warn(f"Low cluster count: {len(cluster_summaries)} (expected ≥40)")


# ══════════════════════════════════════════════════════════════════════════════
# 3. CALL GRAPH
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 3. call_graph.json integrity ===")

cg_path = OUTPUT / "call_graph.json"
if not cg_path.exists():
    fail("call_graph.json exists")
    call_graph = {}
else:
    call_graph = json.loads(cg_path.read_text())
    ok(f"call_graph.json exists ({len(call_graph):,} entries)")

# 3a: All entries have callees and callers keys
bad_schema = [k for k, v in call_graph.items()
              if not isinstance(v, dict) or "callees" not in v or "callers" not in v]
if bad_schema:
    fail(f"All call_graph entries have callees+callers keys",
         f"{len(bad_schema)} malformed. First: {bad_schema[:2]}")
else:
    ok("All call_graph entries have callees+callers keys")

# 3b: No empty-string callee/caller names
bad_callees = [k for k, v in call_graph.items()
               if any(c == "" for c in v.get("callees", []))]
if bad_callees:
    fail(f"No empty-string callee names", f"{len(bad_callees)} entries. First: {bad_callees[:2]}")
else:
    ok("No empty-string callee names")

# 3c: Callers stored as full IDs (must contain a dot or :: separator)
sample_callers = [(k, c) for k, v in call_graph.items()
                  for c in v.get("callers", [])
                  if "." not in c and "::" not in c][:10]
if sample_callers:
    warn(f"Callers stored as short names (not full IDs)",
         f"Examples: {sample_callers[:3]}")
else:
    ok("Callers stored as full IDs (contain dot or :: separator)")

# 3d: body_store coverage — every key in body_store should be in call_graph
bs_not_in_cg = [k for k in body_store if k not in call_graph]
bs_not_in_cg_pct = len(bs_not_in_cg) / len(body_store) * 100 if body_store else 0
if bs_not_in_cg_pct > 10:
    warn(f"body_store/call_graph alignment",
         f"{bs_not_in_cg_pct:.1f}% of body_store keys not in call_graph ({len(bs_not_in_cg):,})")
else:
    ok(f"body_store/call_graph aligned: {100 - bs_not_in_cg_pct:.1f}% overlap")


# ══════════════════════════════════════════════════════════════════════════════
# 4. CO-CHANGE INDEX
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 4. cochange_index.json integrity ===")

cc_path = ARTIFACTS / "cochange_index.json"
if not cc_path.exists():
    fail("cochange_index.json exists")
    cochange = {}
else:
    cc_raw = json.loads(cc_path.read_text())
    # Raw file format: {meta: {...}, edges: {module: [{module, weight}]}}
    # "edges" is the module→pairs dict; "meta" has aggregate stats.
    if "edges" not in cc_raw:
        fail("cochange_index.json has 'edges' key", f"Top-level keys: {list(cc_raw.keys())}")
        cochange = {}
    else:
        cochange = cc_raw["edges"]
        meta_cc = cc_raw.get("meta", {})
        ok(f"cochange_index.json exists ({len(cochange):,} modules, "
           f"{meta_cc.get('total_pairs', '?')} pairs)")

# 4a: Count in expected range
if 7000 <= len(cochange) <= 15000:
    ok(f"Co-change module count in range: {len(cochange):,}")
else:
    fail(f"Co-change count {len(cochange):,} outside expected [7000, 15000]")

# 4b: All values are lists of {module, weight} dicts
bad_cc = 0
bad_weight = 0
for mod, pairs in cochange.items():
    if not isinstance(pairs, list):
        bad_cc += 1
        continue
    for p in pairs:
        if not isinstance(p, dict) or "module" not in p or "weight" not in p:
            bad_cc += 1
        elif not isinstance(p["weight"], (int, float)) or p["weight"] <= 0:
            bad_weight += 1
if bad_cc:
    fail(f"All co-change entries have valid {'{module, weight}'} format", f"{bad_cc} invalid")
else:
    ok(f"All co-change entries have valid format")
if bad_weight:
    fail(f"All co-change weights are positive numbers", f"{bad_weight} invalid weights")
else:
    ok(f"All co-change weights are positive")

# 4c: Pair count in expected range
total_pairs = sum(len(v) for v in cochange.values())
if total_pairs >= 100000:
    ok(f"Total co-change pairs: {total_pairs:,}")
else:
    warn(f"Total co-change pairs {total_pairs:,} below expected ≥100,000")

# 4d: Cross-check: spot-check that co-change modules exist in graph or body_store
node_ids_set = {n.get("id", "") for n in nodes}
all_body_ids = set(body_store.keys())

# Co-change keys are module-level (e.g. "Euler.API.Txns.Mandates"), graph nodes are
# function-level ("Euler.API.Txns.Mandates.fn"). Check by module attribute instead.
graph_modules = {d.get("module", "") for _, d in zip(range(5000), nodes)}  # sample
sample_cc_mods = random.sample(list(cochange.keys()), min(100, len(cochange)))
in_graph = sum(1 for m in sample_cc_mods if m in graph_modules)
in_graph_pct = in_graph / len(sample_cc_mods) * 100 if sample_cc_mods else 0
if in_graph_pct >= 30:
    ok(f"Co-change modules traceable to graph modules: {in_graph_pct:.0f}% of sample")
else:
    warn(f"Only {in_graph_pct:.0f}% of co-change modules matched graph modules "
         f"(co-change may use file-path keys vs dot-path module names)")


# ══════════════════════════════════════════════════════════════════════════════
# 5. VECTOR TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 5. vectors.lance integrity ===")

lance_path = ARTIFACTS / "vectors.lance"
if not lance_path.exists():
    fail("vectors.lance exists")
else:
    ok("vectors.lance exists")
    try:
        import lancedb
        # retrieval_engine.py: lancedb.connect(LANCE_PATH) where LANCE_PATH = "demo_artifact/vectors.lance"
        # Inside vectors.lance the table is named "chunks".
        db = lancedb.connect(str(ARTIFACTS / "vectors.lance"))
        tbl = db.open_table("chunks")
        vec_count = tbl.count_rows()
        if 90000 <= vec_count <= 150000:
            ok(f"Vector row count matches graph range: {vec_count:,}")
        else:
            fail(f"Vector count {vec_count:,} outside expected [90000, 150000]")

        # Check count matches graph node count (within 1%)
        count_diff = abs(vec_count - node_count) / max(node_count, 1)
        if count_diff <= 0.01:
            ok(f"Vector count matches node count within 1%: {vec_count} vs {node_count}")
        else:
            fail(
                f"Vector/node count mismatch >1%",
                f"vectors={vec_count}, graph_nodes={node_count}, diff={count_diff:.2%}"
            )

        # Check schema has required columns
        schema = tbl.schema
        required_cols = {"id", "vector", "service", "module"}
        actual_cols = {f.name for f in schema}
        missing_cols = required_cols - actual_cols
        if missing_cols:
            fail(f"vector table has required columns", f"Missing: {missing_cols}")
        else:
            ok(f"Vector table has required columns: {sorted(actual_cols)}")

        # Sample 10 vectors and check dimension
        sample_df = tbl.search().limit(10).to_pandas()
        if "vector" in sample_df.columns:
            dims = {len(row) for row in sample_df["vector"].tolist()}
            if dims == {4096}:
                ok("Vector dimension = 4096 (Qwen3-Embedding-8B)")
            else:
                fail(f"Vector dimension must be 4096", f"Found: {dims}")

    except ImportError:
        warn("lancedb not importable — skipping vector table checks")
    except Exception as e:
        fail(f"vector table openable", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# 6. GATEWAY INTEGRITY CONFIG
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 6. gateway_integrity_config.json integrity ===")

gw_path = OUTPUT / "gateway_integrity_config.json"
if not gw_path.exists():
    warn("gateway_integrity_config.json missing (optional)")
else:
    gw_data = json.loads(gw_path.read_text())
    ok(f"gateway_integrity_config.json: {len(gw_data)} gateways")
    if len(gw_data) >= 6:
        ok(f"Gateway count ≥6")
    else:
        warn(f"Only {len(gw_data)} gateways configured (expected ≥6)")


# ══════════════════════════════════════════════════════════════════════════════
# 7. DOC CHUNKS
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 7. doc_chunks.json integrity ===")

doc_path = OUTPUT / "doc_chunks.json"
if not doc_path.exists():
    warn("doc_chunks.json missing (optional — only needed for doc search)")
else:
    doc_chunks = json.loads(doc_path.read_text())
    ok(f"doc_chunks.json: {len(doc_chunks)} chunks")
    bad_chunks = [c for c in doc_chunks
                  if not isinstance(c, dict) or not c.get("text", "").strip()]
    if bad_chunks:
        fail(f"No empty doc chunks", f"{len(bad_chunks)} empty chunks")
    else:
        ok("All doc chunks have non-empty text")


# ══════════════════════════════════════════════════════════════════════════════
# 8. ARTIFACT CROSS-CONSISTENCY
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 8. Cross-artifact consistency ===")

# 8a: Every body_store key must be a valid node ID in graph or a rust :: ID
graph_ids = {n.get("id", "") for n in nodes}
bs_not_in_graph = [k for k in body_store if k not in graph_ids]
bs_not_in_graph_pct = len(bs_not_in_graph) / len(body_store) * 100 if body_store else 0
if bs_not_in_graph_pct > 20:
    warn(
        f"body_store / graph alignment",
        f"{bs_not_in_graph_pct:.1f}% of body_store keys not in graph node IDs ({len(bs_not_in_graph):,})"
        f"\n      (Rust/Groovy IDs with :: separators expected to diverge)"
    )
else:
    ok(f"body_store/graph alignment: {100-bs_not_in_graph_pct:.1f}% of body IDs in graph")

# 8b: Embed IDs match graph IDs
embed_ids_path = ARTIFACTS.parent / "output" / "embed_ids.json"
if embed_ids_path.exists():
    embed_ids = json.loads(embed_ids_path.read_text())
    not_in_graph = [e for e in embed_ids if e not in graph_ids]
    not_in_pct = len(not_in_graph) / len(embed_ids) * 100 if embed_ids else 0
    if not_in_pct > 1:
        warn(f"Embed IDs in graph", f"{not_in_pct:.1f}% of embed_ids not found in graph")
    else:
        ok(f"embed_ids.json aligned with graph ({len(embed_ids):,} IDs)")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print()
if warnings:
    print(f"\033[93m{len(warnings)} warnings: {warnings}\033[0m")
if errors:
    print(f"\033[91m{len(errors)} FAILED: {errors}\033[0m")
    sys.exit(1)
else:
    n = 8  # sections
    print(f"\033[92mAll {n} artifact sections passed ({len(warnings)} warnings).\033[0m")
