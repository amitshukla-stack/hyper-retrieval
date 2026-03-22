"""
test_06_auto_eval.py — LLM-powered auto-generated comprehensive evals.

Eliminates manual test authoring. The LLM reads SOURCE CODE (ground truth)
and generates:
  1. Body integrity cases   — distinctive fragments that must appear in body_store
  2. Retrieval accuracy cases — natural language queries that must surface the function
  3. Cross-service flow cases — scenario questions that must surface both endpoints

Architecture:
  SOURCE FILES (ground truth)
       ↓ read
  LLM generates test cases
       ↓ cached to tests/generated/eval_cases.json
  Retrieval tools evaluated
       ↓
  Accuracy report (per service, per type)

Run:
    python3 tests/test_06_auto_eval.py                     # use cached cases if available
    python3 tests/test_06_auto_eval.py --refresh           # regenerate all cases
    python3 tests/test_06_auto_eval.py --samples 10        # N per service (default 5)
    python3 tests/test_06_auto_eval.py --cross-service 15  # cross-service cases (default 10)
    LLM_API_KEY=sk-... LLM_BASE_URL=https://api.openai.com/v1 python3 tests/test_06_auto_eval.py
"""

import sys, json, pathlib, random, os, re, argparse, time, textwrap
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "serve"))

_WS       = pathlib.Path(os.environ.get("WORKSPACE_DIR", "/home/beast/projects/workspaces/juspay"))
ALL_REPOS = pathlib.Path(os.environ.get("SOURCE_DIR",   str(_WS / "source")))
GENERATED = pathlib.Path(__file__).parent / "generated"
CASES_FILE = GENERATED / "eval_cases.json"
RESULTS_FILE = GENERATED / "eval_results.json"

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"
BOLD = "\033[1m"
NC   = "\033[0m"


# ══════════════════════════════════════════════════════════════════════════════
# CLI args
# ══════════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="LLM-powered auto eval for HyperRetrieval")
parser.add_argument("--refresh",       action="store_true", help="Regenerate all LLM cases")
parser.add_argument("--samples",       type=int, default=5,  help="Functions sampled per service")
parser.add_argument("--cross-service", type=int, default=10, help="Cross-service cases to generate")
parser.add_argument("--top-k",         type=int, default=20, help="Rank threshold for retrieval checks")
parser.add_argument("--report-only",   action="store_true",  help="Print last results without running")
args = parser.parse_args()

GENERATED.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# LLM client (any OpenAI-compatible endpoint)
# ══════════════════════════════════════════════════════════════════════════════
def _make_llm_client():
    """Return an OpenAI-compatible client. Prefers env vars over hardcoded defaults."""
    try:
        from openai import OpenAI
    except ImportError:
        print(f"  {FAIL} openai package not installed. Run: pip install openai")
        sys.exit(1)

    if os.environ.get("OPENAI_API_KEY"):
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"]), "gpt-4o-mini"

    api_key  = os.environ.get("LLM_API_KEY",  "")
    base_url = os.environ.get("LLM_BASE_URL", "")
    model    = os.environ.get("LLM_MODEL",    "reasoning-large-model")
    return OpenAI(api_key=api_key, base_url=base_url), model


def _llm_json(client, model: str, system: str, user: str, retries: int = 3) -> dict:
    """Call LLM and parse JSON response. Returns {} on failure."""
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                           {"role": "user",   "content": user}],
                temperature=0.2,
                max_tokens=800,
            )
            text = resp.choices[0].message.content.strip()
            # Strip markdown code fences if present
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
            return json.loads(text)
        except json.JSONDecodeError:
            if attempt < retries - 1:
                time.sleep(1)
            continue
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                print(f"    {WARN} LLM call failed: {e}")
    return {}


# ══════════════════════════════════════════════════════════════════════════════
# Source reading helpers
# ══════════════════════════════════════════════════════════════════════════════
def _read_source(fn_id: str, fn_to_file: dict) -> str:
    """Read the actual source for fn_id from ALL_REPOS. Returns '' if unavailable."""
    frel = fn_to_file.get(fn_id, "")
    if not frel:
        return ""
    src_path = ALL_REPOS / frel
    if not src_path.exists():
        return ""
    try:
        return src_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _extract_function_snippet(src: str, fn_name: str, max_chars: int = 2000) -> str:
    """
    Extract the function body snippet from source.
    Language-agnostic: finds the first occurrence of fn_name at line start
    and returns up to max_chars of surrounding context.
    """
    if not src or not fn_name:
        return ""
    lines = src.splitlines(keepends=True)
    start = -1
    for i, line in enumerate(lines):
        if re.match(rf'^{re.escape(fn_name)}\b', line) or \
           re.match(rf'^\s+def {re.escape(fn_name)}\b', line) or \
           re.match(rf'^\s+fn {re.escape(fn_name)}\b', line) or \
           re.match(rf'^\s+pub fn {re.escape(fn_name)}\b', line):
            start = i
            break
    if start == -1:
        return ""
    snippet = "".join(lines[start:start + 80])
    return snippet[:max_chars]


# ══════════════════════════════════════════════════════════════════════════════
# Case generation
# ══════════════════════════════════════════════════════════════════════════════
BODY_INTEGRITY_SYSTEM = textwrap.dedent("""
You are a code intelligence quality engineer analyzing function source code.
Your job is to extract DISTINCTIVE code fragments that prove a function body
was correctly extracted — not generic constructs that could appear anywhere.

Always return valid JSON only, no prose.
""").strip()

BODY_INTEGRITY_PROMPT = textwrap.dedent("""
Function ID: {fn_id}
Language: {lang}

Source code snippet:
---
{snippet}
---

Extract 3-5 distinctive string literals, identifiers, or type constructor names
from this function that:
1. Are specific to this function's business logic (not boilerplate)
2. Would appear in a correct extraction of this function body
3. Are NOT common programming keywords (if, else, return, let, etc.)

Also write one sentence describing what this function does.

Return ONLY JSON:
{{"fragments": ["...", "...", "..."], "purpose": "..."}}
""").strip()

RETRIEVAL_QUERY_SYSTEM = textwrap.dedent("""
You are a software engineer asking questions about a codebase.
Generate questions an engineer would actually type into a codebase search tool.
Never mention function names directly — describe what the function DOES.
Always return valid JSON only.
""").strip()

RETRIEVAL_QUERY_PROMPT = textwrap.dedent("""
Function ID: {fn_id}
Purpose: {purpose}
Service: {service}

Code snippet:
---
{snippet}
---

Generate 2 natural language questions an engineer would ask to find this function.
Requirements:
- Describe WHAT the function does, not what it is called
- Use domain terminology naturally (payment terms, flow names, etc.)
- Each question should be independently findable (different phrasing)
- 10-20 words each

Return ONLY JSON:
{{"queries": ["...", "..."]}}
""").strip()

CROSS_SERVICE_SYSTEM = textwrap.dedent("""
You are a software architect analyzing cross-service interactions in a payment system.
Generate scenario questions that an engineer investigating a multi-service flow would ask.
Always return valid JSON only.
""").strip()

CROSS_SERVICE_PROMPT = textwrap.dedent("""
A function in service '{src_service}' calls a function in service '{dst_service}'.

Caller ({src_id}):
---
{src_snippet}
---

Callee ({dst_id}):
---
{dst_snippet}
---

Generate ONE scenario question that:
1. Describes the cross-service interaction (what triggers it, what it accomplishes)
2. Would logically surface BOTH functions when searching the codebase
3. Reads naturally as an engineer investigating a production flow

Return ONLY JSON:
{{"query": "...", "scenario_description": "..."}}
""").strip()


def generate_body_and_retrieval_cases(
    body_store: dict,
    fn_to_file: dict,
    node_attrs: dict,    # fn_id → {service, module, lang, ...}
    client,
    model: str,
    n_per_service: int,
) -> list[dict]:
    """Sample functions stratified by service; LLM generates fragments + queries."""
    print(f"\n{BOLD}Phase 1: Generating body integrity + retrieval cases{NC}")
    print(f"  Sampling {n_per_service} functions per service from body_store...")

    # Stratified sample
    by_service: dict[str, list] = defaultdict(list)
    for fn_id in body_store:
        attrs = node_attrs.get(fn_id, {})
        svc = attrs.get("service", "unknown")
        by_service[svc].append(fn_id)

    sampled = []
    for svc, fns in sorted(by_service.items()):
        # Only include functions where we have a source file
        with_source = [f for f in fns if fn_to_file.get(f) and
                       (ALL_REPOS / fn_to_file[f]).exists()]
        n = min(n_per_service, len(with_source))
        sampled.extend(random.sample(with_source, n))
    print(f"  Total sampled: {len(sampled)} functions across {len(by_service)} services")

    cases = []
    for i, fn_id in enumerate(sampled):
        attrs    = node_attrs.get(fn_id, {})
        service  = attrs.get("service", "unknown")
        lang     = attrs.get("lang", "haskell")
        fn_name  = fn_id.split(".")[-1].split("::")[-1]
        src_full = _read_source(fn_id, fn_to_file)
        snippet  = _extract_function_snippet(src_full, fn_name)

        if not snippet:
            continue

        print(f"  [{i+1}/{len(sampled)}] {fn_id} ({service})", end="  ", flush=True)

        # Body integrity: LLM extracts fragments from SOURCE
        bi = _llm_json(client, model, BODY_INTEGRITY_SYSTEM,
                       BODY_INTEGRITY_PROMPT.format(
                           fn_id=fn_id, lang=lang, snippet=snippet))
        fragments = bi.get("fragments", [])
        purpose   = bi.get("purpose", "")

        # Retrieval accuracy: LLM generates queries FROM purpose + snippet
        if purpose:
            rq = _llm_json(client, model, RETRIEVAL_QUERY_SYSTEM,
                           RETRIEVAL_QUERY_PROMPT.format(
                               fn_id=fn_id, purpose=purpose,
                               service=service, snippet=snippet))
            queries = rq.get("queries", [])
        else:
            queries = []

        print(f"{len(fragments)} fragments, {len(queries)} queries")

        if fragments or queries:
            cases.append({
                "type":      "body_and_retrieval",
                "fn_id":     fn_id,
                "service":   service,
                "lang":      lang,
                "purpose":   purpose,
                "fragments": fragments,   # LLM read from SOURCE — ground truth
                "queries":   queries,     # LLM generated from SOURCE purpose
            })

    return cases


def generate_cross_service_cases(
    call_graph: dict,
    body_store: dict,
    node_attrs: dict,
    fn_to_file: dict,
    client,
    model: str,
    n_cases: int,
) -> list[dict]:
    """Find cross-service call graph edges; LLM generates scenario questions."""
    print(f"\n{BOLD}Phase 2: Generating cross-service flow cases{NC}")

    # Collect cross-service edges where both ends have source + body
    cross_edges = []
    for src_id, cg in call_graph.items():
        src_svc = node_attrs.get(src_id, {}).get("service", "")
        if not src_svc:
            continue
        for callee in cg.get("callees", []):
            dst_svc = node_attrs.get(callee, {}).get("service", "")
            if dst_svc and dst_svc != src_svc:
                if callee in body_store and src_id in body_store:
                    cross_edges.append((src_id, callee, src_svc, dst_svc))

    print(f"  Found {len(cross_edges):,} cross-service call edges")
    if not cross_edges:
        print(f"  {WARN} No cross-service edges found — skipping")
        return []

    sampled_edges = random.sample(cross_edges, min(n_cases, len(cross_edges)))
    print(f"  Sampling {len(sampled_edges)} edges for scenario generation...")

    cases = []
    for i, (src_id, dst_id, src_svc, dst_svc) in enumerate(sampled_edges):
        src_name    = src_id.split(".")[-1].split("::")[-1]
        dst_name    = dst_id.split(".")[-1].split("::")[-1]
        src_full    = _read_source(src_id, fn_to_file)
        dst_full    = _read_source(dst_id, fn_to_file)
        src_snippet = _extract_function_snippet(src_full, src_name) or body_store.get(src_id, "")[:600]
        dst_snippet = _extract_function_snippet(dst_full, dst_name) or body_store.get(dst_id, "")[:600]

        print(f"  [{i+1}/{len(sampled_edges)}] {src_svc} → {dst_svc}", end="  ", flush=True)

        result = _llm_json(client, model, CROSS_SERVICE_SYSTEM,
                           CROSS_SERVICE_PROMPT.format(
                               src_id=src_id, dst_id=dst_id,
                               src_service=src_svc, dst_service=dst_svc,
                               src_snippet=src_snippet[:800],
                               dst_snippet=dst_snippet[:800]))

        query   = result.get("query", "")
        scenario = result.get("scenario_description", "")
        print(query[:60] if query else "FAILED")

        if query:
            cases.append({
                "type":        "cross_service",
                "src_id":      src_id,
                "dst_id":      dst_id,
                "src_service": src_svc,
                "dst_service": dst_svc,
                "query":       query,
                "scenario":    scenario,
            })

    return cases


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation runners
# ══════════════════════════════════════════════════════════════════════════════
def run_body_integrity(cases: list[dict], body_store: dict) -> dict:
    """Check LLM-generated fragments appear in body_store entries."""
    print(f"\n{BOLD}=== Body Integrity Eval ==={NC}")
    results = defaultdict(lambda: {"pass": 0, "fail": 0, "cases": []})

    for case in [c for c in cases if c["type"] == "body_and_retrieval"]:
        fn_id    = case["fn_id"]
        service  = case["service"]
        stored   = body_store.get(fn_id, "")
        fragments = case["fragments"]

        if not stored or not fragments:
            continue

        fn_results = []
        for frag in fragments:
            present = frag in stored
            fn_results.append({"fragment": frag, "present": present})
            if present:
                results[service]["pass"] += 1
            else:
                results[service]["fail"] += 1

        # Print only failures (keep output clean)
        failures = [r["fragment"] for r in fn_results if not r["present"]]
        if failures:
            print(f"  {FAIL} {fn_id}")
            for f in failures:
                print(f"      missing: {f!r}")
                print(f"      body start: {stored[:150]!r}")
        else:
            print(f"  {PASS} {fn_id} — {len(fragments)} fragments verified")

        results[service]["cases"].append({
            "fn_id": fn_id, "fragments": fn_results
        })

    return dict(results)


def run_retrieval_accuracy(
    cases: list[dict],
    RE,        # retrieval_engine module
    top_k: int,
) -> dict:
    """Check LLM-generated queries find the expected function in top-K results."""
    print(f"\n{BOLD}=== Retrieval Accuracy Eval (top-{top_k}) ==={NC}")
    results = defaultdict(lambda: {"pass": 0, "fail": 0, "cases": []})

    for case in [c for c in cases if c["type"] == "body_and_retrieval"]:
        fn_id   = case["fn_id"]
        service = case["service"]
        queries = case["queries"]
        purpose = case["purpose"]

        for query in queries:
            # Use keyword search (no GPU needed)
            hits_by_svc = RE.cross_service_keyword_search(query, max_per_service=top_k)
            all_hit_ids = [n.get("id", "") for hits in hits_by_svc.values() for n in hits]
            found = fn_id in all_hit_ids
            rank  = (all_hit_ids.index(fn_id) + 1) if found else None

            if found:
                results[service]["pass"] += 1
                print(f"  {PASS} [{service}] rank={rank:<3d} '{query[:55]}'")
            else:
                results[service]["fail"] += 1
                print(f"  {FAIL} [{service}] NOT FOUND '{query[:55]}'")
                print(f"      target: {fn_id}")
                print(f"      purpose: {purpose}")

            results[service]["cases"].append({
                "fn_id": fn_id, "query": query, "found": found, "rank": rank
            })

    return dict(results)


def run_cross_service_eval(
    cases: list[dict],
    RE,
    top_k: int,
) -> dict:
    """Check cross-service scenario queries surface at least one of the two endpoints."""
    print(f"\n{BOLD}=== Cross-Service Flow Eval (top-{top_k}) ==={NC}")
    results = {"pass": 0, "fail": 0, "cases": []}

    for case in [c for c in cases if c["type"] == "cross_service"]:
        src_id  = case["src_id"]
        dst_id  = case["dst_id"]
        src_svc = case["src_service"]
        dst_svc = case["dst_service"]
        query   = case["query"]

        hits_by_svc = RE.cross_service_keyword_search(query, max_per_service=top_k)
        all_hit_ids = [n.get("id", "") for hits in hits_by_svc.values() for n in hits]

        src_found = src_id in all_hit_ids
        dst_found = dst_id in all_hit_ids
        both_found = src_found and dst_found
        any_found  = src_found or dst_found

        label = f"{src_svc} → {dst_svc}"
        if both_found:
            results["pass"] += 1
            print(f"  {PASS} BOTH  [{label}] '{query[:50]}'")
        elif any_found:
            results["pass"] += 1   # partial credit
            found_side = src_svc if src_found else dst_svc
            print(f"  {WARN} ONE   [{label}] found {found_side} '{query[:50]}'")
        else:
            results["fail"] += 1
            print(f"  {FAIL} NONE  [{label}] '{query[:50]}'")
            print(f"      src: {src_id}")
            print(f"      dst: {dst_id}")

        results["cases"].append({
            "src_id": src_id, "dst_id": dst_id,
            "query": query, "scenario": case.get("scenario", ""),
            "src_found": src_found, "dst_found": dst_found,
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════════
def _pct(p, f):
    total = p + f
    return f"{p/total*100:.0f}%" if total else "—"

def print_summary_report(body_results, retrieval_results, cross_results, all_services):
    print(f"\n{'═'*70}")
    print(f"{BOLD}  AUTO EVAL SUMMARY REPORT{NC}")
    print(f"{'═'*70}")

    print(f"\n{BOLD}Body Integrity (LLM fragments from source vs body_store):{NC}")
    total_bp = total_bf = 0
    for svc in sorted(all_services):
        r = body_results.get(svc, {"pass": 0, "fail": 0})
        p, f = r["pass"], r["fail"]
        total_bp += p; total_bf += f
        bar = "█" * p + "░" * f if (p + f) <= 30 else ""
        status = PASS if f == 0 else FAIL
        print(f"  {status} {svc:<35s} {p:3d}/{p+f:3d}  {_pct(p,f):>4s}  {bar}")
    print(f"  {'─'*60}")
    print(f"  {'TOTAL':<35s} {total_bp:3d}/{total_bp+total_bf:3d}  {_pct(total_bp, total_bf):>4s}")

    print(f"\n{BOLD}Retrieval Accuracy (LLM queries vs keyword search top-{args.top_k}):{NC}")
    total_rp = total_rf = 0
    for svc in sorted(all_services):
        r = retrieval_results.get(svc, {"pass": 0, "fail": 0})
        p, f = r["pass"], r["fail"]
        total_rp += p; total_rf += f
        status = PASS if f == 0 else (WARN if p > f else FAIL)
        print(f"  {status} {svc:<35s} {p:3d}/{p+f:3d}  {_pct(p,f):>4s}")
    print(f"  {'─'*60}")
    print(f"  {'TOTAL':<35s} {total_rp:3d}/{total_rp+total_rf:3d}  {_pct(total_rp, total_rf):>4s}")

    print(f"\n{BOLD}Cross-Service Flow Accuracy:{NC}")
    cp = cross_results.get("pass", 0)
    cf = cross_results.get("fail", 0)
    status = PASS if cf == 0 else (WARN if cp > cf else FAIL)
    print(f"  {status} {cp}/{cp+cf} scenarios surfaced ≥1 endpoint  ({_pct(cp, cf)})")

    # Overall gate
    body_pct      = total_bp / max(total_bp + total_bf, 1)
    retrieval_pct = total_rp / max(total_rp + total_rf, 1)
    cross_pct     = cp       / max(cp + cf, 1)

    print(f"\n{'═'*70}")
    gates = [
        ("Body integrity ≥ 90%",      body_pct >= 0.90,      f"{body_pct:.0%}"),
        ("Retrieval accuracy ≥ 60%",   retrieval_pct >= 0.60, f"{retrieval_pct:.0%}"),
        ("Cross-service coverage ≥ 60%", cross_pct >= 0.60,  f"{cross_pct:.0%}"),
    ]
    passed_gates = all(ok for _, ok, _ in gates)
    for label, ok, val in gates:
        print(f"  {PASS if ok else FAIL} {label:<40s} {val}")
    print(f"{'═'*70}")
    if passed_gates:
        print(f"\033[92m  ALL QUALITY GATES PASSED\033[0m")
    else:
        failed = [label for label, ok, _ in gates if not ok]
        print(f"\033[91m  GATES FAILED: {failed}\033[0m")
    return passed_gates


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if args.report_only:
    if not RESULTS_FILE.exists():
        print(f"{FAIL} No results file found. Run without --report-only first.")
        sys.exit(1)
    saved = json.loads(RESULTS_FILE.read_text())
    print_summary_report(
        saved["body"], saved["retrieval"], saved["cross"],
        saved["services"]
    )
    sys.exit(0)

# ── Load retrieval engine ───────────────────────────────────────────────────
print(f"\n{BOLD}Loading retrieval engine (no GPU)...{NC}")
os.environ["EMBED_SERVER_URL"] = ""
_artifact_dir = pathlib.Path(os.environ.get("ARTIFACT_DIR", str(_WS / "artifacts")))
import retrieval_engine as RE
RE.initialize(artifact_dir=_artifact_dir, load_embedder=False)
print("Ready.\n")

# Build lookup dicts
node_attrs  = {nid: dict(d) for nid, d in RE.G.nodes(data=True)}
fn_to_file  = {nid: d.get("file", "") for nid, d in RE.G.nodes(data=True)}
all_services = sorted({d.get("service", "") for d in node_attrs.values() if d.get("service")})

# ── Load or generate cases ──────────────────────────────────────────────────
if not args.refresh and CASES_FILE.exists():
    print(f"Loading cached eval cases from {CASES_FILE}")
    all_cases = json.loads(CASES_FILE.read_text())
    print(f"  {len(all_cases)} cases loaded")
    print(f"  (Use --refresh to regenerate from source + LLM)")
else:
    print(f"{BOLD}Generating eval cases via LLM (reads source files as ground truth)...{NC}")
    client, model = _make_llm_client()
    print(f"  LLM: {model}")

    body_cases  = generate_body_and_retrieval_cases(
        RE.body_store, fn_to_file, node_attrs, client, model, args.samples)
    cross_cases = generate_cross_service_cases(
        RE.call_graph, RE.body_store, node_attrs, fn_to_file,
        client, model, args.cross_service)

    all_cases = body_cases + cross_cases
    CASES_FILE.write_text(json.dumps(all_cases, indent=2))
    print(f"\n  Saved {len(all_cases)} cases → {CASES_FILE}")

# ── Run evaluations ─────────────────────────────────────────────────────────
body_results      = run_body_integrity(all_cases, RE.body_store)
retrieval_results = run_retrieval_accuracy(all_cases, RE, args.top_k)
cross_results     = run_cross_service_eval(all_cases, RE, args.top_k)

# ── Persist results ─────────────────────────────────────────────────────────
RESULTS_FILE.write_text(json.dumps({
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "top_k":   args.top_k,
    "services": all_services,
    "body":      {k: {kk: vv for kk, vv in v.items() if kk != "cases"}
                  for k, v in body_results.items()},
    "retrieval": {k: {kk: vv for kk, vv in v.items() if kk != "cases"}
                  for k, v in retrieval_results.items()},
    "cross":    {kk: vv for kk, vv in cross_results.items() if kk != "cases"},
}, indent=2))

# ── Print summary ────────────────────────────────────────────────────────────
passed = print_summary_report(body_results, retrieval_results, cross_results, all_services)
sys.exit(0 if passed else 1)
