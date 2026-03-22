"""
Stage 1 — Extract symbols, edges, and git signals from all services.
Supports: Haskell (.hs), Rust (.rs), Python (.py)
Output:  pipeline/output/raw_graph.json
"""
import re, ast, json, subprocess, collections, pathlib, sys, os

REPO_ROOT = pathlib.Path(os.environ.get("REPO_ROOT", pathlib.Path(__file__).parent.parent / "repo"))
OUT_DIR   = pathlib.Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

# ── Haskell source parser ────────────────────────────────────────────────────

def parse_haskell_file(path: pathlib.Path) -> tuple[list, list]:
    src = path.read_text(encoding="utf-8", errors="replace")
    symbols, edges = [], []

    # Module name
    m = re.search(r"^module\s+([\w.]+)", src, re.MULTILINE)
    module = m.group(1) if m else str(path.stem)
    file_id = str(path.relative_to(REPO_ROOT))

    # Imports
    for imp in re.finditer(r"^import\s+(?:qualified\s+)?([\w.]+)", src, re.MULTILINE):
        edges.append({"from": module, "to": imp.group(1), "kind": "import", "lang": "haskell"})

    # Type signatures  ->  name :: Type
    seen_sigs = set()
    for sig in re.finditer(r"^([a-z_][a-zA-Z0-9_']*(?:,\s*[a-z_][a-zA-Z0-9_']*)*)\s*::\s*(.+)", src, re.MULTILINE):
        names_raw = sig.group(1)
        type_str  = sig.group(2).strip()
        for name in [n.strip() for n in names_raw.split(",")]:
            nid = f"{module}.{name}"
            if nid not in seen_sigs:
                seen_sigs.add(nid)
                symbols.append({
                    "id":     nid,
                    "name":   name,
                    "module": module,
                    "kind":   "function",
                    "type":   type_str,
                    "file":   file_id,
                    "lang":   "haskell",
                    "service": path.parts[len(REPO_ROOT.parts)] if len(path.parts) > len(REPO_ROOT.parts) else "unknown"
                })

    # Data / newtype / type declarations
    for decl in re.finditer(r"^(?:data|newtype|type)\s+([A-Z][a-zA-Z0-9_']*)", src, re.MULTILINE):
        nid = f"{module}.{decl.group(1)}"
        symbols.append({
            "id":     nid,
            "name":   decl.group(1),
            "module": module,
            "kind":   "type",
            "type":   "",
            "file":   file_id,
            "lang":   "haskell",
            "service": path.parts[len(REPO_ROOT.parts)] if len(path.parts) > len(REPO_ROOT.parts) else "unknown"
        })

    # Class instances — reveals type class relationships
    for inst in re.finditer(r"^instance\s+.*?([A-Z][a-zA-Z0-9_']*)\s+([A-Z][a-zA-Z0-9_']*)", src, re.MULTILINE):
        edges.append({
            "from": f"{module}.{inst.group(2)}",
            "to":   f"{module}.{inst.group(1)}",
            "kind": "instance",
            "lang": "haskell"
        })

    return symbols, edges


# ── Rust source parser ───────────────────────────────────────────────────────

def parse_rust_file(path: pathlib.Path, service: str) -> tuple[list, list]:
    src = path.read_text(encoding="utf-8", errors="replace")
    symbols, edges = [], []
    file_id = str(path.relative_to(REPO_ROOT))
    # Derive module path from file path
    module = str(path.relative_to(REPO_ROOT)).replace("/", "::").replace("\\", "::").removesuffix(".rs")

    # use statements
    for use in re.finditer(r"^use\s+([\w::{}, *]+);", src, re.MULTILINE):
        edges.append({"from": module, "to": use.group(1), "kind": "import", "lang": "rust"})

    # fn declarations
    for fn_match in re.finditer(
        r"(?:pub(?:\([\w]+\))?\s+)?(?:async\s+)?fn\s+([a-z_][a-zA-Z0-9_]*)\s*(<[^>]*>)?\s*\(([^)]*)\)\s*(?:->\s*([^{;]+))?",
        src
    ):
        name    = fn_match.group(1)
        params  = fn_match.group(3).strip()
        ret     = (fn_match.group(4) or "").strip()
        type_str = f"({params}) -> {ret}" if ret else f"({params})"
        symbols.append({
            "id":     f"{module}::{name}",
            "name":   name,
            "module": module,
            "kind":   "function",
            "type":   type_str,
            "file":   file_id,
            "lang":   "rust",
            "service": service
        })

    # struct / enum / trait
    for decl in re.finditer(r"^(?:pub(?:\([\w]+\))?\s+)?(?:struct|enum|trait)\s+([A-Z][a-zA-Z0-9_]*)", src, re.MULTILINE):
        symbols.append({
            "id":     f"{module}::{decl.group(1)}",
            "name":   decl.group(1),
            "module": module,
            "kind":   "type",
            "type":   "",
            "file":   file_id,
            "lang":   "rust",
            "service": service
        })

    return symbols, edges


# ── Python source parser ─────────────────────────────────────────────────────

def parse_python_file(path: pathlib.Path, service: str) -> tuple[list, list]:
    symbols, edges = [], []
    file_id = str(path.relative_to(REPO_ROOT))
    module  = str(path.relative_to(REPO_ROOT)).replace("/", ".").replace("\\", ".").removesuffix(".py")

    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return symbols, edges

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = node.module if isinstance(node, ast.ImportFrom) else None
            for alias in (node.names if isinstance(node, ast.Import) else [ast.alias(name=mod or "")]):
                target = mod or alias.name
                if target:
                    edges.append({"from": module, "to": target, "kind": "import", "lang": "python"})

        elif isinstance(node, ast.FunctionDef):
            args = [a.arg for a in node.args.args]
            symbols.append({
                "id":     f"{module}.{node.name}",
                "name":   node.name,
                "module": module,
                "kind":   "function",
                "type":   f"({', '.join(args)})",
                "file":   file_id,
                "lang":   "python",
                "service": service,
                "docstring": ast.get_docstring(node) or ""
            })

        elif isinstance(node, ast.ClassDef):
            symbols.append({
                "id":     f"{module}.{node.name}",
                "name":   node.name,
                "module": module,
                "kind":   "class",
                "type":   "",
                "file":   file_id,
                "lang":   "python",
                "service": service,
                "docstring": ast.get_docstring(node) or ""
            })

    return symbols, edges


# ── Ghost dependency detector ────────────────────────────────────────────────

GHOST_PATTERNS = {
    "postgres":    re.compile(r'postgres(?:ql)?://|"postgresql|PG_|DATABASE_URL|psycopg|pgpool', re.I),
    "redis":       re.compile(r'redis://|REDIS_|RedisClient|hset|hget|zadd|lpush', re.I),
    "kafka":       re.compile(r'kafka|KafkaProducer|KafkaConsumer|bootstrap_servers|topic', re.I),
    "sqs":         re.compile(r'sqs|SQS|boto3_client_sqs|queue_url|send_message', re.I),
    "s3":          re.compile(r's3://|upload_file|download_file|S3_BUCKET', re.I),
    "http_client": re.compile(r'requests[.](get|post|put|delete)|http_client|aiohttp|reqwest', re.I),
    "grpc":        re.compile(r'grpc[.]|[.]proto|grpc_channel', re.I),
    "mysql":       re.compile(r'mysql://|MYSQL_|mysqlclient', re.I),
}

def detect_ghosts(path: pathlib.Path) -> list[str]:
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    return [name for name, pat in GHOST_PATTERNS.items() if pat.search(src)]


# ── Git signals ──────────────────────────────────────────────────────────────

def extract_git_signals(repo_path: pathlib.Path) -> dict:
    try:
        log = subprocess.run(
            ["git", "-C", str(repo_path), "log", "--all", "--name-only",
             "--format=COMMIT|%H|%s|%ae|%ai"],
            capture_output=True, text=True, timeout=60
        ).stdout
    except Exception as e:
        print(f"  [git] skipped: {e}")
        return {"co_change_edges": [], "file_history": {}}

    commits, current = [], None
    for line in log.splitlines():
        if line.startswith("COMMIT|"):
            parts = line.split("|", 4)
            if len(parts) == 5:
                _, sha, msg, author, date = parts
                current = {"sha": sha, "msg": msg, "author": author, "date": date, "files": []}
                commits.append(current)
        elif line.strip() and current:
            current["files"].append(line.strip())

    co_change = collections.Counter()
    for c in commits:
        files = c["files"]
        for i, f1 in enumerate(files):
            for f2 in files[i+1:]:
                co_change[tuple(sorted([f1, f2]))] += 1

    file_history = collections.defaultdict(list)
    for c in commits:
        for f in c["files"]:
            file_history[f].append({"sha": c["sha"][:7], "msg": c["msg"], "date": c["date"][:10]})

    return {
        "co_change_edges": [
            {"from": k[0], "to": k[1], "weight": v, "kind": "co_change"}
            for k, v in co_change.most_common(3000)
        ],
        "file_history": {k: v[:8] for k, v in file_history.items()}
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    all_symbols, all_edges = [], []
    ghost_map = collections.defaultdict(list)  # symbol_id -> [ghost_deps]

    services = [p for p in REPO_ROOT.iterdir() if p.is_dir() and not p.name.startswith(".")]
    print(f"Found {len(services)} service directories: {[s.name for s in services]}")

    for service_dir in services:
        service_name = service_dir.name
        print(f"\nProcessing: {service_name}")

        # Haskell
        hs_files = [f for f in service_dir.rglob("*.hs") if f.is_file()]
        for f in hs_files:
            if any(p in f.parts for p in ["dist", "dist-newstyle", ".stack-work", ".juspay"]):
                continue
            s, e = parse_haskell_file(f)
            all_symbols.extend(s)
            all_edges.extend(e)
            ghosts = detect_ghosts(f)
            if ghosts:
                for sym in s:
                    ghost_map[sym["id"]].extend(ghosts)
        print(f"  Haskell: {len(hs_files)} files → {sum(1 for s in all_symbols if s.get('lang')=='haskell' and s.get('service')==service_name)} symbols")

        # Rust
        rs_files = list(service_dir.rglob("*.rs"))
        for f in rs_files:
            if any(p in str(f) for p in ["target/", "/target\\"]):
                continue
            s, e = parse_rust_file(f, service_name)
            all_symbols.extend(s)
            all_edges.extend(e)
            ghosts = detect_ghosts(f)
            if ghosts:
                for sym in s:
                    ghost_map[sym["id"]].extend(ghosts)
        print(f"  Rust:    {len(rs_files)} files → {sum(1 for s in all_symbols if s.get('lang')=='rust' and s.get('service')==service_name)} symbols")

        # Python
        py_files = list(service_dir.rglob("*.py"))
        for f in py_files:
            if any(p in str(f) for p in ["venv/", "__pycache__", ".venv/"]):
                continue
            s, e = parse_python_file(f, service_name)
            all_symbols.extend(s)
            all_edges.extend(e)
            ghosts = detect_ghosts(f)
            if ghosts:
                for sym in s:
                    ghost_map[sym["id"]].extend(ghosts)
        print(f"  Python:  {len(py_files)} files → {sum(1 for s in all_symbols if s.get('lang')=='python' and s.get('service')==service_name)} symbols")

    # Attach ghost deps to symbols
    for sym in all_symbols:
        sym["ghost_deps"] = list(set(ghost_map.get(sym["id"], [])))

    # Git signals (run on each service with git, and on root)
    print("\nExtracting git signals...")
    git_data = extract_git_signals(REPO_ROOT)
    all_edges.extend(git_data["co_change_edges"])
    file_history = git_data["file_history"]

    # Attach recent commit history to symbols
    for sym in all_symbols:
        fkey = sym.get("file", "")
        sym["commit_history"] = file_history.get(fkey, [])[:5]

    # Deduplicate symbols by id
    seen_ids = {}
    deduped = []
    for s in all_symbols:
        if s["id"] not in seen_ids:
            seen_ids[s["id"]] = True
            deduped.append(s)

    result = {
        "nodes": deduped,
        "edges": all_edges,
        "stats": {
            "total_symbols": len(deduped),
            "total_edges":   len(all_edges),
            "services":      [s.name for s in services]
        }
    }

    out_path = OUT_DIR / "raw_graph.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n✓ Wrote {len(deduped)} symbols, {len(all_edges)} edges → {out_path}")
    print(f"  Services: {result['stats']['services']}")


if __name__ == "__main__":
    main()
