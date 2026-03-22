"""
Stage 1 — Extract symbols, bodies, call graph, log patterns, and edges.
Supports: Haskell (.hs), Rust (.rs), Python (.py), Groovy (.groovy/.grails)
Output:  $OUTPUT_DIR/raw_graph.json
         $OUTPUT_DIR/body_store.json      (fn_id → body text)
         $OUTPUT_DIR/call_graph.json      (fn_id → {callees, callers})
         $OUTPUT_DIR/log_patterns.json    (fn_id → [log patterns])

Env vars:
  REPO_ROOT   — path to workspace source/ dir (default: workspaces/juspay/source)
  OUTPUT_DIR  — path to workspace output/ dir (default: workspaces/juspay/output)
"""
import re, ast, json, subprocess, collections, pathlib, sys, os, textwrap

REPO_ROOT = pathlib.Path(os.environ.get("REPO_ROOT",
    "/home/beast/projects/workspaces/juspay/source"))
OUT_DIR   = pathlib.Path(os.environ.get("OUTPUT_DIR",
    "/home/beast/projects/workspaces/juspay/output"))
OUT_DIR.mkdir(exist_ok=True, parents=True)

MAX_BODY_CHARS = 8000   # truncate very long bodies to keep memory sane
MAX_BODY_LINES = 200    # ~5 screens of code


# ═══════════════════════════════════════════════════════════════════════════════
# HASKELL PARSER
# ═══════════════════════════════════════════════════════════════════════════════

# Log patterns used in Juspay Haskell codebase
_HS_LOG_RE = re.compile(
    r'(?:logInfo|logError|logWarn|logDebug|Logger\.log|L\.log|runIO\s+\$\s+log)\s*'
    r'(?:\'[A-Z\w]+\')?\s*'
    r'(?:\$\s*)?'
    r'("(?:[^"\\]|\\.)*"|[A-Z_][A-Z_0-9]{3,})',
    re.MULTILINE
)

_HS_CALL_RE = re.compile(
    r'(?<![A-Z.\'"#])(?<![a-z0-9_])'       # not preceded by type/module chars
    r'([a-z_][a-zA-Z0-9_\']*)'              # lowercase fn name
    r'(?![a-zA-Z0-9_\'"=<>])',              # not followed by continuation
)


def _extract_hs_body(src: str, name: str, sig_pos: int) -> str:
    """
    Extract the body of a Haskell function given its name and position of its
    type signature.  Captures from the first definition line (name = ...) until
    the next top-level definition or EOF.  Returns at most MAX_BODY_CHARS chars.
    """
    lines = src[sig_pos:].splitlines(keepends=True)
    body_lines = []
    in_body = False
    top_level_re = re.compile(r'^[a-z_][a-zA-Z0-9_\']*\s*(?:[^=\n]*=|::)')

    for i, line in enumerate(lines):
        # Skip the type signature itself (line starts with name ::)
        if i == 0 and re.match(rf'^{re.escape(name)}\s*::', line):
            continue

        # Start capturing at the definition line
        if not in_body and re.match(rf'^{re.escape(name)}\b', line):
            in_body = True

        if in_body:
            # Stop at next top-level definition (different name)
            if i > 0 and top_level_re.match(line) and not re.match(rf'^{re.escape(name)}\b', line):
                break
            body_lines.append(line)
            if len(body_lines) >= MAX_BODY_LINES:
                body_lines.append("  -- [body truncated]\n")
                break

    body = "".join(body_lines).strip()
    if len(body) > MAX_BODY_CHARS:
        return body[:MAX_BODY_CHARS] + "\n  -- [body truncated at char limit]"
    return body


def parse_haskell_file(path: pathlib.Path) -> tuple[list, list, dict, dict, dict]:
    """Returns: (symbols, edges, body_store, call_store, log_store)"""
    src = path.read_text(encoding="utf-8", errors="replace")
    symbols, edges = [], []
    body_store, call_store, log_store = {}, {}, {}

    m = re.search(r"^module\s+([\w.]+)", src, re.MULTILINE)
    module = m.group(1) if m else str(path.stem)
    file_id = str(path.relative_to(REPO_ROOT))
    service = path.parts[len(REPO_ROOT.parts)] if len(path.parts) > len(REPO_ROOT.parts) else "unknown"

    # Imports
    for imp in re.finditer(r"^import\s+(?:qualified\s+)?([\w.]+)", src, re.MULTILINE):
        edges.append({"from": module, "to": imp.group(1), "kind": "import", "lang": "haskell"})

    # Type signatures
    seen_sigs = set()
    for sig in re.finditer(r"^([a-z_][a-zA-Z0-9_']*(?:,\s*[a-z_][a-zA-Z0-9_']*)*)\s*::\s*(.+)", src, re.MULTILINE):
        names_raw = sig.group(1)
        type_str  = sig.group(2).strip()
        sig_pos   = sig.start()

        for name in [n.strip() for n in names_raw.split(",")]:
            nid = f"{module}.{name}"
            if nid in seen_sigs:
                continue
            seen_sigs.add(nid)

            symbols.append({
                "id":      nid,
                "name":    name,
                "module":  module,
                "kind":    "function",
                "type":    type_str,
                "file":    file_id,
                "lang":    "haskell",
                "service": service,
            })

            # Extract body
            body = _extract_hs_body(src, name, sig_pos)
            if body:
                body_store[nid] = body

                # Extract callees from body (approximate — lowercase identifiers)
                called = set()
                for cm in _HS_CALL_RE.finditer(body):
                    fn = cm.group(1)
                    if fn not in {name, "do", "let", "in", "where", "case", "of",
                                  "if", "then", "else", "return", "pure", "when",
                                  "unless", "void", "fmap", "map", "filter",
                                  "foldr", "foldl", "mapM", "mapM_", "forM",
                                  "forM_", "sequence", "catch", "handle",
                                  "try", "throwIO", "evaluate", "liftIO"}:
                        called.add(fn)
                call_store[nid] = {"callees": sorted(called), "callers": []}

                # Extract log patterns
                logs = []
                for lm in _HS_LOG_RE.finditer(body):
                    logs.append(lm.group(1).strip('"'))
                if logs:
                    log_store[nid] = logs

    # Data / newtype / type
    for decl in re.finditer(r"^(?:data|newtype|type)\s+([A-Z][a-zA-Z0-9_']*)", src, re.MULTILINE):
        nid = f"{module}.{decl.group(1)}"
        if nid not in seen_sigs:
            symbols.append({
                "id":      nid,
                "name":    decl.group(1),
                "module":  module,
                "kind":    "type",
                "type":    "",
                "file":    file_id,
                "lang":    "haskell",
                "service": service,
            })

    # Class instances
    for inst in re.finditer(r"^instance\s+.*?([A-Z][a-zA-Z0-9_']*)\s+([A-Z][a-zA-Z0-9_']*)", src, re.MULTILINE):
        edges.append({
            "from": f"{module}.{inst.group(2)}",
            "to":   f"{module}.{inst.group(1)}",
            "kind": "instance",
            "lang": "haskell"
        })

    return symbols, edges, body_store, call_store, log_store


# ═══════════════════════════════════════════════════════════════════════════════
# RUST PARSER
# ═══════════════════════════════════════════════════════════════════════════════

_RS_LOG_RE = re.compile(
    r'(?:log::(?:info|error|warn|debug|trace)|tracing::(?:info|error|warn|debug|span)|'
    r'info!|error!|warn!|debug!|trace!)\s*\([^;]{0,200}',
    re.MULTILINE
)

_RS_CALL_RE = re.compile(
    r'(?:'
    r'([a-z_][a-zA-Z0-9_]*)\s*\('       # direct call: fn_name(
    r'|'
    r'\.([a-z_][a-zA-Z0-9_]*)\s*\('     # method call: .method(
    r')',
)


def _extract_rust_body(src: str, fn_start: int) -> str:
    """
    Extract the body of a Rust function starting at fn_start.
    Finds the matching opening { and returns everything until the matching }.
    """
    pos = fn_start
    # Find opening brace
    brace_pos = src.find('{', pos)
    if brace_pos == -1:
        return ""

    depth = 0
    i = brace_pos
    body_chars = 0
    while i < len(src):
        ch = src[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                body = src[fn_start:i+1].strip()
                if len(body) > MAX_BODY_CHARS:
                    return body[:MAX_BODY_CHARS] + "\n// [body truncated at char limit]"
                return body
        body_chars += 1
        if body_chars > MAX_BODY_CHARS * 2:
            return src[fn_start:fn_start + MAX_BODY_CHARS] + "\n// [body truncated]"
        i += 1
    return ""


def parse_rust_file(path: pathlib.Path, service: str) -> tuple[list, list, dict, dict, dict]:
    src = path.read_text(encoding="utf-8", errors="replace")
    symbols, edges = [], []
    body_store, call_store, log_store = {}, {}, {}
    file_id = str(path.relative_to(REPO_ROOT))
    module  = str(path.relative_to(REPO_ROOT)).replace("/", "::").replace("\\", "::").removesuffix(".rs")

    for use in re.finditer(r"^use\s+([\w::{}, *]+);", src, re.MULTILINE):
        edges.append({"from": module, "to": use.group(1), "kind": "import", "lang": "rust"})

    fn_re = re.compile(
        r'(?:pub(?:\([\w]+\))?\s+)?(?:async\s+)?fn\s+([a-z_][a-zA-Z0-9_]*)\s*'
        r'(?:<[^>]*>)?\s*\(([^)]*)\)\s*(?:->\s*([^{;]+))?'
    )
    for fn_match in fn_re.finditer(src):
        name    = fn_match.group(1)
        params  = fn_match.group(2).strip()
        ret     = (fn_match.group(3) or "").strip()
        type_str = f"({params}) -> {ret}" if ret else f"({params})"
        nid     = f"{module}::{name}"

        symbols.append({
            "id":      nid,
            "name":    name,
            "module":  module,
            "kind":    "function",
            "type":    type_str,
            "file":    file_id,
            "lang":    "rust",
            "service": service,
        })

        body = _extract_rust_body(src, fn_match.start())
        if body:
            body_store[nid] = body

            called = set()
            for cm in _RS_CALL_RE.finditer(body):
                fn_called = cm.group(1) or cm.group(2)
                if fn_called and fn_called not in {name, "new", "from", "into",
                                                    "clone", "unwrap", "expect",
                                                    "map", "and_then", "ok_or",
                                                    "iter", "collect", "len",
                                                    "push", "get", "insert"}:
                    called.add(fn_called)
            call_store[nid] = {"callees": sorted(called), "callers": []}

            logs = []
            for lm in _RS_LOG_RE.finditer(body):
                logs.append(lm.group(0)[:120].strip())
            if logs:
                log_store[nid] = logs

    for decl in re.finditer(r"^(?:pub(?:\([\w]+\))?\s+)?(?:struct|enum|trait)\s+([A-Z][a-zA-Z0-9_]*)", src, re.MULTILINE):
        symbols.append({
            "id":      f"{module}::{decl.group(1)}",
            "name":    decl.group(1),
            "module":  module,
            "kind":    "type",
            "type":    "",
            "file":    file_id,
            "lang":    "rust",
            "service": service,
        })

    return symbols, edges, body_store, call_store, log_store


# ═══════════════════════════════════════════════════════════════════════════════
# PYTHON PARSER
# ═══════════════════════════════════════════════════════════════════════════════

def parse_python_file(path: pathlib.Path, service: str) -> tuple[list, list, dict, dict, dict]:
    symbols, edges = [], []
    body_store, call_store, log_store = {}, {}, {}
    file_id = str(path.relative_to(REPO_ROOT))
    module  = str(path.relative_to(REPO_ROOT)).replace("/", ".").replace("\\", ".").removesuffix(".py")
    src     = path.read_text(encoding="utf-8", errors="replace")

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return symbols, edges, body_store, call_store, log_store

    src_lines = src.splitlines()

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = node.module if isinstance(node, ast.ImportFrom) else None
            for alias in (node.names if isinstance(node, ast.Import) else [ast.alias(name=mod or "")]):
                target = mod or alias.name
                if target:
                    edges.append({"from": module, "to": target, "kind": "import", "lang": "python"})

        elif isinstance(node, ast.FunctionDef):
            args = [a.arg for a in node.args.args]
            nid  = f"{module}.{node.name}"
            symbols.append({
                "id":        nid,
                "name":      node.name,
                "module":    module,
                "kind":      "function",
                "type":      f"({', '.join(args)})",
                "file":      file_id,
                "lang":      "python",
                "service":   service,
                "docstring": ast.get_docstring(node) or "",
            })
            # Extract body as source lines
            if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                body_lines = src_lines[node.lineno - 1 : min(node.end_lineno, node.lineno + MAX_BODY_LINES)]
                body = "\n".join(body_lines)
                if body:
                    if len(body) > MAX_BODY_CHARS:
                        body_store[nid] = body[:MAX_BODY_CHARS] + "\n# [body truncated at char limit]"
                    else:
                        body_store[nid] = body

                    # Callees: find function calls in the body
                    called = set()
                    for call_node in ast.walk(node):
                        if isinstance(call_node, ast.Call):
                            if isinstance(call_node.func, ast.Name):
                                called.add(call_node.func.id)
                            elif isinstance(call_node.func, ast.Attribute):
                                called.add(call_node.func.attr)
                    call_store[nid] = {"callees": sorted(called - {node.name}), "callers": []}

        elif isinstance(node, ast.ClassDef):
            symbols.append({
                "id":        f"{module}.{node.name}",
                "name":      node.name,
                "module":    module,
                "kind":      "class",
                "type":      "",
                "file":      file_id,
                "lang":      "python",
                "service":   service,
                "docstring": ast.get_docstring(node) or "",
            })

    return symbols, edges, body_store, call_store, log_store


# ═══════════════════════════════════════════════════════════════════════════════
# GROOVY PARSER (for graphh — legacy Grails service)
# ═══════════════════════════════════════════════════════════════════════════════

_GROOVY_LOG_RE = re.compile(
    r'(?:log\.(?:info|error|warn|debug|trace))\s*\([^;]{0,200}',
    re.MULTILINE
)


def parse_groovy_file(path: pathlib.Path, service: str) -> tuple[list, list, dict, dict, dict]:
    src = path.read_text(encoding="utf-8", errors="replace")
    symbols, edges = [], []
    body_store, call_store, log_store = {}, {}, {}
    file_id = str(path.relative_to(REPO_ROOT))

    # Derive module from file path
    # e.g. graphh/grails-app/controllers/FooController.groovy → graphh.FooController
    try:
        rel = path.relative_to(REPO_ROOT / service)
        parts = [p for p in rel.parts if p not in ("grails-app", "controllers",
                                                     "services", "domain",
                                                     "src", "main", "groovy")]
        module = service + "." + ".".join(parts).removesuffix(".groovy")
    except Exception:
        module = service + "." + path.stem

    # Class declaration
    for cls in re.finditer(r'(?:class|interface)\s+([A-Z][a-zA-Z0-9_]*)', src):
        symbols.append({
            "id":      f"{module}.{cls.group(1)}",
            "name":    cls.group(1),
            "module":  module,
            "kind":    "class",
            "type":    "",
            "file":    file_id,
            "lang":    "groovy",
            "service": service,
        })

    # Method declarations: def methodName(...) or TypeName methodName(...)
    method_re = re.compile(
        r'(?:def|void|String|boolean|int|long|Map|List|Object)\s+([a-z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)',
        re.MULTILINE
    )
    for fn in method_re.finditer(src):
        name    = fn.group(1)
        params  = fn.group(2).strip()
        nid     = f"{module}.{name}"

        symbols.append({
            "id":      nid,
            "name":    name,
            "module":  module,
            "kind":    "function",
            "type":    f"({params})",
            "file":    file_id,
            "lang":    "groovy",
            "service": service,
        })

        # Try to get body
        body = _extract_rust_body(src, fn.start())  # same brace-matching logic
        if body:
            body_store[nid] = body
            logs = [lm.group(0)[:120] for lm in _GROOVY_LOG_RE.finditer(body)]
            if logs:
                log_store[nid] = logs

    # Imports
    for imp in re.finditer(r'^import\s+([\w.]+)', src, re.MULTILINE):
        edges.append({"from": module, "to": imp.group(1), "kind": "import", "lang": "groovy"})

    return symbols, edges, body_store, call_store, log_store


# ═══════════════════════════════════════════════════════════════════════════════
# GHOST DEPENDENCY DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# GIT SIGNALS
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# CALL GRAPH: REVERSE (build callers index from callees)
# ═══════════════════════════════════════════════════════════════════════════════

def build_caller_index(call_store: dict) -> dict:
    """
    For each function that has callees, add itself as a caller of each callee.
    Only resolves within same-module or fully-qualified matches.
    Returns updated call_store with `callers` populated.
    """
    # Build name -> [nid] index for fast lookup
    name_to_ids = collections.defaultdict(list)
    for nid in call_store:
        short_name = nid.split(".")[-1].split("::")[-1]
        name_to_ids[short_name].append(nid)

    caller_map = collections.defaultdict(set)  # nid -> set of caller nids

    for caller_nid, info in call_store.items():
        for callee_name in info["callees"]:
            for callee_nid in name_to_ids.get(callee_name, []):
                caller_map[callee_nid].add(caller_nid)

    # Merge back
    for nid, callers in caller_map.items():
        if nid in call_store:
            call_store[nid]["callers"] = sorted(callers)

    return call_store


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    all_symbols, all_edges = [], []
    all_bodies:  dict = {}
    all_calls:   dict = {}
    all_logs:    dict = {}
    ghost_map = collections.defaultdict(list)

    services = [p for p in REPO_ROOT.iterdir() if p.is_dir() and not p.name.startswith(".")]
    print(f"Found {len(services)} service directories: {[s.name for s in services]}")

    for service_dir in sorted(services):
        service_name = service_dir.name
        print(f"\nProcessing: {service_name}")

        before_sym = len(all_symbols)

        # Haskell
        hs_files = [
            f for f in service_dir.rglob("*.hs") if f.is_file()
            and not any(p in f.parts for p in ["dist", "dist-newstyle", ".stack-work", ".juspay"])
        ]
        for f in hs_files:
            s, e, b, c, l = parse_haskell_file(f)
            all_symbols.extend(s); all_edges.extend(e)
            all_bodies.update(b); all_calls.update(c); all_logs.update(l)
            for sym in s:
                ghosts = detect_ghosts(f)
                if ghosts:
                    ghost_map[sym["id"]].extend(ghosts)
        hs_sym = len(all_symbols) - before_sym
        hs_with_body = sum(1 for s in all_symbols[before_sym:] if s["id"] in all_bodies)
        print(f"  Haskell: {len(hs_files)} files → {hs_sym} symbols, {hs_with_body} with bodies")
        before_sym = len(all_symbols)

        # Rust
        rs_files = [
            f for f in service_dir.rglob("*.rs") if f.is_file()
            and not any(p in str(f) for p in ["target/", "/target\\", "\\target\\"])
        ]
        for f in rs_files:
            s, e, b, c, l = parse_rust_file(f, service_name)
            all_symbols.extend(s); all_edges.extend(e)
            all_bodies.update(b); all_calls.update(c); all_logs.update(l)
            for sym in s:
                ghosts = detect_ghosts(f)
                if ghosts:
                    ghost_map[sym["id"]].extend(ghosts)
        rs_sym = len(all_symbols) - before_sym
        rs_with_body = sum(1 for s in all_symbols[before_sym:] if s["id"] in all_bodies)
        print(f"  Rust:    {len(rs_files)} files → {rs_sym} symbols, {rs_with_body} with bodies")
        before_sym = len(all_symbols)

        # Groovy
        groovy_files = [
            f for f in service_dir.rglob("*.groovy") if f.is_file()
        ]
        groovy_files += [
            f for f in service_dir.rglob("*.grails") if f.is_file()
        ]
        for f in groovy_files:
            s, e, b, c, l = parse_groovy_file(f, service_name)
            all_symbols.extend(s); all_edges.extend(e)
            all_bodies.update(b); all_calls.update(c); all_logs.update(l)
        gv_sym = len(all_symbols) - before_sym
        if gv_sym > 0:
            print(f"  Groovy:  {len(groovy_files)} files → {gv_sym} symbols")
        before_sym = len(all_symbols)

        # Python
        py_files = [
            f for f in service_dir.rglob("*.py") if f.is_file()
            and not any(p in str(f) for p in ["venv/", "__pycache__", ".venv/"])
        ]
        for f in py_files:
            s, e, b, c, l = parse_python_file(f, service_name)
            all_symbols.extend(s); all_edges.extend(e)
            all_bodies.update(b); all_calls.update(c); all_logs.update(l)
            for sym in s:
                ghosts = detect_ghosts(f)
                if ghosts:
                    ghost_map[sym["id"]].extend(ghosts)
        py_sym = len(all_symbols) - before_sym
        if py_sym > 0:
            print(f"  Python:  {len(py_files)} files → {py_sym} symbols")

    # Attach ghost deps
    for sym in all_symbols:
        sym["ghost_deps"] = list(set(ghost_map.get(sym["id"], [])))

    # Git signals
    print("\nExtracting git signals...")
    git_data = extract_git_signals(REPO_ROOT)
    all_edges.extend(git_data["co_change_edges"])
    file_history = git_data["file_history"]
    for sym in all_symbols:
        sym["commit_history"] = file_history.get(sym.get("file", ""), [])[:5]

    # Build caller index
    print("Building caller index...")
    all_calls = build_caller_index(all_calls)

    # Deduplicate symbols by id
    seen_ids: dict = {}
    deduped = []
    for s in all_symbols:
        if s["id"] not in seen_ids:
            seen_ids[s["id"]] = True
            deduped.append(s)

    # Add call_edges to edges list (for graph analysis)
    for caller_nid, info in all_calls.items():
        for callee_name in info["callees"]:
            all_edges.append({
                "from": caller_nid,
                "to":   callee_name,
                "kind": "calls",
            })

    # ── Write outputs ──────────────────────────────────────────────────────────
    result = {
        "nodes": deduped,
        "edges": all_edges,
        "stats": {
            "total_symbols":   len(deduped),
            "total_edges":     len(all_edges),
            "with_body":       len(all_bodies),
            "with_calls":      len(all_calls),
            "with_log_patterns": len(all_logs),
            "services":        [s.name for s in services]
        }
    }

    raw_path = OUT_DIR / "raw_graph.json"
    raw_path.write_text(json.dumps(result, indent=2))
    print(f"\n✓ raw_graph.json: {len(deduped)} symbols, {len(all_edges)} edges")

    body_path = OUT_DIR / "body_store.json"
    body_path.write_text(json.dumps(all_bodies, indent=2))
    print(f"✓ body_store.json: {len(all_bodies)} function bodies")

    call_path = OUT_DIR / "call_graph.json"
    call_path.write_text(json.dumps(all_calls, indent=2))
    print(f"✓ call_graph.json: {len(all_calls)} entries (callees + callers)")

    log_path = OUT_DIR / "log_patterns.json"
    log_path.write_text(json.dumps(all_logs, indent=2))
    print(f"✓ log_patterns.json: {len(all_logs)} functions with log patterns")

    print(f"\nBody extraction rate: {len(all_bodies)/max(1,len(deduped))*100:.1f}%")
    print(f"Services: {result['stats']['services']}")


if __name__ == "__main__":
    main()
