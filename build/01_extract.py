"""
Stage 1 — Extract symbols, bodies, call graph, log patterns, and edges.
Supports: Haskell (.hs), Rust (.rs), Python (.py), Groovy (.groovy/.grails)
Output:  $OUTPUT_DIR/raw_graph.json
         $OUTPUT_DIR/body_store.json      (fn_id → body text)
         $OUTPUT_DIR/call_graph.json      (fn_id → {callees, callers})
         $OUTPUT_DIR/log_patterns.json    (fn_id → [log patterns])

Env vars:
  REPO_ROOT    — path to workspace source/ dir
  OUTPUT_DIR   — path to workspace output/ dir
  CONFIG_PATH  — path to config.yaml (optional; falls back to WORKSPACE_DIR/config.yaml)
  WORKSPACE_DIR — path to workspace root (optional)
"""
import re, ast, json, subprocess, collections, pathlib, sys, os, textwrap
import multiprocessing as _mp

# ── Tree-sitter imports (graceful fallback if not installed) ─────────────────
try:
    import tree_sitter_haskell
    from tree_sitter import Language, Parser
    HS_LANGUAGE = Language(tree_sitter_haskell.language())
    _TS_HASKELL = True
except Exception as _e:
    print(f"[warn] tree-sitter-haskell unavailable ({_e}); falling back to regex")
    _TS_HASKELL = False

try:
    import tree_sitter_rust
    from tree_sitter import Language, Parser
    RS_LANGUAGE = Language(tree_sitter_rust.language())
    _TS_RUST = True
except Exception as _e:
    print(f"[warn] tree-sitter-rust unavailable ({_e}); falling back to regex")
    _TS_RUST = False

try:
    import tree_sitter_groovy
    from tree_sitter import Language, Parser
    GV_LANGUAGE = Language(tree_sitter_groovy.language())
    _TS_GROOVY = True
except Exception as _e:
    print(f"[warn] tree-sitter-groovy unavailable ({_e}); falling back to regex")
    _TS_GROOVY = False

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = pathlib.Path(os.environ.get("REPO_ROOT", "workspaces/source"))
OUT_DIR   = pathlib.Path(os.environ.get("OUTPUT_DIR", "workspaces/output"))
OUT_DIR.mkdir(exist_ok=True, parents=True)

MAX_BODY_CHARS = 8000   # truncate very long bodies to keep memory sane
MAX_BODY_LINES = 200    # ~5 screens of code

# ── Config (service profiles) ────────────────────────────────────────────────
def _load_config() -> dict:
    """Load config.yaml; return empty dict on any failure."""
    candidates = []
    if os.environ.get("CONFIG_PATH"):
        candidates.append(pathlib.Path(os.environ["CONFIG_PATH"]))
    if os.environ.get("WORKSPACE_DIR"):
        candidates.append(pathlib.Path(os.environ["WORKSPACE_DIR"]) / "config.yaml")
    for p in candidates:
        if p.exists():
            try:
                import yaml  # type: ignore
                return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            except Exception as e:
                print(f"[warn] Could not load config {p}: {e}")
    return {}

_CONFIG = _load_config()
_SERVICE_PROFILES: dict = _CONFIG.get("service_profiles", {})

def _service_meta(service: str) -> tuple[str, float]:
    """Return (service_role, traffic_weight) for a service, with defaults."""
    prof = _SERVICE_PROFILES.get(service, {})
    return prof.get("role", ""), float(prof.get("traffic_weight", 1.0))


# ── Tree-sitter helper ───────────────────────────────────────────────────────
def _find_nodes(root, node_type: str) -> list:
    results = []
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type == node_type:
            results.append(n)
        stack.extend(reversed(n.children))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# HASKELL PARSER  (tree-sitter primary, regex fallback)
# ═══════════════════════════════════════════════════════════════════════════════

_HS_LOG_RE = re.compile(
    r'(?:logInfo|logError|logWarn|logDebug|Logger\.log|L\.log|runIO\s+\$\s+log)\s*'
    r'(?:\'[A-Z\w]+\')?\s*'
    r'(?:\$\s*)?'
    r'("(?:[^"\\]|\\.)*"|[A-Z_][A-Z_0-9]{3,})',
    re.MULTILINE
)

_HS_CALL_RE = re.compile(
    r'(?<![A-Z.\'"#])(?<![a-z0-9_])'
    r'([a-z_][a-zA-Z0-9_\']*)'
    r'(?![a-zA-Z0-9_\'"=<>])',
)

_HS_CALL_SKIP = frozenset({
    "do", "let", "in", "where", "case", "of", "if", "then", "else",
    "return", "pure", "when", "unless", "void", "fmap", "map", "filter",
    "foldr", "foldl", "mapM", "mapM_", "forM", "forM_", "sequence",
    "catch", "handle", "try", "throwIO", "evaluate", "liftIO",
})


def _truncate_body(body: str, comment_prefix: str = "--") -> str:
    if len(body) > MAX_BODY_CHARS:
        return body[:MAX_BODY_CHARS] + f"\n{comment_prefix} [truncated]"
    lines = body.splitlines(keepends=True)
    if len(lines) > MAX_BODY_LINES:
        return "".join(lines[:MAX_BODY_LINES]) + f"\n{comment_prefix} [truncated]"
    return body


def _hs_extract_calls_logs(body: str, name: str) -> tuple[list, list]:
    called = set()
    for cm in _HS_CALL_RE.finditer(body):
        fn = cm.group(1)
        if fn != name and fn not in _HS_CALL_SKIP:
            called.add(fn)
    logs = [lm.group(1).strip('"') for lm in _HS_LOG_RE.finditer(body)]
    return sorted(called), logs


def _parse_haskell_ts(path: pathlib.Path, src: str, src_bytes: bytes,
                      module: str, file_id: str, service: str,
                      service_role: str, traffic_weight: float
                      ) -> tuple[list, list, dict, dict, dict]:
    """Tree-sitter Haskell parser."""
    symbols, edges = [], []
    body_store, call_store, log_store = {}, {}, {}

    parser = Parser(HS_LANGUAGE)
    tree = parser.parse(src_bytes)
    root = tree.root_node

    # Imports
    for imp_node in _find_nodes(root, "import"):
        mod_node = imp_node.child_by_field_name("module")
        if mod_node:
            edges.append({
                "from": module,
                "to":   mod_node.text.decode("utf-8", errors="replace"),
                "kind": "import",
                "lang": "haskell",
            })

    seen_ids: set = set()

    # Type signatures → function symbols
    for sig_node in _find_nodes(root, "signature"):
        # The variable names appear before '::'; collect all variable children
        names = []
        type_parts = []
        past_colons = False
        for child in sig_node.children:
            text = child.text.decode("utf-8", errors="replace").strip()
            if child.type in ("::", "::"):
                past_colons = True
                continue
            if text == "::":
                past_colons = True
                continue
            if not past_colons:
                if child.type == "variable":
                    names.append(text)
                elif child.type == "operator" and text == "::":
                    past_colons = True
            else:
                type_parts.append(text)
        type_str = " ".join(type_parts).strip()

        for name in names:
            nid = f"{module}.{name}"
            if nid in seen_ids:
                continue
            seen_ids.add(nid)
            symbols.append({
                "id":             nid,
                "name":           name,
                "module":         module,
                "kind":           "function",
                "type":           type_str,
                "file":           file_id,
                "lang":           "haskell",
                "service":        service,
                "service_role":   service_role,
                "traffic_weight": traffic_weight,
                "ghost_deps":     [],
                "commit_history": [],
                "docstring":      "",
            })

    # Function definitions — extract bodies using byte offsets
    for fn_node in _find_nodes(root, "function"):
        name_node = fn_node.child_by_field_name("name")
        if name_node is None:
            # fallback: first variable child
            for child in fn_node.children:
                if child.type == "variable":
                    name_node = child
                    break
        if name_node is None:
            continue
        name = name_node.text.decode("utf-8", errors="replace").strip()
        nid = f"{module}.{name}"

        # Ensure symbol exists even without a preceding signature
        if nid not in seen_ids:
            seen_ids.add(nid)
            symbols.append({
                "id":             nid,
                "name":           name,
                "module":         module,
                "kind":           "function",
                "type":           "",
                "file":           file_id,
                "lang":           "haskell",
                "service":        service,
                "service_role":   service_role,
                "traffic_weight": traffic_weight,
                "ghost_deps":     [],
                "commit_history": [],
                "docstring":      "",
            })

        if nid not in body_store:
            body_bytes = src_bytes[fn_node.start_byte:fn_node.end_byte]
            body = body_bytes.decode("utf-8", errors="replace")
            body = _truncate_body(body, "--")
            if body:
                body_store[nid] = body
                called, logs = _hs_extract_calls_logs(body, name)
                call_store[nid] = {"callees": called, "callers": []}
                if logs:
                    log_store[nid] = logs

    # Data / newtype / type aliases
    for node_type in ("data_type", "type_alias", "newtype"):
        for decl_node in _find_nodes(root, node_type):
            name_node = decl_node.child_by_field_name("name")
            if name_node is None:
                for child in decl_node.children:
                    if child.type == "constructor" or (child.type[0].isupper() and len(child.type) > 1):
                        name_node = child
                        break
                    if child.type == "name":
                        name_node = child
                        break
            if name_node is None:
                continue
            name = name_node.text.decode("utf-8", errors="replace").strip()
            if not name or not name[0].isupper():
                continue
            nid = f"{module}.{name}"
            if nid not in seen_ids:
                seen_ids.add(nid)
                symbols.append({
                    "id":             nid,
                    "name":           name,
                    "module":         module,
                    "kind":           "type",
                    "type":           "",
                    "file":           file_id,
                    "lang":           "haskell",
                    "service":        service,
                    "service_role":   service_role,
                    "traffic_weight": traffic_weight,
                    "ghost_deps":     [],
                    "commit_history": [],
                    "docstring":      "",
                })

    # Fallback: also capture data/newtype/type via regex for names tree-sitter misses
    for decl in re.finditer(r"^(?:data|newtype|type)\s+([A-Z][a-zA-Z0-9_']*)", src, re.MULTILINE):
        name = decl.group(1)
        nid = f"{module}.{name}"
        if nid not in seen_ids:
            seen_ids.add(nid)
            symbols.append({
                "id":             nid,
                "name":           name,
                "module":         module,
                "kind":           "type",
                "type":           "",
                "file":           file_id,
                "lang":           "haskell",
                "service":        service,
                "service_role":   service_role,
                "traffic_weight": traffic_weight,
                "ghost_deps":     [],
                "commit_history": [],
                "docstring":      "",
            })

    # Class instances → edges
    for inst in re.finditer(
        r"^instance\s+.*?([A-Z][a-zA-Z0-9_']*)\s+([A-Z][a-zA-Z0-9_']*)", src, re.MULTILINE
    ):
        edges.append({
            "from": f"{module}.{inst.group(2)}",
            "to":   f"{module}.{inst.group(1)}",
            "kind": "instance",
            "lang": "haskell",
        })

    return symbols, edges, body_store, call_store, log_store


def _parse_haskell_regex(path: pathlib.Path, src: str,
                         module: str, file_id: str, service: str,
                         service_role: str, traffic_weight: float
                         ) -> tuple[list, list, dict, dict, dict]:
    """Pure-regex Haskell parser (fallback)."""
    symbols, edges = [], []
    body_store, call_store, log_store = {}, {}, {}

    for imp in re.finditer(r"^import\s+(?:qualified\s+)?([\w.]+)", src, re.MULTILINE):
        edges.append({"from": module, "to": imp.group(1), "kind": "import", "lang": "haskell"})

    seen_ids: set = set()
    top_level_re = re.compile(r'^[a-z_][a-zA-Z0-9_\']*\s*(?:[^=\n]*=|::)')

    for sig in re.finditer(
        r"^([a-z_][a-zA-Z0-9_']*(?:,\s*[a-z_][a-zA-Z0-9_']*)*)\s*::\s*(.+)",
        src, re.MULTILINE
    ):
        names_raw = sig.group(1)
        type_str  = sig.group(2).strip()
        sig_pos   = sig.start()

        for name in [n.strip() for n in names_raw.split(",")]:
            nid = f"{module}.{name}"
            if nid in seen_ids:
                continue
            seen_ids.add(nid)
            symbols.append({
                "id":             nid,
                "name":           name,
                "module":         module,
                "kind":           "function",
                "type":           type_str,
                "file":           file_id,
                "lang":           "haskell",
                "service":        service,
                "service_role":   service_role,
                "traffic_weight": traffic_weight,
                "ghost_deps":     [],
                "commit_history": [],
                "docstring":      "",
            })

            # Body extraction (regex — starts after the sig line)
            lines = src[sig_pos:].splitlines(keepends=True)
            body_lines = []
            in_body = False
            for i, line in enumerate(lines):
                if i == 0 and re.match(rf'^{re.escape(name)}\s*::', line):
                    continue
                if not in_body and re.match(rf'^{re.escape(name)}\b', line):
                    in_body = True
                if in_body:
                    if i > 0 and top_level_re.match(line) and not re.match(rf'^{re.escape(name)}\b', line):
                        break
                    body_lines.append(line)
                    if len(body_lines) >= MAX_BODY_LINES:
                        body_lines.append("  -- [truncated]\n")
                        break
            body = "".join(body_lines).strip()
            if len(body) > MAX_BODY_CHARS:
                body = body[:MAX_BODY_CHARS] + "\n  -- [truncated]"
            if body:
                body_store[nid] = body
                called, logs = _hs_extract_calls_logs(body, name)
                call_store[nid] = {"callees": called, "callers": []}
                if logs:
                    log_store[nid] = logs

    for decl in re.finditer(r"^(?:data|newtype|type)\s+([A-Z][a-zA-Z0-9_']*)", src, re.MULTILINE):
        nid = f"{module}.{decl.group(1)}"
        if nid not in seen_ids:
            seen_ids.add(nid)
            symbols.append({
                "id":             nid,
                "name":           decl.group(1),
                "module":         module,
                "kind":           "type",
                "type":           "",
                "file":           file_id,
                "lang":           "haskell",
                "service":        service,
                "service_role":   service_role,
                "traffic_weight": traffic_weight,
                "ghost_deps":     [],
                "commit_history": [],
                "docstring":      "",
            })

    for inst in re.finditer(
        r"^instance\s+.*?([A-Z][a-zA-Z0-9_']*)\s+([A-Z][a-zA-Z0-9_']*)", src, re.MULTILINE
    ):
        edges.append({
            "from": f"{module}.{inst.group(2)}",
            "to":   f"{module}.{inst.group(1)}",
            "kind": "instance",
            "lang": "haskell",
        })

    return symbols, edges, body_store, call_store, log_store


def parse_haskell_file(path: pathlib.Path) -> tuple[list, list, dict, dict, dict]:
    """Returns: (symbols, edges, body_store, call_store, log_store)"""
    src_bytes = path.read_bytes()
    src = src_bytes.decode("utf-8", errors="replace")

    m = re.search(r"^module\s+([\w.]+)", src, re.MULTILINE)
    module = m.group(1) if m else str(path.stem)
    file_id = str(path.relative_to(REPO_ROOT))
    service = path.parts[len(REPO_ROOT.parts)] if len(path.parts) > len(REPO_ROOT.parts) else "unknown"
    service_role, traffic_weight = _service_meta(service)

    if _TS_HASKELL:
        try:
            return _parse_haskell_ts(path, src, src_bytes, module, file_id,
                                     service, service_role, traffic_weight)
        except Exception as e:
            print(f"  [warn] tree-sitter Haskell failed for {path.name}: {e}; using regex")

    return _parse_haskell_regex(path, src, module, file_id,
                                service, service_role, traffic_weight)


# ═══════════════════════════════════════════════════════════════════════════════
# RUST PARSER  (tree-sitter primary, regex fallback)
# ═══════════════════════════════════════════════════════════════════════════════

_RS_LOG_RE = re.compile(
    r'(?:log::(?:info|error|warn|debug|trace)|tracing::(?:info|error|warn|debug|span)|'
    r'info!|error!|warn!|debug!|trace!)\s*\([^;]{0,200}',
    re.MULTILINE
)

_RS_CALL_SKIP = frozenset({
    "new", "from", "into", "clone", "unwrap", "expect",
    "map", "and_then", "ok_or", "iter", "collect", "len",
    "push", "get", "insert",
})


def _rs_body_from_node(src_bytes: bytes, node) -> str:
    body_bytes = src_bytes[node.start_byte:node.end_byte]
    body = body_bytes.decode("utf-8", errors="replace")
    return _truncate_body(body, "//")


def _rs_extract_calls_logs(body: str, name: str) -> tuple[list, list]:
    called = set()
    for cm in re.finditer(
        r'(?:([a-z_][a-zA-Z0-9_]*)\s*\(|\.([a-z_][a-zA-Z0-9_]*)\s*\()', body
    ):
        fn = cm.group(1) or cm.group(2)
        if fn and fn != name and fn not in _RS_CALL_SKIP:
            called.add(fn)
    logs = [lm.group(0)[:120].strip() for lm in _RS_LOG_RE.finditer(body)]
    return sorted(called), logs


def _parse_rust_ts(path: pathlib.Path, src: str, src_bytes: bytes,
                   module: str, file_id: str, service: str,
                   service_role: str, traffic_weight: float
                   ) -> tuple[list, list, dict, dict, dict]:
    """Tree-sitter Rust parser."""
    symbols, edges = [], []
    body_store, call_store, log_store = {}, {}, {}

    parser = Parser(RS_LANGUAGE)
    tree = parser.parse(src_bytes)
    root = tree.root_node

    # Imports
    for use_node in _find_nodes(root, "use_declaration"):
        edges.append({
            "from": module,
            "to":   use_node.text.decode("utf-8", errors="replace").strip().lstrip("use ").rstrip(";"),
            "kind": "import",
            "lang": "rust",
        })

    seen_ids: set = set()

    # Functions
    for fn_node in _find_nodes(root, "function_item"):
        name_node = fn_node.child_by_field_name("name")
        if name_node is None:
            continue
        name = name_node.text.decode("utf-8", errors="replace").strip()

        params_node = fn_node.child_by_field_name("parameters")
        params = params_node.text.decode("utf-8", errors="replace").strip() if params_node else ""

        ret_node = fn_node.child_by_field_name("return_type")
        ret = ret_node.text.decode("utf-8", errors="replace").strip() if ret_node else ""
        type_str = f"{params} -> {ret}".strip(" ->") if ret else params

        nid = f"{module}::{name}"
        if nid in seen_ids:
            continue
        seen_ids.add(nid)

        symbols.append({
            "id":             nid,
            "name":           name,
            "module":         module,
            "kind":           "function",
            "type":           type_str,
            "file":           file_id,
            "lang":           "rust",
            "service":        service,
            "service_role":   service_role,
            "traffic_weight": traffic_weight,
            "ghost_deps":     [],
            "commit_history": [],
            "docstring":      "",
        })

        body = _rs_body_from_node(src_bytes, fn_node)
        if body:
            body_store[nid] = body
            called, logs = _rs_extract_calls_logs(body, name)
            call_store[nid] = {"callees": called, "callers": []}
            if logs:
                log_store[nid] = logs

    # Types: struct, enum, trait
    for type_node_type in ("struct_item", "enum_item", "trait_item"):
        for decl_node in _find_nodes(root, type_node_type):
            name_node = decl_node.child_by_field_name("name")
            if name_node is None:
                continue
            name = name_node.text.decode("utf-8", errors="replace").strip()
            nid = f"{module}::{name}"
            if nid not in seen_ids:
                seen_ids.add(nid)
                symbols.append({
                    "id":             nid,
                    "name":           name,
                    "module":         module,
                    "kind":           "type",
                    "type":           "",
                    "file":           file_id,
                    "lang":           "rust",
                    "service":        service,
                    "service_role":   service_role,
                    "traffic_weight": traffic_weight,
                    "ghost_deps":     [],
                    "commit_history": [],
                    "docstring":      "",
                })

    return symbols, edges, body_store, call_store, log_store


def _extract_rust_body_regex(src: str, fn_start: int) -> str:
    """Brace-matching body extraction for Rust/Groovy regex fallback."""
    brace_pos = src.find('{', fn_start)
    if brace_pos == -1:
        return ""
    depth = 0
    i = brace_pos
    while i < len(src):
        ch = src[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                body = src[fn_start:i+1].strip()
                if len(body) > MAX_BODY_CHARS:
                    return body[:MAX_BODY_CHARS] + "\n// [truncated]"
                return body
        if i - fn_start > MAX_BODY_CHARS * 2:
            return src[fn_start:fn_start + MAX_BODY_CHARS] + "\n// [truncated]"
        i += 1
    return ""


def _parse_rust_regex(path: pathlib.Path, src: str,
                      module: str, file_id: str, service: str,
                      service_role: str, traffic_weight: float
                      ) -> tuple[list, list, dict, dict, dict]:
    """Pure-regex Rust parser (fallback)."""
    symbols, edges = [], []
    body_store, call_store, log_store = {}, {}, {}

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
            "id":             nid,
            "name":           name,
            "module":         module,
            "kind":           "function",
            "type":           type_str,
            "file":           file_id,
            "lang":           "rust",
            "service":        service,
            "service_role":   service_role,
            "traffic_weight": traffic_weight,
            "ghost_deps":     [],
            "commit_history": [],
            "docstring":      "",
        })

        body = _extract_rust_body_regex(src, fn_match.start())
        if body:
            body_store[nid] = body
            called, logs = _rs_extract_calls_logs(body, name)
            call_store[nid] = {"callees": called, "callers": []}
            if logs:
                log_store[nid] = logs

    for decl in re.finditer(
        r"^(?:pub(?:\([\w]+\))?\s+)?(?:struct|enum|trait)\s+([A-Z][a-zA-Z0-9_]*)",
        src, re.MULTILINE
    ):
        symbols.append({
            "id":             f"{module}::{decl.group(1)}",
            "name":           decl.group(1),
            "module":         module,
            "kind":           "type",
            "type":           "",
            "file":           file_id,
            "lang":           "rust",
            "service":        service,
            "service_role":   service_role,
            "traffic_weight": traffic_weight,
            "ghost_deps":     [],
            "commit_history": [],
            "docstring":      "",
        })

    return symbols, edges, body_store, call_store, log_store


def parse_rust_file(path: pathlib.Path, service: str) -> tuple[list, list, dict, dict, dict]:
    src_bytes = path.read_bytes()
    src = src_bytes.decode("utf-8", errors="replace")
    file_id = str(path.relative_to(REPO_ROOT))
    module  = file_id.replace("/", "::").replace("\\", "::").removesuffix(".rs")
    service_role, traffic_weight = _service_meta(service)

    if _TS_RUST:
        try:
            return _parse_rust_ts(path, src, src_bytes, module, file_id,
                                  service, service_role, traffic_weight)
        except Exception as e:
            print(f"  [warn] tree-sitter Rust failed for {path.name}: {e}; using regex")

    return _parse_rust_regex(path, src, module, file_id,
                             service, service_role, traffic_weight)


# ═══════════════════════════════════════════════════════════════════════════════
# PYTHON PARSER  (ast module — already correct, kept as-is with minor fixes)
# ═══════════════════════════════════════════════════════════════════════════════

def parse_python_file(path: pathlib.Path, service: str) -> tuple[list, list, dict, dict, dict]:
    symbols, edges = [], []
    body_store, call_store, log_store = {}, {}, {}
    file_id = str(path.relative_to(REPO_ROOT))
    module  = file_id.replace("/", ".").replace("\\", ".").removesuffix(".py")
    src     = path.read_text(encoding="utf-8", errors="replace")
    service_role, traffic_weight = _service_meta(service)

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return symbols, edges, body_store, call_store, log_store

    src_lines = src.splitlines()

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = node.module if isinstance(node, ast.ImportFrom) else None
            for alias in (node.names if isinstance(node, ast.Import)
                          else [ast.alias(name=mod or "")]):
                target = mod or alias.name
                if target:
                    edges.append({"from": module, "to": target, "kind": "import", "lang": "python"})

        elif isinstance(node, ast.FunctionDef):
            args = [a.arg for a in node.args.args]
            nid  = f"{module}.{node.name}"
            symbols.append({
                "id":             nid,
                "name":           node.name,
                "module":         module,
                "kind":           "function",
                "type":           f"({', '.join(args)})",
                "file":           file_id,
                "lang":           "python",
                "service":        service,
                "service_role":   service_role,
                "traffic_weight": traffic_weight,
                "ghost_deps":     [],
                "commit_history": [],
                "docstring":      ast.get_docstring(node) or "",
            })
            if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                body_lines = src_lines[node.lineno - 1 : min(node.end_lineno, node.lineno + MAX_BODY_LINES)]
                body = "\n".join(body_lines)
                if body:
                    if len(body) > MAX_BODY_CHARS:
                        body = body[:MAX_BODY_CHARS] + "\n# [truncated]"
                    body_store[nid] = body

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
                "id":             f"{module}.{node.name}",
                "name":           node.name,
                "module":         module,
                "kind":           "class",
                "type":           "",
                "file":           file_id,
                "lang":           "python",
                "service":        service,
                "service_role":   service_role,
                "traffic_weight": traffic_weight,
                "ghost_deps":     [],
                "commit_history": [],
                "docstring":      ast.get_docstring(node) or "",
            })

    return symbols, edges, body_store, call_store, log_store


# ═══════════════════════════════════════════════════════════════════════════════
# GROOVY PARSER  (tree-sitter primary, regex fallback)
# ═══════════════════════════════════════════════════════════════════════════════

_GROOVY_LOG_RE = re.compile(
    r'(?:log\.(?:info|error|warn|debug|trace))\s*\([^;]{0,200}',
    re.MULTILINE
)

_GROOVY_SKIP_PARTS = frozenset({
    "grails-app", "controllers", "services", "domain", "src", "main", "groovy"
})


def _groovy_module(path: pathlib.Path, service: str) -> str:
    try:
        rel = path.relative_to(REPO_ROOT / service)
        parts = [p for p in rel.parts if p not in _GROOVY_SKIP_PARTS]
        return service + "." + ".".join(parts).removesuffix(".groovy")
    except Exception:
        return service + "." + path.stem


def _parse_groovy_ts(path: pathlib.Path, src: str, src_bytes: bytes,
                     module: str, file_id: str, service: str,
                     service_role: str, traffic_weight: float
                     ) -> tuple[list, list, dict, dict, dict]:
    """Tree-sitter Groovy parser."""
    symbols, edges = [], []
    body_store, call_store, log_store = {}, {}, {}

    parser = Parser(GV_LANGUAGE)
    tree = parser.parse(src_bytes)
    root = tree.root_node

    # Imports
    for imp_node in _find_nodes(root, "import_declaration"):
        imp_text = imp_node.text.decode("utf-8", errors="replace").strip()
        imp_text = re.sub(r'^import\s+', '', imp_text).rstrip(";")
        edges.append({"from": module, "to": imp_text, "kind": "import", "lang": "groovy"})

    seen_ids: set = set()

    # Classes
    for cls_node in _find_nodes(root, "class_declaration"):
        name_node = cls_node.child_by_field_name("name")
        if name_node is None:
            for child in cls_node.children:
                if child.type == "identifier":
                    name_node = child
                    break
        if name_node is None:
            continue
        name = name_node.text.decode("utf-8", errors="replace").strip()
        nid = f"{module}.{name}"
        if nid not in seen_ids:
            seen_ids.add(nid)
            symbols.append({
                "id":             nid,
                "name":           name,
                "module":         module,
                "kind":           "class",
                "type":           "",
                "file":           file_id,
                "lang":           "groovy",
                "service":        service,
                "service_role":   service_role,
                "traffic_weight": traffic_weight,
                "ghost_deps":     [],
                "commit_history": [],
                "docstring":      "",
            })

    # Methods
    for meth_node in _find_nodes(root, "method_declaration"):
        name_node = meth_node.child_by_field_name("name")
        if name_node is None:
            for child in meth_node.children:
                if child.type == "identifier":
                    name_node = child
                    break
        if name_node is None:
            continue
        name = name_node.text.decode("utf-8", errors="replace").strip()
        nid = f"{module}.{name}"
        if nid in seen_ids:
            continue
        seen_ids.add(nid)

        body_bytes = src_bytes[meth_node.start_byte:meth_node.end_byte]
        body = body_bytes.decode("utf-8", errors="replace")
        body = _truncate_body(body, "//")

        symbols.append({
            "id":             nid,
            "name":           name,
            "module":         module,
            "kind":           "function",
            "type":           "",
            "file":           file_id,
            "lang":           "groovy",
            "service":        service,
            "service_role":   service_role,
            "traffic_weight": traffic_weight,
            "ghost_deps":     [],
            "commit_history": [],
            "docstring":      "",
        })
        if body:
            body_store[nid] = body
            logs = [lm.group(0)[:120] for lm in _GROOVY_LOG_RE.finditer(body)]
            if logs:
                log_store[nid] = logs

    return symbols, edges, body_store, call_store, log_store


def _parse_groovy_regex(path: pathlib.Path, src: str,
                        module: str, file_id: str, service: str,
                        service_role: str, traffic_weight: float
                        ) -> tuple[list, list, dict, dict, dict]:
    """Pure-regex Groovy parser (fallback)."""
    symbols, edges = [], []
    body_store, call_store, log_store = {}, {}, {}

    for cls in re.finditer(r'(?:class|interface)\s+([A-Z][a-zA-Z0-9_]*)', src):
        symbols.append({
            "id":             f"{module}.{cls.group(1)}",
            "name":           cls.group(1),
            "module":         module,
            "kind":           "class",
            "type":           "",
            "file":           file_id,
            "lang":           "groovy",
            "service":        service,
            "service_role":   service_role,
            "traffic_weight": traffic_weight,
            "ghost_deps":     [],
            "commit_history": [],
            "docstring":      "",
        })

    method_re = re.compile(
        r'(?:def|void|String|boolean|int|long|Map|List|Object)\s+'
        r'([a-z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)',
        re.MULTILINE
    )
    for fn in method_re.finditer(src):
        name   = fn.group(1)
        params = fn.group(2).strip()
        nid    = f"{module}.{name}"
        symbols.append({
            "id":             nid,
            "name":           name,
            "module":         module,
            "kind":           "function",
            "type":           f"({params})",
            "file":           file_id,
            "lang":           "groovy",
            "service":        service,
            "service_role":   service_role,
            "traffic_weight": traffic_weight,
            "ghost_deps":     [],
            "commit_history": [],
            "docstring":      "",
        })
        body = _extract_rust_body_regex(src, fn.start())
        if body:
            body_store[nid] = body
            logs = [lm.group(0)[:120] for lm in _GROOVY_LOG_RE.finditer(body)]
            if logs:
                log_store[nid] = logs

    for imp in re.finditer(r'^import\s+([\w.]+)', src, re.MULTILINE):
        edges.append({"from": module, "to": imp.group(1), "kind": "import", "lang": "groovy"})

    return symbols, edges, body_store, call_store, log_store


def parse_groovy_file(path: pathlib.Path, service: str) -> tuple[list, list, dict, dict, dict]:
    src_bytes = path.read_bytes()
    src = src_bytes.decode("utf-8", errors="replace")
    file_id = str(path.relative_to(REPO_ROOT))
    module = _groovy_module(path, service)
    service_role, traffic_weight = _service_meta(service)

    if _TS_GROOVY:
        try:
            return _parse_groovy_ts(path, src, src_bytes, module, file_id,
                                    service, service_role, traffic_weight)
        except Exception as e:
            print(f"  [warn] tree-sitter Groovy failed for {path.name}: {e}; using regex")

    return _parse_groovy_regex(path, src, module, file_id,
                               service, service_role, traffic_weight)


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


def detect_ghosts(path: pathlib.Path) -> list:
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

    co_change: collections.Counter = collections.Counter()
    for c in commits:
        files = c["files"]
        for i, f1 in enumerate(files):
            for f2 in files[i+1:]:
                co_change[tuple(sorted([f1, f2]))] += 1

    file_history: dict = collections.defaultdict(list)
    for c in commits:
        for f in c["files"]:
            file_history[f].append({
                "sha":  c["sha"][:7],
                "msg":  c["msg"],
                "date": c["date"][:10],
            })

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
    name_to_ids: dict = collections.defaultdict(list)
    for nid in call_store:
        short_name = nid.split(".")[-1].split("::")[-1]
        name_to_ids[short_name].append(nid)

    caller_map: dict = collections.defaultdict(set)
    for caller_nid, info in call_store.items():
        for callee_name in info["callees"]:
            for callee_nid in name_to_ids.get(callee_name, []):
                caller_map[callee_nid].add(caller_nid)

    for nid, callers in caller_map.items():
        if nid in call_store:
            call_store[nid]["callers"] = sorted(callers)

    return call_store


# ═══════════════════════════════════════════════════════════════════════════════
# PER-SERVICE WORKER  (runs in a pool worker process)
# ═══════════════════════════════════════════════════════════════════════════════

def _process_service_dir(service_dir: pathlib.Path) -> tuple:
    """Process one service directory in a worker process.

    Returns:
        (service_name, symbols, edges, body_store, call_store, log_store, ghost_map_dict)
    """
    service_name = service_dir.name
    symbols: list = []
    edges:   list = []
    body_store: dict = {}
    call_store: dict = {}
    log_store:  dict = {}
    ghost_map: dict = collections.defaultdict(list)

    before = 0

    # ── Haskell ─────────────────────────────────────────────────────────────
    hs_files = [
        f for f in service_dir.rglob("*.hs") if f.is_file()
        and not any(p in f.parts for p in ["dist", "dist-newstyle", ".stack-work", ".juspay"])
    ]
    for f in hs_files:
        try:
            s, e, b, c, l = parse_haskell_file(f)
            symbols.extend(s); edges.extend(e)
            body_store.update(b); call_store.update(c); log_store.update(l)
            ghosts = detect_ghosts(f)
            if ghosts:
                for sym in s:
                    ghost_map[sym["id"]].extend(ghosts)
        except Exception as exc:
            print(f"  [error] {f}: {exc}", flush=True)
    hs_sym = len(symbols) - before
    hs_with_body = sum(1 for s in symbols[before:] if s["id"] in body_store)
    print(f"  {service_name} Haskell: {len(hs_files)} files → {hs_sym} symbols, {hs_with_body} with bodies", flush=True)
    before = len(symbols)

    # ── Rust ─────────────────────────────────────────────────────────────────
    rs_files = [
        f for f in service_dir.rglob("*.rs") if f.is_file()
        and not any(p in str(f) for p in ["target/", "/target\\", "\\target\\"])
    ]
    for f in rs_files:
        try:
            s, e, b, c, l = parse_rust_file(f, service_name)
            symbols.extend(s); edges.extend(e)
            body_store.update(b); call_store.update(c); log_store.update(l)
            ghosts = detect_ghosts(f)
            if ghosts:
                for sym in s:
                    ghost_map[sym["id"]].extend(ghosts)
        except Exception as exc:
            print(f"  [error] {f}: {exc}", flush=True)
    rs_sym = len(symbols) - before
    rs_with_body = sum(1 for s in symbols[before:] if s["id"] in body_store)
    print(f"  {service_name} Rust: {len(rs_files)} files → {rs_sym} symbols, {rs_with_body} with bodies", flush=True)
    before = len(symbols)

    # ── Groovy ───────────────────────────────────────────────────────────────
    groovy_files = (
        [f for f in service_dir.rglob("*.groovy") if f.is_file()]
        + [f for f in service_dir.rglob("*.grails") if f.is_file()]
    )
    for f in groovy_files:
        try:
            s, e, b, c, l = parse_groovy_file(f, service_name)
            symbols.extend(s); edges.extend(e)
            body_store.update(b); call_store.update(c); log_store.update(l)
        except Exception as exc:
            print(f"  [error] {f}: {exc}", flush=True)
    gv_sym = len(symbols) - before
    if gv_sym > 0:
        print(f"  {service_name} Groovy: {len(groovy_files)} files → {gv_sym} symbols", flush=True)
    before = len(symbols)

    # ── Python ───────────────────────────────────────────────────────────────
    py_files = [
        f for f in service_dir.rglob("*.py") if f.is_file()
        and not any(p in str(f) for p in ["venv/", "__pycache__", ".venv/"])
    ]
    for f in py_files:
        try:
            s, e, b, c, l = parse_python_file(f, service_name)
            symbols.extend(s); edges.extend(e)
            body_store.update(b); call_store.update(c); log_store.update(l)
            ghosts = detect_ghosts(f)
            if ghosts:
                for sym in s:
                    ghost_map[sym["id"]].extend(ghosts)
        except Exception as exc:
            print(f"  [error] {f}: {exc}", flush=True)
    py_sym = len(symbols) - before
    if py_sym > 0:
        print(f"  {service_name} Python: {len(py_files)} files → {py_sym} symbols", flush=True)

    return (service_name, symbols, edges, body_store, call_store, log_store, dict(ghost_map))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    all_symbols: list = []
    all_edges:   list = []
    all_bodies:  dict = {}
    all_calls:   dict = {}
    all_logs:    dict = {}
    ghost_map: dict = collections.defaultdict(list)

    services = sorted(p for p in REPO_ROOT.iterdir() if p.is_dir() and not p.name.startswith("."))
    print(f"Found {len(services)} service directories: {[s.name for s in services]}")
    if _SERVICE_PROFILES:
        print(f"Loaded service profiles for: {list(_SERVICE_PROFILES.keys())}")

    # ── Parallel extraction across services ──────────────────────────────────
    n_workers = min(len(services), os.cpu_count() or 1)
    print(f"\nProcessing {len(services)} services with {n_workers} parallel workers...")

    with _mp.Pool(processes=n_workers) as pool:
        results = pool.map(_process_service_dir, services)

    # ── Merge results from all workers ───────────────────────────────────────
    for svc_name, syms, edges, bodies, calls, logs, ghosts in results:
        print(f"  Merged {svc_name}: {len(syms)} symbols, {len(bodies)} bodies")
        all_symbols.extend(syms)
        all_edges.extend(edges)
        all_bodies.update(bodies)
        all_calls.update(calls)
        all_logs.update(logs)
        for nid, g in ghosts.items():
            ghost_map[nid].extend(g)

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
    deduped: list = []
    for s in all_symbols:
        if s["id"] not in seen_ids:
            seen_ids[s["id"]] = True
            deduped.append(s)

    # Add call edges to edges list (for graph analysis)
    for caller_nid, info in all_calls.items():
        for callee_name in info["callees"]:
            all_edges.append({"from": caller_nid, "to": callee_name, "kind": "calls"})

    # ── Write outputs ─────────────────────────────────────────────────────────
    result = {
        "nodes": deduped,
        "edges": all_edges,
        "stats": {
            "total_symbols":     len(deduped),
            "total_edges":       len(all_edges),
            "with_body":         len(all_bodies),
            "with_calls":        len(all_calls),
            "with_log_patterns": len(all_logs),
            "services":          [s.name for s in services],
        }
    }

    raw_path = OUT_DIR / "raw_graph.json"
    raw_path.write_text(json.dumps(result, indent=2))
    print(f"\n[ok] raw_graph.json: {len(deduped)} symbols, {len(all_edges)} edges")

    body_path = OUT_DIR / "body_store.json"
    body_path.write_text(json.dumps(all_bodies, indent=2))
    print(f"[ok] body_store.json: {len(all_bodies)} function bodies")

    call_path = OUT_DIR / "call_graph.json"
    call_path.write_text(json.dumps(all_calls, indent=2))
    print(f"[ok] call_graph.json: {len(all_calls)} entries (callees + callers)")

    log_path = OUT_DIR / "log_patterns.json"
    log_path.write_text(json.dumps(all_logs, indent=2))
    print(f"[ok] log_patterns.json: {len(all_logs)} functions with log patterns")

    print(f"\nBody extraction rate: {len(all_bodies)/max(1,len(deduped))*100:.1f}%")
    print(f"Services: {result['stats']['services']}")


if __name__ == "__main__":
    main()
