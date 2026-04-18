"""
LLM-powered comment-code alignment checker.

Extracts (comment, code) pairs from Python/Rust/Haskell source and asks a local
LLM (LM Studio at 172.19.144.1:1234/v1) whether the comment accurately describes
the code. Language-agnostic: the LLM handles syntax differences.

O-8 milestone 3: multi-language comment-code alignment.
Gate: TP rate >= 0.80, FP rate <= 0.20, latency <= 2s/file.
"""
import ast
import re
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import openai
    _openai_available = True
except ImportError:
    _openai_available = False

from comment_code_checker import Finding

LM_STUDIO_BASE = "http://172.19.144.1:1234/v1"
LM_MODEL = "mistralai/devstral-small-2-2512"

_ALIGNMENT_PROMPT = """\
Review this comment against the code. MISALIGNED means the code does the OPPOSITE or NOTHING \
of what the comment claims — not just that the implementation could be more thorough.

Comment: {comment}
Code: {code}

Q1: Does the code perform the core claimed actions? (even imperfectly) YES/NO
Q2: Does the code actively CONTRADICT the comment (e.g. claims to validate but skips all checks)? YES/NO

Output exactly one line: VERDICT: ALIGNED or VERDICT: MISALIGNED
(MISALIGNED only if Q1=NO or Q2=YES)
"""


@dataclass
class CommentCodePair:
    language: str
    filename: str
    line: int
    comment: str
    code: str


# ── extractors ────────────────────────────────────────────────────────────────

def _extract_python(source: str, filename: str) -> list[CommentCodePair]:
    pairs = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return pairs
    lines = source.splitlines()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        docstring = ast.get_docstring(node)
        if docstring:
            func_src = "\n".join(lines[node.lineno - 1: node.end_lineno])
            pairs.append(CommentCodePair("python", filename, node.lineno, docstring, func_src[:600]))
            continue
        # inline comment on the line before the def
        def_line = node.lineno - 1  # 0-indexed
        if def_line > 0 and lines[def_line - 1].strip().startswith("#"):
            comment = lines[def_line - 1].strip().lstrip("# ").strip()
            func_src = "\n".join(lines[def_line: node.end_lineno])
            pairs.append(CommentCodePair("python", filename, node.lineno, comment, func_src[:600]))
    return pairs


def _extract_rust(source: str, filename: str) -> list[CommentCodePair]:
    pairs = []
    lines = source.splitlines()
    i = 0
    while i < len(lines):
        # collect consecutive /// or // comment block
        comment_lines = []
        while i < len(lines) and re.match(r"\s*//[/!]?\s?(.*)", lines[i]):
            m = re.match(r"\s*//[/!]?\s?(.*)", lines[i])
            comment_lines.append(m.group(1))
            i += 1
        if not comment_lines:
            i += 1
            continue
        # next non-blank line should be a fn / pub fn / async fn
        j = i
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j < len(lines) and re.search(r"\bfn\s+\w+", lines[j]):
            # grab up to 20 lines of the function body
            func_src = "\n".join(lines[j: j + 20])
            comment = " ".join(comment_lines).strip()
            if len(comment) > 10:
                pairs.append(CommentCodePair("rust", filename, j + 1, comment, func_src[:600]))
        i = j + 1 if j > i else i
    return pairs


def _extract_haskell(source: str, filename: str) -> list[CommentCodePair]:
    pairs = []
    lines = source.splitlines()
    i = 0
    while i < len(lines):
        # Haskell Haddock: -- | or {- | blocks
        comment_lines = []
        while i < len(lines) and re.match(r"\s*--\s*(.*)", lines[i]):
            m = re.match(r"\s*--\s*(.*)", lines[i])
            comment_lines.append(m.group(1))
            i += 1
        if not comment_lines:
            i += 1
            continue
        # next non-blank, non-comment line: Haskell function definition
        j = i
        while j < len(lines) and (not lines[j].strip() or lines[j].strip().startswith("--")):
            j += 1
        if j < len(lines) and re.match(r"^[a-z_]\w*\s+(::|\w)", lines[j]):
            func_src = "\n".join(lines[j: j + 15])
            comment = " ".join(comment_lines).strip()
            if len(comment) > 10:
                pairs.append(CommentCodePair("haskell", filename, j + 1, comment, func_src[:600]))
        i = j + 1 if j > i else i
    return pairs


def extract_pairs(source: str, filename: str, language: Optional[str] = None) -> list[CommentCodePair]:
    if language is None:
        ext = Path(filename).suffix.lower()
        language = {"py": "python", ".rs": "rust", ".hs": "haskell"}.get(ext, ext.lstrip("."))
    if language == "python":
        return _extract_python(source, filename)
    elif language == "rust":
        return _extract_rust(source, filename)
    elif language == "haskell":
        return _extract_haskell(source, filename)
    return []


# ── LLM judge ─────────────────────────────────────────────────────────────────

def _call_llm(comment: str, code: str, timeout: float = 8.0) -> tuple[str, str]:
    """Returns (verdict, reason). verdict ∈ {ALIGNED, MISALIGNED, PARTIAL, ERROR}."""
    if not _openai_available:
        return "ERROR", "openai package not installed"
    try:
        client = openai.OpenAI(base_url=LM_STUDIO_BASE, api_key="lm-studio")
        prompt = _ALIGNMENT_PROMPT.format(comment=comment[:300], code=code[:600])
        resp = client.chat.completions.create(
            model=LM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0.0,
            timeout=timeout,
        )
        raw = resp.choices[0].message.content.strip()
        # find "VERDICT: ALIGNED" or "VERDICT: MISALIGNED" anywhere in response
        import re as _re
        m = _re.search(r"VERDICT:\s*(ALIGNED|MISALIGNED|PARTIAL)", raw, _re.IGNORECASE)
        if m:
            verdict = m.group(1).upper()
            reason = raw[m.end():].strip()[:80]
        else:
            # fallback: scan for bare verdict word
            for word in ("MISALIGNED", "PARTIAL", "ALIGNED"):
                if word in raw.upper():
                    verdict = word
                    reason = raw[:80]
                    break
            else:
                verdict = "ERROR"
                reason = raw[:80]
        return verdict, reason
    except Exception as e:
        return "ERROR", str(e)[:80]


# ── main entry ────────────────────────────────────────────────────────────────

def check_llm_alignment(source: str, filename: str = "<stdin>",
                         language: Optional[str] = None) -> list[Finding]:
    pairs = extract_pairs(source, filename, language)
    findings = []
    for pair in pairs:
        verdict, reason = _call_llm(pair.comment, pair.code)
        if verdict in ("MISALIGNED", "PARTIAL"):
            severity = "critical" if verdict == "MISALIGNED" else "warning"
            findings.append(Finding(
                file=filename,
                line=pair.line,
                pattern="llm_comment_alignment",
                severity=severity,
                message=f"Comment-code {verdict.lower()}: {reason}",
                comment=pair.comment[:100],
                code=pair.code[:100],
            ))
    return findings


# ── benchmark harness ─────────────────────────────────────────────────────────

PYTHON_TP = '''
def process_payment(amount, account_id):
    """Validates the payment amount and debits the account atomically."""
    # No validation, no transaction — just a direct write
    db.write(f"UPDATE accounts SET balance = balance - {amount} WHERE id = {account_id}")
    return True
'''

PYTHON_TN = '''
def validate_and_debit(amount, account_id):
    """Validates the payment amount and debits the account atomically."""
    if amount <= 0:
        raise ValueError("Amount must be positive")
    with db.transaction():
        balance = db.query(f"SELECT balance FROM accounts WHERE id = {account_id}")
        if balance < amount:
            raise InsufficientFundsError()
        db.execute(f"UPDATE accounts SET balance = balance - {amount} WHERE id = {account_id}")
    return True
'''

RUST_TP = '''
/// Acquires the payment lock and processes the transaction safely.
pub fn process_payment(lock: &Mutex<()>, amount: f64) -> Result<(), PaymentError> {
    let _guard = lock.lock().unwrap();
    drop(_guard);  // lock released immediately
    // no critical section — payment processed without protection
    db::write_payment(amount)
}
'''

RUST_TN = '''
/// Acquires the payment lock and processes the transaction safely.
pub fn process_payment(lock: &Mutex<()>, amount: f64) -> Result<(), PaymentError> {
    let _guard = lock.lock().unwrap();
    // critical section: validate and write under lock
    if amount <= 0.0 { return Err(PaymentError::InvalidAmount); }
    db::write_payment(amount)?;
    Ok(())  // _guard dropped here, lock released
}
'''

HASKELL_TP = '''
-- Authenticates the user before processing the refund request.
processRefund :: UserId -> Amount -> IO (Either Error ())
processRefund userId amount = do
  -- auth check skipped for performance
  executeRefund userId amount
'''

HASKELL_TN = '''
-- Authenticates the user before processing the refund request.
processRefund :: UserId -> Amount -> IO (Either Error ())
processRefund userId amount = do
  authResult <- authenticate userId
  case authResult of
    Left err -> return (Left err)
    Right _  -> executeRefund userId amount
'''


def run_benchmark():
    fixtures = [
        ("Python TP", PYTHON_TP, "python", True),   # expect MISALIGNED
        ("Python TN", PYTHON_TN, "python", False),  # expect ALIGNED
        ("Rust TP", RUST_TP, "rust", True),
        ("Rust TN", RUST_TN, "rust", False),
        ("Haskell TP", HASKELL_TP, "haskell", True),
        ("Haskell TN", HASKELL_TN, "haskell", False),
    ]
    results = []
    print(f"\n{'='*60}")
    print("LLM Alignment Checker Benchmark")
    print(f"Model: {LM_MODEL}")
    print(f"{'='*60}")
    for name, source, lang, expect_finding in fixtures:
        t0 = time.time()
        findings = check_llm_alignment(source, f"test.{lang[:2]}", language=lang)
        latency = time.time() - t0
        got_finding = len(findings) > 0
        correct = got_finding == expect_finding
        verdict_str = findings[0].message[:60] if findings else "(no finding)"
        status = "PASS" if correct else "FAIL"
        print(f"  [{status}] {name:15s} | {'expect finding' if expect_finding else 'expect clean  '} | got={got_finding} | {latency:.1f}s | {verdict_str}")
        results.append({"name": name, "correct": correct, "latency": latency, "finding": got_finding})

    tp_cases = [r for r in results if "TP" in r["name"]]
    tn_cases = [r for r in results if "TN" in r["name"]]
    tp_rate = sum(r["correct"] for r in tp_cases) / len(tp_cases)
    fp_rate = sum(not r["correct"] for r in tn_cases) / len(tn_cases)
    avg_latency = sum(r["latency"] for r in results) / len(results)

    print(f"\n{'='*60}")
    print(f"  TP rate  : {tp_rate:.2f}  (gate: >= 0.80)  {'PASS' if tp_rate >= 0.80 else 'FAIL'}")
    print(f"  FP rate  : {fp_rate:.2f}  (gate: <= 0.20)  {'PASS' if fp_rate <= 0.20 else 'FAIL'}")
    print(f"  avg lat  : {avg_latency:.1f}s (gate: <= 2.0s)  {'PASS' if avg_latency <= 2.0 else 'FAIL'}")
    print(f"{'='*60}")
    overall = tp_rate >= 0.80 and fp_rate <= 0.20 and avg_latency <= 2.0
    print(f"  VERDICT  : {'PASS — T-018 gate MET' if overall else 'FAIL — needs tuning'}")
    return overall, {"tp_rate": tp_rate, "fp_rate": fp_rate, "avg_latency_s": avg_latency, "results": results}


if __name__ == "__main__":
    run_benchmark()
