"""
Comment-Code Consistency Checker — Prototype

Detects mismatches between what comments claim and what code does.
Starting with the highest-value pattern: lock/resource acquire-release.

The incident: AI generated code that acquired and released a lock on the same
line with comment "acquiring lock for payment processing". Every reviewer
trusted the comment. The race condition caused false auto-refunds.

Pattern 1: Lock scope checker
- Detects lock acquire followed by immediate release (empty critical section)
- Flags when comment claims protection but lock scope is empty/trivial

Pattern 2: Transaction scope checker
- Detects begin/commit with no mutations between them

Pattern 3: Comment-action mismatch
- Comment says "validate" but no validation logic follows
- Comment says "check auth" but no auth check follows
"""
import re
import ast
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Finding:
    file: str
    line: int
    pattern: str
    severity: str  # "critical", "warning", "info"
    message: str
    comment: str = ""
    code: str = ""


def check_lock_patterns(source: str, filename: str = "<stdin>") -> list[Finding]:
    """Detect locks that are acquired and immediately released."""
    findings = []
    lines = source.split("\n")

    # Pattern: lock.acquire() ... lock.release() with nothing meaningful between
    # Also catches: with lock: pass, async with lock: pass
    lock_keywords = [
        r"\.acquire\s*\(",
        r"\.lock\s*\(",
        r"acquire_lock\s*\(",
        r"get_lock\s*\(",
        r"withLock\s*\(",  # Haskell/Scala style
        r"synchronized\s*\(",  # Java
        r"mutex\.lock\s*\(",  # C++/Rust
        r"Lock\s*\(\s*\)",
    ]
    release_keywords = [
        r"\.release\s*\(",
        r"\.unlock\s*\(",
        r"release_lock\s*\(",
        r"mutex\.unlock\s*\(",
    ]

    lock_pattern = "|".join(lock_keywords)
    release_pattern = "|".join(release_keywords)

    for i, line in enumerate(lines):
        # Check: acquire and release on the SAME line
        if re.search(lock_pattern, line) and re.search(release_pattern, line):
            # Look for nearby comment claiming protection
            comment = _find_nearby_comment(lines, i, window=3)
            findings.append(Finding(
                file=filename, line=i + 1,
                pattern="lock-same-line",
                severity="critical",
                message="Lock acquired and released on the same line — critical section is empty",
                comment=comment,
                code=line.strip(),
            ))
            continue

        # Check: acquire followed by release within 2 lines (trivial scope)
        if re.search(lock_pattern, line):
            for j in range(i + 1, min(i + 3, len(lines))):
                stripped = lines[j].strip()
                if re.search(release_pattern, stripped):
                    # Count non-empty, non-comment lines between acquire and release
                    between = [
                        lines[k].strip() for k in range(i + 1, j)
                        if lines[k].strip() and not lines[k].strip().startswith(("#", "//", "--"))
                    ]
                    if len(between) == 0:
                        comment = _find_nearby_comment(lines, i, window=3)
                        findings.append(Finding(
                            file=filename, line=i + 1,
                            pattern="lock-empty-scope",
                            severity="critical",
                            message=f"Lock acquired (line {i+1}) and released (line {j+1}) with no code between them",
                            comment=comment,
                            code=f"{line.strip()} ... {stripped}",
                        ))
                    break

        # Check: `with lock:` followed by `pass` or trivial body
        if re.match(r"\s*(?:async\s+)?with\s+.*lock.*:", line, re.IGNORECASE):
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line in ("pass", "...") or not next_line:
                    comment = _find_nearby_comment(lines, i, window=3)
                    findings.append(Finding(
                        file=filename, line=i + 1,
                        pattern="lock-with-pass",
                        severity="critical",
                        message="Lock context manager with empty/pass body — no critical section protected",
                        comment=comment,
                        code=f"{line.strip()} → {next_line}",
                    ))

    return findings


def check_comment_action_mismatch(source: str, filename: str = "<stdin>") -> list[Finding]:
    """Detect comments that claim actions the code doesn't perform.

    Only fires when the comment is an *imperative directive* ("Lock the mutex",
    "Encrypt the password"), not narrative text that happens to mention the
    keyword ("# security checks live in the middleware"). This kills the large
    class of FPs where tests, section headers, and docstring-style comments
    simply *reference* a concept without implying the next lines implement it.
    """
    # Skip test files entirely — tests discuss patterns narratively without
    # implementing them, producing near-100% FP rate. Production code paths
    # (serve/, apps/) are where the signal lives.
    if "/tests/" in filename.replace("\\", "/") or filename.endswith("_test.py") or "/test_" in filename:
        return []

    findings = []
    lines = source.split("\n")

    # Imperative-form patterns: comment must START with an action verb
    # (after '#' and whitespace). This is the key to killing narrative FPs.
    claim_patterns = [
        {
            "comment_regex": r"^\s*#\s*(?:acquire|lock|take|grab|hold|release)\b.*\b(?:lock|mutex|semaphore)\b",
            "code_regex": r"(?:\.acquire|\.lock|\.release|Lock\s*\(|mutex|synchroniz|with\s+\w*lock)",
            "claim": "locking/synchronization",
        },
        {
            "comment_regex": r"^\s*#\s*(?:validate|sanitize|verify|assert|reject|require)\s+(?:the\s+)?(?:input|request|user|payload|data|param)",
            "code_regex": r"(?:validate|sanitize|\.check|\.verify|assert\s+|raise\s+\w*Error|if\s+not\s+)",
            "claim": "input validation",
        },
        {
            "comment_regex": r"^\s*#\s*(?:authenticate|authorize|check\s+auth|verify\s+auth|require\s+auth|check\s+permission)",
            "code_regex": r"(?:auth|permission|token|credential|is_authenticated|has_perm|require_auth|@login_required|@permission_required)",
            "claim": "authentication/authorization",
        },
        {
            "comment_regex": r"^\s*#\s*(?:encrypt|decrypt|hash|sign|hmac)\s+(?:the\s+)?(?:password|data|payload|token|secret|key|message)",
            "code_regex": r"(?:encrypt|decrypt|hash|hmac|bcrypt|sha\d|aes|nacl|cipher|sign\s*\()",
            "claim": "encryption/security",
        },
    ]

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue

        # Skip shebang, encoding, and type-hint pragmas
        if stripped.startswith("#!") or "coding:" in stripped or stripped.startswith("# type:"):
            continue

        # Skip section-header-style comments (all caps, separators, decorative)
        # These are narrative, not directives: "# ===== SECURITY ====="
        body = stripped.lstrip("#").strip()
        if not body:
            continue
        if re.match(r"^[=\-*#]+$", body):  # decorative separator
            continue
        if body.isupper() and len(body) > 3:  # ALL-CAPS section headers
            continue

        for pattern in claim_patterns:
            if re.search(pattern["comment_regex"], stripped, re.IGNORECASE):
                # Check next 5 lines for the claimed action
                lookahead = "\n".join(lines[i+1:i+6])
                if not re.search(pattern["code_regex"], lookahead, re.IGNORECASE):
                    findings.append(Finding(
                        file=filename, line=i + 1,
                        pattern="comment-action-mismatch",
                        severity="warning",
                        message=f"Comment claims {pattern['claim']} but next 5 lines contain no related code",
                        comment=stripped,
                        code=lines[i+1].strip() if i+1 < len(lines) else "",
                    ))

    return findings


def check_transaction_patterns(source: str, filename: str = "<stdin>") -> list[Finding]:
    """Detect transactions that begin and commit with no mutations between them."""
    findings = []
    lines = source.split("\n")

    begin_patterns = [
        r"\.begin\s*\(",
        r"BEGIN\s*;",
        r"begin_transaction\s*\(",
        r"start_transaction\s*\(",
        r"@transaction\.atomic",
        r"with\s+.*(?:transaction|atomic|session\.begin)",
    ]
    commit_patterns = [
        r"\.commit\s*\(",
        r"COMMIT\s*;",
        r"commit_transaction\s*\(",
    ]
    mutation_patterns = [
        r"\.(?:save|update|insert|delete|execute|write|put|set|remove|add|create)\s*\(",
        r"(?:INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\s",
        r"\.query\s*\(",
    ]

    begin_re = "|".join(begin_patterns)
    commit_re = "|".join(commit_patterns)
    mutation_re = "|".join(mutation_patterns)

    for i, line in enumerate(lines):
        if re.search(begin_re, line, re.IGNORECASE):
            # Look for commit within 5 lines
            for j in range(i + 1, min(i + 6, len(lines))):
                if re.search(commit_re, lines[j], re.IGNORECASE):
                    # Check for mutations between begin and commit
                    between = "\n".join(lines[i+1:j])
                    if not re.search(mutation_re, between, re.IGNORECASE):
                        comment = _find_nearby_comment(lines, i, window=3)
                        findings.append(Finding(
                            file=filename, line=i + 1,
                            pattern="transaction-empty-scope",
                            severity="critical",
                            message=f"Transaction begins (line {i+1}) and commits (line {j+1}) with no mutations between",
                            comment=comment,
                            code=f"{line.strip()} ... {lines[j].strip()}",
                        ))
                    break

    return findings


def check_auth_before_action(source: str, filename: str = "<stdin>") -> list[Finding]:
    """Detect sensitive actions performed before auth checks."""
    findings = []
    lines = source.split("\n")

    # Sensitive actions that should happen AFTER auth
    sensitive_actions = [
        r"charge_card\s*\(",
        r"process_payment\s*\(",
        r"execute_refund\s*\(",
        r"transfer_funds\s*\(",
        r"delete_account\s*\(",
        r"update_password\s*\(",
        r"\.delete\s*\(",
        r"send_money\s*\(",
        r"withdraw\s*\(",
    ]
    auth_checks = [
        r"is_authenticated",
        r"check_auth",
        r"verify_token",
        r"require_auth",
        r"has_permission",
        r"authorize\s*\(",
        r"@login_required",
        r"@require_auth",
    ]

    sensitive_re = "|".join(sensitive_actions)
    auth_re = "|".join(auth_checks)

    # Within each function, check if sensitive action appears before auth check
    func_start = None
    for i, line in enumerate(lines):
        if re.match(r"\s*(?:def |async def |function |fn )", line):
            func_start = i

        if func_start is not None and re.search(sensitive_re, line, re.IGNORECASE):
            # Check if auth happened between func_start and here
            preceding = "\n".join(lines[func_start:i])
            if not re.search(auth_re, preceding, re.IGNORECASE):
                # Check if auth happens AFTER (wrong order)
                following = "\n".join(lines[i:min(i+10, len(lines))])
                if re.search(auth_re, following, re.IGNORECASE):
                    comment = _find_nearby_comment(lines, i, window=3)
                    findings.append(Finding(
                        file=filename, line=i + 1,
                        pattern="auth-after-action",
                        severity="critical",
                        message="Sensitive action performed BEFORE auth check — auth appears later in the function",
                        comment=comment,
                        code=line.strip(),
                    ))

    return findings


def check_error_swallowing(source: str, filename: str = "<stdin>") -> list[Finding]:
    """Detect try/except blocks that silently swallow errors."""
    findings = []
    lines = source.split("\n")

    for i, line in enumerate(lines):
        stripped = line.strip()
        # Python: except ... : pass
        if re.match(r"except\s*.*:", stripped):
            # Check if the except body is just pass, continue, or empty
            for j in range(i + 1, min(i + 4, len(lines))):
                body = lines[j].strip()
                if not body or body.startswith("#"):
                    continue
                if body in ("pass", "continue", "..."):
                    comment = _find_nearby_comment(lines, i, window=3)
                    findings.append(Finding(
                        file=filename, line=i + 1,
                        pattern="error-swallowed",
                        severity="warning",
                        message=f"Exception caught and silently ignored ({body}). Errors may be hidden.",
                        comment=comment,
                        code=f"{stripped} → {body}",
                    ))
                elif body.startswith("log") or body.startswith("print") or body.startswith("logger"):
                    # Log-only handlers in critical paths are also suspicious
                    # Check if there's a return/raise after
                    remaining = "\n".join(lines[j+1:j+3])
                    if not re.search(r"\b(raise|return|sys\.exit|abort)\b", remaining):
                        comment = _find_nearby_comment(lines, i, window=3)
                        findings.append(Finding(
                            file=filename, line=i + 1,
                            pattern="error-logged-not-raised",
                            severity="info",
                            message="Exception caught, logged, but not re-raised. Processing continues with potentially invalid state.",
                            comment=comment,
                            code=f"{stripped} → {body}",
                        ))
                break  # only check first non-empty line of except body

    return findings


def _find_nearby_comment(lines: list[str], line_idx: int, window: int = 3) -> str:
    """Find the nearest comment within a window of lines."""
    for offset in range(-window, window + 1):
        idx = line_idx + offset
        if 0 <= idx < len(lines):
            stripped = lines[idx].strip()
            if stripped.startswith("#") or stripped.startswith("//") or stripped.startswith("--"):
                return stripped
    return ""


# Directory-based severity stratification.
# Error swallowing in a batch script (build/, tools/, tests/) is usually
# intentional ("skip bad row and continue"); in a request handler (serve/,
# apps/chat/, apps/api/) it hides production bugs. Same pattern, different
# blast radius — severity reflects that.
_BATCH_DIR_SEGMENTS = ("/build/", "/tools/", "/tests/", "/scripts/", "/bench/",
                       "/benchmarks/", "/eval/", "/experiments/", "/lab/")
_REQUEST_PATH_SEGMENTS = ("/serve/", "/apps/chat/", "/apps/api/", "/server/",
                          "/handlers/", "/routes/", "/views/", "/controllers/")


def _classify_dir(filepath: str) -> str:
    """Return 'batch', 'request', or 'other' based on filepath segments."""
    norm = filepath.replace("\\", "/").lower()
    for seg in _REQUEST_PATH_SEGMENTS:
        if seg in norm:
            return "request"
    for seg in _BATCH_DIR_SEGMENTS:
        if seg in norm:
            return "batch"
    return "other"


def _stratify(finding: "Finding") -> "Finding":
    """Downgrade severity for swallowed/logged errors in batch-script dirs.
    Keep critical-path (locks/transactions/auth) severity untouched — those
    are blast-radius patterns that matter regardless of directory."""
    if finding.pattern not in ("error-swallowed", "error-logged-not-raised"):
        return finding
    bucket = _classify_dir(finding.file)
    if bucket == "batch" and finding.severity == "warning":
        return Finding(
            file=finding.file, line=finding.line, pattern=finding.pattern,
            severity="info",
            message=finding.message + " (batch dir — downgraded to info)",
            comment=finding.comment, code=finding.code,
        )
    if bucket == "request" and finding.severity == "info":
        return Finding(
            file=finding.file, line=finding.line, pattern=finding.pattern,
            severity="warning",
            message=finding.message + " (request path — escalated to warning)",
            comment=finding.comment, code=finding.code,
        )
    return finding


def check_file(filepath: str) -> list[Finding]:
    """Run all checks on a file and apply directory-based severity stratification."""
    source = Path(filepath).read_text()
    findings = []
    findings.extend(check_lock_patterns(source, filepath))
    findings.extend(check_transaction_patterns(source, filepath))
    findings.extend(check_auth_before_action(source, filepath))
    findings.extend(check_error_swallowing(source, filepath))
    findings.extend(check_comment_action_mismatch(source, filepath))
    return [_stratify(f) for f in findings]


def main():
    # Demo with synthetic examples that mirror the real incident
    test_code = '''
# Payment processing module

def process_payment(payment_id, amount):
    """Process a payment with proper locking."""

    # Acquire lock for payment processing
    lock = redis.lock(f"payment:{payment_id}")
    lock.acquire()
    lock.release()  # Released immediately!

    # Now "protected" code runs without lock
    result = charge_card(payment_id, amount)
    if result.success:
        update_payment_status(payment_id, "success")
        send_webhook(payment_id, "success")
    return result


def process_refund(payment_id):
    """Process refund with synchronization."""

    # Acquiring lock to prevent race condition
    with redis.lock(f"refund:{payment_id}"):
        pass

    # This runs without any lock protection
    refund = execute_refund(payment_id)
    notify_merchant(payment_id, refund)
    return refund


def validate_and_charge(user_input):
    # Validate and sanitize user input
    amount = float(user_input["amount"])

    # Check authentication before proceeding
    charge_card(user_input["card"], amount)

    return {"status": "charged"}


def save_order(order_data):
    """Save order with transaction safety."""
    # Begin transaction to ensure atomicity
    db.begin_transaction()
    db.commit_transaction()

    # Actual writes happen outside transaction!
    db.save(order_data)
    db.update("inventory", {"stock": order_data["stock"] - 1})


def charge_and_verify(card, amount):
    """Charge card with proper auth."""
    # Process the payment first, verify identity after
    charge_card(card, amount)
    result = send_to_gateway(card, amount)

    # Now check if user is authorized
    if not is_authenticated(card.user):
        log.error("Unauthorized charge!")


def fetch_payment_status(payment_id):
    """Get payment status from gateway."""
    try:
        response = gateway.get_status(payment_id)
        return response.status
    except ConnectionError:
        # Handle connection error gracefully
        pass
    except TimeoutError:
        # Log timeout and continue
        logger.warning("Gateway timeout")
        # processing continues with no status...


def risky_batch_process(payments):
    """Process payments in batch."""
    for p in payments:
        try:
            result = process_single(p)
        except Exception:
            # Error handling for batch item
            continue
    return "all done"
'''

    print("=" * 70)
    print("AI Code Guardrails — Comment-Code Consistency Checker v2 (5 patterns)")
    print("=" * 70)

    findings = []
    findings.extend(check_lock_patterns(test_code, "payment_service.py"))
    findings.extend(check_transaction_patterns(test_code, "payment_service.py"))
    findings.extend(check_auth_before_action(test_code, "payment_service.py"))
    findings.extend(check_error_swallowing(test_code, "payment_service.py"))
    findings.extend(check_comment_action_mismatch(test_code, "payment_service.py"))

    for f in findings:
        icon = "🔴" if f.severity == "critical" else "🟡" if f.severity == "warning" else "🔵"
        print(f"\n{icon} [{f.severity.upper()}] {f.file}:{f.line}")
        print(f"   Pattern: {f.pattern}")
        print(f"   {f.message}")
        if f.comment:
            print(f"   Comment: {f.comment}")
        if f.code:
            print(f"   Code:    {f.code}")

    print(f"\n{'=' * 70}")
    print(f"Total: {len(findings)} findings ({sum(1 for f in findings if f.severity == 'critical')} critical, {sum(1 for f in findings if f.severity == 'warning')} warning)")

    return findings


def print_findings(findings: list[Finding]):
    """Pretty-print findings."""
    for f in findings:
        icon = "!" if f.severity == "critical" else "?" if f.severity == "warning" else "i"
        print(f"\n[{f.severity.upper()}] {f.file}:{f.line}")
        print(f"  Pattern: {f.pattern}")
        print(f"  {f.message}")
        if f.comment:
            print(f"  Comment: {f.comment}")
        if f.code:
            print(f"  Code:    {f.code}")


def scan_path(target: str, extensions: tuple = (".py", ".js", ".ts", ".rs", ".hs", ".java", ".go")) -> list[Finding]:
    """Scan a file or directory recursively."""
    target_path = Path(target)
    all_findings = []

    if target_path.is_file():
        all_findings.extend(check_file(str(target_path)))
    elif target_path.is_dir():
        for ext in extensions:
            for fp in target_path.rglob(f"*{ext}"):
                # Skip common non-source dirs
                parts = fp.parts
                if any(skip in parts for skip in ("node_modules", ".git", "__pycache__", "venv", ".venv", "target", "dist")):
                    continue
                try:
                    all_findings.extend(check_file(str(fp)))
                except Exception as e:
                    print(f"  Skipping {fp}: {e}", file=sys.stderr)
    else:
        print(f"Error: {target} is not a file or directory", file=sys.stderr)
        sys.exit(2)

    return all_findings


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="AI Code Guardrails — Comment-Code Consistency Checker",
        epilog="Exit code: 1 if critical findings, 0 otherwise. Use in CI: guard path/to/src/",
    )
    parser.add_argument("targets", nargs="*", help="Files or directories to scan. If none, runs built-in demo.")
    parser.add_argument("--severity", choices=["critical", "warning", "info"], default="info",
                        help="Minimum severity to report (default: info)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    severity_rank = {"critical": 3, "warning": 2, "info": 1}
    min_rank = severity_rank[args.severity]

    if not args.targets:
        # Run demo
        findings = main()
    else:
        print(f"Scanning {len(args.targets)} target(s)...")
        findings = []
        for target in args.targets:
            findings.extend(scan_path(target))

        # Filter by severity
        findings = [f for f in findings if severity_rank.get(f.severity, 0) >= min_rank]

        if args.json:
            import json
            json_out = [{"file": f.file, "line": f.line, "pattern": f.pattern,
                         "severity": f.severity, "message": f.message,
                         "comment": f.comment, "code": f.code} for f in findings]
            print(json.dumps(json_out, indent=2))
        else:
            print(f"\n{'='*70}")
            print(f"Guard Scan Results")
            print(f"{'='*70}")
            print_findings(findings)
            n_crit = sum(1 for f in findings if f.severity == "critical")
            n_warn = sum(1 for f in findings if f.severity == "warning")
            n_info = sum(1 for f in findings if f.severity == "info")
            print(f"\n{'='*70}")
            print(f"Total: {len(findings)} findings ({n_crit} critical, {n_warn} warning, {n_info} info)")
            if n_crit:
                print(f"EXIT 1 — {n_crit} critical finding(s) would block merge")

    sys.exit(1 if any(f.severity == "critical" for f in findings) else 0)
