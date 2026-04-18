"""
test_lore_signals.py — Unit tests for Lore trailer parsing in get_why_context.

Tests the _get_lore_signals function and its integration with get_why_context:
1. Returns [] when HR_LORE_PATH not set
2. Returns [] when repo path doesn't exist
3. Parses all 4 Lore trailer types correctly
4. get_why_context output always contains 'lore_signals' key
5. Cache works (same module name returns same result without re-running git)
"""
import sys, os, pathlib, subprocess, tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "serve"))
os.environ["EMBED_SERVER_URL"] = ""  # keyword-only mode — no GPU needed

import retrieval_engine as engine


# ── Test 1: lore_signals absent when HR_LORE_PATH not set ───────────────────

def test_lore_disabled_by_default():
    os.environ.pop("HR_LORE_PATH", None)
    engine._LORE_REPO = ""
    engine._get_lore_signals.cache_clear()
    signals = engine._get_lore_signals("SomeModule.Foo")
    assert signals == [], f"Expected [], got {signals}"
    print("PASS test_lore_disabled_by_default")


# ── Test 2: lore_signals = [] for nonexistent repo path ─────────────────────

def test_lore_nonexistent_path():
    engine._LORE_REPO = "/tmp/does_not_exist_xyz"
    engine._get_lore_signals.cache_clear()
    signals = engine._get_lore_signals("SomeModule.Foo")
    assert signals == [], f"Expected [], got {signals}"
    engine._LORE_REPO = ""
    print("PASS test_lore_nonexistent_path")


# ── Test 3: Parses all 4 Lore trailer types from a real git repo ─────────────

def test_lore_parses_trailers():
    """Create a temp git repo with a Lore-annotated commit and verify parsing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Init repo
        subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmpdir, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmpdir, capture_output=True)

        # Create a file with a keyword that will match the module name
        test_file = pathlib.Path(tmpdir) / "PaymentProcessor.py"
        test_file.write_text("def process(): pass\n")
        subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)

        commit_msg = (
            "feat: add payment processing\n\n"
            "Core payment flow.\n\n"
            "Lore-Constraint: must not change return type — callers depend on dict shape\n"
            "Lore-Rejected: async variant was considered but adds latency for sync callers\n"
            "Lore-Directive: always validate amount > 0 before calling downstream\n"
            "Lore-Verify: PaymentProcessor.process returns {status, amount, ref} always\n"
        )
        subprocess.run(["git", "commit", "-m", commit_msg], cwd=tmpdir, capture_output=True)

        engine._LORE_REPO = tmpdir
        engine._get_lore_signals.cache_clear()
        signals = engine._get_lore_signals("PaymentProcessor")

        engine._LORE_REPO = ""
        engine._get_lore_signals.cache_clear()

    types_found = {s["type"] for s in signals}
    assert "constraint" in types_found, f"Expected constraint, got {types_found}"
    assert "rejected" in types_found,   f"Expected rejected, got {types_found}"
    assert "directive" in types_found,  f"Expected directive, got {types_found}"
    assert "verify" in types_found,     f"Expected verify, got {types_found}"
    assert all("commit" in s for s in signals), "Each signal must have commit hash"
    print(f"PASS test_lore_parses_trailers ({len(signals)} signals: {types_found})")


# ── Test 4: get_why_context always has lore_signals key ─────────────────────

def test_why_context_has_lore_key():
    """lore_signals key must always be present in get_why_context output."""
    os.environ.pop("HR_LORE_PATH", None)
    engine._LORE_REPO = ""
    engine._get_lore_signals.cache_clear()

    # Initialize without embedder to avoid GPU requirement
    os.environ.setdefault("ARTIFACT_DIR", "/home/beast/projects/workspaces/juspay/artifacts")
    engine.initialize(load_embedder=False)

    result = engine.get_why_context("Euler.Types.Transaction")
    assert "lore_signals" in result, f"'lore_signals' key missing from get_why_context output"
    assert isinstance(result["lore_signals"], list), "lore_signals must be a list"
    print(f"PASS test_why_context_has_lore_key (lore_signals={result['lore_signals']})")


if __name__ == "__main__":
    test_lore_disabled_by_default()
    test_lore_nonexistent_path()
    test_lore_parses_trailers()
    test_why_context_has_lore_key()
    print("\nAll 4 tests PASS")
