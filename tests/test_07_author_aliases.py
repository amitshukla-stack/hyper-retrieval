"""
Test 07 — Author alias detection and resolution.

Tests the alias detection algorithm and ownership pipeline integration.
Does NOT require servers (embed/chainlit) — pure unit tests.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "build"))


def test_alias_detection_basic():
    """Same name, different email → should merge."""
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("detect", str(pathlib.Path(__file__).parent.parent / "build" / "01b_detect_author_aliases.py"))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)

    authors = {
        "john@company.com": "John Doe",
        "john@johndoe-macbookpro.local": "John Doe",
        "john.doe@personal.com": "John Doe",
        "jane@company.com": "Jane Smith",
        "bot@ci.company.com": "CI Bot",
    }

    result = mod.detect_aliases(authors)

    # John's local and personal emails should alias to corporate
    assert "john@company.com" not in result["aliases"], "Corporate email should be canonical, not aliased"
    assert result["aliases"].get("john@johndoe-macbookpro.local") == "john@company.com"
    assert result["aliases"].get("john.doe@personal.com") == "john@company.com"

    # Jane should not be aliased (only one email)
    assert "jane@company.com" not in result["aliases"]

    print("PASS: basic alias detection")


def test_bot_detection():
    """Bot emails should be excluded."""
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("detect", str(pathlib.Path(__file__).parent.parent / "build" / "01b_detect_author_aliases.py"))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod.is_bot_email("dependabot[bot]@users.noreply.github.com")
    assert mod.is_bot_email("noreply@github.com")
    assert not mod.is_bot_email("john@company.com")
    print("PASS: bot detection")


def test_local_email_detection():
    """Local machine emails should be detected."""
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("detect", str(pathlib.Path(__file__).parent.parent / "build" / "01b_detect_author_aliases.py"))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod.is_local_email("john@johns-macbook-pro.local")
    assert mod.is_local_email("johndoe@juspays-macbook-pro.local")
    assert mod.is_local_email("john@example.com")
    assert not mod.is_local_email("john@company.com")
    print("PASS: local email detection")


def test_resolve_email_chain():
    """Alias chains should resolve transitively."""
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("ownership", str(pathlib.Path(__file__).parent.parent / "build" / "08_build_ownership.py"))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)

    aliases = {
        "a@x.com": "b@x.com",
        "b@x.com": "c@x.com",
    }
    assert mod.resolve_email("a@x.com", aliases) == "c@x.com"
    assert mod.resolve_email("c@x.com", aliases) == "c@x.com"
    assert mod.resolve_email("d@x.com", aliases) == "d@x.com"
    print("PASS: resolve email chain")


def test_ownership_merge():
    """Alias resolution should merge commit counts."""
    from importlib.util import spec_from_file_location, module_from_spec
    from collections import defaultdict

    spec = spec_from_file_location("ownership", str(pathlib.Path(__file__).parent.parent / "build" / "08_build_ownership.py"))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)

    ownership = defaultdict(lambda: defaultdict(int))
    ownership["repo::src::Module"]["john@company.com"] = 10
    ownership["repo::src::Module"]["john@macbook.local"] = 5
    ownership["repo::src::Other"]["jane@company.com"] = 3

    author_names = {
        "john@company.com": "John Doe",
        "john@macbook.local": "John Doe",
        "jane@company.com": "Jane Smith",
    }

    aliases = {"john@macbook.local": "john@company.com"}
    display_names = {"john@company.com": "John Doe"}
    exclude = set()

    merged_own, merged_names = mod.apply_aliases(
        ownership, author_names, aliases, display_names, exclude
    )

    # John's commits should be merged
    assert merged_own["repo::src::Module"]["john@company.com"] == 15
    assert "john@macbook.local" not in merged_own["repo::src::Module"]
    # Jane untouched
    assert merged_own["repo::src::Other"]["jane@company.com"] == 3
    print("PASS: ownership merge")


if __name__ == "__main__":
    test_bot_detection()
    test_local_email_detection()
    test_alias_detection_basic()
    test_resolve_email_chain()
    test_ownership_merge()
    print("\nAll author alias tests passed!")
