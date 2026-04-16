"""
Stage 1b — Detect author aliases and generate canonical mapping.

Analyzes the ownership index to find duplicate authors (same person,
different email) using multiple signals:
  1. Exact name match with different emails
  2. Email prefix similarity on same domain
  3. Local machine emails (macbookpro, .local) matching corporate emails
  4. Bot/placeholder account detection

Outputs:
  - config/author_aliases.yaml  (canonical mapping for pipeline use)
  - suggested_aliases_report.txt (human-readable audit report)

Usage:
  python3 build/01b_detect_author_aliases.py [--artifact-dir DIR] [--dry-run]
"""
import argparse
import json
import pathlib
import re
from collections import defaultdict
from difflib import SequenceMatcher

# Patterns that indicate a local/machine-generated email
LOCAL_EMAIL_PATTERNS = [
    r"macbookpro",
    r"macbook-pro",
    r"\.local$",
    r"juspays-macbook",
    r"johndoe@",
    r"example\.com$",
]

# Known bot patterns
BOT_PATTERNS = [
    r"dependabot\[bot\]",
    r"noreply@",
    r"no-reply@",
    r"bot@",
    r"ci-bot",
    r"github-actions",
    r"renovate\[bot\]",
]


def is_local_email(email: str) -> bool:
    """Check if email looks like a local machine / placeholder."""
    return any(re.search(p, email, re.IGNORECASE) for p in LOCAL_EMAIL_PATTERNS)


def is_bot_email(email: str) -> bool:
    """Check if email looks like a bot / CI account."""
    return any(re.search(p, email, re.IGNORECASE) for p in BOT_PATTERNS)


def is_corporate_email(email: str) -> bool:
    """Check if email is a real corporate email (not local, not bot)."""
    return "@" in email and not is_local_email(email) and not is_bot_email(email)


def email_prefix(email: str) -> str:
    """Extract the local part before @."""
    return email.split("@")[0].lower().strip() if "@" in email else email.lower().strip()


def name_similarity(name_a: str, name_b: str) -> float:
    """Similarity ratio between two names (0.0 - 1.0)."""
    a = name_a.lower().strip()
    b = name_b.lower().strip()
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def prefix_similarity(email_a: str, email_b: str) -> float:
    """Similarity between email prefixes."""
    a = email_prefix(email_a)
    b = email_prefix(email_b)
    if a == b:
        return 1.0
    # Also check without dots/hyphens
    a_clean = re.sub(r"[.\-_]", "", a)
    b_clean = re.sub(r"[.\-_]", "", b)
    if a_clean == b_clean:
        return 0.95
    return SequenceMatcher(None, a, b).ratio()


def detect_aliases(authors: dict[str, str]) -> dict:
    """
    Detect author aliases using multiple signals.

    Returns:
        {
            "aliases": {variant_email: canonical_email, ...},
            "display_names": {canonical_email: display_name, ...},
            "exclude": [bot_emails...],
            "groups": [{canonical, variants, reason}, ...],  # for reporting
        }
    """
    aliases = {}
    display_names = {}
    exclude = []
    groups = []

    # Step 1: Identify bots
    for email in authors:
        if is_bot_email(email):
            exclude.append(email)

    # Step 2: Group by exact name match
    name_to_emails = defaultdict(list)
    for email, name in authors.items():
        if email in exclude:
            continue
        name_to_emails[name.lower().strip()].append(email)

    # Step 3: For each name group, pick canonical email
    for name_lower, emails in name_to_emails.items():
        if len(emails) < 2:
            continue

        # Prefer corporate email as canonical
        corporate = [e for e in emails if is_corporate_email(e)]
        local = [e for e in emails if is_local_email(e)]
        other = [e for e in emails if e not in corporate and e not in local]

        if corporate:
            canonical = corporate[0]  # first corporate email
        elif other:
            canonical = other[0]
        else:
            canonical = emails[0]

        variants = [e for e in emails if e != canonical]
        if variants:
            for v in variants:
                aliases[v] = canonical
            display_names[canonical] = authors.get(canonical, name_lower)
            groups.append({
                "canonical": canonical,
                "variants": variants,
                "name": authors.get(canonical, name_lower),
                "reason": "exact_name_match",
            })

    # Step 4: Find prefix-similar emails on same domain (not yet aliased)
    remaining = [e for e in authors if e not in aliases and e not in exclude]
    domain_groups = defaultdict(list)
    for email in remaining:
        if "@" in email:
            domain = email.split("@")[1]
            domain_groups[domain].append(email)

    for domain, domain_emails in domain_groups.items():
        if len(domain_emails) < 2:
            continue
        for i, email_a in enumerate(domain_emails):
            for email_b in domain_emails[i + 1:]:
                if email_a in aliases or email_b in aliases:
                    continue
                psim = prefix_similarity(email_a, email_b)
                nsim = name_similarity(
                    authors.get(email_a, ""), authors.get(email_b, "")
                )
                # High prefix similarity + reasonable name similarity
                if psim > 0.8 and nsim > 0.6:
                    canonical = email_a
                    aliases[email_b] = canonical
                    display_names[canonical] = authors.get(canonical, email_prefix(canonical))
                    groups.append({
                        "canonical": canonical,
                        "variants": [email_b],
                        "name": authors.get(canonical, ""),
                        "reason": f"prefix_sim={psim:.2f}, name_sim={nsim:.2f}",
                    })

    # Step 5: Handle no-domain emails (e.g., "username" without @)
    no_domain = [e for e in remaining if "@" not in e and e not in aliases]
    for nd_email in no_domain:
        # Try to match against corporate emails by prefix
        nd_clean = re.sub(r"[.\-_]", "", nd_email.lower())
        for corp_email in remaining:
            if "@" not in corp_email or corp_email in aliases:
                continue
            corp_clean = re.sub(r"[.\-_]", "", email_prefix(corp_email))
            if nd_clean == corp_clean:
                aliases[nd_email] = corp_email
                groups.append({
                    "canonical": corp_email,
                    "variants": [nd_email],
                    "name": authors.get(corp_email, ""),
                    "reason": "no_domain_prefix_match",
                })
                break

    return {
        "aliases": aliases,
        "display_names": display_names,
        "exclude": sorted(exclude),
        "groups": groups,
    }


def write_yaml(result: dict, out_path: pathlib.Path):
    """Write author_aliases.yaml config file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Author alias mapping for HyperRetrieval",
        "# Generated by build/01b_detect_author_aliases.py",
        "# Review and edit before use — auto-detection may have false positives.",
        "",
        "# Maps variant emails to their canonical email.",
        "# When building ownership index, all variant emails are resolved",
        "# to their canonical form before counting.",
        "aliases:",
    ]
    for variant, canonical in sorted(result["aliases"].items()):
        lines.append(f"  {variant}: {canonical}")

    lines.append("")
    lines.append("# Canonical email to display name mapping.")
    lines.append("display_names:")
    for email, name in sorted(result["display_names"].items()):
        lines.append(f'  {email}: "{name}"')

    lines.append("")
    lines.append("# Emails to exclude from ownership (bots, CI, placeholders).")
    lines.append("exclude:")
    for email in result["exclude"]:
        lines.append(f"  - {email}")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Written: {out_path}", flush=True)


def write_report(result: dict, authors: dict, out_path: pathlib.Path):
    """Write human-readable audit report."""
    total_authors = len(authors)
    total_aliases = len(result["aliases"])
    total_bots = len(result["exclude"])
    canonical_count = total_authors - total_aliases - total_bots

    lines = [
        "=" * 60,
        "Author Alias Detection Report",
        "=" * 60,
        "",
        f"Total emails in ownership index:  {total_authors}",
        f"Aliases detected (duplicates):    {total_aliases}",
        f"Bot/CI accounts excluded:         {total_bots}",
        f"Canonical unique authors:         {canonical_count}",
        f"Reduction:                        {(total_aliases + total_bots) / total_authors * 100:.1f}%",
        "",
        "-" * 60,
        "ALIAS GROUPS",
        "-" * 60,
    ]

    for g in sorted(result["groups"], key=lambda x: len(x["variants"]), reverse=True):
        lines.append(f"\n  Canonical: {g['canonical']}  ({g['name']})")
        lines.append(f"  Reason:    {g['reason']}")
        for v in g["variants"]:
            lines.append(f"    <- {v}  ({authors.get(v, '?')})")

    if result["exclude"]:
        lines.append("")
        lines.append("-" * 60)
        lines.append("EXCLUDED (bots/CI)")
        lines.append("-" * 60)
        for e in result["exclude"]:
            lines.append(f"  {e}  ({authors.get(e, '?')})")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Written: {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Detect author aliases in ownership index")
    parser.add_argument("--artifact-dir", default=None,
                        help="Directory containing ownership_index.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print report only, don't write alias file")
    args = parser.parse_args()

    # Load config for default paths
    try:
        import yaml
        config_path = pathlib.Path(__file__).parent.parent / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
        else:
            cfg = {}
    except ImportError:
        cfg = {}

    artifact_dir = pathlib.Path(
        args.artifact_dir or cfg.get("artifact_dir",
            "/home/beast/projects/workspaces/juspay/artifacts")
    )
    ownership_path = artifact_dir / "ownership_index.json"

    if not ownership_path.exists():
        print(f"ERROR: {ownership_path} not found. Run 08_build_ownership.py first.")
        return

    print(f"Loading {ownership_path}...", flush=True)
    data = json.loads(ownership_path.read_text())
    authors = data.get("authors", {})
    print(f"Found {len(authors)} unique author emails", flush=True)

    result = detect_aliases(authors)

    total_aliases = len(result["aliases"])
    total_bots = len(result["exclude"])
    canonical = len(authors) - total_aliases - total_bots
    print(f"\nResults:")
    print(f"  Aliases found:     {total_aliases}")
    print(f"  Bots excluded:     {total_bots}")
    print(f"  Canonical authors: {canonical}")
    print(f"  Reduction:         {(total_aliases + total_bots) / len(authors) * 100:.1f}%")

    # Write report
    report_path = artifact_dir / "suggested_aliases_report.txt"
    write_report(result, authors, report_path)

    if not args.dry_run:
        config_dir = pathlib.Path(__file__).parent.parent / "config"
        yaml_path = config_dir / "author_aliases.yaml"
        write_yaml(result, yaml_path)
    else:
        print("\n(dry-run: alias file not written)")


if __name__ == "__main__":
    main()
