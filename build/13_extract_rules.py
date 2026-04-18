"""
13_extract_rules.py — Offline feedback-to-rules extraction (T-030 Phase 2)

Reads ~/.hyperretrieval/feedback_signals.jsonl
Extracts learned rules via signal accumulation (Bugbot-style):
  - helpful signal → positive vote for this (tool, pattern) pair
  - not_helpful signal → negative vote
  - confidence = positive / total
  - status: candidate (1-2 signals), active (≥3 AND confidence ≥ 0.7),
             disabled (≥3 AND confidence ≤ 0.3), contested (in between)

Writes ~/.hyperretrieval/active_rules.json

Usage:
    python3 build/13_extract_rules.py [--signals PATH] [--output PATH] [--min-signals N]
"""
import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ── Thresholds ─────────────────────────────────────────────────────────────────
PROMOTE_MIN_SIGNALS   = 3    # signals before promotion is considered
PROMOTE_CONFIDENCE    = 0.70  # confidence threshold for "active"
DEMOTE_CONFIDENCE     = 0.30  # confidence threshold for "disabled"


# ── Query normalisation ────────────────────────────────────────────────────────

def _extract_files(query: str) -> list[str]:
    """Pull file paths from a query string — handles JSON arrays and plain text."""
    # JSON array: ["a.hs", "b.py"]
    m = re.search(r'\[([^\]]+)\]', query)
    if m:
        try:
            items = json.loads('[' + m.group(1) + ']')
            return sorted(str(i) for i in items if i)
        except Exception:
            pass
    # Bare paths: any token containing / or ending in known extension
    tokens = re.findall(r'\S+', query)
    paths = [t.strip('",') for t in tokens
             if '/' in t or re.search(r'\.(hs|py|rs|ts|js|go|java|groovy)$', t)]
    return sorted(set(paths)) if paths else []


def _normalise_key(tool: str, query: str) -> str:
    """Stable key for grouping: tool + sorted file list (or lowercased query)."""
    files = _extract_files(query)
    if files:
        payload = tool + "|files:" + ",".join(files)
    else:
        payload = tool + "|q:" + query.lower().strip()[:120]
    return hashlib.md5(payload.encode()).hexdigest()[:12]


def _human_label(tool: str, query: str) -> str:
    files = _extract_files(query)
    if files:
        short = [f.split("/")[-1] for f in files[:3]]
        suffix = "…" if len(files) > 3 else ""
        return f"{tool}({', '.join(short)}{suffix})"
    return f"{tool}({query[:60]})"


# ── Core extraction ────────────────────────────────────────────────────────────

class RuleExtractor:
    def __init__(self, min_signals: int = PROMOTE_MIN_SIGNALS):
        self.min_signals = min_signals
        # key → {"tool", "label", "positive", "negative", "last_ts", "summaries"}
        self._groups: dict[str, dict] = defaultdict(lambda: {
            "tool": "", "label": "", "positive": 0, "negative": 0,
            "last_ts": 0.0, "summaries": [],
        })

    def feed(self, entry: dict) -> None:
        tool   = entry.get("tool", "unknown")
        query  = entry.get("query", "")
        signal = entry.get("signal", "")
        key    = _normalise_key(tool, query)
        g      = self._groups[key]
        g["tool"]  = tool
        g["label"] = _human_label(tool, query)
        if signal == "helpful":
            g["positive"] += 1
        elif signal == "not_helpful":
            g["negative"] += 1
        ts = entry.get("ts", 0.0)
        if ts > g["last_ts"]:
            g["last_ts"] = ts
        summary = entry.get("result_summary", "").strip()
        if summary and summary not in g["summaries"]:
            g["summaries"].append(summary)

    def rules(self) -> list[dict]:
        out = []
        for key, g in self._groups.items():
            total = g["positive"] + g["negative"]
            if total == 0:
                continue
            confidence = g["positive"] / total

            if total >= self.min_signals:
                if confidence >= PROMOTE_CONFIDENCE:
                    status = "active"
                elif confidence <= DEMOTE_CONFIDENCE:
                    status = "disabled"
                else:
                    status = "contested"
            else:
                status = "candidate"

            last_ts = g["last_ts"]
            last_seen = (
                datetime.fromtimestamp(last_ts, tz=timezone.utc).isoformat()
                if last_ts else None
            )

            out.append({
                "id":         key,
                "tool":       g["tool"],
                "label":      g["label"],
                "positive":   g["positive"],
                "negative":   g["negative"],
                "total":      total,
                "confidence": round(confidence, 3),
                "status":     status,
                "last_seen":  last_seen,
                "sample_summaries": g["summaries"][:3],
            })

        # Sort: active first, then by confidence desc, then total desc
        order = {"active": 0, "candidate": 1, "contested": 2, "disabled": 3}
        out.sort(key=lambda r: (order.get(r["status"], 9), -r["confidence"], -r["total"]))
        return out


# ── I/O ────────────────────────────────────────────────────────────────────────

def load_signals(path: Path) -> list[dict]:
    if not path.exists():
        return []
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def save_rules(rules: list[dict], signals: list[dict], output: Path) -> None:
    active    = sum(1 for r in rules if r["status"] == "active")
    candidate = sum(1 for r in rules if r["status"] == "candidate")
    contested = sum(1 for r in rules if r["status"] == "contested")
    disabled  = sum(1 for r in rules if r["status"] == "disabled")

    payload = {
        "version":   1,
        "generated": datetime.now(tz=timezone.utc).isoformat(),
        "stats": {
            "total_signals":  len(signals),
            "pattern_groups": len(rules),
            "active":         active,
            "candidate":      candidate,
            "contested":      contested,
            "disabled":       disabled,
        },
        "rules": rules,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(payload, f, indent=2)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Extract feedback rules from signals")
    p.add_argument("--signals",     default=str(Path.home() / ".hyperretrieval" / "feedback_signals.jsonl"))
    p.add_argument("--output",      default=str(Path.home() / ".hyperretrieval" / "active_rules.json"))
    p.add_argument("--min-signals", type=int, default=PROMOTE_MIN_SIGNALS,
                   help="Min signals before active/disabled promotion (default 3)")
    p.add_argument("--summary",     action="store_true", help="Print summary to stdout")
    args = p.parse_args()

    signals_path = Path(args.signals)
    output_path  = Path(args.output)

    signals = load_signals(signals_path)
    if not signals:
        print(f"No signals at {signals_path}. Nothing to extract.", file=sys.stderr)
        # Still write an empty rules file so downstream consumers don't crash
        save_rules([], [], output_path)
        print(f"Wrote empty rules: {output_path}")
        return

    extractor = RuleExtractor(min_signals=args.min_signals)
    for entry in signals:
        extractor.feed(entry)

    rules = extractor.rules()
    save_rules(rules, signals, output_path)

    active   = [r for r in rules if r["status"] == "active"]
    disabled = [r for r in rules if r["status"] == "disabled"]

    print(f"Signals: {len(signals)} → {len(rules)} patterns")
    print(f"  active={len(active)}  candidate={len(rules)-len(active)-len(disabled)}  disabled={len(disabled)}")
    print(f"Rules written: {output_path}")

    if args.summary and active:
        print("\nActive rules:")
        for r in active:
            print(f"  [{r['confidence']:.0%} conf, {r['total']} signals] {r['label']}")


if __name__ == "__main__":
    main()
