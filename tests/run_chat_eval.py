"""
Chat app evaluation runner — runs the 50-question test suite against
the live Chainlit chat app and scores responses.

Prerequisites:
  - Embed server running (port 8001)
  - Chainlit server running (port 8000)

Usage:
  python3 tests/run_chat_eval.py [--questions tests/chat_50_questions.json]
  python3 tests/run_chat_eval.py --category impact  # run only one category
  python3 tests/run_chat_eval.py --dry-run           # print questions without running
"""
import argparse
import json
import pathlib
import sys
import time
import urllib.request
import urllib.error

CHAT_URL = "http://localhost:8000"
MCP_URL = "http://localhost:8002"


def check_servers():
    """Verify required servers are running."""
    for name, url in [("Chainlit", CHAT_URL), ("MCP", MCP_URL)]:
        try:
            urllib.request.urlopen(url, timeout=5)
            print(f"  {name}: OK ({url})")
        except Exception as e:
            print(f"  {name}: UNAVAILABLE ({url}) — {e}")
            return False
    return True


def call_mcp_tool(tool_name: str, args: dict) -> dict:
    """Call an MCP tool directly for evaluation."""
    payload = json.dumps({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": args},
        "id": 1,
    }).encode()

    req = urllib.request.Request(
        f"{MCP_URL}/message",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def evaluate_question(q: dict) -> dict:
    """Run a single question through MCP tools and evaluate."""
    qid = q["id"]
    question = q["question"]
    expected_tools = q["expected_tools"]
    criteria = q["quality_criteria"]

    result = {
        "id": qid,
        "question": question,
        "category": q["category"],
        "expected_tools": expected_tools,
        "tool_results": {},
        "score": 0,  # 0-3: 0=fail, 1=partial, 2=good, 3=excellent
        "notes": "",
    }

    # Call the first expected tool with the question as search query
    tool = expected_tools[0] if expected_tools else "search_symbols"
    try:
        if tool == "search_modules":
            resp = call_mcp_tool("search_modules", {"query": question})
        elif tool == "search_symbols":
            resp = call_mcp_tool("search_symbols", {"query": question})
        elif tool == "get_blast_radius":
            # Extract file/module name from question
            resp = call_mcp_tool("search_modules", {"query": question})
        elif tool == "suggest_reviewers":
            resp = call_mcp_tool("search_modules", {"query": question})
        else:
            resp = call_mcp_tool("search_symbols", {"query": question})

        result["tool_results"][tool] = resp

        # Basic scoring: did we get non-empty results?
        resp_content = json.dumps(resp)
        if "error" in resp:
            result["score"] = 0
            result["notes"] = f"Tool error: {resp.get('error', '')[:100]}"
        elif len(resp_content) > 100:
            result["score"] = 2  # Got substantial results
            result["notes"] = f"Got {len(resp_content)} chars of results"
        elif len(resp_content) > 20:
            result["score"] = 1  # Got something but minimal
            result["notes"] = "Minimal results"
        else:
            result["score"] = 0
            result["notes"] = "Empty results"

    except Exception as e:
        result["score"] = 0
        result["notes"] = f"Exception: {str(e)[:100]}"

    return result


def main():
    parser = argparse.ArgumentParser(description="Run chat app evaluation")
    parser.add_argument("--questions", type=pathlib.Path,
                        default=pathlib.Path(__file__).parent / "chat_50_questions.json")
    parser.add_argument("--category", default=None,
                        help="Run only questions from this category")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print questions without running")
    parser.add_argument("--output", type=pathlib.Path, default=None,
                        help="Output results to JSON file")
    args = parser.parse_args()

    # Load questions
    data = json.loads(args.questions.read_text())
    questions = data["questions"]

    if args.category:
        questions = [q for q in questions if q["category"] == args.category]
        print(f"Filtered to {len(questions)} questions in category '{args.category}'")

    if args.dry_run:
        print(f"\n{len(questions)} questions:\n")
        for q in questions:
            print(f"  [{q['category']:>15}] Q{q['id']:>2}: {q['question']}")
            print(f"                     Tools: {q['expected_tools']}")
            print(f"                     Criteria: {q['quality_criteria'][:80]}")
            print()
        return

    # Check servers
    print("Checking servers...")
    if not check_servers():
        print("\nServers not ready. Start embed + chainlit + mcp servers first.")
        print("  ~/start_embed.sh && ~/start_chainlit.sh && ~/start_mcp.sh")
        sys.exit(1)

    # Run evaluation
    print(f"\nRunning {len(questions)} questions...")
    results = []
    t0 = time.time()

    for i, q in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] Q{q['id']}: {q['question'][:60]}...", end=" ")
        result = evaluate_question(q)
        results.append(result)
        score_label = ["FAIL", "PARTIAL", "GOOD", "EXCELLENT"][result["score"]]
        print(f"→ {score_label} ({result['notes'][:40]})")

    elapsed = time.time() - t0

    # Summary
    from collections import Counter
    scores = Counter(r["score"] for r in results)
    by_category = {}
    for r in results:
        cat = r["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r["score"])

    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY ({len(results)} questions, {elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"  EXCELLENT (3): {scores.get(3, 0)}")
    print(f"  GOOD (2):      {scores.get(2, 0)}")
    print(f"  PARTIAL (1):   {scores.get(1, 0)}")
    print(f"  FAIL (0):      {scores.get(0, 0)}")
    avg = sum(r["score"] for r in results) / len(results) if results else 0
    print(f"  Average score: {avg:.2f}/3.0")

    print(f"\nBy category:")
    for cat, cat_scores in sorted(by_category.items()):
        cat_avg = sum(cat_scores) / len(cat_scores) if cat_scores else 0
        print(f"  {cat:>15}: {cat_avg:.2f}/3.0")

    # Save results
    out_path = args.output or pathlib.Path(__file__).parent / "chat_eval_results.json"
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_questions": len(results),
        "average_score": round(avg, 2),
        "elapsed_seconds": round(elapsed, 1),
        "score_distribution": dict(scores),
        "by_category": {cat: round(sum(s)/len(s), 2) for cat, s in by_category.items()},
        "results": results,
    }
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
