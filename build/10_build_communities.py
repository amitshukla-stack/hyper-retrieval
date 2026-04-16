"""
Stage 10 — Build functional community index from co-change graph.

Uses Louvain community detection to discover functional domains —
groups of modules that change together regardless of which service
they belong to. These communities reveal architectural boundaries
that cut across service boundaries.

Reads: cochange_index.json + cross_cochange_index.json (optional)
Outputs: community_index.json

Usage:
  python3 build/10_build_communities.py [--artifact-dir DIR] [--resolution 1.0]
"""
import argparse
import json
import pathlib
import re
from collections import Counter, defaultdict

try:
    import networkx as nx
    import community as community_louvain
except ImportError:
    raise SystemExit("pip install python-louvain networkx")


def build_graph(artifact_dir: pathlib.Path) -> nx.Graph:
    """Build weighted graph from co-change indexes."""
    G = nx.Graph()

    cc_path = artifact_dir / "cochange_index.json"
    if cc_path.exists():
        cc = json.loads(cc_path.read_text())
        for mod, partners in cc.get("edges", {}).items():
            for p in partners:
                if G.has_edge(mod, p["module"]):
                    G[mod][p["module"]]["weight"] += p["weight"]
                else:
                    G.add_edge(mod, p["module"], weight=p["weight"])
        print(f"  Intra-repo co-change: {cc.get('meta', {}).get('total_pairs', 0):,} pairs", flush=True)

    xcc_path = artifact_dir / "cross_cochange_index.json"
    if xcc_path.exists():
        xcc = json.loads(xcc_path.read_text())
        for mod, partners in xcc.get("edges", {}).items():
            for p in partners:
                if G.has_edge(mod, p["module"]):
                    G[mod][p["module"]]["weight"] += p["weight"]
                else:
                    G.add_edge(mod, p["module"], weight=p["weight"])
        print(f"  Cross-repo co-change: {xcc.get('meta', {}).get('total_pairs', 0):,} pairs", flush=True)

    print(f"  Combined graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges", flush=True)
    return G


def auto_label(members: list[str], all_members_count: int) -> str:
    """Generate a human-readable label for a community from member names."""
    # Extract service composition
    services = Counter(m.split("::")[0] for m in members)

    # Extract path segments (skip repo name and very short segments)
    path_segments = Counter()
    for m in members:
        parts = m.split("::")
        for part in parts[1:]:  # skip repo name
            cleaned = part.strip()
            if len(cleaned) > 2 and cleaned[0].isupper():
                path_segments[cleaned] += 1

    # Top services
    top_svcs = [s for s, _ in services.most_common(3)]
    svc_str = "+".join(top_svcs)

    # Top domain terms
    top_terms = [t for t, c in path_segments.most_common(5)
                 if c > len(members) * 0.05]  # appears in >5% of members

    if top_terms:
        domain = ", ".join(top_terms[:3])
        return f"{svc_str} ({domain})"
    else:
        return svc_str


def main():
    parser = argparse.ArgumentParser(description="Build community index from co-change graph")
    parser.add_argument("--artifact-dir", type=pathlib.Path, default=None)
    parser.add_argument("--resolution", type=float, default=1.0,
                        help="Louvain resolution parameter (higher = more communities)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    # Load config for default artifact dir
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

    print("Building community index...", flush=True)
    G = build_graph(artifact_dir)

    if G.number_of_nodes() == 0:
        print("ERROR: No co-change data found. Run 06_build_cochange.py first.")
        return

    print(f"Running Louvain community detection (resolution={args.resolution}, seed={args.seed})...",
          flush=True)

    partition = community_louvain.best_partition(
        G, resolution=args.resolution, random_state=args.seed
    )

    n_communities = len(set(partition.values()))
    modularity = community_louvain.modularity(partition, G)
    print(f"  Communities found: {n_communities}", flush=True)
    print(f"  Modularity: {modularity:.4f}", flush=True)

    # Build community details
    communities = {}
    for comm_id in sorted(set(partition.values())):
        members = [m for m, c in partition.items() if c == comm_id]
        services = Counter(m.split("::")[0] for m in members)

        communities[str(comm_id)] = {
            "size": len(members),
            "services": dict(services.most_common()),
            "cross_service": len(services) > 1,
            "label": auto_label(members, len(partition)),
        }

    # Build module→community map
    module_to_community = {mod: comm for mod, comm in partition.items()}

    # Summary
    cross_service = sum(1 for c in communities.values() if c["cross_service"])
    print(f"\n  Cross-service communities: {cross_service}/{n_communities}", flush=True)
    print(f"\n  Top communities:", flush=True)
    for cid, cdata in sorted(communities.items(),
                              key=lambda x: -x[1]["size"])[:10]:
        label = cdata["label"]
        cs = " [CROSS-SERVICE]" if cdata["cross_service"] else ""
        print(f"    {cid}: {cdata['size']:,} modules — {label}{cs}", flush=True)

    # Write output
    index = {
        "meta": {
            "n_communities": n_communities,
            "modularity": round(modularity, 4),
            "total_modules": len(partition),
            "cross_service_communities": cross_service,
            "resolution": args.resolution,
            "seed": args.seed,
        },
        "communities": communities,
        "module_to_community": module_to_community,
    }

    out_path = artifact_dir / "community_index.json"
    out_path.write_text(json.dumps(index, separators=(",", ":")))
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nWritten: {out_path}  ({size_mb:.1f}MB)", flush=True)


if __name__ == "__main__":
    main()
