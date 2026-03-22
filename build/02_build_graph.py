"""
Stage 2 — Build NetworkX graph, run Louvain at module level, propagate to nodes.
Input:  pipeline/output/raw_graph.json
Output: pipeline/output/graph_clustered.json
"""
import json, pathlib, collections

OUT_DIR = pathlib.Path(__file__).parent / "output"

def main():
    import networkx as nx
    import community as community_louvain

    raw = json.loads((OUT_DIR / "raw_graph.json").read_text())
    nodes = raw["nodes"]
    edges = raw["edges"]

    print(f"Input: {len(nodes)} nodes, {len(edges)} edges")

    # ── Build MODULE-LEVEL graph ──────────────────────────────────────────────
    # Nodes are functions/types, but import edges connect module names.
    # Cluster at module level, then assign functions to their module's cluster.

    # Collect all modules
    all_modules = set(n["module"] for n in nodes if n.get("module"))

    # Build module graph
    MG = nx.Graph()
    for m in all_modules:
        MG.add_node(m)

    import_edges_added = 0
    co_change_by_file  = collections.defaultdict(set)   # file -> set of modules

    for e in edges:
        kind = e.get("kind", "")
        src  = e.get("from", "")
        dst  = e.get("to", "")

        if kind == "import":
            # Both should be module names — add edge if both exist in our modules
            # or if dst is a prefix of a known module (external import → skip)
            if src in all_modules and dst in all_modules:
                MG.add_edge(src, dst, kind="import", weight=1)
                import_edges_added += 1
            elif src in all_modules:
                # External import — still add but as a phantom
                if not MG.has_node(dst):
                    MG.add_node(dst, external=True)
                MG.add_edge(src, dst, kind="import", weight=1)

        elif kind == "co_change":
            # src/dst are file paths — map back to modules
            # Find which modules live in those files
            weight = e.get("weight", 1)
            if weight >= 2:   # only meaningful co-changes
                co_change_by_file[src]
                co_change_by_file[dst]

    # Map files → modules for co-change edges
    file_to_modules = collections.defaultdict(set)
    for n in nodes:
        if n.get("file") and n.get("module"):
            file_to_modules[n["file"]].add(n["module"])

    co_change_added = 0
    for e in edges:
        if e.get("kind") == "co_change" and e.get("weight", 1) >= 2:
            src_mods = file_to_modules.get(e["from"], set())
            dst_mods = file_to_modules.get(e["to"],   set())
            for sm in src_mods:
                for dm in dst_mods:
                    if sm != dm and sm in all_modules and dm in all_modules:
                        if MG.has_edge(sm, dm):
                            MG[sm][dm]["weight"] = MG[sm][dm].get("weight", 1) + 1
                        else:
                            MG.add_edge(sm, dm, kind="co_change", weight=e["weight"])
                            co_change_added += 1

    print(f"Module graph: {MG.number_of_nodes()} nodes, {MG.number_of_edges()} edges")
    print(f"  Import edges: {import_edges_added}, co-change edges: {co_change_added}")

    # Keep only internal modules for Louvain (remove phantom external nodes)
    internal_modules = [m for m in all_modules if MG.has_node(m)]
    sub = MG.subgraph(internal_modules)

    # Handle disconnected graph — Louvain needs connected components
    # Run on the largest connected component, assign isolated nodes to their own cluster
    components = list(nx.connected_components(sub))
    print(f"  Connected components: {len(components)}")
    print(f"  Largest component: {max(len(c) for c in components)} modules")

    # Run Louvain on each component that has >1 node
    partition = {}
    cluster_offset = 0
    for comp in sorted(components, key=len, reverse=True):
        if len(comp) == 1:
            # Isolated module — give it own cluster
            partition[list(comp)[0]] = cluster_offset
            cluster_offset += 1
        else:
            comp_sub = sub.subgraph(comp)
            comp_partition = community_louvain.best_partition(
                comp_sub, random_state=42, resolution=1.2
            )
            # Offset cluster IDs to avoid collisions
            max_local = max(comp_partition.values()) + 1
            for mod, cid in comp_partition.items():
                partition[mod] = cid + cluster_offset
            cluster_offset += max_local

    cluster_counts = collections.Counter(partition.values())
    real_clusters   = {cid: cnt for cid, cnt in cluster_counts.items() if cnt >= 2}
    print(f"\nLouvain: {len(cluster_counts)} total clusters")
    print(f"  Clusters with 2+ modules: {len(real_clusters)}")
    print(f"  Top 10 by size:")
    for cid, cnt in sorted(real_clusters.items(), key=lambda x: -x[1])[:10]:
        # Show dominant service for this cluster
        mods_in = [m for m, c in partition.items() if c == cid]
        print(f"    cluster_{cid}: {cnt} modules")

    # ── Propagate cluster IDs down to individual nodes ────────────────────────
    node_index = {n["id"]: n for n in nodes}
    for n in nodes:
        mod = n.get("module", "")
        n["cluster"] = partition.get(mod, -1)

    # ── Build full node-level directed graph (for runtime traversal) ──────────
    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n["id"], **{k: v for k, v in n.items()
                               if isinstance(v, (str, int, float, bool))})

    all_node_ids = set(n["id"] for n in nodes)
    for e in edges:
        src, dst, kind = e.get("from",""), e.get("to",""), e.get("kind","")
        if src in all_node_ids and dst in all_node_ids:
            G.add_edge(src, dst, kind=kind, weight=e.get("weight", 1))

    print(f"\nFull node graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ── Build cluster metadata ────────────────────────────────────────────────
    clusters = collections.defaultdict(lambda: {"nodes": [], "summary": None})
    for n in nodes:
        clusters[n["cluster"]]["nodes"].append(n["id"])

    for cid, data in clusters.items():
        svc_counter = collections.Counter()
        lang_counter = collections.Counter()
        ghost_counter = collections.Counter()
        for nid in data["nodes"]:
            nd = node_index.get(nid, {})
            svc_counter[nd.get("service", "unknown")] += 1
            lang_counter[nd.get("lang", "unknown")] += 1
            for g in nd.get("ghost_deps", []):
                ghost_counter[g] += 1
        data["services"]         = dict(svc_counter.most_common())
        data["langs"]            = dict(lang_counter.most_common())
        data["ghost_deps"]       = [g for g, _ in ghost_counter.most_common(5)]
        data["dominant_service"] = svc_counter.most_common(1)[0][0] if svc_counter else "unknown"

    nx_data = nx.node_link_data(G)

    result = {
        "nodes":    nodes,
        "edges":    edges,
        "clusters": {str(k): v for k, v in clusters.items()},
        "networkx": nx_data,
        "stats": {
            "n_nodes":            G.number_of_nodes(),
            "n_edges":            G.number_of_edges(),
            "n_clusters":         len(cluster_counts),
            "n_real_clusters":    len(real_clusters),
            "n_module_nodes":     MG.number_of_nodes(),
            "n_module_edges":     MG.number_of_edges(),
        }
    }

    out_path = OUT_DIR / "graph_clustered.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n✓ Wrote clustered graph → {out_path}")
    print(f"  {len(nodes)} symbol nodes | {len(edges)} edges | "
          f"{len(real_clusters)} meaningful clusters")


if __name__ == "__main__":
    main()
