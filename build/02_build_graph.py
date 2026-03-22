"""
Stage 2 — Build NetworkX graph, run Leiden at module level, propagate to nodes.
Input:  output/raw_graph.json
Output: output/graph_clustered.json
"""
import json, pathlib, collections, os

import networkx as nx
import leidenalg, igraph as ig

OUT_DIR = pathlib.Path(os.environ.get("OUTPUT_DIR", pathlib.Path(__file__).parent / "output"))


def load_service_profiles():
    """Load service_profiles from config.yaml if available."""
    config_path = os.environ.get("CONFIG_PATH", "")
    if not config_path:
        workspace_dir = os.environ.get("WORKSPACE_DIR", "")
        if workspace_dir:
            config_path = str(pathlib.Path(workspace_dir) / "config.yaml")
    if config_path and pathlib.Path(config_path).exists():
        try:
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            return cfg.get("service_profiles", {})
        except Exception as e:
            print(f"  Warning: could not load service_profiles from config: {e}")
    return {}


def main():
    service_profiles = load_service_profiles()
    print(f"  Service profiles loaded: {list(service_profiles.keys()) or '(none)'}")

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

        # co_change edges are handled in the second pass below (dead lines removed)

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

    # Keep only internal modules for Leiden (remove phantom external nodes)
    internal_modules = [m for m in all_modules if MG.has_node(m)]
    sub = MG.subgraph(internal_modules)

    # Handle disconnected graph — run on each connected component
    components = list(nx.connected_components(sub))
    print(f"  Connected components: {len(components)}")
    print(f"  Largest component: {max(len(c) for c in components)} modules")

    # Run Leiden on each component that has >1 node
    partition = {}
    cluster_offset = 0
    for comp in sorted(components, key=len, reverse=True):
        if len(comp) == 1:
            # Isolated module — give it own cluster
            partition[list(comp)[0]] = cluster_offset
            cluster_offset += 1
        else:
            comp_sub = sub.subgraph(comp)

            # Convert networkx subgraph to igraph
            nodes_list = list(comp_sub.nodes())
            node_idx = {n: i for i, n in enumerate(nodes_list)}
            edges_list = [(node_idx[u], node_idx[v]) for u, v in comp_sub.edges()]
            weights = [comp_sub[u][v].get('weight', 1) for u, v in comp_sub.edges()]
            ig_graph = ig.Graph(n=len(nodes_list), edges=edges_list)
            ig_graph.es['weight'] = weights
            leiden_partition = leidenalg.find_partition(
                ig_graph,
                leidenalg.RBConfigurationVertexPartition,
                weights='weight',
                resolution_parameter=1.2,
                seed=42,
            )
            comp_partition = {nodes_list[i]: leiden_partition.membership[i]
                              for i in range(len(nodes_list))}

            # Offset cluster IDs to avoid collisions
            max_local = (max(comp_partition.values()) + 1) if comp_partition else 1
            for mod, cid in comp_partition.items():
                partition[mod] = cid + cluster_offset
            cluster_offset += max_local

    cluster_counts = collections.Counter(partition.values())
    real_clusters   = {cid: cnt for cid, cnt in cluster_counts.items() if cnt >= 2}
    print(f"\nLeiden: {len(cluster_counts)} total clusters")
    print(f"  Clusters with 2+ modules: {len(real_clusters)}")
    print(f"  Top 10 by size:")
    for cid, cnt in sorted(real_clusters.items(), key=lambda x: -x[1])[:10]:
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

        # Attach traffic_weight from service_profiles for the dominant service
        dominant_svc = data["dominant_service"]
        if service_profiles and dominant_svc in service_profiles:
            profile = service_profiles[dominant_svc]
            data["traffic_weight"] = profile.get("traffic_weight")
            data["dominant_service_role"] = profile.get("role")
            data["dominant_service_region"] = profile.get("region")
        else:
            data["traffic_weight"] = None
            data["dominant_service_role"] = None
            data["dominant_service_region"] = None

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
    print(f"\nWrote clustered graph -> {out_path}")
    print(f"  {len(nodes)} symbol nodes | {len(edges)} edges | "
          f"{len(real_clusters)} meaningful clusters")


if __name__ == "__main__":
    main()
