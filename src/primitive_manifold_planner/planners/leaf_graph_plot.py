from __future__ import annotations

from typing import Optional, List, Tuple, Dict, Iterable, Any
import matplotlib.pyplot as plt
import numpy as np


Node = Tuple[str, object]
EdgeKey = Tuple[Node, Node]


def _edge_key(a: Node, b: Node) -> EdgeKey:
    return (a, b) if str(a) <= str(b) else (b, a)


def _node_positions_from_keys(nodes: List[Node]) -> Dict[Node, np.ndarray]:
    """
    Build simple 2D positions for graph nodes so the leaf-graph can be visualized.

    x-position groups by family name
    y-position stacks lambdas / leaves within each family
    """
    family_names = sorted(list({fam for fam, _ in nodes}))
    family_to_x = {fam: i for i, fam in enumerate(family_names)}

    grouped: Dict[str, List[Node]] = {}
    for node in nodes:
        grouped.setdefault(node[0], []).append(node)

    pos: Dict[Node, np.ndarray] = {}
    for fam, fam_nodes in grouped.items():
        fam_nodes_sorted = sorted(fam_nodes, key=lambda x: str(x[1]))
        ys = np.linspace(0.0, 1.0, max(len(fam_nodes_sorted), 2))
        if len(fam_nodes_sorted) == 1:
            ys = np.array([0.5])
        for idx, node in enumerate(fam_nodes_sorted):
            pos[node] = np.array([family_to_x[fam], ys[idx]], dtype=float)

    return pos


def _route_to_edge_keys(route: Optional[Iterable[Any]]) -> List[EdgeKey]:
    """
    Accept either:
      - list of node tuples: [(fam, lam), (fam, lam), ...]
      - list of edge-like objects with .src and .dst
      - list of (src, dst) tuples
    """
    if route is None:
        return []

    route = list(route)
    if len(route) == 0:
        return []

    # case 1: node sequence
    if isinstance(route[0], tuple) and len(route[0]) == 2 and isinstance(route[0][0], str):
        return [_edge_key(route[i], route[i + 1]) for i in range(len(route) - 1)]

    out: List[EdgeKey] = []
    for item in route:
        if hasattr(item, "src") and hasattr(item, "dst"):
            out.append(_edge_key(item.src, item.dst))
        elif isinstance(item, tuple) and len(item) == 2:
            out.append(_edge_key(item[0], item[1]))
    return out


def plot_leaf_graph_from_routes(
    nodes: List[Node],
    edges: List[Tuple[Node, Node]],
    primary_route: Optional[Iterable[Any]] = None,
    secondary_route: Optional[Iterable[Any]] = None,
    edge_labels: Optional[Dict[EdgeKey, str]] = None,
    title: str = "Leaf graph",
    primary_label: str = "primary",
    secondary_label: str = "secondary",
):
    """
    Plot a leaf graph from explicit node/edge lists and optionally overlay two routes.

    - all edges: light gray
    - primary_route: solid red
    - secondary_route: dashed blue
    - optional edge_labels: annotate edge midpoints
    """
    unique_nodes = list(dict.fromkeys(nodes))
    pos = _node_positions_from_keys(unique_nodes)

    primary_edges = set(_route_to_edge_keys(primary_route))
    secondary_edges = set(_route_to_edge_keys(secondary_route))

    fig, ax = plt.subplots(figsize=(11, 6))

    # Base graph
    drawn_edge_keys = set()
    for src, dst in edges:
        if src not in pos or dst not in pos:
            continue
        ek = _edge_key(src, dst)
        if ek in drawn_edge_keys:
            continue
        drawn_edge_keys.add(ek)

        p_src = pos[src]
        p_dst = pos[dst]

        ax.plot(
            [p_src[0], p_dst[0]],
            [p_src[1], p_dst[1]],
            color="lightgray",
            linewidth=1.2,
            alpha=0.8,
            zorder=1,
        )

        if edge_labels is not None and ek in edge_labels:
            mid = 0.5 * (p_src + p_dst)
            ax.text(
                mid[0],
                mid[1] + 0.03,
                str(edge_labels[ek]),
                ha="center",
                va="bottom",
                fontsize=8,
                color="dimgray",
                zorder=2,
            )

    # Primary route
    for ek in primary_edges:
        src, dst = ek
        if src not in pos or dst not in pos:
            continue
        p_src = pos[src]
        p_dst = pos[dst]
        ax.plot(
            [p_src[0], p_dst[0]],
            [p_src[1], p_dst[1]],
            color="tab:red",
            linewidth=3.0,
            alpha=0.95,
            zorder=3,
        )

    # Secondary route
    for ek in secondary_edges:
        src, dst = ek
        if src not in pos or dst not in pos:
            continue
        p_src = pos[src]
        p_dst = pos[dst]
        ax.plot(
            [p_src[0], p_dst[0]],
            [p_src[1], p_dst[1]],
            color="tab:blue",
            linewidth=2.6,
            linestyle="--",
            alpha=0.95,
            zorder=4,
        )

    # Draw nodes
    for node, p in pos.items():
        fam, lam = node
        ax.scatter(p[0], p[1], s=180, color="white", edgecolor="black", zorder=5)
        ax.text(
            p[0],
            p[1],
            f"{fam}\n[{lam}]",
            ha="center",
            va="center",
            fontsize=8,
            zorder=6,
        )

    # Legend handles
    h1, = ax.plot([], [], color="tab:red", linewidth=3.0, label=primary_label)
    h2, = ax.plot([], [], color="tab:blue", linewidth=2.6, linestyle="--", label=secondary_label)
    ax.legend(handles=[h1, h2], loc="upper left")

    ax.set_title(title)
    ax.set_xticks(range(len(sorted(list({fam for fam, _ in unique_nodes})))))
    ax.set_xticklabels(sorted(list({fam for fam, _ in unique_nodes})), rotation=20)
    ax.set_yticks([])
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig, ax


def plot_leaf_graph(graph, route=None, title="Leaf graph"):
    """
    Backward-compatible wrapper for the original graph plotting style.
    """
    nodes = list(graph.adjacency.keys())
    edges: List[Tuple[Node, Node]] = []
    for src, out_edges in graph.adjacency.items():
        for edge in out_edges:
            edges.append((src, edge.dst))

    return plot_leaf_graph_from_routes(
        nodes=nodes,
        edges=edges,
        primary_route=route,
        secondary_route=None,
        edge_labels=None,
        title=title,
        primary_label="route",
        secondary_label="",
    )