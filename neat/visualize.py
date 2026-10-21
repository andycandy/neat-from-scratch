"""Draw a genome as a left-to-right network diagram (matplotlib only)."""
import matplotlib

matplotlib.use("Agg")                     # render to file, never open a window
import matplotlib.pyplot as plt  # noqa: E402

from neat.genes import Genome, NodeType  # noqa: E402
from neat.network import topo_order  # noqa: E402

ACT_COLOURS = {
    None: "lightgray",                    # inputs
    "identity": "white",
    "sigmoid": "tab:green",
    "tanh": "tab:orange",
    "relu": "tab:purple",
    "sin": "tab:cyan",
    "gauss": "tab:pink",
    "abs": "tab:olive",
    "step": "tab:brown",
}


def node_positions(genome: Genome) -> dict[int, tuple[float, float]]:
    """x = depth (longest path from an input), outputs pushed to the far right;
    y = nodes spread evenly within each depth column."""
    enabled = [c for c in genome.conns.values() if c.enabled]
    depth: dict[int, int] = {}
    for n in topo_order(genome.nodes, enabled):
        srcs = [c.src for c in enabled if c.dest == n]
        depth[n] = 1 + max(depth[s] for s in srcs) if srcs else 0
    max_d = max([depth[n] for n in depth if genome.nodes[n].type != NodeType.OUTPUT], default=0) + 1
    for n, g in genome.nodes.items():
        if g.type == NodeType.OUTPUT:
            depth[n] = max_d

    columns: dict[int, list[int]] = {}
    for n, d in depth.items():
        columns.setdefault(d, []).append(n)
    pos = {}
    for d, ids in columns.items():
        for k, n in enumerate(sorted(ids)):
            pos[n] = (float(d), k - (len(ids) - 1) / 2)    # centred around y = 0
    return pos


def draw_genome(genome: Genome, path: str, input_names: list[str] | None = None,
                output_names: list[str] | None = None, title: str | None = None) -> None:
    pos = node_positions(genome)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # edges first (zorder 1) so nodes sit on top
    for c in genome.conns.values():
        (x1, y1), (x2, y2) = pos[c.src], pos[c.dest]
        if c.enabled:
            ax.plot([x1, x2], [y1, y2], color="tab:blue" if c.weight > 0 else "tab:red",
                    linewidth=0.5 + 1.5 * abs(c.weight), alpha=0.8, zorder=1)
        else:
            ax.plot([x1, x2], [y1, y2], color="gray", linewidth=0.8, linestyle="--",
                    alpha=0.3, zorder=1)

    inputs = sorted(n for n, g in genome.nodes.items() if g.type == NodeType.INPUT)
    outputs = sorted(n for n, g in genome.nodes.items() if g.type == NodeType.OUTPUT)
    for n, g in genome.nodes.items():
        x, y = pos[n]
        ax.scatter(x, y, s=700, color=ACT_COLOURS.get(g.activation, "white"),
                   edgecolors="black", zorder=2)
        if g.type == NodeType.INPUT:
            i = inputs.index(n)
            label = input_names[i] if input_names else f"x{i}"
        elif g.type == NodeType.OUTPUT:
            j = outputs.index(n)
            label = output_names[j] if output_names else f"out{j}"
        else:
            label = str(n)
        ax.text(x, y, label, ha="center", va="center", fontsize=8, zorder=3)

    used = sorted({g.activation for g in genome.nodes.values() if g.activation is not None})
    for a in used:                                    # invisible points just for the legend
        ax.scatter([], [], s=80, color=ACT_COLOURS.get(a, "white"), edgecolors="black", label=a)
    ax.plot([], [], color="tab:blue", label="w > 0")
    ax.plot([], [], color="tab:red", label="w < 0")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False)

    hidden = sum(g.type == NodeType.HIDDEN for g in genome.nodes.values())
    enabled = sum(c.enabled for c in genome.conns.values())
    ax.set_title(title or f"{hidden} hidden, {enabled} enabled connections", fontsize=10)
    ax.axis("off")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
