"""Figures for the Part 1 Google Doc, sized to stay readable at half-page width
(3.4 in wide, 8 pt text, 200 dpi). Reads only saved results; trains nothing.

uv run python experiments/figures_doc_part1.py
"""
import csv
import json
import pickle
import re
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_slime import HallOfFame, load_latest_checkpoint  # noqa: F401  (HallOfFame: for unpickling)

from backprop.datasets import make_splits
from backprop.train import accuracy
from neat.genes import NodeType
from neat.network import compile_genome, run
from neat.visualize import ACT_COLOURS, node_positions

OUT = Path("doc_assets/part1/figures")
W = 3.4                                    # inches: half of a Google Doc page
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 200})
SLIME_INPUTS = ["my x", "my y", "my vx", "my vy", "ball x", "ball y", "ball vx", "ball vy",
                "opp x", "opp y", "opp vx", "opp vy"]


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / name)


def draw_network(ax, genome, in_names, out_names):
    """Compact network drawing: x = depth, colour = activation, edge colour = sign, width = |w|."""
    pos = node_positions(genome)
    enabled = [c for c in genome.conns.values() if c.enabled]
    w_max = max((abs(c.weight) for c in enabled), default=1.0) or 1.0
    for c in enabled:
        (x1, y1), (x2, y2) = pos[c.src], pos[c.dest]
        ax.plot([x1, x2], [y1, y2], color="tab:blue" if c.weight > 0 else "tab:red",
                lw=0.3 + 2.2 * abs(c.weight) / w_max, alpha=0.7, zorder=1)
    inputs = sorted(n for n, g in genome.nodes.items() if g.type == NodeType.INPUT)
    outputs = sorted(n for n, g in genome.nodes.items() if g.type == NodeType.OUTPUT)
    for n, g in genome.nodes.items():
        x, y = pos[n]
        ax.scatter(x, y, s=60, color=ACT_COLOURS.get(g.activation, "white"), edgecolors="black",
                   linewidths=0.6, zorder=2)
        if n in inputs:
            ax.text(x - 0.12, y, in_names[inputs.index(n)], ha="right", va="center", fontsize=7)
        elif n in outputs:
            ax.text(x + 0.12, y, out_names[outputs.index(n)], ha="left", va="center", fontsize=7)
    used = sorted({g.activation for g in genome.nodes.values()
                   if g.type == NodeType.HIDDEN and g.activation})
    for a in used:
        ax.scatter([], [], s=30, color=ACT_COLOURS[a], edgecolors="black", linewidths=0.6, label=a)
    ax.plot([], [], color="tab:blue", label="w > 0")
    ax.plot([], [], color="tab:red", label="w < 0")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False, handlelength=1.2)
    ax.axis("off")


def hof_history(*logs):
    """[(generation, hall-of-fame score)] from runner logs, one point per logged generation."""
    pts = []
    for f in logs:
        for line in Path(f).read_text(encoding="utf-8", errors="ignore").splitlines():
            m = re.search(r"gen\s+(\d+).*hof\s+(-?[\d.]+)", line)
            if m:
                pts.append((int(m.group(1)), float(m.group(2))))
    return np.array(sorted(dict(pts).items()))


# ---------------------------------------------------------------- Unit 2: XOR sweep
def xor_sweep():
    text = Path("runs/sweep_xor/runs.csv").read_text().splitlines()
    rows = [r for r in csv.DictReader(text) if r["param"] == "p_activation"]
    values = sorted({float(r["value"]) for r in rows})
    fig, ax = plt.subplots(figsize=(W, 2.2))
    for i, v in enumerate(values):
        g = np.array([int(r["gens"]) for r in rows if float(r["value"]) == v])
        q1, med, q3 = np.percentile(g, [25, 50, 75])
        ax.errorbar(i, med, yerr=[[med - q1], [q3 - med]], color="tab:blue", marker="o", ms=4, capsize=3)
        ax.text(i + 0.08, med, f"{med:.0f}", va="center", fontsize=7)
    ax.set_xticks(range(len(values)), [f"{v:g}" + (" (default)" if v == 0.05 else "") for v in values])
    ax.set_xlabel("p_activation (chance to change a hidden node's activation)")
    ax.set_ylabel("generations to solve XOR\n(median, IQR; 20 seeds)")
    ax.set_xlim(-0.4, len(values) - 0.6)
    ax.grid(axis="y", alpha=0.2)
    save(fig, "fig01_xor_p_activation.png")


# ---------------------------------------------------------------- Unit 3: Backprop-NEAT
def best_seed(dataset):
    results = [json.loads(p.read_text()) for p in Path("runs/backprop").glob(f"{dataset}_s*/result.json")]
    return max(results, key=lambda r: r["val_fitness"])["seed"]


def backprop_dataset(dataset, num):
    seed = best_seed(dataset)
    genome = pickle.loads(Path(f"runs/backprop/{dataset}_s{seed}/best.pkl").read_bytes())
    X, y = make_splits(dataset, seed)["test"]
    acc = accuracy(genome, X, y)
    lo, hi = X.min(0) - 1, X.max(0) + 1
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 200), np.linspace(lo[1], hi[1], 200))
    net = compile_genome(genome)
    prob = 1 / (1 + np.exp(-np.clip(run(net, np.c_[gx.ravel(), gy.ravel()])[:, 0], -60, 60)))
    wrong = (run(net, X)[:, 0] > 0) != (y > 0.5)
    hidden = sum(n.type == NodeType.HIDDEN for n in genome.nodes.values())
    conns = sum(c.enabled for c in genome.conns.values())

    fig, (a, b) = plt.subplots(2, 1, figsize=(W, 5.6), gridspec_kw={"height_ratios": [1.15, 1]})
    a.contourf(gx, gy, prob.reshape(gx.shape), levels=20, cmap="RdBu_r", alpha=0.55)
    a.contour(gx, gy, prob.reshape(gx.shape), levels=[0.5], colors="k", linewidths=1.2)
    a.scatter(X[~wrong, 0], X[~wrong, 1], c=y[~wrong], cmap="RdBu_r", edgecolors="k", linewidths=0.3, s=9)
    a.scatter(X[wrong, 0], X[wrong, 1], marker="x", c="k", s=22, linewidths=0.9)
    a.set_title(f"{dataset}, seed {seed}: test accuracy {acc:.3f}\n(x = misclassified, {wrong.sum()} of {len(y)})")
    a.set_aspect("equal")
    a.tick_params(length=2)
    draw_network(b, genome, ["x", "y"], ["class"])
    b.set_title(f"{hidden} hidden nodes, {conns} enabled connections")
    fig.tight_layout()
    save(fig, f"fig{num:02d}_backprop_{dataset}.png")


def pareto():
    X, y = make_splits("spiral", 0)["val"]
    pop = pickle.loads(Path("runs/backprop/spiral_s0/final_population.pkl").read_bytes())
    pts = [(sum(c.enabled for c in g.conns.values()), accuracy(g, X, y)) for g in pop]
    top = {}
    for c, a in pts:
        top[c] = max(a, top.get(c, 0.0))
    front, best = [], -1.0
    for c, a in sorted(top.items()):
        if a > best:
            front.append((c, a))
            best = a
    fig, ax = plt.subplots(figsize=(W, 2.4))
    ax.scatter(*zip(*pts), s=8, color="0.6", label=f"final population ({len(pts)} genomes)")
    ax.plot(*zip(*front), color="tab:blue", lw=1.5, marker="o", ms=4, label="Pareto front")
    for c, a in front:
        ax.annotate(f"{a:.3f}", (c, a), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=7)
    ax.set_xlabel("enabled connections")
    ax.set_ylabel("validation accuracy")
    ax.set_ylim(top=1.06)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    ax.grid(alpha=0.2)
    save(fig, "fig05_backprop_spiral_pareto.png")


# ---------------------------------------------------------------- Unit 4: Slime Volleyball
LEADERBOARD = [(-4.866, "random policy"), (-0.071, "CMA-ES self-play"), (0.353, "GA self-play")]


def reference_lines(ax, x_text):
    for v, name in LEADERBOARD:
        ax.axhline(v, color="0.55", ls="--", lw=0.7)
        ax.text(x_text, v + 0.07, f"{name} ({v:+.3f})", fontsize=6.5, color="0.35", ha="right")


def slime_runs():
    runs = [("slime2: direct, fitness v1", ["runs/slime2.log"], "tab:gray"),
            ("slime3: fitness v2", ["runs/slime3.log"], "tab:purple"),
            ("slime4: 16 games/genome", ["runs/slime4.log"], "tab:orange"),
            ("slime5: self-play", ["runs/slime5.log"], "tab:blue")]
    fig, ax = plt.subplots(figsize=(W, 2.8))
    for name, logs, colour in runs:
        h = hof_history(*logs)
        ax.step(h[:, 0], h[:, 1], where="post", color=colour, lw=1.3, label=name)
    reference_lines(ax, 1660)
    ax.set_xlabel("generation")
    ax.set_ylabel("hall-of-fame score vs built-in AI\n(best kept agent, 20 games)")
    ax.set_ylim(-5.1, 0.7)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.42), frameon=False)
    ax.grid(alpha=0.2)
    save(fig, "fig06_slime_run_history.png")


def slime_network():
    _, hof = load_latest_checkpoint(Path("runs/slime5"))
    genome = hof.ranked()[0][1]
    hidden = sum(n.type == NodeType.HIDDEN for n in genome.nodes.values())
    conns = sum(c.enabled for c in genome.conns.values())
    fig, ax = plt.subplots(figsize=(W, 4.4))
    draw_network(ax, genome, SLIME_INPUTS, ["forward", "backward", "jump"])
    ax.set_title(f"reported agent: {hidden} hidden nodes, {conns} enabled connections")
    save(fig, "fig07_slime_best_network.png")


def slime_complexification():
    d = np.genfromtxt("runs/slime5/stats.csv", delimiter=",", names=True)
    acts = [n for n in d.dtype.names if n.startswith("act_")]
    total = sum(d[a] for a in acts)
    total[total == 0] = 1
    fig, (a, b) = plt.subplots(2, 1, figsize=(W, 4.2), sharex=True)
    a.plot(d["gen"], d["mean_hidden"], color="tab:blue", lw=1, label="population mean")
    a.step(d["gen"], d["champ_hidden"], where="post", color="tab:orange", lw=1, label="generation champion")
    a.set_ylabel("hidden nodes")
    a.legend(frameon=False, loc="upper left")
    a.grid(alpha=0.2)
    shares = [d[x] / total for x in acts]
    colours = [ACT_COLOURS[x[4:]] if x != "act_identity" else "0.75" for x in acts]   # white would vanish
    b.stackplot(d["gen"], *shares, labels=[x[4:] for x in acts], colors=colours, edgecolor="white", linewidth=0.2)
    b.set_ylabel("share of hidden nodes")
    b.set_xlabel("generation")
    b.set_ylim(0, 1)
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=4, frameon=False, handlelength=1)
    save(fig, "fig08_slime_complexification.png")


def slime_champion_snapshots():
    """Generation champion at 4 points of slime5, 2x2. Inputs/outputs in the same order as Fig 7."""
    fig, axs = plt.subplots(2, 2, figsize=(W, 4.6))
    used = set()
    for ax, gen in zip(axs.flat, (10, 50, 200, 786)):
        g = pickle.loads(Path(f"runs/slime5/best/gen_{gen:04d}.pkl").read_bytes())
        draw_network(ax, g, [""] * 12, [""] * 3)
        ax.get_legend().remove()
        hidden = [n.activation for n in g.nodes.values() if n.type == NodeType.HIDDEN]
        used |= set(hidden)
        ax.set_title(f"gen {gen}: {len(hidden)} hidden, {sum(c.enabled for c in g.conns.values())} conns")
    dot = {"marker": "o", "ls": "", "ms": 6, "markeredgecolor": "black", "markeredgewidth": 0.6}
    handles = [Line2D([], [], color=ACT_COLOURS[a], label=a, **dot) for a in sorted(used | {"tanh"})]
    handles += [Line2D([], [], color="tab:blue", label="w > 0"), Line2D([], [], color="tab:red", label="w < 0")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02),
               handlelength=1.2, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "fig08b_slime_champion_snapshots.png")


def slime_ablations():
    runs = [("slime5: chosen settings", ["runs/slime5.log"], "tab:blue"),
            ("slime6: paper's distance coefficients", ["runs/slime6/log.txt", "runs/slime6/log_resume1.txt"], "tab:orange"),
            ("slime7: p_activation 0.2", ["runs/slime7/log.txt"], "tab:green")]
    fig, ax = plt.subplots(figsize=(W, 2.6))
    for name, logs, colour in runs:
        h = hof_history(*logs)
        h = h[h[:, 0] <= 700]
        ax.step(h[:, 0], h[:, 1], where="post", color=colour, lw=1.3, label=name)
    ax.set_xlabel("generation")
    ax.set_ylabel("hall-of-fame score vs built-in AI")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.2)
    save(fig, "fig09_slime_ablations.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    xor_sweep()
    for num, dataset in ((2, "circle"), (3, "xor"), (4, "spiral")):
        backprop_dataset(dataset, num)
    pareto()
    slime_runs()
    slime_network()
    slime_complexification()
    slime_champion_snapshots()
    slime_ablations()
    shutil.copy("runs/slime5/gen00786_rank0_seed10014.gif", OUT / "fig10_slime_game.gif")
    print("copied GIF")


if __name__ == "__main__":
    main()
