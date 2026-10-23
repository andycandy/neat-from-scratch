"""Unit 4 figures from runs/slime5 (and the slime6 ablation, if present).

- training.png:      champion score vs the built-in AI, species, hidden nodes, hidden activations
- ablation.png:      slime5 vs slime6 (champion score and hidden nodes)
- network_best.png:  the reported agent (hall of fame #0), inputs and outputs labelled
- network_genXXXX.png: champions at a few generations, to show complexification
- input_usage.json:  per input of the reported agent: enabled connections and their total |w|

Champion scores are measured here (stats.csv has no pure score): every STEP generations the
champion plays the same GAMES fixed games vs the built-in AI. Cached in champ_scores.json.

uv run python experiments/figures_slime.py
"""
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_slime import scores
from run_slime import HallOfFame, load_latest_checkpoint  # noqa: F401  (HallOfFame: for unpickling)

from neat.genes import NodeType
from neat.visualize import draw_genome

RUN, ABLATION = Path("runs/slime5"), Path("runs/slime6")
OUT = Path("figures/slime")
STEP, GAMES = 25, 40
SEEDS = range(300_000, 300_000 + GAMES)        # fixed, never used for selection
INPUTS = ["my x", "my y", "my vx", "my vy", "ball x", "ball y", "ball vx", "ball vy",
          "opp x", "opp y", "opp vx", "opp vy"]
OUTPUTS = ["forward", "backward", "jump"]
SNAPSHOTS = [10, 50, 200, 786]


def champ_scores(pool, run):
    """{generation: mean pure score of that generation's champion on SEEDS}, cached."""
    cache = run / "champ_scores.json"
    done = {int(k): v for k, v in json.loads(cache.read_text()).items()} if cache.exists() else {}
    last = len(list((run / "best").glob("gen_*.pkl"))) - 1
    for gen in [*range(0, last + 1, STEP), last]:
        if gen not in done:
            genome = pickle.loads((run / "best" / f"gen_{gen:04d}.pkl").read_bytes())
            done[gen] = float(scores(pool, genome, SEEDS).mean())
            print(f"{run.name} gen {gen}: {done[gen]:+.3f}", flush=True)
            cache.write_text(json.dumps(done))
    return dict(sorted(done.items()))


def stats(run):
    return np.genfromtxt(run / "stats.csv", delimiter=",", names=True)


def training_figure(champ):
    d = stats(RUN)
    fig, axs = plt.subplots(2, 2, figsize=(12, 7.5))
    ax = axs[0, 0]
    ax.plot(list(champ), list(champ.values()), color="tab:blue", lw=2, marker="o", ms=3)
    ax.axhline(-4.866, color="0.5", ls="--", lw=1)
    ax.text(1000, -4.75, "random policy (−4.87)", fontsize=8, color="0.4")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_title(f"champion's pure score vs built-in AI ({GAMES} fixed games)")
    ax.set_ylabel("mean score per game")

    axs[0, 1].plot(d["gen"], d["n_species"], color="tab:blue", lw=1)
    axs[0, 1].set_title("number of species")

    ax = axs[1, 0]
    ax.plot(d["gen"], d["mean_hidden"], color="tab:blue", lw=1.5, label="population mean")
    ax.plot(d["gen"], d["champ_hidden"], color="tab:orange", lw=1, alpha=0.8, label="champion")
    ax.set_title("hidden nodes")
    ax.legend(fontsize=8, frameon=False)

    acts = [n for n in d.dtype.names if n.startswith("act_")]
    axs[1, 1].stackplot(d["gen"], *[d[a] for a in acts], labels=[a[4:] for a in acts],
                        edgecolor="white", linewidth=0.3)
    axs[1, 1].set_title("hidden-node activations (whole population)")
    axs[1, 1].legend(loc="upper left", fontsize=7, ncol=2, frameon=False)

    for ax in axs.flat:
        ax.set_xlabel("generation")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "training.png", dpi=130)
    plt.close(fig)


def ablation_figure(champ5, champ6):
    d5, d6 = stats(RUN), stats(ABLATION)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4))
    for champ, d, name, colour in ((champ5, d5, "slime5: (0.5, 0.5, 0.3)", "tab:blue"),
                                   (champ6, d6, "slime6: paper (1.0, 1.0, 0.4)", "tab:orange")):
        a.plot(list(champ), list(champ.values()), color=colour, lw=2, marker="o", ms=3, label=name)
        b.plot(d["gen"], d["mean_hidden"], color=colour, lw=1.5, label=name)
    a.set_title(f"champion's pure score vs built-in AI ({GAMES} fixed games)")
    b.set_title("mean hidden nodes")
    for ax in (a, b):
        ax.set_xlabel("generation")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "ablation.png", dpi=130)
    plt.close(fig)


def input_usage(genome):
    """Per input: number of enabled outgoing connections and their total |weight|."""
    inputs = sorted(n for n, g in genome.nodes.items() if g.type == NodeType.INPUT)
    usage = {}
    for name, n in zip(INPUTS, inputs):
        out = [c for c in genome.conns.values() if c.enabled and c.src == n]
        usage[name] = {"connections": len(out), "sum_abs_w": round(sum(abs(c.weight) for c in out), 2)}
    return usage


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool() as pool:
        champ5 = champ_scores(pool, RUN)
        champ6 = champ_scores(pool, ABLATION) if ABLATION.exists() else None
    training_figure(champ5)
    if champ6:
        ablation_figure(champ5, champ6)

    _, hof = load_latest_checkpoint(RUN)
    _, best = hof.ranked()[0]
    hidden = sum(g.type == NodeType.HIDDEN for g in best.nodes.values())
    enabled = sum(c.enabled for c in best.conns.values())
    draw_genome(best, str(OUT / "network_best.png"), INPUTS, OUTPUTS,
                title=f"reported agent (hall of fame #0): {hidden} hidden, {enabled} enabled connections")
    for gen in SNAPSHOTS:
        g = pickle.loads((RUN / "best" / f"gen_{gen:04d}.pkl").read_bytes())
        draw_genome(g, str(OUT / f"network_gen{gen:04d}.png"), INPUTS, OUTPUTS,
                    title=f"champion, generation {gen}")

    usage = input_usage(best)
    acts = sorted(g.activation for g in best.nodes.values() if g.type == NodeType.HIDDEN)
    summary = {"hidden": hidden, "enabled_connections": enabled, "hidden_activations": acts,
               "inputs": usage}
    (OUT / "input_usage.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"saved figures in {OUT}")


if __name__ == "__main__":
    main()
