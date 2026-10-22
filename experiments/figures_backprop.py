"""Unit 3 figures from runs/backprop/*: one decision-boundary + architecture figure per dataset,
plus complexification curves. The genome shown per dataset is the run with the best VALIDATION
fitness; its title reports the TEST accuracy (the test set is used only here).

uv run python experiments/figures_backprop.py
"""
import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backprop.datasets import make_splits
from backprop.train import accuracy
from neat.network import compile_genome, run
from neat.visualize import draw_genome

RUNS = Path("runs/backprop")
OUT = Path("figures/backprop")


def best_run(dataset):
    """The seed whose best genome has the highest validation fitness."""
    results = [json.loads(p.read_text()) for p in RUNS.glob(f"{dataset}_s*/result.json")]
    return max(results, key=lambda r: r["val_fitness"])


def boundary_and_architecture(dataset):
    r = best_run(dataset)
    folder = RUNS / f"{dataset}_s{r['seed']}"
    genome = pickle.loads((folder / "best.pkl").read_bytes())
    X, y = make_splits(dataset, r["seed"])["test"]
    test_acc = accuracy(genome, X, y)

    # predicted probability on a grid covering the data
    lo, hi = X.min(0) - 1, X.max(0) + 1
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 200), np.linspace(lo[1], hi[1], 200))
    logits = run(compile_genome(genome), np.c_[gx.ravel(), gy.ravel()])[:, 0]
    prob = 1 / (1 + np.exp(-np.clip(logits, -60, 60))).reshape(gx.shape)
    pred = run(compile_genome(genome), X)[:, 0] > 0
    wrong = pred != (y > 0.5)

    arch_png = folder / "architecture.png"
    draw_genome(genome, str(arch_png), input_names=["x", "y"], output_names=["class"],
                title=f"{r['hidden']} hidden, {r['connections']} connections")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.contourf(gx, gy, prob, levels=20, cmap="RdBu_r", alpha=0.6)
    ax1.contour(gx, gy, prob, levels=[0.5], colors="k", linewidths=2)        # the boundary
    ax1.scatter(X[~wrong, 0], X[~wrong, 1], c=y[~wrong], cmap="RdBu_r", edgecolors="k", s=25)
    ax1.scatter(X[wrong, 0], X[wrong, 1], marker="x", c="k", s=60, label="misclassified")
    ax1.set_title(f"{dataset}: test accuracy {test_acc:.3f}  (seed {r['seed']})")
    ax1.set_aspect("equal")
    ax1.legend(loc="upper right", fontsize=8)
    ax2.imshow(plt.imread(arch_png))
    ax2.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / f"{dataset}_boundary_architecture.png", dpi=130)
    plt.close(fig)
    return dataset, r["seed"], test_acc, r["hidden"], r["connections"]


def complexification():
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for csv in sorted(RUNS.glob("*_s*/stats.csv")):
        d = np.genfromtxt(csv, delimiter=",", names=True)
        label = csv.parent.name
        axs[0].plot(d["gen"], d["best"], label=label)
        axs[1].plot(d["gen"], d["mean_enabled_conns"], label=label)
        axs[2].plot(d["gen"], d["mean_hidden"], label=label)
    for ax, title in zip(axs, ["best fitness (val acc - penalty)", "mean enabled connections",
                               "mean hidden nodes"]):
        ax.set_title(title)
        ax.set_xlabel("generation")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "complexification.png", dpi=130)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for row in [boundary_and_architecture(d) for d in ("circle", "xor", "spiral")]:
        d, seed, acc, hidden, conns = row
        print(f"dataset {d:<6} seed {seed}  test acc {acc:.3f}  hidden {hidden}  connections {conns}")
    complexification()
    print(f"saved figures in {OUT}")


if __name__ == "__main__":
    main()
