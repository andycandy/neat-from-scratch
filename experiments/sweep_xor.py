"""Mutation-rate sweep on XOR: change ONE setting at a time from the defaults, 20 seeds each.

uv run python experiments/sweep_xor.py
"""
import csv
import dataclasses
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from xor import fitness, solved

from neat.config import NeatConfig
from neat.genes import NodeType
from neat.population import Population

SEEDS = range(20)
MAX_GENS = 300
DEFAULT = NeatConfig(num_inputs=2, num_outputs=1)
SWEEP = {                                    # default value is included in each list
    "p_weight": [0.5, 0.8, 0.95],
    "sigma": [0.1, 0.5, 1.0, 2.0],
    "p_add_conn": [0.03, 0.1, 0.3],
    "p_add_node": [0.01, 0.05, 0.1, 0.2],
    "p_activation": [0.0, 0.05, 0.2],
    "p_toggle": [0.0, 0.01, 0.05],
    "p_crossover": [0.0, 0.25, 0.75, 1.0],
}
OUT = Path("runs/sweep_xor")


def one_run(job):
    param, value, seed = job
    cfg = dataclasses.replace(DEFAULT, **{param: value})
    pop = Population(cfg, seed)
    best = pop.run(fitness, MAX_GENS, stop=solved)
    ok = solved(best)
    return {"param": param, "value": value, "seed": seed, "solved": ok,
            "gens": pop.generation if ok else MAX_GENS,
            "hidden": sum(n.type == NodeType.HIDDEN for n in best.nodes.values()),
            "acts": " ".join(sorted(n.activation for n in best.nodes.values() if n.type == NodeType.HIDDEN))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(p, v, s) for p, values in SWEEP.items() for v in values for s in SEEDS]
    with Pool() as pool:
        rows = pool.map(one_run, jobs)
    with open(OUT / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fig, axs = plt.subplots(1, len(SWEEP), figsize=(3 * len(SWEEP), 3.2), sharey=True)
    print(f"{'setting':<14}{'value':>7}  solved  gens median [IQR]  hidden")
    for ax, (param, values) in zip(axs, SWEEP.items()):
        meds, lo, hi = [], [], []
        for v in values:
            r = [x for x in rows if x["param"] == param and x["value"] == v]
            g = np.array([x["gens"] for x in r])          # unsolved runs count as MAX_GENS
            q1, med, q3 = np.percentile(g, [25, 50, 75])
            meds.append(med)
            lo.append(med - q1)
            hi.append(q3 - med)
            mark = " (default)" if v == getattr(DEFAULT, param) else ""
            print(f"{param:<14}{v:>7}  {sum(x['solved'] for x in r):>3}/{len(r)}   "
                  f"{med:5.1f} [{q1:.0f}-{q3:.0f}]   {np.mean([x['hidden'] for x in r]):.1f}{mark}")
        x = range(len(values))
        ax.errorbar(x, meds, yerr=[lo, hi], color="tab:blue", marker="o", capsize=3)
        ax.set_xticks(list(x), [str(v) for v in values])
        ax.axvline(values.index(getattr(DEFAULT, param)), color="0.7", ls="--", lw=1)
        ax.set_title(param, fontsize=10)
        ax.grid(alpha=0.2)
    axs[0].set_ylabel("generations to solve\n(median, IQR; dashed = default)")
    fig.tight_layout()
    fig.savefig(OUT / "sweep_xor.png", dpi=130)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
