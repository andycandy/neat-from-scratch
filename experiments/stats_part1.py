"""Significance checks quoted in the Part 1 write-up. Reads saved results only.

uv run python experiments/stats_part1.py > logs/stats_part1_2026-10-25.txt
"""
import csv
import json
from pathlib import Path

import numpy as np

rng = np.random.default_rng(0)


def perm_test(a, b, n=100_000):
    """Two-sided permutation test on the difference of means."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled, obs = np.concatenate([a, b]), abs(a.mean() - b.mean())
    hits = sum(abs((p := rng.permutation(pooled))[:len(a)].mean() - p[len(a):].mean()) >= obs for _ in range(n))
    return (hits + 1) / (n + 1)


# Decision 8: per-seed generations to solve XOR (logs/decision8_coefficients_xor_2026-10-23.txt)
ours = [10, 18, 15, 19, 40, 21, 21, 7, 24, 6]
paper = [13, 16, 9, 11, 5, 9, 10, 12, 7, 6]
print(f"decision 8: ours {np.mean(ours):.1f} +- {np.std(ours):.1f}, paper {np.mean(paper):.1f} +- {np.std(paper):.1f}, "
      f"medians {np.median(ours):.1f} vs {np.median(paper):.1f}, permutation p = {perm_test(ours, paper):.4f} (10 vs 10 seeds)")

# XOR sweep: each setting vs the default, median difference with a bootstrap 95% CI
rows = list(csv.DictReader(Path("runs/sweep_xor/runs.csv").read_text().splitlines()))
defaults = {"p_weight": "0.8", "sigma": "0.5", "p_add_conn": "0.1", "p_add_node": "0.05",
            "p_activation": "0.05", "p_toggle": "0.01", "p_crossover": "0.75"}
for param, default in defaults.items():
    base = np.array([int(r["gens"]) for r in rows if r["param"] == param and r["value"] == default])
    for v in sorted({r["value"] for r in rows if r["param"] == param} - {default}, key=float):
        x = np.array([int(r["gens"]) for r in rows if r["param"] == param and r["value"] == v])
        boot = [np.median(rng.choice(x, len(x))) - np.median(rng.choice(base, len(base))) for _ in range(10_000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        flag = "" if lo <= 0 <= hi else "  <- CI excludes 0"
        print(f"sweep {param}={v} vs {default}: median diff {np.median(x) - np.median(base):+.1f} gens, 95% CI [{lo:+.1f}, {hi:+.1f}]{flag}")

# Slime final evaluation
r = json.loads(Path("runs/slime5/final_eval.json").read_text())
print(f"slime final: {r['mean']} +- {r['std']} over {r['games']} games, standard error {r['std'] / np.sqrt(r['games']):.3f}")
