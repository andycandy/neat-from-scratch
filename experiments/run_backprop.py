"""Backprop-NEAT experiments (Unit 3 §14): datasets x seeds, one result row per run.

uv run python experiments/run_backprop.py                       # all 3 datasets x 2 seeds
uv run python experiments/run_backprop.py --datasets circle --seeds 0 --generations 3   # smoke test
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backprop import train  # noqa: E402
from backprop.datasets import make_splits  # noqa: E402
from backprop.padded import ACT_NAMES  # noqa: E402
from neat.config import NeatConfig  # noqa: E402
from neat.genes import NodeType  # noqa: E402
from neat.population import Population  # noqa: E402


def make_config() -> NeatConfig:
    """XOR defaults, changed only where Unit 3 asks (reasons in DECISIONS.md, Unit 3)."""
    return NeatConfig(
        num_inputs=2, num_outputs=1,
        output_activation="identity",   # the output is a logit; the sigmoid lives in the loss (§4)
        activations=ACT_NAMES,          # differentiable only: no "step" (§11)
        max_nodes=train.N_MAX,          # add-node skipped at the padding size (§9)
        p_weight=0.2, sigma=0.1,        # gradients tune weights now; keep a little exploration (§7)
        distance_coefficients=(0.5, 0.5, 0.1),   # lower c3: trained weights shouldn't split species (§7)
    )


def run_one(dataset: str, seed: int, generations: int, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    splits = make_splits(dataset, seed)
    pop = Population(make_config(), seed)
    fitness_fn = train.make_fitness_fn(splits)
    t0 = time.time()
    for _ in range(generations):
        pop.run(fitness_fn, 1)
        row = pop.stats[-1]
        print(f"{dataset} s{seed} gen {row['gen']:3d}  best fit {row['best']:.3f}  "
              f"best val acc {fitness_fn.val_acc.max():.3f}  hidden {row['mean_hidden']:.1f}  "
              f"species {row['n_species']}", flush=True)
    best = pop.best                        # chosen by VALIDATION fitness; weights already trained
    X_te, y_te = splits["test"]
    result = {
        "dataset": dataset, "seed": seed, "generations": generations,
        "test_acc": train.accuracy(best, X_te, y_te),
        "val_fitness": best.fitness,
        "hidden": sum(n.type == NodeType.HIDDEN for n in best.nodes.values()),
        "connections": sum(c.enabled for c in best.conns.values()),
        "seconds": round(time.time() - t0, 1),
    }
    pop.save_stats(str(out / "stats.csv"))
    with open(out / "best.pkl", "wb") as f:
        pickle.dump(best, f)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print("RESULT", json.dumps(result), flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["circle", "xor", "spiral"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--generations", type=int, default=50)
    ap.add_argument("--out", default="runs/backprop")
    args = ap.parse_args()
    results = [run_one(d, s, args.generations, Path(args.out) / f"{d}_s{s}")
               for d in args.datasets for s in args.seeds]
    (Path(args.out) / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
