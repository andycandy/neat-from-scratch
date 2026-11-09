"""Slime analysis numbers quoted in docs/report.md. Writes to results/slime/.

- champ_scores_<run>.json: every STEP generations, that generation's champion plays the same
  GAMES fixed games against the built-in AI (stats.csv has no pure score). Cached.
- input_usage.json: per input of the reported agent, its enabled connections and their total |w|.

uv run python experiments/analyze_slime.py
"""
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_slime import scores
from run_slime import HallOfFame, load_latest_checkpoint  # noqa: F401  (HallOfFame: for unpickling)

from neat.genes import NodeType

RUNS = [Path("runs/slime5"), Path("runs/slime6"), Path("runs/slime7")]   # reported run + 2 ablations
OUT = Path("results/slime")
STEP, GAMES = 25, 40
SEEDS = range(300_000, 300_000 + GAMES)        # fixed, never used for selection
INPUTS = ["my x", "my y", "my vx", "my vy", "ball x", "ball y", "ball vx", "ball vy",
          "opp x", "opp y", "opp vx", "opp vy"]


def champ_scores(pool, run):
    """{generation: mean pure score of that generation's champion on SEEDS}, cached."""
    cache = OUT / f"champ_scores_{run.name}.json"
    done = {int(k): v for k, v in json.loads(cache.read_text()).items()} if cache.exists() else {}
    last = len(list((run / "best").glob("gen_*.pkl"))) - 1
    for gen in [*range(0, last + 1, STEP), last]:
        if gen not in done:
            genome = pickle.loads((run / "best" / f"gen_{gen:04d}.pkl").read_bytes())
            done[gen] = float(scores(pool, genome, SEEDS).mean())
            print(f"{run.name} gen {gen}: {done[gen]:+.3f}", flush=True)
            cache.write_text(json.dumps(done))
    return dict(sorted(done.items()))


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
        for run in RUNS:
            if run.exists():
                champ_scores(pool, run)

    _, hof = load_latest_checkpoint(RUNS[0])
    _, best = hof.ranked()[0]
    summary = {"hidden": sum(g.type == NodeType.HIDDEN for g in best.nodes.values()),
               "enabled_connections": sum(c.enabled for c in best.conns.values()),
               "hidden_activations": sorted(g.activation for g in best.nodes.values() if g.type == NodeType.HIDDEN),
               "inputs": input_usage(best)}
    (OUT / "input_usage.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
