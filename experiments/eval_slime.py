"""Final Slime evaluation. Selection and reporting use separate fresh seed ranges:
select: the top-5 hall-of-fame agents each play the same 200 games, keep the best;
report: that one agent on 1,000 different games (the leaderboard's protocol).

uv run python experiments/eval_slime.py --run runs/slime5
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_slime import HallOfFame, evaluate_all, load_latest_checkpoint  # noqa: F401  (HallOfFame: for unpickling)

SELECT_SEEDS = range(100_000, 100_200)
REPORT_SEEDS = range(200_000, 201_000)
CHUNK = 20      # games per worker job (played in lockstep)


def scores(pool, genome, seeds):
    seeds = list(seeds)
    jobs = [(genome, seeds[i:i + CHUNK], 0) for i in range(0, len(seeds), CHUNK)]
    return np.concatenate([r[1] for r in evaluate_all(pool, jobs)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/slime5")
    args = ap.parse_args()
    pop, hof = load_latest_checkpoint(Path(args.run))

    with Pool() as pool:
        select = [(scores(pool, g, SELECT_SEEDS).mean(), hof_score, g) for hof_score, g in hof.ranked()]
        for rank, (s, hof_score, _) in enumerate(select):
            print(f"hof #{rank}: hof score {hof_score:+.2f}  select (200 games) {s:+.3f}")
        best = max(range(len(select)), key=lambda i: select[i][0])
        report = scores(pool, select[best][2], REPORT_SEEDS)

    result = {
        "generation": pop.generation, "chosen_hof_rank": best,
        "select_mean_200": round(float(select[best][0]), 3),
        "games": len(report), "mean": round(float(report.mean()), 3), "std": round(float(report.std()), 3),
        "win_rate": round(float((report > 0).mean()), 3), "draw_rate": round(float((report == 0).mean()), 3),
        "loss_rate": round(float((report < 0).mean()), 3),
    }
    print(json.dumps(result, indent=2))
    (Path(args.run) / "final_eval.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
