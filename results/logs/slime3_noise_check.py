# Scratch script that produced results/logs/slime3_noise_check_2026-10-22.txt (22 Oct 14:26 IST).
# Kept as evidence. It targets an earlier run_slime.py (_init_worker, 2-item jobs) and does not run on the current code.
"""Is 4 games per genome enough? Score the same population on two disjoint seed sets
and check whether the rankings agree (Spearman rank correlation)."""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import __main__  # noqa: E402
import run_slime  # noqa: E402

__main__.HallOfFame = run_slime.HallOfFame


def ranks(x):
    """Ranks with ties averaged."""
    x = np.asarray(x, dtype=float)
    order = x.argsort()
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in np.unique(x):
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


if __name__ == "__main__":
    pop, _ = run_slime.load_latest_checkpoint(ROOT / "runs/slime3")
    genomes = pop.genomes
    A, B, C = [101, 102, 103, 104], [201, 202, 203, 204], list(range(301, 317))
    with Pool(6, initializer=run_slime._init_worker) as pool:
        ra = pool.map(run_slime.evaluate_genome, [(g, A) for g in genomes])
        rb = pool.map(run_slime.evaluate_genome, [(g, B) for g in genomes])
        rc = pool.map(run_slime.evaluate_genome, [(g, C) for g in genomes])
    fa, fb, fc = (np.array([r[0] for r in rr]) for rr in (ra, rb, rc))
    sa, sb = (np.array([np.mean(r[1]) for r in rr]) for rr in (ra, rb))
    ta, tb = (np.array([np.mean(r[3]) for r in rr]) for rr in (ra, rb))

    print(f"generation {pop.generation}, {len(genomes)} genomes")
    print(f"fitness:  set A std {fa.std():.3f}, set B std {fb.std():.3f}")
    print(f"rank agreement A vs B (Spearman):  fitness {spearman(fa, fb):+.2f}   "
          f"pure score {spearman(sa, sb):+.2f}   returns {spearman(ta, tb):+.2f}")
    top = 15
    topA, topB = set(np.argsort(fa)[-top:]), set(np.argsort(fb)[-top:])
    print(f"top-{top} by fitness: {len(topA & topB)} of {top} genomes appear in both sets")
    print(f"4-game fitness vs 16-game fitness (Spearman): {spearman(fa, fc):+.2f}")
    print(f"16-game: mean score {np.mean([np.mean(r[1]) for r in rc]):+.3f}, "
          f"returns/game {np.mean([np.mean(r[3]) for r in rc]):.2f}, "
          f"genomes with mean returns >= 1: {sum(np.mean(r[3]) >= 1 for r in rc)}")
