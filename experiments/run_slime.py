"""Slime Volleyball with NEAT (Unit 4). Evolve against the built-in AI.

Run:      uv run python experiments/run_slime.py --out runs/slime1
Resume:   uv run python experiments/run_slime.py --out runs/slime1 --resume
"""
from __future__ import annotations   # lets `Pool | None` etc. appear in type hints

import argparse
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neat.config import NeatConfig  # noqa: E402
from neat.genes import Genome  # noqa: E402
from neat.network import compile_genome, run  # noqa: E402
from neat.population import Population  # noqa: E402
from slime.slimevolley import SlimeVolleyEnv  # noqa: E402

# ---------------------------------------------------------------------------
# Settings (every value here goes into DECISIONS.md, Unit 4 §18)
# ---------------------------------------------------------------------------
BETA = 0.5                 # survival bonus weight, must be < 1 (Unit 4 §5)
EPISODES = 4               # E: episodes per genome per generation
INPUT_SCALE = 1.0          # TODO step 2: fixed divisor(s) after measuring obs ranges
                           #   (the env ALREADY divides raw values by 10 internally)
HOF_SIZE = 5               # hall of fame: keep this many
HOF_EVERY = 10             # re-evaluate every N generations
HOF_EPISODES = 20          # ... on this many fresh episodes
KEEP_CHECKPOINTS = 3       # keep the last few, a crash while saving can corrupt one


def make_config() -> NeatConfig:
    # TODO: start from XOR defaults, change only what Unit 4 §10 says:
    #   12 inputs, 3 outputs, output_activation "tanh",
    #   activations += ("identity", "step"), max_stagnation=20, pop_size from your budget
    return NeatConfig(num_inputs=12, num_outputs=3)


# ---------------------------------------------------------------------------
# Playing (runs inside worker processes)
# ---------------------------------------------------------------------------
_ENV: SlimeVolleyEnv | None = None   # one env per worker process, made by _init_worker


def _init_worker() -> None:
    """Pool initializer: create the environment once per worker."""
    global _ENV
    _ENV = SlimeVolleyEnv()


def play(net, env: SlimeVolleyEnv, seed: int) -> tuple[float, int]:
    """One full game against the built-in AI. Returns (pure score, number of steps).
    `net` is the result of compile_genome(genome)."""
    # HINT: env.seed(seed) BEFORE env.reset() (unseeded env uses global np.random -> not
    #       reproducible). Loop until done:
    #         out = run(net, obs[None, :] / INPUT_SCALE)[0]     # (1, 12) -> (3,)
    #         action = (out > 0).astype(int)                     # tanh outputs, threshold 0
    #         obs, reward, done, info = env.step(action)
    #       sum the rewards, count the steps.
    raise NotImplementedError


def evaluate_genome(args: tuple[Genome, list[int]]) -> tuple[float, list[float], list[int]]:
    """Worker job: (genome, seeds) -> (fitness, per-episode scores, per-episode lengths).
    Must depend ONLY on (genome, seeds), so parallel == serial."""
    genome, seeds = args
    # HINT: compile ONCE here, then play() each seed with the module-level _ENV
    #       (call _init_worker() first if _ENV is None, so this also works without a pool).
    #       fitness = mean(score_e + BETA * T_e / 3000)
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Fitness function for Population.run
# ---------------------------------------------------------------------------
def make_fitness_fn(pool: Pool | None, pop: Population, last: dict):
    """Returns fitness_fn(genomes) -> np.ndarray, as Population.run expects.
    `last` is a dict the caller reads afterwards (this generation's genomes + results),
    because after pop.run(...) returns, pop.genomes are already the NEXT (unevaluated) generation."""
    def fitness_fn(genomes: list[Genome]) -> np.ndarray:
        # HINT: common random numbers: draw EPISODES seeds ONCE per call from pop.rng
        #       (pop.rng is checkpointed, so resume stays deterministic):
        #         seeds = [int(s) for s in pop.rng.integers(0, 2**31 - 1, size=EPISODES)]
        #       jobs = [(g, seeds) for g in genomes]
        #       results = pool.map(evaluate_genome, jobs) if pool else list(map(evaluate_genome, jobs))
        #       store in `last`: genomes, results, seeds  (for the hall of fame + logging)
        #       return np.array of the fitness values
        raise NotImplementedError
    return fitness_fn


# ---------------------------------------------------------------------------
# Hall of fame
# ---------------------------------------------------------------------------
def update_hall_of_fame(hof: list[tuple[float, Genome]], candidates: list[Genome],
                        pool: Pool | None, rng: np.random.Generator) -> list[tuple[float, Genome]]:
    """Re-evaluate candidates on HOF_EPISODES fresh seeds; keep the best HOF_SIZE overall,
    sorted best first by that re-evaluated PURE score (no survival bonus)."""
    # HINT: fresh seeds from rng; reuse evaluate_genome (its 2nd return value = pure scores);
    #       score = mean of pure scores; merge with hof; sort by score, reverse=True; slice.
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def save_checkpoint(out: Path, pop: Population, hof: list) -> None:
    # HINT: one file per generation: out / "checkpoints" / f"gen_{pop.generation:05d}.pkl"
    #       pickle.dump({"pop": pop, "hof": hof}, f)   (pop already holds rng, tracker, species)
    #       write to a temp name first, then os.replace(tmp, final): a crash mid-write
    #       can't corrupt the final file.  Then delete all but the newest KEEP_CHECKPOINTS.
    raise NotImplementedError


def load_latest_checkpoint(out: Path) -> tuple[Population, list]:
    # HINT: sorted((out / "checkpoints").glob("gen_*.pkl"))[-1]; pickle.load; return pop, hof
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/slime1")
    ap.add_argument("--generations", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    # HINT, the whole loop:
    #   pop, hof = load_latest_checkpoint(out) if args.resume else (Population(make_config(), args.seed), [])
    #   with Pool(args.workers, initializer=_init_worker) as pool:
    #       last = {}
    #       fitness_fn = make_fitness_fn(pool, pop, last)
    #       while pop.generation < args.generations:
    #           t0 = time.time()
    #           pop.run(fitness_fn, 1, snapshot_dir=str(out / "best"))   # exactly one generation
    #           if pop.generation % HOF_EVERY == 0:
    #               top = sorted(last["genomes"], key=lambda g: g.fitness, reverse=True)[:HOF_SIZE]
    #               hof = update_hall_of_fame(hof, top, pool, pop.rng)
    #           pop.save_stats(str(out / "stats.csv"))
    #           save_checkpoint(out, pop, hof)
    #           print(gen, best fitness, mean, n_species, seconds, flush=True)  # flush: log file
    raise NotImplementedError


if __name__ == "__main__":   # REQUIRED on Windows: Pool workers re-import this file
    main()
