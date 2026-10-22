"""Slime Volleyball with NEAT (Unit 4). Evolve against the built-in AI.

Run:      uv run python experiments/run_slime.py --out runs/slime1
Resume:   uv run python experiments/run_slime.py --out runs/slime1 --resume
"""
from __future__ import annotations   # lets `Pool | None` etc. appear in type hints

import argparse
import heapq
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neat.config import NeatConfig  # noqa: E402
from neat.genes import Genome, NodeType  # noqa: E402
from neat.network import compile_genome, run  # noqa: E402
from neat.population import Population  # noqa: E402
from slime.slimevolley import SlimeVolleyEnv  # noqa: E402

# ---------------------------------------------------------------------------
# Settings (every value here goes into DECISIONS.md, Unit 4 §18)
# ---------------------------------------------------------------------------
# Fitness shaping (fixed for the whole run; reasons in DECISIONS.md, "Slime fitness v2")
SURVIVAL_A = 0.75          # max survival bonus per rally; < 1 so a point always beats a rally
SURVIVAL_TAU = 300         # steps (10 s at 30 fps); bonus = A * (1 - exp(-rally_steps / TAU))
RETURN_GAMMA = 0.2         # per return (we touched the ball, then it crossed to the AI's side)
NODE_LAMBDA = 0.002        # per hidden node: structure must pay for itself
TOUCH_MARGIN = 0.3         # the game pushes the ball just outside the agent after a bounce
EPISODES = 4               # E: episodes per genome per generation
INPUT_SCALE = 1.0          # TODO step 2: fixed divisor(s) after measuring obs ranges
                           #   (the env ALREADY divides raw values by 10 internally)
HOF_SIZE = 5               # hall of fame: keep this many
HOF_EVERY = 10             # re-evaluate every N generations
HOF_EPISODES = 20          # ... on this many fresh episodes
KEEP_CHECKPOINTS = 3       # keep the last few, a crash while saving can corrupt one


def make_config() -> NeatConfig:
    """XOR defaults, changing only what Unit 4 §10 asks for."""
    return NeatConfig(
        num_inputs=12,                    # agent, ball, opponent: x, y, vx, vy each
        num_outputs=3,                    # forward, backward, jump
        pop_size=150,                     # revisit after measuring steps/s (step 5)
        output_activation="tanh",         # press a button when its output > 0
        activations=("identity", "sigmoid", "tanh", "relu", "sin", "gauss", "abs", "step"),
        max_stagnation=20,                # noisy fitness: give species longer
        target_species=(8, 12),           # dynamic threshold aims at ~10 species
        # Speciation scale (see DECISIONS, Unit 4 #10). N = 1 always, as in Stanley's original
        # code: dividing by N = 36 conns made a new node worth 0.03, below weight noise (~0.4).
        # With N = 1: weight-only differences 0.33 avg / 0.45 max; one new node +1.0.
        n_small_cutoff=10**9,             # i.e. never normalise by N
        compat_threshold=3.0,             # Stanley's start value
        threshold_step=0.3,               # Stanley's step
        threshold_min=1.0,                # above weight noise: weights alone never split species
        threshold_max=6.0,                # cap prevents controller windup
    )


# ---------------------------------------------------------------------------
# Playing (runs inside worker processes)
# ---------------------------------------------------------------------------
_ENV: SlimeVolleyEnv | None = None   # one env per worker process, made by _init_worker


def _init_worker() -> None:
    """Pool initializer: create the environment once per worker."""
    global _ENV
    _ENV = SlimeVolleyEnv()


def play(net, env: SlimeVolleyEnv, seed: int) -> tuple[float, int, int, float]:
    """One full game against the built-in AI.
    Returns (pure score, steps, returns over the net, survival bonus summed over rallies).
    `net` is the result of compile_genome(genome)."""
    env.seed(seed)       # BEFORE env.reset(): an unseeded env uses global np.random
    obs = env.reset()
    env.policy.reset()   # built-in AI is an RNN: env.reset() does NOT clear its memory,
                         # so without this a game depends on the worker's previous game
    me = env.game.agent_right                     # our agent (same object all game)
    touch_dist2 = (me.r + 0.5 + TOUCH_MARGIN) ** 2  # ball radius 0.5
    done = False
    total, step = 0.0, 0
    returns, rally, survival, touched = 0, 0, 0.0, False
    prev_bx = env.game.ball.x
    while not done:
        out = run(net, obs[None, :] / INPUT_SCALE)[0]
        action = (out > 0).astype(int)
        obs, reward, done, info = env.step(action)
        total += reward
        step += 1
        rally += 1
        ball = env.game.ball
        if (ball.x - me.x) ** 2 + (ball.y - me.y) ** 2 < touch_dist2:
            touched = True
        if touched and prev_bx > 0 > ball.x:      # crossed from our side to the AI's side
            returns += 1
            touched = False
        prev_bx = ball.x
        if reward != 0:                           # a point ended this rally
            survival += SURVIVAL_A * (1 - np.exp(-rally / SURVIVAL_TAU))
            rally, touched = 0, False
            prev_bx = env.game.ball.x              # ball was re-served at the net
    if rally > 0:                                 # unfinished last rally (3000-step limit)
        survival += SURVIVAL_A * (1 - np.exp(-rally / SURVIVAL_TAU))
    return total, step, returns, survival


def evaluate_genome(args: tuple[Genome, list[int]]) -> tuple[float, list[float], list[int], list[int]]:
    """Worker job: (genome, seeds) -> (fitness, per-episode pure scores, lengths, returns).
    fitness = mean over games of (score + survival bonus + RETURN_GAMMA * returns)
              - NODE_LAMBDA * hidden nodes.
    Must depend ONLY on (genome, seeds), so parallel == serial."""
    genome, seeds = args
    net = compile_genome(genome=genome)
    scores = []
    steps = []
    rets = []
    fitness = 0.0
    if _ENV is None: _init_worker()
    for s in seeds:
        score, step, returns, survival = play(net=net, env=_ENV, seed=s)
        scores.append(score)
        steps.append(step)
        rets.append(returns)
        fitness += score + survival + RETURN_GAMMA * returns
    hidden = sum(n.type == NodeType.HIDDEN for n in genome.nodes.values())
    return fitness / len(seeds) - NODE_LAMBDA * hidden, scores, steps, rets


# ---------------------------------------------------------------------------
# Fitness function for Population.run
# ---------------------------------------------------------------------------
def make_fitness_fn(pool: Pool | None, pop: Population, hof: HallOfFame):
    """Returns fitness_fn(genomes) -> np.ndarray, as Population.run expects.
    Every HOF_EVERY generations it also feeds this generation's top genomes to the hall of fame
    (here, while the scored genomes are still at hand)."""
    def fitness_fn(genomes: list[Genome]) -> np.ndarray:
        seeds = [int(s) for s in pop.rng.integers(0, 2**31 - 1, size=EPISODES)]
        jobs = [(g, seeds) for g in genomes]
        results = pool.map(evaluate_genome, jobs) if pool else list(map(evaluate_genome, jobs))
        fitnesses = np.array([r[0] for r in results])     # r = (fitness, scores, steps, returns)
        fitness_fn.mean_returns = float(np.mean([np.mean(r[3]) for r in results]))  # for the log
        fitness_fn.mean_score = float(np.mean([np.mean(r[1]) for r in results]))

        if pop.generation % HOF_EVERY == 0:
            # g.fitness is NOT set yet (Population.run sets it after we return): rank by `fitnesses`
            top = [genomes[i] for i in np.argsort(fitnesses)[::-1][:HOF_SIZE]]
            hof.update(top, pool, pop.rng)
        return fitnesses
    return fitness_fn


# ---------------------------------------------------------------------------
# Hall of fame
# ---------------------------------------------------------------------------
class HallOfFame:
    """The HOF_SIZE most reliable genomes seen so far, scored by mean PURE score
    (no survival bonus) over HOF_EPISODES fresh games.

    Bounded top-k = min-heap of size k: heap[0] is the WEAKEST of the kept ones,
    so "does the newcomer beat the worst?" is O(1) and evicting it is O(log k)."""

    def __init__(self, size: int = HOF_SIZE) -> None:
        self.size = size
        self.heap: list[tuple[float, int, Genome]] = []   # (score, tie-breaker, genome)
        self._tie = 0                                      # unique: ties never compare Genomes

    def add(self, score: float, genome: Genome) -> None:
        if any(g.conns == genome.conns and g.nodes == genome.nodes for _, _, g in self.heap):
            return                                         # elitism re-sends identical copies
        self._tie += 1
        item = (score, self._tie, genome)
        if len(self.heap) < self.size:
            heapq.heappush(self.heap, item)
        elif score > self.heap[0][0]:                      # beats the current worst?
            heapq.heapreplace(self.heap, item)             # pop worst + push new, one step

    def update(self, candidates: list[Genome], pool: Pool | None,
               rng: np.random.Generator) -> None:
        """Re-evaluate candidates on fresh seeds; add each by its mean pure score."""
        seeds = [int(x) for x in rng.integers(0, 2**31 - 1, size=HOF_EPISODES)]
        jobs = [(g, seeds) for g in candidates]
        results = pool.map(evaluate_genome, jobs) if pool else list(map(evaluate_genome, jobs))
        for g, (_, scores, _, _) in zip(candidates, results):
            self.add(float(np.mean(scores)), g)

    def ranked(self) -> list[tuple[float, Genome]]:
        """Best first. (The heap itself is NOT sorted; only heap[0] is guaranteed.)"""
        return [(score, g) for score, _, g in sorted(self.heap, key=lambda t: t[:2], reverse=True)]


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def save_checkpoint(out: Path, pop: Population, hof: HallOfFame) -> None:
    """Pickle everything needed to resume (pop holds rng, tracker, species, stats)."""
    ckpt_dir = out / "checkpoints"
    final = ckpt_dir / f"gen_{pop.generation:05d}.pkl"
    tmp = final.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump({"pop": pop, "hof": hof}, f)
    os.replace(tmp, final)                       # atomic: never a half-written .pkl
    for old in sorted(ckpt_dir.glob("gen_*.pkl"))[:-KEEP_CHECKPOINTS]:
        old.unlink()


def load_latest_checkpoint(out: Path) -> tuple[Population, HallOfFame]:
    latest = sorted((out / "checkpoints").glob("gen_*.pkl"))[-1]
    with open(latest, "rb") as f:
        data = pickle.load(f)
    print(f"resumed from {latest}", flush=True)
    return data["pop"], data["hof"]


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

    if args.resume:
        pop, hof = load_latest_checkpoint(out)
    else:
        pop, hof = Population(make_config(), args.seed), HallOfFame()

    (out / "best").mkdir(exist_ok=True)
    with Pool(args.workers, initializer=_init_worker) as pool:
        fitness_fn = make_fitness_fn(pool, pop, hof)      # hof is updated inside it
        while pop.generation < args.generations:
            t0 = time.time()
            pop.run(fitness_fn, 1)                        # exactly one generation
            with open(out / "best" / f"gen_{pop.generation - 1:04d}.pkl", "wb") as f:
                pickle.dump(pop.champ, f)                 # that generation's best, for figures
            pop.save_stats(str(out / "stats.csv"))
            save_checkpoint(out, pop, hof)
            row = pop.stats[-1]
            hof_best = hof.ranked()[0][0] if hof.heap else float("nan")
            print(f"gen {row['gen']:5d}  best {row['best']:6.3f}  mean {row['mean']:6.3f}  "
                  f"score {fitness_fn.mean_score:6.3f}  returns {fitness_fn.mean_returns:5.2f}  "
                  f"species {row['n_species']:3d}  hidden {row['mean_hidden']:5.2f}  "
                  f"hof {hof_best:6.3f}  {time.time() - t0:5.1f}s", flush=True)


if __name__ == "__main__":   # REQUIRED on Windows: Pool workers re-import this file
    main()
