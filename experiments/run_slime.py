"""Slime Volleyball with NEAT. Reasons for every setting: DECISIONS.md, Unit 4.

uv run python experiments/run_slime.py --out runs/slime5
uv run python experiments/run_slime.py --out runs/slime5 --resume
"""
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

from neat.config import NeatConfig
from neat.genes import NodeType
from neat.network import compile_genome, run
from neat.population import Population
from slime.slimevolley import SlimeVolleyEnv

SURVIVAL_A = 0.75      # max survival bonus per rally (< 1: a point always beats a rally)
SURVIVAL_TAU = 300     # steps (10 s) for the survival bonus to saturate
RETURN_GAMMA = 0.2     # bonus per return over the net
NODE_LAMBDA = 0.01     # penalty per hidden node
TOUCH_MARGIN = 0.3     # the game pushes the ball just outside the agent after a bounce
EPISODES = 16          # games per genome per generation
MIRROR_GAMES = 12      # of which vs itself; the rest vs the built-in AI
HOF_SIZE, HOF_EVERY, HOF_EPISODES = 5, 10, 20
KEEP_CHECKPOINTS = 3


def make_config() -> NeatConfig:
    return NeatConfig(
        num_inputs=12, num_outputs=3, pop_size=150,
        output_activation="tanh",
        activations=("identity", "sigmoid", "tanh", "relu", "sin", "gauss", "abs", "step"),
        max_stagnation=20,
        target_species=(8, 12),
        n_small_cutoff=10**9,          # never divide the distance by N
        distance_coefficients=(1.0, 1.0, 0.4),   # the paper's (slime6); slime5 used (0.5, 0.5, 0.3)
        # threshold range doubled with the coefficients (slime5: 3.0, 0.3, 1.0, 6.0), so its cap doesn't bind
        compat_threshold=6.0, threshold_step=0.6, threshold_min=2.0, threshold_max=12.0,
    )


_ENVS = []   # one game object per game, reused by this process


def _envs(n):
    while len(_ENVS) < n:
        _ENVS.append(SlimeVolleyEnv())
    return _ENVS[:n]


class Tracker:
    """Score, steps, returns and survival bonus of OUR (right) agent in one game."""

    def __init__(self, env):
        self.env, self.me = env, env.game.agent_right
        self.touch_dist2 = (self.me.r + 0.5 + TOUCH_MARGIN) ** 2
        self.score, self.steps, self.returns, self.rally, self.survival = 0.0, 0, 0, 0, 0.0
        self.touched, self.prev_bx = False, env.game.ball.x

    def update(self, reward):
        ball = self.env.game.ball
        self.score += reward
        self.steps += 1
        self.rally += 1
        if (ball.x - self.me.x) ** 2 + (ball.y - self.me.y) ** 2 < self.touch_dist2:
            self.touched = True
        if self.touched and self.prev_bx > 0 > ball.x:     # we hit it over the net
            self.returns += 1
            self.touched = False
        self.prev_bx = ball.x
        if reward != 0:                                     # point scored: rally over
            self.survival += SURVIVAL_A * (1 - np.exp(-self.rally / SURVIVAL_TAU))
            self.rally, self.touched, self.prev_bx = 0, False, ball.x

    def result(self):
        last = SURVIVAL_A * (1 - np.exp(-self.rally / SURVIVAL_TAU))   # unfinished rally
        return self.score, self.steps, self.returns, self.survival + last


def play_batch(net, envs, seeds, n_mirror):
    """All games step together: one network call per step.
    Games 0..n_mirror-1 are vs itself, the rest vs the built-in AI.
    Returns (score, steps, returns, survival) per game."""
    games = []
    for env, seed in zip(envs, seeds):
        env.seed(seed)
        obs = env.reset()
        env.policy.reset()          # the built-in AI is an RNN; reset() doesn't clear it
        games.append({"env": env, "obs_r": obs, "obs_l": obs, "track": Tracker(env), "done": False})

    while not all(g["done"] for g in games):
        rows, who = [], []                                   # the batch, and who each row belongs to
        for k, g in enumerate(games):
            if g["done"]:
                continue                                     # finished games sit out
            rows.append(g["obs_r"]); who.append((k, "r"))    # right player's view
            if k < n_mirror:
                rows.append(g["obs_l"]); who.append((k, "l"))  # mirror game: left player is us too

        acts = (run(net, np.stack(rows)) > 0).astype(int)   # ONE call for all rows
        act_of = {kw: acts[i] for i, kw in enumerate(who)}  # (game k, side) -> its 3 buttons

        for k, g in enumerate(games):
            if g["done"]:
                continue
            if k < n_mirror:                                 # we control both players
                obs_r, reward, done, info = g["env"].step(act_of[(k, "r")], act_of[(k, "l")])
                g["obs_l"] = info["otherObs"]                # left player's next view
            else:                                            # built-in AI plays the left side
                obs_r, reward, done, info = g["env"].step(act_of[(k, "r")])
            g["obs_r"] = obs_r
            g["track"].update(reward)
            g["done"] = done

    return [g["track"].result() for g in games]


def play(net, env, seed):
    """One game vs the built-in AI."""
    return play_batch(net, [env], [seed], 0)[0]


def evaluate_genome(job):
    """(genome, seeds, n_mirror) -> (fitness, AI-game scores, steps, returns).
    Mirror-game scores are left out of the fitness: vs yourself they're zero-sum noise."""
    genome, seeds, n_mirror = job
    results = play_batch(compile_genome(genome), _envs(len(seeds)), seeds, n_mirror)
    fitness = 0.0
    for k, (score, _, returns, survival) in enumerate(results):
        if k >= n_mirror:
            fitness += score
        fitness += survival + RETURN_GAMMA * returns
    hidden = sum(n.type == NodeType.HIDDEN for n in genome.nodes.values())
    return (fitness / len(seeds) - NODE_LAMBDA * hidden,
            [r[0] for r in results[n_mirror:]],
            [r[1] for r in results],
            [r[2] for r in results])


def evaluate_all(pool, jobs):
    return pool.map(evaluate_genome, jobs) if pool else [evaluate_genome(j) for j in jobs]


def make_fitness_fn(pool, pop, hof):
    def fitness_fn(genomes):
        seeds = [int(s) for s in pop.rng.integers(0, 2**31 - 1, size=EPISODES)]
        results = evaluate_all(pool, [(g, seeds, MIRROR_GAMES) for g in genomes])
        fitnesses = np.array([r[0] for r in results])
        fitness_fn.mean_score = float(np.mean([np.mean(r[1]) for r in results]))
        fitness_fn.mean_returns = float(np.mean([np.mean(r[3]) for r in results]))
        if pop.generation % HOF_EVERY == 0:
            top = [genomes[i] for i in np.argsort(fitnesses)[::-1][:HOF_SIZE]]
            hof.update(top, pool, pop.rng)
        return fitnesses
    return fitness_fn


class HallOfFame:
    """Best HOF_SIZE genomes by mean pure score vs the built-in AI over HOF_EPISODES games.
    Min-heap: heap[0] is the weakest kept genome, the one a newcomer has to beat."""

    def __init__(self, size=HOF_SIZE):
        self.size = size
        self.heap = []      # (score, tie-breaker, genome)
        self._tie = 0

    def add(self, score, genome):
        if any(g.conns == genome.conns and g.nodes == genome.nodes for _, _, g in self.heap):
            return          # elitism re-sends identical copies
        self._tie += 1
        item = (score, self._tie, genome)
        if len(self.heap) < self.size:
            heapq.heappush(self.heap, item)
        elif score > self.heap[0][0]:
            heapq.heapreplace(self.heap, item)

    def update(self, candidates, pool, rng):
        seeds = [int(x) for x in rng.integers(0, 2**31 - 1, size=HOF_EPISODES)]
        results = evaluate_all(pool, [(g, seeds, 0) for g in candidates])
        for g, r in zip(candidates, results):
            self.add(float(np.mean(r[1])), g)

    def ranked(self):
        return [(s, g) for s, _, g in sorted(self.heap, key=lambda t: t[:2], reverse=True)]


def save_checkpoint(out, pop, hof):
    final = out / "checkpoints" / f"gen_{pop.generation:05d}.pkl"
    tmp = final.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump({"pop": pop, "hof": hof}, f)
    os.replace(tmp, final)          # a crash mid-write can't leave a broken checkpoint
    for old in sorted((out / "checkpoints").glob("gen_*.pkl"))[:-KEEP_CHECKPOINTS]:
        old.unlink()


def load_latest_checkpoint(out):
    latest = max((out / "checkpoints").glob("gen_*.pkl"))
    with open(latest, "rb") as f:
        data = pickle.load(f)
    print(f"resumed from {latest}", flush=True)
    return data["pop"], data["hof"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/slime6")
    ap.add_argument("--generations", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "best").mkdir(exist_ok=True)

    if args.resume:
        pop, hof = load_latest_checkpoint(out)
    else:
        pop, hof = Population(make_config(), args.seed), HallOfFame()

    with Pool(args.workers) as pool:
        fitness_fn = make_fitness_fn(pool, pop, hof)
        while pop.generation < args.generations:
            t0 = time.time()
            pop.run(fitness_fn, 1)
            with open(out / "best" / f"gen_{pop.generation - 1:04d}.pkl", "wb") as f:
                pickle.dump(pop.champ, f)
            pop.save_stats(str(out / "stats.csv"))
            save_checkpoint(out, pop, hof)
            row = pop.stats[-1]
            hof_best = hof.ranked()[0][0] if hof.heap else float("nan")
            print(f"gen {row['gen']:5d}  best {row['best']:6.3f}  score {fitness_fn.mean_score:6.3f}  "
                  f"returns {fitness_fn.mean_returns:5.2f}  species {row['n_species']:3d}  "
                  f"hidden {row['mean_hidden']:5.2f}  hof {hof_best:6.3f}  {time.time() - t0:5.1f}s",
                  flush=True)


if __name__ == "__main__":      # required on Windows: workers re-import this file
    main()
