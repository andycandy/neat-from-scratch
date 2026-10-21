"""Unit 4 §16 tests. Fill these in as you build experiments/slime.py."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from slime.slimevolley import SlimeVolleyEnv  # noqa: E402


def test_env_random_policy():
    env = SlimeVolleyEnv()
    rng = np.random.default_rng(0)
    scores = []
    for ep in range(10):
        env.seed(ep)
        obs, done, total = env.reset(), False, 0
        while not done:
            obs, r, done, _ = env.step(rng.integers(0, 2, 3))
            total += r
        scores.append(total)
    assert obs.shape == (12,)
    assert np.mean(scores) < -4                      # README: random ~ -4.87


def test_same_genome_same_seeds_same_fitness():
    ...  # evaluate_genome((g, [1, 2])) twice -> identical tuples


def test_parallel_equals_serial():
    ...  # make_fitness_fn(None, ...) vs make_fitness_fn(Pool(2, initializer=_init_worker), ...)
    #     on the same genomes; give both a Population with the same seed so seeds match


def test_survival_bonus_never_beats_a_point():
    ...  # with BETA < 1: (score+1) + BETA*0  >  score + BETA*1   for the worst case


def test_resume_matches_uninterrupted():
    ...  # run 3 gens; vs run 2 gens, save, load, run 1 more -> same stats rows


def test_hall_of_fame_sorted():
    ...  # after update_hall_of_fame, scores are non-increasing and len <= HOF_SIZE
