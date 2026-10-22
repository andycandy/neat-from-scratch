"""Unit 4 §16 tests. Fill these in as you build experiments/slime.py."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from slime.slimevolley import SlimeVolleyEnv


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


def _genomes(n=6):
    from run_slime import make_config

    from neat.population import Population
    return Population(make_config(), 0).genomes[:n]


def test_same_genome_same_seeds_same_fitness():
    from run_slime import evaluate_genome
    g = _genomes(1)[0]
    a = evaluate_genome((g, [1, 2, 3], 1))
    evaluate_genome((_genomes(2)[1], [7], 0))    # play something else in between
    assert evaluate_genome((g, [1, 2, 3], 1)) == a   # no state leaks between games


def test_parallel_equals_serial():
    from multiprocessing import Pool

    from run_slime import evaluate_genome
    jobs = [(g, [3, 4, 5], 2) for g in _genomes()]
    serial = list(map(evaluate_genome, jobs))
    with Pool(2) as pool:
        parallel = pool.map(evaluate_genome, jobs)
    assert parallel == serial


def test_survival_bonus_never_beats_a_point():
    from run_slime import SURVIVAL_A
    assert SURVIVAL_A < 1                     # a full-length rally is worth less than one point


def test_returns_counted_for_a_real_player():
    """Drive OUR side with the built-in AI's own policy: it rallies, so returns must be > 0;
    a frozen agent (no buttons) never touches the ball, so returns must be 0."""
    import run_slime

    from slime.slimevolley import BaselinePolicy

    class PolicyNet:            # stand-in for a compiled network
        def __init__(self, policy): self.policy = policy

    def play_with(policy_fn, seed):
        env = SlimeVolleyEnv()
        orig_run = run_slime.run
        run_slime.run = lambda net, x: np.array([policy_fn(x[0])], dtype=float) * 2 - 1
        try:
            return run_slime.play(None, env, seed)
        finally:
            run_slime.run = orig_run

    bp = BaselinePolicy()
    good = sum(play_with(lambda o: bp.predict(o), s)[2] for s in range(3))
    frozen = sum(play_with(lambda o: [0, 0, 0], s)[2] for s in range(3))
    assert good > 3 and frozen == 0


def test_resume_matches_uninterrupted(tmp_path, monkeypatch):
    import dataclasses

    import run_slime
    from run_slime import (
        HallOfFame,
        load_latest_checkpoint,
        make_fitness_fn,
        save_checkpoint,
    )

    from neat.population import Population
    monkeypatch.setattr(run_slime, "EPISODES", 2)       # small and fast; restored after
    monkeypatch.setattr(run_slime, "MIRROR_GAMES", 1)   # 1 mirror + 1 AI game
    cfg = dataclasses.replace(run_slime.make_config(), pop_size=12)
    a, ha = Population(cfg, 0), HallOfFame()
    a.run(make_fitness_fn(None, a, ha), 3)
    b, hb = Population(cfg, 0), HallOfFame()
    b.run(make_fitness_fn(None, b, hb), 2)
    (tmp_path / "checkpoints").mkdir()
    save_checkpoint(tmp_path, b, hb)
    b, hb = load_latest_checkpoint(tmp_path)
    b.run(make_fitness_fn(None, b, hb), 1)
    assert a.stats == b.stats


def test_hall_of_fame_keeps_top_k_sorted():
    from run_slime import HallOfFame

    from neat.config import NeatConfig
    from neat.innovation import InnovationTracker
    from neat.population import make_initial_population
    cfg = NeatConfig(num_inputs=12, num_outputs=3, pop_size=8)
    gs = make_initial_population(cfg, InnovationTracker(15), np.random.default_rng(0))
    hof = HallOfFame(size=3)
    for score, g in zip([0.5, -1.0, 2.0, 2.0, 1.5, -3.0, 0.0, 4.0], gs):  # includes a tie
        hof.add(score, g)
    assert [s for s, _ in hof.ranked()] == [4.0, 2.0, 2.0]
    hof.add(9.0, gs[7])                  # identical genome already kept -> ignored
    assert len(hof.ranked()) == 3 and hof.ranked()[0][0] == 4.0


def test_play_batch_matches_play_on_ai_games():
    """Lockstep batching must not change anything: same results as one game at a time."""
    from run_slime import _envs, play, play_batch

    from neat.network import compile_genome
    for g in _genomes(3):
        net = compile_genome(g)
        seeds = [11, 12, 13, 14]
        one_by_one = [play(net, SlimeVolleyEnv(), s) for s in seeds]
        assert play_batch(net, _envs(4), seeds, 0) == one_by_one


def test_mirror_games_run_and_are_deterministic():
    from run_slime import _envs, play_batch

    from neat.network import compile_genome
    net = compile_genome(_genomes(1)[0])
    res = play_batch(net, _envs(6), list(range(6)), 6)
    assert all(-5 <= r[0] <= 5 and r[1] > 0 for r in res)       # real games were played
    assert res == play_batch(net, _envs(6), list(range(6)), 6)
