import numpy as np
from neat.config import NeatConfig
from neat.innovation import InnovationTracker
from neat.population import make_initial_population


def test_initial_population_shares_innovations():
    cfg = NeatConfig(num_inputs=2, num_outputs=1, pop_size=10)
    pop = make_initial_population(cfg, InnovationTracker(3), np.random.default_rng(0))
    assert len({frozenset(g.conns) for g in pop}) == 1
    assert len({g.conns[0].weight for g in pop}) > 1


def test_same_seed_same_run():
    from neat.population import Population

    def fit(genomes):
        from neat.network import compile_genome, run
        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1.]])
        return np.array([-np.abs(run(compile_genome(g), x)[:, 0] - [0, 1, 1, 0]).sum()
                         for g in genomes])

    cfg = NeatConfig(num_inputs=2, num_outputs=1, pop_size=30)
    a, b = Population(cfg, 7), Population(cfg, 7)
    assert a.run(fit, 10).fitness == b.run(fit, 10).fitness
    assert a.stats == b.stats


def test_save_load_roundtrip(tmp_path):
    from neat.population import Population
    cfg = NeatConfig(num_inputs=2, num_outputs=1, pop_size=20)
    p = Population(cfg, 1)
    p.save(str(tmp_path / "ck.pkl"))
    q = Population.load(str(tmp_path / "ck.pkl"))
    assert q.rng.random() == p.rng.random() and q.tracker.conn_registry == p.tracker.conn_registry


def test_adjust_threshold():
    from neat.population import adjust_threshold
    cfg = NeatConfig(num_inputs=2, num_outputs=1, compat_threshold=1.0)
    assert adjust_threshold(cfg, 20).compat_threshold > 1.0       # too many -> raise
    assert adjust_threshold(cfg, 3).compat_threshold < 1.0        # too few -> lower
    assert adjust_threshold(cfg, 10).compat_threshold == 1.0      # in range -> keep
    low = NeatConfig(num_inputs=2, num_outputs=1, compat_threshold=0.3)
    assert adjust_threshold(low, 1).compat_threshold == 0.3       # clamped at min
    high = NeatConfig(num_inputs=2, num_outputs=1, compat_threshold=2.0, threshold_max=2.0)
    assert adjust_threshold(high, 50).compat_threshold == 2.0     # clamped at max (no windup)


def test_threshold_pulls_species_into_range():
    from neat.population import Population
    rng = np.random.default_rng(0)
    cfg = NeatConfig(num_inputs=2, num_outputs=1, compat_threshold=0.3, threshold_step=0.3)
    p = Population(cfg, 0)
    p.run(lambda gs: rng.random(len(gs)), 40)
    counts = [r["n_species"] for r in p.stats]
    assert max(counts) > 12                                # starts with too many species
    assert np.mean(counts[-10:]) < max(counts)             # threshold rose and pulled it down
    assert p.stats[-1]["threshold"] > 0.3


def test_snapshots_and_stats_columns(tmp_path):
    from neat.population import Population
    cfg = NeatConfig(num_inputs=2, num_outputs=1, pop_size=30)
    p = Population(cfg, 0)
    p.run(lambda gs: np.zeros(len(gs)), 3, snapshot_dir=str(tmp_path / "best"))
    assert sorted(f.name for f in (tmp_path / "best").iterdir()) ==         ["gen_0000.pkl", "gen_0001.pkl", "gen_0002.pkl"]
    p.save_stats(str(tmp_path / "s.csv"))
    header = (tmp_path / "s.csv").read_text().splitlines()[0]
    assert "act_tanh" in header and "champ_hidden" in header and "threshold" in header


def test_draw_genome_writes_png(tmp_path):
    from neat.population import make_initial_population
    from neat.innovation import InnovationTracker
    from neat.mutation import mutate
    from neat.visualize import draw_genome
    cfg = NeatConfig(num_inputs=2, num_outputs=1, pop_size=1, p_add_node=0.5, p_toggle=0.2)
    t, rng = InnovationTracker(3), np.random.default_rng(0)
    g = make_initial_population(cfg, t, rng)[0]
    for _ in range(30):
        g = mutate(g, cfg, t, rng)
    draw_genome(g, str(tmp_path / "g.png"))
    assert (tmp_path / "g.png").stat().st_size > 0


def test_stop_checks_every_genome():
    from neat.population import Population
    cfg = NeatConfig(num_inputs=2, num_outputs=1, pop_size=20)
    p = Population(cfg, 0)
    target = p.genomes[5]                                      # not the fittest
    fits = lambda gs: np.array([1.0 if g is gs[0] else 0.0 for g in gs])
    got = p.run(fits, 5, stop=lambda g: g is target)
    assert got is target and p.generation == 0
