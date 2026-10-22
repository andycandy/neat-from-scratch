import numpy as np

from neat.config import NeatConfig
from neat.reproduction import allocate

CFG = NeatConfig(num_inputs=2, num_outputs=1)


def test_allocate_example():
    assert allocate([3.4, 3.3, 3.3], 10) == [4, 3, 3]


def test_allocate_always_sums():
    rng = np.random.default_rng(0)
    for _ in range(500):
        shares = list(rng.random(int(rng.integers(1, 12))))
        assert sum(allocate(shares, 150)) == 150


def test_allocate_all_zero():
    assert sum(allocate([0.0, 0.0, 0.0], 10)) == 10


def _setup(fitness_fn, n_species=3, per=10, seed=0):
    """n_species hand-made species of `per` genomes each; fitness_fn(species_idx, j) -> float."""
    from neat.innovation import InnovationTracker
    from neat.population import make_initial_population
    from neat.species import Species
    cfg = NeatConfig(num_inputs=2, num_outputs=1, pop_size=n_species * per)
    tracker = InnovationTracker(3)
    rng = np.random.default_rng(seed)
    pop = make_initial_population(cfg, tracker, rng)
    species = []
    for k in range(n_species):
        members = pop[k * per:(k + 1) * per]
        for j, g in enumerate(members):
            g.fitness = fitness_fn(k, j)
        species.append(Species(k, members[0], members, float("-inf"), 0))
    return cfg, tracker, rng, species


def test_children_equal_pop_size():
    from neat.reproduction import reproduce
    for seed in range(20):
        r = np.random.default_rng(seed)
        cfg, t, rng, sp = _setup(lambda k, j, r=r: float(r.normal(0, 5)), n_species=int(r.integers(1, 6)),
                                 per=int(r.integers(1, 12)), seed=seed)
        kids, _, _ = reproduce(sp, cfg, t, rng, generation=1, next_key=1000)
        assert len(kids) == cfg.pop_size
        assert len({id(k) for k in kids}) == len(kids)          # distinct objects
        assert len({k.id for k in kids}) == len(kids)           # distinct keys


def test_negative_fitness_favours_better_species():
    from neat.reproduction import allocate, species_scores
    _, _, _, sp = _setup(lambda k, j: -1.0 if k == 0 else -3.0, n_species=2)
    counts = allocate(species_scores(sp), 20)
    assert counts[0] >= counts[1]


def test_elite_survives_unchanged():
    from neat.reproduction import reproduce
    cfg, t, rng, sp = _setup(lambda k, j: float(j), n_species=1, per=10)
    best = max(sp[0].members, key=lambda g: g.fitness)
    kids, _, _ = reproduce(sp, cfg, t, rng, generation=1, next_key=1000)
    assert any(k.conns == best.conns and k.nodes == best.nodes for k in kids)


def test_culled_species_not_returned():
    from neat.reproduction import reproduce
    cfg, t, rng, sp = _setup(lambda k, j: float(k), n_species=3)
    sp[0].last_improved = -100            # stagnant, and not the best species
    sp[0].best_fitness = 0.0              # record already 0 -> no improvement this gen
    _, survivors, _ = reproduce(sp, cfg, t, rng, generation=50, next_key=1000)
    assert sp[0] not in survivors and len(survivors) == 2


def test_min_species_size_guaranteed():
    from neat.reproduction import breed_species, reproduce  # noqa: F401
    cfg, t, rng, sp = _setup(lambda k, j: float(k), n_species=3)
    _, survivors, _ = reproduce(sp, cfg, t, rng, generation=1, next_key=1000)
    assert len(survivors) == 3           # worst species (score 0) still gets 2 children
