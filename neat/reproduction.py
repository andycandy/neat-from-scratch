import numpy as np

from neat.config import NeatConfig
from neat.crossover import crossover
from neat.genes import Genome
from neat.innovation import InnovationTracker
from neat.mutation import mutate
from neat.species import Species


def allocate(shares: list[float], total: int) -> list[int]:
    """Largest remainder: whole counts proportional to `shares`, summing exactly to `total`.
    All shares 0 -> split evenly."""
    k = len(shares)
    summ = sum(shares)
    if summ == 0:
        return [total // k + (1 if i < total % k else 0) for i in range(k)]
    raw = [total * x / summ for x in shares]
    base = [int(r) for r in raw]
    left = total - sum(base)
    for i in sorted(range(k), key=lambda i: raw[i] - base[i], reverse=True)[:left]:
        base[i] += 1
    return base


def species_scores(species_list: list[Species]) -> list[float]:
    """Mean fitness per species after shifting so the worst genome overall scores 0."""
    lo = min(g.fitness for s in species_list for g in s.members)
    return [float(np.mean([g.fitness - lo for g in s.members])) for s in species_list]


def update_and_cull(species_list: list[Species], cfg: NeatConfig,
                    generation: int) -> list[Species]:
    """Update best_fitness/last_improved; drop species stagnant for >= cfg.max_stagnation
    generations, but never the species holding the best genome."""
    for s in species_list:
        best_now = max(g.fitness for g in s.members)
        if best_now > s.best_fitness:
            s.last_improved = generation
            s.best_fitness = best_now
    best_s = max(species_list, key=lambda s: s.best_fitness)
    species_list = [s for s in species_list
                    if s is best_s or generation - s.last_improved < cfg.max_stagnation]
    return species_list


def breed_species(s: Species, n: int, cfg: NeatConfig, tracker: InnovationTracker,
                  rng: np.random.Generator, next_key: int) -> tuple[list[Genome], int]:
    """Exactly n children: elite (if > cfg.elitism_min_size members), then crossover or
    clone from the top cfg.survival_threshold, each mutated. Every child gets a new id."""
    children = []
    members = sorted(s.members, key=lambda g: g.fitness, reverse=True)
    if n == 0: return children, next_key
    if len(members) > cfg.elitism_min_size:
        e = members[0]
        children.append(Genome(next_key, dict(e.nodes), dict(e.conns), dict(e.conns_broken)))
        next_key += 1
    pool = members[:max(2, int(len(members) * cfg.survival_threshold))]
    while len(children) < n:
        if len(pool) >= 2 and rng.random() < cfg.p_crossover:
            i, j = rng.choice(len(pool), 2, replace=False)
            a, b = pool[i], pool[j]
            if b.fitness > a.fitness or (b.fitness == a.fitness and rng.random() < 0.5):
                a, b = b, a
            child = crossover(a, b, cfg, rng)
        else:
            p = pool[int(rng.integers(len(pool)))]
            child = Genome(p.id, dict(p.nodes), dict(p.conns), dict(p.conns_broken))  # copy
        child = mutate(child, cfg, tracker, rng)
        child.id = next_key; child.fitness = None; next_key += 1
        children.append(child)
    return children, next_key
    


def reproduce(species_list: list[Species], cfg: NeatConfig, tracker: InnovationTracker,
              rng: np.random.Generator, generation: int,
              next_key: int) -> tuple[list[Genome], list[Species], int]:
    """cull -> scores -> allocate -> breed each species.
    Returns (children, surviving species, next_key). Species that were culled or got
    0 children are extinct and must not be passed to the next speciate."""
    species_list = update_and_cull(species_list, cfg, generation)
    scores = species_scores(species_list)
    floor = cfg.min_species_size
    if floor * len(species_list) <= cfg.pop_size:   # guarantee a minimum, share the rest
        counts = [floor + c for c in allocate(scores, cfg.pop_size - floor * len(species_list))]
    else:
        counts = allocate(scores, cfg.pop_size)
    children: list[Genome] = []
    survivors: list[Species] = []
    for s, n in zip(species_list, counts):
        kids, next_key = breed_species(s, n, cfg, tracker, rng, next_key)
        children += kids
        if n > 0:
            survivors.append(s)
    return children, survivors, next_key
