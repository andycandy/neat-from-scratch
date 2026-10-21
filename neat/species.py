from dataclasses import dataclass
from dataclasses import dataclass

from neat.config import NeatConfig
from neat.genes import Genome

@dataclass
class Species:
    key: int
    representative: Genome
    members: list[Genome]
    best_fitness: float
    last_improved: int

def compatibility_distance(a: Genome, b: Genome, cfg: NeatConfig) -> float:
    c1, c2, c3 = cfg.distance_coefficients
    matching = a.conns.keys() & b.conns.keys()
    left = a.conns.keys() ^ b.conns.keys()
    ceiling = min(max(a.conns, default=0), max(b.conns, default=0))
    E = {i for i in left if i > ceiling}
    D = left - E
    W = 0.0
    if matching:
        W = sum(abs(a.conns[i].weight - b.conns[i].weight) for i in matching) / len(matching)
    N = max(len(a.conns), len(b.conns))
    if N < cfg.n_small_cutoff:
        N = 1
    return c1 * len(E) / N + c2 * len(D) / N + c3 * W


@dataclass
class Species:
    key: int
    representative: Genome
    members: list[Genome]
    best_fitness: float          # best ever seen in this species
    last_improved: int           # generation of last improvement


def speciate(genomes: list[Genome], species_list: list[Species], cfg: NeatConfig,
             generation: int, next_key: int) -> tuple[list[Species], int]:
    """Assign every genome to exactly one species. Returns (species, next_key).

    Stanley's rule: each genome joins the FIRST species whose (previous-generation)
    representative is within the threshold, else founds a new species. Species left
    with no members die. So a higher threshold lets older species absorb newer ones,
    which is what lets the dynamic threshold actually reduce the species count."""
    for s in species_list:
        s.members = []

    for g in genomes:
        joined = False
        for s in species_list:
            if (compatibility_distance(g, s.representative, cfg) < cfg.compat_threshold):
                s.members.append(g)
                joined = True
                break
        if joined is False:
            new_specie = Species(
                key=next_key, 
                representative=g, 
                members=[g], 
                best_fitness=-1e9, 
                last_improved=generation
            )
            species_list.append(new_specie)
            next_key += 1

    species_list = [s for s in species_list if s.members]
    for s in species_list:   # next generation compares against the member closest to the old rep
        s.representative = min(s.members, key=lambda g: compatibility_distance(g, s.representative, cfg))
    return species_list, next_key
