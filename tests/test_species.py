import dataclasses

from neat.config import NeatConfig
from neat.genes import ConnGene, Genome, NodeGene, NodeType
from neat.species import compatibility_distance as dist

CFG = NeatConfig(num_inputs=2, num_outputs=1)


def make(weights: dict[int, float]) -> Genome:
    nodes = {0: NodeGene(0, NodeType.INPUT, None, 0.0),
             2: NodeGene(2, NodeType.OUTPUT, "sigmoid", 0.0)}
    return Genome(0, nodes, {i: ConnGene(i, 0, 2, w, True) for i, w in weights.items()}, {})


def test_self_distance_zero():
    g = make({1: 0.5, 2: -1.0})
    assert dist(g, g, CFG) == 0


def test_symmetric():
    a, b = make({1: 0.5, 2: 1.0, 5: 1.0}), make({1: 0.1, 3: 1.0, 4: 2.0})
    assert dist(a, b, CFG) == dist(b, a, CFG)


def test_unit1_example():
    # matching {1,2,3} (mean |dw| = 0.3), a-only {4,5} excess... build E=2, D=3, N=6
    a = make({1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 8: 0.0, 9: 0.0})        # 6 conns
    b = make({1: 0.3, 2: 0.3, 3: 0.3, 5: 0.0, 6: 0.0, 7: 0.0}) # 6 conns
    # a-only {4,8,9}, b-only {5,6,7}; ceiling = min(9,7)=7 -> excess {8,9}, disjoint {4,5,6,7}
    cfg = dataclasses.replace(CFG, distance_coefficients=(1.0, 1.0, 0.4), n_small_cutoff=0)
    assert abs(dist(a, b, cfg) - (2 / 6 + 4 / 6 + 0.4 * 0.3)) < 1e-9


def test_identical_genomes_one_species():
    from neat.species import speciate
    gs = [make({1: .5, 2: .5}) for _ in range(10)]
    species, _ = speciate(gs, [], CFG, 0, 0)
    assert len(species) == 1 and len(species[0].members) == 10


def test_far_apart_genomes_two_species():
    from neat.species import speciate
    a = make({1: 0.0})
    b = make({i: 0.0 for i in range(2, 12)})          # 11 non-shared genes, N=1 -> dist 5.5 > 3
    species, _ = speciate([a, b], [], CFG, 0, 0)
    assert len(species) == 2


def test_everyone_in_exactly_one_species():
    from neat.species import speciate
    gs = [make({i: 0.0 for i in range(k + 1)}) for k in range(0, 40, 3)]
    species, _ = speciate(gs, [], CFG, 0, 0)
    species, _ = speciate(gs, species, CFG, 1, 100)   # second round reuses old species
    members = [id(g) for s in species for g in s.members]
    assert len(members) == len(gs) and set(members) == {id(g) for g in gs}
