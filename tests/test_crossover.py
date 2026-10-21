import numpy as np

from neat.config import NeatConfig
from neat.crossover import crossover
from neat.genes import ConnGene, Genome, NodeGene, NodeType

CFG = NeatConfig(num_inputs=2, num_outputs=1)


def make(innos, extra_nodes=()):
    """Genome with inputs 0,1, output 2, given hidden ids; one conn per innovation number."""
    nodes = {0: NodeGene(0, NodeType.INPUT, None, 0.0),
             1: NodeGene(1, NodeType.INPUT, None, 0.0),
             2: NodeGene(2, NodeType.OUTPUT, "sigmoid", 0.0)}
    for h in extra_nodes:
        nodes[h] = NodeGene(h, NodeType.HIDDEN, "tanh", 0.0)
    conns = {i: ConnGene(i, 0, 2, float(i), True) for i in innos}
    return Genome(0, nodes, conns, {})


def test_self_crossover_same_genome():
    g = make([1, 2, 3])
    c = crossover(g, g, CFG, np.random.default_rng(0))
    assert c.conns == g.conns and c.nodes == g.nodes


def test_child_innovations_equal_fitter():
    p1, p2 = make([1, 2, 4, 5]), make([1, 2, 3, 6, 7])
    c = crossover(p1, p2, CFG, np.random.default_rng(0))
    assert set(c.conns) == {1, 2, 4, 5}


def test_all_referenced_nodes_exist():
    fitter = make([1], extra_nodes=(7,))
    fitter.conns[2] = ConnGene(2, 0, 7, 1.0, True)
    fitter.conns[3] = ConnGene(3, 7, 2, 1.0, True)
    c = crossover(fitter, make([1]), CFG, np.random.default_rng(0))
    assert all(x.src in c.nodes and x.dest in c.nodes for x in c.conns.values())
    assert 7 in c.nodes


def test_disabled_gene_mostly_stays_disabled():
    a, b = make([1]), make([1])
    b.conns[1] = ConnGene(1, 0, 2, 1.0, False)
    rng = np.random.default_rng(0)
    on = sum(crossover(a, b, CFG, rng).conns[1].enabled for _ in range(2000))
    assert 0.2 < on / 2000 < 0.3   # ~25% re-enabled


def test_parents_untouched():
    a, b = make([1, 2]), make([1, 3])
    before = (dict(a.conns), dict(b.conns))
    crossover(a, b, CFG, np.random.default_rng(0))
    assert (a.conns, b.conns) == before
