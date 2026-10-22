import numpy as np

from neat.genes import ConnGene, Genome, NodeGene, NodeType
from neat.network import compile_genome, run


def genome(extra_enabled=True):
    nodes = {
        0: NodeGene(0, NodeType.INPUT, None, 0.0),
        1: NodeGene(1, NodeType.INPUT, None, 0.0),
        2: NodeGene(2, NodeType.OUTPUT, "sigmoid", 0.0),
        3: NodeGene(3, NodeType.HIDDEN, "tanh", 0.0),
    }
    conns = {c.inno_id: c for c in [
        ConnGene(0, 0, 3, -0.5, True), ConnGene(1, 1, 3, -1, True),
        ConnGene(2, 3, 2, 2, True), ConnGene(3, 0, 2, 1, True),
    ]}
    return Genome(0, nodes, conns, {})


def test_hand_example():
    out = run(compile_genome(genome()), np.array([[1.0, 0.5]]))
    expected = 1 / (1 + np.exp(-(2 * np.tanh(-1.0) + 1.0)))  # by hand
    assert abs(out[0, 0] - expected) < 1e-9


def test_batch_equals_rows():
    g = genome()
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1.]])
    rows = np.vstack([run(compile_genome(g), r[None]) for r in x])
    assert np.allclose(run(compile_genome(g), x), rows)


def test_disabled_conn_ignored():
    g = genome()
    base = run(compile_genome(g), np.array([[1.0, 0.5]]))
    g.conns[9] = ConnGene(9, 1, 2, 1000.0, False)
    assert np.allclose(run(compile_genome(g), np.array([[1.0, 0.5]])), base)
