from neat.genes import Genome, NodeGene, NodeType
from neat.innovation import InnovationTracker


def g():
    return Genome(id=0, nodes={}, conns={}, conns_broken={})


def test_same_conn_same_number():
    t = InnovationTracker(10)
    assert t.conn_innovation(3, 4) == t.conn_innovation(3, 4)
    assert t.conn_innovation(3, 4) != t.conn_innovation(4, 3)


def test_same_split_same_node():
    t = InnovationTracker(10)
    assert t.split_node_id(7, g()) == t.split_node_id(7, g())


def test_same_genome_split_twice():
    t = InnovationTracker(10)
    a, b = g(), g()
    first = t.split_node_id(7, a)
    a.nodes[first] = NodeGene(first, NodeType.HIDDEN, "tanh", 0.0)
    a.conns_broken[7] = 1
    second = t.split_node_id(7, a)
    assert second != first
    # another genome's 2nd split gets the same id
    b.nodes[first] = a.nodes[first]
    b.conns_broken[7] = 1
    assert t.split_node_id(7, b) == second
