import dataclasses

import numpy as np

from neat.config import NeatConfig
from neat.genes import ConnGene, Genome, NodeGene, NodeType
from neat.mutation import mutate_weights

CFG = NeatConfig(num_inputs=2, num_outputs=1)


def genome():
    nodes = {
        0: NodeGene(0, NodeType.INPUT, None, 0.0),
        1: NodeGene(1, NodeType.INPUT, None, 0.0),
        2: NodeGene(2, NodeType.OUTPUT, "sigmoid", 0.5),
    }
    conns = {0: ConnGene(0, 0, 2, 1.0, True), 1: ConnGene(1, 1, 2, -1.0, False)}
    return Genome(0, nodes, conns, {})


def test_parent_unchanged_and_not_shared():
    g = genome()
    before = (dict(g.nodes), dict(g.conns))
    child = mutate_weights(g, CFG, np.random.default_rng(0))
    assert (g.nodes, g.conns) == before
    assert child is not g and child.conns is not g.conns and child.nodes is not g.nodes
    assert child.conns[0].weight != g.conns[0].weight


def test_clipped():
    cfg = dataclasses.replace(CFG, sigma=100.0)
    child = mutate_weights(genome(), cfg, np.random.default_rng(0))
    assert all(abs(c.weight) <= cfg.w_max for c in child.conns.values())
    assert all(abs(n.bias) <= cfg.w_max for n in child.nodes.values())


def test_inputs_unchanged_and_seeded():
    a = mutate_weights(genome(), CFG, np.random.default_rng(1))
    b = mutate_weights(genome(), CFG, np.random.default_rng(1))
    assert a.conns == b.conns and a.nodes == b.nodes
    assert a.nodes[0].bias == 0.0


def test_reachable_follows_disabled():
    from neat.mutation import reachable
    nodes = {i: NodeGene(i, NodeType.HIDDEN, "tanh", 0.0) for i in (4, 5, 6)}
    conns = {0: ConnGene(0, 4, 5, 1.0, True), 1: ConnGene(1, 5, 6, 1.0, False)}
    g = Genome(0, nodes, conns, {})
    assert reachable(g, 4, 5) and not reachable(g, 5, 4)
    assert reachable(g, 5, 6) and reachable(g, 4, 6)


def test_add_connection_never_cycles_or_targets_input():
    from neat.innovation import InnovationTracker
    from neat.mutation import mutate_add_connection
    from neat.network import topo_order
    from neat.population import make_initial_population
    cfg = NeatConfig(num_inputs=2, num_outputs=1, pop_size=1)
    tracker = InnovationTracker(3)
    rng = np.random.default_rng(0)
    g = make_initial_population(cfg, tracker, rng)[0]
    # add hidden nodes by hand so there is room for new edges
    for h in (3, 4, 5):
        g.nodes[h] = NodeGene(h, NodeType.HIDDEN, "tanh", 0.0)
    for _ in range(1000):
        g = mutate_add_connection(g, cfg, tracker, rng)
        topo_order(g.nodes, g.conns.values())  # raises on cycle
    assert all(g.nodes[c.dest].type != NodeType.INPUT for c in g.conns.values())
    assert len(g.conns) > 3


def test_add_node_structure_and_parent_untouched():
    from neat.innovation import InnovationTracker
    from neat.mutation import add_node
    g = genome()                      # conn 0 enabled (0->2, w=1.0), conn 1 disabled
    before = (dict(g.nodes), dict(g.conns), dict(g.conns_broken))
    t = InnovationTracker(3)
    t.conn_innovation(0, 2)           # register genome's conns so numbers 0, 1 are taken
    t.conn_innovation(1, 2)
    c = add_node(g, CFG, t, np.random.default_rng(0))
    assert (g.nodes, g.conns, g.conns_broken) == before
    assert len(c.nodes) == 4 and len(c.conns) == 4
    assert not c.conns[0].enabled and c.conns_broken == {0: 1}
    new = [x for x in c.conns.values() if x.inno_id not in (0, 1)]
    assert sorted(x.weight for x in new) == [1.0, 1.0]   # old w was 1.0
    assert all(x.enabled for x in new)


def test_2000_mutations_stay_valid():
    from neat.innovation import InnovationTracker
    from neat.mutation import mutate
    from neat.network import topo_order, compile_genome, run
    from neat.population import make_initial_population
    cfg = NeatConfig(num_inputs=2, num_outputs=1, pop_size=1, p_add_node=0.3, p_add_conn=0.3,
                     p_toggle=0.2, p_activation=0.2)
    tracker = InnovationTracker(3)
    rng = np.random.default_rng(0)
    g = make_initial_population(cfg, tracker, rng)[0]
    for _ in range(2000):
        parent = (dict(g.nodes), dict(g.conns), dict(g.conns_broken))
        g = mutate(g, cfg, tracker, rng)
        topo_order(g.nodes, g.conns.values())                    # raises on cycle
        assert all(c.src in g.nodes and c.dest in g.nodes for c in g.conns.values())
    assert all(g.nodes[c.dest].type != NodeType.INPUT for c in g.conns.values())
    assert len(g.nodes) > 3
    run(compile_genome(g), np.zeros((4, 2)))                      # still runnable


def test_mutate_leaves_parent_alone():
    from neat.innovation import InnovationTracker
    from neat.mutation import mutate
    cfg = NeatConfig(num_inputs=2, num_outputs=1, p_weight=1, p_add_conn=1, p_add_node=1,
                     p_activation=1, p_toggle=1)
    g = genome()
    before = (dict(g.nodes), dict(g.conns), dict(g.conns_broken))
    mutate(g, cfg, InnovationTracker(3), np.random.default_rng(0))
    assert (g.nodes, g.conns, g.conns_broken) == before
