"""Unit 3 §18 equivalence tests for backprop/padded.py. They fail until pack/forward/unpack exist."""

import numpy as np
import pytest

from neat.config import NeatConfig
from neat.genes import ConnGene, Genome, NodeGene, NodeType
from neat.innovation import InnovationTracker
from neat.mutation import mutate
from neat.network import compile_genome, run
from neat.population import make_initial_population

padded = pytest.importorskip("backprop.padded")

CFG = NeatConfig(num_inputs=2, num_outputs=1, pop_size=1, output_activation="identity",
                 activations=padded.ACT_NAMES, p_add_node=0.3, p_add_conn=0.3, p_toggle=0.1,
                 p_activation=0.2, max_nodes=32)


def _fwd(g, x, n_max=32):
    p = padded.pack(g, n_max)
    return np.asarray(padded.forward(p.W, p.M, p.b, p.act, p.out_idx, x))


def unit1_genome():
    nodes = {0: NodeGene(0, NodeType.INPUT, None, 0.0), 1: NodeGene(1, NodeType.INPUT, None, 0.0),
             2: NodeGene(2, NodeType.OUTPUT, "sigmoid", 0.0), 3: NodeGene(3, NodeType.HIDDEN, "tanh", 0.0)}
    conns = {0: ConnGene(0, 0, 3, 0.5, True), 1: ConnGene(1, 1, 3, -1.0, True),
             2: ConnGene(2, 0, 2, 1.0, True), 3: ConnGene(3, 3, 2, 2.0, True)}
    return Genome(0, nodes, conns, {})


def random_genomes(n=100):
    tracker, rng = InnovationTracker(3), np.random.default_rng(0)
    gs = []
    for k in range(n):
        g = make_initial_population(CFG, tracker, rng)[0]
        for _ in range(int(rng.integers(1, 40))):
            g = mutate(g, CFG, tracker, rng)
        gs.append(g)
    return gs


def test_unit1_example_0731():
    out = _fwd(unit1_genome(), np.array([[1.0, 0.5]], dtype=np.float32), n_max=5)
    assert out.shape == (1, 1) and abs(out[0, 0] - 0.731) < 1e-3


def test_matches_numpy_on_100_random_genomes():
    x = np.random.default_rng(1).uniform(-5, 5, (50, 2)).astype(np.float32)
    for g in random_genomes():
        want = run(compile_genome(g), x)
        assert np.allclose(_fwd(g, x), want, atol=1e-4), "padded != numpy (transposed W? wrong order?)"


def test_padding_is_inert():
    x = np.random.default_rng(2).uniform(-5, 5, (20, 2)).astype(np.float32)
    for g in random_genomes(20):
        if len(g.nodes) <= 16:
            assert np.allclose(_fwd(g, x, 16), _fwd(g, x, 32), atol=1e-5)


def test_pack_unpack_roundtrip():
    for g in random_genomes(20):
        p = padded.pack(g, 32)
        back = padded.unpack(g, p, p.W, p.b)
        for k in g.conns:                                    # float32 arrays: equal to ~7 digits
            assert np.isclose(back.conns[k].weight, g.conns[k].weight, rtol=1e-6, atol=1e-7)
        for k in g.nodes:
            assert np.isclose(back.nodes[k].bias, g.nodes[k].bias, rtol=1e-6, atol=1e-7)


def test_masked_entries_get_zero_gradient():
    import jax
    import jax.numpy as jnp
    p = padded.pack(random_genomes(1)[0], 32)
    x = jnp.ones((4, 2))
    grad = jax.grad(lambda W: padded.forward(W, p.M, p.b, p.act, p.out_idx, x).sum())(jnp.asarray(p.W))
    assert float(jnp.abs(grad * (1 - p.M)).max()) == 0.0
