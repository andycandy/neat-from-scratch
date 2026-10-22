from collections import deque

import numpy as np

from neat.activations import ACTIVATIONS
from neat.genes import Genome, NodeType


def topo_order(node_ids, conns):
    indeg = {n: 0 for n in node_ids}
    out = {n: [] for n in node_ids}
    for c in conns:
        if c.enabled:
            indeg[c.dest] += 1
            out[c.src].append(c.dest)
    queue = deque(n for n, d in indeg.items() if d == 0)
    order = []
    while queue:
        n = queue.popleft()
        order.append(n)
        for m in out[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                queue.append(m)
    if len(order) != len(indeg):
        raise ValueError("cycle")
    return order

def compile_genome(genome: Genome):
    """Do the graph work once. Returns (num_inputs, output_ids, steps)."""
    enabled = [c for c in genome.conns.values() if c.enabled]
    order = topo_order(genome.nodes, enabled)

    incoming = {n: [] for n in genome.nodes}      # node id -> [(src, weight), ...]
    for c in enabled:
        incoming[c.dest].append((c.src, c.weight))

    steps = []                                    # non-input nodes, in eval order
    for n in order:
        node = genome.nodes[n]
        if node.type != NodeType.INPUT:
            steps.append((n, ACTIVATIONS[node.activation], node.bias, incoming[n]))

    num_inputs = sum(g.type == NodeType.INPUT for g in genome.nodes.values())
    output_ids = sorted(n for n, g in genome.nodes.items() if g.type == NodeType.OUTPUT)
    return num_inputs, output_ids, steps


def run(net, x: np.ndarray) -> np.ndarray:
    """x: (batch, num_inputs) -> (batch, num_outputs)."""
    num_inputs, output_ids, steps = net
    x = np.asarray(x, dtype=float)
    values = {i: x[:, i] for i in range(num_inputs)}   # node id -> array over batch
    for n, act, bias, srcs in steps:
        total = np.full(x.shape[0], bias)
        for src, w in srcs:
            total += w * values[src]
        values[n] = act(total)
    return np.stack([values[o] for o in output_ids], axis=1)
