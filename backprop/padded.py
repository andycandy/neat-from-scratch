"""Padded fixed-size genomes for JAX (Unit 3 §9–11, §13).

Every genome becomes arrays of the SAME shape, so one compiled function trains the
whole population with vmap. Positions follow a topological order, so W and M are
strictly upper triangular (i < j): no cycles, and not-yet-computed positions are 0.
"""
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from neat.genes import Genome, NodeType
from neat.network import topo_order

# ONE shared ordered list: the activation id used in `pack` and in `forward` must match.
ACT_NAMES = ("identity", "sigmoid", "tanh", "relu", "sin", "gauss", "abs")
ACT_ID = {name: i for i, name in enumerate(ACT_NAMES)}


def all_activations(s: jnp.ndarray) -> jnp.ndarray:
    """Every activation applied to s, stacked on a new LAST axis: shape s.shape + (7,).
    All branches stay finite for finite s (jax.nn.sigmoid is the overflow-safe sigmoid), so an
    unselected branch can't leak NaN into gradients (§11). No clipping: it would change
    identity/relu/abs/sin for |s| > clip and break equality with the NumPy network."""
    return jnp.stack([s, jax.nn.sigmoid(s), jnp.tanh(s), jnp.maximum(s, 0),
                      jnp.sin(s), jnp.exp(-s * s), jnp.abs(s)], axis=-1)


@dataclass
class Packed:
    """One genome as fixed-size arrays (stack many along a new axis 0 for vmap)."""
    W: np.ndarray        # (n_max, n_max) float: W[i, j] = weight from position i to position j
    M: np.ndarray        # (n_max, n_max) float: 1 where an ENABLED connection exists
    b: np.ndarray        # (n_max,) float: bias per position (0 for inputs and padding)
    act: np.ndarray      # (n_max,) int: activation id per position (identity for padding)
    out_idx: np.ndarray  # (n_outputs,) int: positions of the output nodes, in output-id order
    pos: dict[int, int]  # node id -> position (kept for unpack)


def pack(genome: Genome, n_max: int) -> Packed:
    """Genome -> fixed-size arrays.

    HINTS (§13):
      1. order = topo_order(genome.nodes, enabled conns)   (from neat.network)
         BUT inputs must sit at positions 0..num_inputs-1 in id order (x -> 0, y -> 1):
         put sorted input ids first, then the non-input nodes in topological order.
      2. pos = {node_id: position}
      3. for each ENABLED conn: W[pos[src], pos[dest]] = weight;  M[...] = 1.0
         (src's position is always lower than dest's -> upper triangular)
      4. b[pos[n]] = node.bias, act[pos[n]] = ACT_ID[node.activation] for non-input nodes;
         inputs and padding: bias 0, ACT_ID["identity"]
      5. out_idx = positions of the OUTPUT nodes (sorted by id), as an int array
    Pitfall: W[dst, src] in one place and W[src, dst] in another -> silently wrong network.
    """
    inputs = sorted(n for n, g in genome.nodes.items() if g.type == NodeType.INPUT)
    rest = [n for n in topo_order(genome.nodes, genome.conns.values()) if n not in inputs]
    order = inputs + rest                         # x -> 0, y -> 1, then topological order
    if len(order) > n_max:
        raise ValueError(f"genome has {len(order)} nodes > n_max={n_max}")
    pos = {node_id: i for i, node_id in enumerate(order)}

    W = np.zeros((n_max, n_max), dtype=np.float32)
    M = np.zeros((n_max, n_max), dtype=np.float32)
    for c in genome.conns.values():
        W[pos[c.src], pos[c.dest]] = c.weight * c.enabled
        M[pos[c.src], pos[c.dest]] = c.enabled

    b = np.zeros(n_max, dtype=np.float32)
    act = np.full(n_max, ACT_ID["identity"], dtype=np.int32)   # inputs + padding: identity, bias 0
    for n in rest:
        node = genome.nodes[n]
        b[pos[n]] = node.bias
        act[pos[n]] = ACT_ID[node.activation]

    outputs = sorted(n for n, g in genome.nodes.items() if g.type == NodeType.OUTPUT)
    out_idx = np.array([pos[n] for n in outputs], dtype=np.int32)   # one lookup per output
    return Packed(W, M, b, act, out_idx, pos)


def forward(W, M, b, act, out_idx, x, num_inputs: int = 2):
    """Padded forward pass for ONE genome on a batch. x: (batch, num_inputs) -> (batch, n_outputs),
    the same contract as neat.network.run.

    HINTS (§10):
      a = zeros((batch, n_max));  a = a.at[:, :num_inputs].set(x)
      WM = W * M
      for j in range(num_inputs, n_max):           # becomes jax.lax.fori_loop(num_inputs, n_max, body, a)
          s = b[j] + a @ WM[:, j]                  # (batch,)
          a = a.at[:, j].set(all_activations(s)[:, act[j]])   # pick this node's activation
      return a[:, out_idx]
    Why the full dot product is safe: WM is upper triangular, and positions >= j are still 0.
    """
    x, W, M, b, act = (jnp.asarray(v) for v in (x, W, M, b, act))   # numpy can't be indexed by a traced j
    n_max = W.shape[0]
    a = jnp.zeros((len(x), n_max))
    a = a.at[:, :num_inputs].set(x)
    WM = W * M
    def body(j, a):
        s = b[j] + a @ WM[:, j]
        return a.at[:, j].set(all_activations(s)[:, act[j]])
    a = jax.lax.fori_loop(num_inputs, n_max, body, a)
    return a[:, out_idx]


def unpack(genome: Genome, packed: Packed, W_trained, b_trained) -> Genome:
    """Write trained weights back into a NEW genome (Lamarckian inheritance, §7).

    HINTS: for each ENABLED conn: dataclasses.replace(conn, weight=float(W_trained[pos[src], pos[dest]]))
           for each non-input node: dataclasses.replace(node, bias=float(b_trained[pos[n]]))
           disabled conns keep their old weights. Return a new Genome (new dicts).
    """
    pos = packed.pos
    conns = {k: replace(c, weight=float(W_trained[pos[c.src], pos[c.dest]])) if c.enabled else c
             for k, c in genome.conns.items()}
    nodes = {k: n if n.type == NodeType.INPUT else replace(n, bias=float(b_trained[pos[k]]))
             for k, n in genome.nodes.items()}
    return Genome(genome.id, nodes, conns, dict(genome.conns_broken))
