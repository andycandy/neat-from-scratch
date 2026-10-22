"""Padded fixed-size genomes for JAX (Unit 3 §9–11, §13).

Every genome becomes arrays of the SAME shape, so one compiled function trains the
whole population with vmap. Positions follow a topological order, so W and M are
strictly upper triangular (i < j): no cycles, and not-yet-computed positions are 0.
"""
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from neat.genes import Genome, NodeType

# ONE shared ordered list: the activation id used in `pack` and in `forward` must match.
ACT_NAMES = ("identity", "sigmoid", "tanh", "relu", "sin", "gauss", "abs")
ACT_ID = {name: i for i, name in enumerate(ACT_NAMES)}


def all_activations(s: jnp.ndarray) -> jnp.ndarray:
    """Every activation applied to s, stacked on a new LAST axis: shape s.shape + (7,).
    Clipped input keeps every branch finite (an inf in an unselected branch can still
    leak NaN into gradients through the selection, §11)."""
    s = jnp.clip(s, -30.0, 30.0)
    return jnp.stack([s, 1 / (1 + jnp.exp(-s)), jnp.tanh(s), jnp.maximum(s, 0),
                      jnp.sin(s), jnp.exp(-s * s), jnp.abs(s)], axis=-1)


@dataclass
class Packed:
    """One genome as fixed-size arrays (stack many along a new axis 0 for vmap)."""
    W: np.ndarray        # (n_max, n_max) float: W[i, j] = weight from position i to position j
    M: np.ndarray        # (n_max, n_max) float: 1 where an ENABLED connection exists
    b: np.ndarray        # (n_max,) float: bias per position (0 for inputs and padding)
    act: np.ndarray      # (n_max,) int: activation id per position (identity for padding)
    out_idx: int         # position of the output node
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
      5. out_idx = pos[the OUTPUT node's id]
    Pitfall: W[dst, src] in one place and W[src, dst] in another -> silently wrong network.
    """
    raise NotImplementedError


def forward(W, M, b, act, out_idx, x, num_inputs: int = 2):
    """Padded forward pass for ONE genome on a batch. x: (batch, num_inputs) -> (batch,) logits.

    HINTS (§10):
      a = zeros((batch, n_max));  a = a.at[:, :num_inputs].set(x)
      WM = W * M
      for j in range(num_inputs, n_max):           # becomes jax.lax.fori_loop(num_inputs, n_max, body, a)
          s = b[j] + a @ WM[:, j]                  # (batch,)
          a = a.at[:, j].set(all_activations(s)[:, act[j]])   # pick this node's activation
      return a[:, out_idx]
    Why the full dot product is safe: WM is upper triangular, and positions >= j are still 0.
    """
    raise NotImplementedError


def unpack(genome: Genome, packed: Packed, W_trained, b_trained) -> Genome:
    """Write trained weights back into a NEW genome (Lamarckian inheritance, §7).

    HINTS: for each ENABLED conn: dataclasses.replace(conn, weight=float(W_trained[pos[src], pos[dest]]))
           for each non-input node: dataclasses.replace(node, bias=float(b_trained[pos[n]]))
           disabled conns keep their old weights. Return a new Genome (new dicts).
    """
    raise NotImplementedError
