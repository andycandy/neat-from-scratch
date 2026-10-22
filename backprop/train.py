"""Inner loop of Backprop-NEAT (Unit 3 §4, §6, §7, §12).

fitness_fn(genomes):
    pack every genome  ->  stack into (P, ...) arrays  ->  ONE jitted, vmapped call trains all P
    ->  write trained weights back (Lamarckian)  ->  fitness = val accuracy - lambda * connections
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax

from backprop.padded import forward, pack, unpack
from neat.genes import Genome

N_MAX = 32
STEPS = 200          # gradient steps per genome per generation
LR = 0.02            # Adam learning rate
CLIP = 1.0           # global gradient-norm clip (cheap NaN protection)
LAMBDA = 0.002       # fitness penalty per enabled connection

compile_count = 0    # incremented only while JAX traces (= compiles) train_one


def train_one(W, M, b, act, out_idx, X_tr, y_tr, X_val, y_val):
    """Train ONE padded genome; return (trained W, trained b, val loss, val accuracy).
    Parameters are W and b; M, act, out_idx come from evolution and stay fixed."""
    global compile_count
    compile_count += 1                                   # Python side effect: runs at trace time only

    def loss_fn(params, X, y):
        logits = forward(params[0], M, params[1], act, out_idx, X)[:, 0]
        return optax.sigmoid_binary_cross_entropy(logits, y).mean(), logits

    opt = optax.chain(optax.clip_by_global_norm(CLIP), optax.adam(LR))
    params = (W, b)
    state = opt.init(params)

    def step(carry, _):                                  # one gradient step
        params, state = carry
        grads = jax.grad(lambda p: loss_fn(p, X_tr, y_tr)[0])(params)
        updates, state = opt.update(grads, state)
        return (optax.apply_updates(params, updates), state), None

    (params, _), _ = jax.lax.scan(step, (params, state), None, length=STEPS)
    val_loss, logits = loss_fn(params, X_val, y_val)
    val_acc = ((logits > 0) == (y_val > 0.5)).mean()
    return params[0], params[1], val_loss, val_acc


# vmap: arrays with a leading P axis are split per genome (0); the data is shared (None).
train_population = jax.jit(jax.vmap(train_one, in_axes=(0, 0, 0, 0, 0, None, None, None, None)))


def stack(genomes: list[Genome]):
    """Pack each genome, then stack the arrays along a new leading axis of size P."""
    packs = [pack(g, N_MAX) for g in genomes]
    arrays = [np.stack([getattr(p, f) for p in packs]) for f in ("W", "M", "b", "act", "out_idx")]
    return packs, arrays


def make_fitness_fn(splits: dict):
    """splits = backprop.datasets.make_splits(...). Returns fitness_fn for Population.run."""
    X_tr, y_tr = (jnp.asarray(a) for a in splits["train"])
    X_val, y_val = (jnp.asarray(a) for a in splits["val"])

    def fitness_fn(genomes: list[Genome]) -> np.ndarray:
        packs, (W, M, b, act, out_idx) = stack(genomes)
        W_t, b_t, val_loss, val_acc = train_population(W, M, b, act, out_idx, X_tr, y_tr, X_val, y_val)
        W_t, b_t = np.asarray(W_t), np.asarray(b_t)
        for i, g in enumerate(genomes):                  # Lamarckian: children inherit trained weights
            trained = unpack(g, packs[i], W_t[i], b_t[i])
            g.conns, g.nodes = trained.conns, trained.nodes
        conns = np.array([sum(c.enabled for c in g.conns.values()) for g in genomes])
        fitness_fn.val_acc = np.asarray(val_acc)         # kept for logging / picking the best
        return np.asarray(val_acc) - LAMBDA * conns

    return fitness_fn


def accuracy(genome: Genome, X, y) -> float:
    """Accuracy of an (already trained) genome on any split, e.g. the test set."""
    p = pack(genome, N_MAX)
    logits = np.asarray(forward(p.W, p.M, p.b, p.act, p.out_idx, X))[:, 0]
    return float(((logits > 0) == (y > 0.5)).mean())
