"""Engineering improvement (Unit 3 §17): naive per-genome jit vs padded + vmap.

Both train the SAME genomes from the SAME starting weights on the SAME data with the same
optimizer and step count, so the only difference is how the computation is compiled.

uv run python experiments/bench_backprop.py
"""
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_backprop import make_config

from backprop import train
from backprop.datasets import make_splits
from neat.genes import Genome, NodeType
from neat.innovation import InnovationTracker
from neat.mutation import mutate
from neat.network import topo_order
from neat.population import make_initial_population

N_GENOMES = 20
ACT = {"identity": lambda s: s, "sigmoid": jax.nn.sigmoid, "tanh": jnp.tanh,
       "relu": lambda s: jnp.maximum(s, 0), "sin": jnp.sin, "gauss": lambda s: jnp.exp(-s * s),
       "abs": jnp.abs}
naive_compiles = 0


def sample_genomes(n: int) -> list[Genome]:
    """Varied topologies: random mutation chains from minimal genomes (as in the tests)."""
    cfg = make_config()
    cfg = type(cfg)(**{**cfg.__dict__, "p_add_node": 0.3, "p_add_conn": 0.3, "pop_size": 1})
    tracker, rng = InnovationTracker(3), np.random.default_rng(0)
    out = []
    for _ in range(n):
        g = make_initial_population(cfg, tracker, rng)[0]
        for _ in range(int(rng.integers(5, 40))):
            g = mutate(g, cfg, tracker, rng)
        out.append(g)
    return out


def naive_train(genome: Genome, X_tr, y_tr, X_val, y_val):
    """Baseline: the genome's graph is baked into Python code, so jit compiles per genome."""
    conns = [c for c in genome.conns.values() if c.enabled]
    order = [n for n in topo_order(genome.nodes, conns) if genome.nodes[n].type != NodeType.INPUT]
    inputs = sorted(n for n, g in genome.nodes.items() if g.type == NodeType.INPUT)
    out_id = next(n for n, g in genome.nodes.items() if g.type == NodeType.OUTPUT)
    params = (jnp.array([c.weight for c in conns], jnp.float32),
              jnp.array([genome.nodes[n].bias for n in order], jnp.float32))

    def forward(params, X):
        w, b = params
        vals = {n: X[:, i] for i, n in enumerate(inputs)}
        for k, n in enumerate(order):                     # Python loop: unrolled at trace time
            s = b[k] + sum((w[i] * vals[c.src] for i, c in enumerate(conns) if c.dest == n),
                           jnp.zeros(X.shape[0]))
            vals[n] = ACT[genome.nodes[n].activation](s)
        return vals[out_id]

    def loss(params, X, y):
        return optax.sigmoid_binary_cross_entropy(forward(params, X), y).mean()

    @jax.jit                                              # a NEW function per genome -> new compile
    def fit(params):
        global naive_compiles
        naive_compiles += 1
        opt = optax.chain(optax.clip_by_global_norm(train.CLIP), optax.adam(train.LR))

        def step(carry, _):
            p, s = carry
            u, s = opt.update(jax.grad(loss)(p, X_tr, y_tr), s)
            return (optax.apply_updates(p, u), s), None

        (p, _), _ = jax.lax.scan(step, (params, opt.init(params)), None, length=train.STEPS)
        return ((forward(p, X_val) > 0) == (y_val > 0.5)).mean()

    return float(fit(params))


def main() -> None:
    splits = make_splits("circle", 0)
    X_tr, y_tr = (jnp.asarray(a) for a in splits["train"])
    X_val, y_val = (jnp.asarray(a) for a in splits["val"])
    genomes = sample_genomes(N_GENOMES)
    sizes = [len(g.nodes) for g in genomes]

    t = time.perf_counter()
    naive_acc = np.array([naive_train(g, X_tr, y_tr, X_val, y_val) for g in genomes])
    naive_s = time.perf_counter() - t

    before = train.compile_count
    _, arrays = train.stack(genomes)
    t = time.perf_counter()
    out = train.train_population(*arrays, X_tr, y_tr, X_val, y_val)
    jax.block_until_ready(out)
    padded_first_s = time.perf_counter() - t                       # includes the one compile
    t = time.perf_counter()
    out = train.train_population(*arrays, X_tr, y_tr, X_val, y_val)
    jax.block_until_ready(out)
    padded_warm_s = time.perf_counter() - t                        # already compiled
    padded_acc = np.asarray(out[3])

    result = {
        "genomes": N_GENOMES, "nodes_min_max": [min(sizes), max(sizes)], "steps": train.STEPS,
        "naive": {"seconds": round(naive_s, 2), "genomes_per_s": round(N_GENOMES / naive_s, 2),
                  "compiles": naive_compiles},
        "padded_vmap_first_call": {"seconds": round(padded_first_s, 2),
                                   "genomes_per_s": round(N_GENOMES / padded_first_s, 2),
                                   "compiles": train.compile_count - before},
        "padded_vmap_compiled": {"seconds": round(padded_warm_s, 2),
                                 "genomes_per_s": round(N_GENOMES / padded_warm_s, 2)},
        "speedup_first_call": round(naive_s / padded_first_s, 1),
        "speedup_compiled": round(naive_s / padded_warm_s, 1),
        "max_val_acc_difference": float(np.abs(naive_acc - padded_acc).max()),
        "val_acc_naive": naive_acc.round(3).tolist(),
        "val_acc_padded": padded_acc.round(3).tolist(),
    }
    print(json.dumps(result, indent=2))
    Path("runs").mkdir(exist_ok=True)
    Path("runs/backprop_bench.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
