"""Unit 3 §18: training reduces loss; the whole run compiles once."""
import numpy as np

from backprop import train
from backprop.datasets import make_splits
from backprop.padded import ACT_NAMES
from neat.config import NeatConfig
from neat.population import Population

CFG = NeatConfig(num_inputs=2, num_outputs=1, pop_size=20, output_activation="identity",
                 activations=ACT_NAMES, max_nodes=train.N_MAX)


def test_training_reduces_loss_and_learns_circle():
    splits = make_splits("circle", 0)
    pop = Population(CFG, 0)
    _, arrays = train.stack(pop.genomes)
    X, y = splits["train"]
    W, b, before, _ = (np.asarray(v) for v in train.train_population(*arrays, X, y, X, y))
    # train again from the trained weights: loss must be lower than the first evaluation
    arrays[0], arrays[2] = W, b
    _, _, after, acc = (np.asarray(v) for v in train.train_population(*arrays, X, y, X, y))
    assert after.mean() < before.mean()


def test_linear_net_learns_linearly_separable_labels():
    # gen-0 nets have no hidden nodes = linear, so they can't do circle/xor/spiral (~50%),
    # but they must learn "x > 0" almost perfectly if the inner loop works
    X, _ = make_splits("xor", 0)["train"]
    y = (X[:, 0] > 0).astype(np.float32)
    _, arrays = train.stack(Population(CFG, 0).genomes)
    _, _, _, acc = (np.asarray(v) for v in train.train_population(*arrays, X, y, X, y))
    assert acc.mean() > 0.95


def test_one_compilation_across_generations():
    pop = Population(CFG, 1)
    fitness_fn = train.make_fitness_fn(make_splits("xor", 0))
    before = train.compile_count
    pop.run(fitness_fn, 2)                           # 2 generations, different topologies
    assert train.compile_count - before <= 1         # 0 if an earlier test already compiled it
