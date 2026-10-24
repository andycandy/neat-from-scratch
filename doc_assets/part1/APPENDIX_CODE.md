# Part 1 code appendix: files

## NEAT core (`neat/`)
- `neat/config.py`: `NeatConfig`, every hyperparameter in one frozen dataclass.
- `neat/genes.py`: node and connection genes (immutable) and the `Genome` container.
- `neat/innovation.py`: global innovation registry; split-node ids keyed by (connection, times split).
- `neat/activations.py`: name → NumPy activation function table.
- `neat/network.py`: Kahn topological sort, compile-once genome → feed-forward evaluator, batched `run`.
- `neat/mutation.py`: weight/bias, add-connection (with cycle check), add-node, activation and enable mutations.
- `neat/crossover.py`: crossover aligned by innovation number; disabled genes stay disabled 75% of the time.
- `neat/species.py`: compatibility distance, Stanley-rule speciation, adaptive clamped threshold.
- `neat/reproduction.py`: fitness sharing, stagnation, largest-remainder child counts with a 2-child floor, elitism, breeding.
- `neat/population.py`: initial population and the generation loop with per-generation statistics.
- `neat/visualize.py`: network drawing (depth layout, colour by activation, edge colour/width by weight).

## XOR
- `experiments/xor.py`: XOR benchmark over 10 seeds (fitness, "solved" test, per-seed CSV and drawing).
- `experiments/sweep_xor.py`: one-at-a-time mutation-rate sweep, 20 seeds per value.

## Backprop-NEAT (`backprop/`)
- `backprop/datasets.py`: exact port of backprop-neat-js circle/XOR/spiral generators, plus a validation split.
- `backprop/padded.py`: pack a genome into fixed 32-node arrays, JAX forward pass (`fori_loop`), unpack trained weights.
- `backprop/train.py`: `jit(vmap(train_one))` population training (Adam, 200 steps), Lamarckian write-back, fitness.
- `experiments/run_backprop.py`: datasets × seeds experiment runner; saves best genome, stats, final population.
- `experiments/bench_backprop.py`: per-genome jit vs padded + vmap benchmark.
- `experiments/figures_backprop.py`: decision-boundary + architecture figures, complexification, accuracy vs connections.

## Slime Volleyball
- `slime/slimevolley.py`: vendored slimevolleygym game with `gym` removed (physics, rules and built-in AI unchanged).
- `experiments/run_slime.py`: training runner: fitness shaping, self-play + AI games in lockstep, parallel evaluation, hall of fame, checkpoints.
- `experiments/eval_slime.py`: final evaluation (select on 200 games, report on 1,000 different games).
- `experiments/watch.py`: plays an agent on fresh seeds, reports button usage, saves a GIF of its best game.
- `experiments/figures_slime.py`: training curves, network drawings, input usage, ablation comparison.

## Write-up support
- `experiments/stats_part1.py`: permutation test and bootstrap CIs quoted in the write-up.
- `experiments/figures_doc_part1.py`: all figures in this section, sized for half-page width.

## Tests (`tests/`, 63 tests)
- `tests/test_innovation.py`: shared innovation numbers and split-node ids.
- `tests/test_population.py`: initial population, determinism, threshold control, stop check, stats.
- `tests/test_network.py`: hand-computed output, batch = row-by-row, disabled connections ignored.
- `tests/test_mutation.py`: parent unchanged, clipping, acyclicity over 1,000–2,000 mutations.
- `tests/test_crossover.py`: alignment, node consistency, disabled-gene rule, parents unchanged.
- `tests/test_species.py`: distance properties, speciation, threshold merging.
- `tests/test_reproduction.py`: child counts, negative fitness, elitism, culling, minimum children.
- `tests/test_slime.py`: determinism, parallel = serial, resume = uninterrupted, lockstep = one-by-one, hall of fame.
- `tests/test_backprop_datasets.py`, `tests/test_backprop_padded.py`, `tests/test_backprop_train.py`: dataset port, padded forward = NumPy network, inert padding, zero masked gradients, single compile.
