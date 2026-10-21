# Unit 2 — Design decisions log

The judgement calls made while building the NEAT core, with the reason for each.
Section numbers refer to the Unit 2 build guide.

## Algorithm decisions

1. **Bias: per-node bias** (not a bias input node). Simpler to implement, and it mutates exactly like a weight. The original paper used a bias input. (§4.1)
2. **Innovation numbers: one global registry for the whole run**, keyed by `(src, dst)`. "Same structure ⇒ same number" holds across all genomes and generations by construction. The paper reuses numbers only within a generation. (§5.2)
3. **Output nodes cannot be connection sources.** Sources are input or hidden nodes; targets are hidden or output nodes. This is simpler, and outputs never feed back into the network. (§11.1)
4. **Add-connection on an existing disabled pair re-adds it enabled**, with the same innovation number and a fresh random weight. Only enabled duplicates are rejected. (§11.2)
5. **New hidden nodes get a fixed `tanh` activation and bias 0.** This disturbs the network less than a random activation. Only hidden nodes change activation through mutation. Inputs have no activation (`None`), and outputs keep `cfg.output_activation`. (§12, §13)
6. **Equal-fitness crossover takes one random parent's structure.** This departs from the paper, which takes the union of both parents' disjoint and excess genes. The union can combine `4→5` from one parent with `5→4` from the other and create a cycle. (§14.2)
7. **`N` in the distance formula** is the larger genome's connection count, or 1 when both genomes have fewer than `n_small_cutoff = 20` connections (the paper's rule). Disabled genes are counted. (§15)
8. **Distance coefficients** `c1 = 0.5, c2 = 0.5, c3 = 0.3`, instead of the paper's 1.0 / 1.0 / 0.4. Reason: **TODO, write your reason here.** **The species threshold is dynamic:** it starts at `δ_t = 3.0`. After each speciation it rises by 0.1 if there are more than 12 species and falls by 0.1 (to a minimum of 0.3) if there are fewer than 8. A fixed threshold can give 1 species or dozens, depending on the task and input count. (§16.1)
9. **Negative fitness: shift by the population minimum** before computing species shares, so the worst genome scores 0 and the ranking is preserved. (§17.1)
10. **No interspecies crossover.** The paper's 0.001 rate is dropped. (§17.4)
11. **Hyperparameters used for XOR:** see the table below.
12. **Hidden-node ids are keyed by `(split connection, number of prior splits of that connection in this genome)`.** Each genome counts its splits in `conns_broken`. The guide keys only by the split connection, so a second split of the same connection gets unshared ids in different genomes. With the count in the key, the k-th split of a connection gets the same node id in every genome, and repeated structure still lines up in crossover and distance. `conns_broken` is inherited from the fitter parent in crossover. If the registry's node id is already present in the genome (it should not happen with the count), a fresh, unshared id is allocated as a safety net. (§5.3, §5.4)

## Other choices

- **Stagnation is measured on best-ever fitness** per species (the paper's rule). neat-python uses mean fitness instead. A species is removed after `max_stagnation = 15` generations without improvement. The species that holds the best genome is never removed.
- **Child counts:** every surviving species is guaranteed `min_species_size = 2` children. The rest are shared in proportion to each species' mean shifted fitness (fitness sharing) and rounded with the largest-remainder method, so the total is exactly the population size. Without the minimum, the shift gives the worst species a score of 0, so it, and any newborn species, would go extinct at once.
- **Extinction:** species that are culled for stagnation or get no children are removed from the population's species list. They are not carried into the next speciation.
- **Elitism:** a species with more than 5 members keeps its best genome unchanged (with a new id).
- **Parent pool:** the top 30% of each species, at least 2 genomes.
- **Weights and biases** start from N(0, 1), clipped to ±5. Input-node biases are never mutated.

- **XOR stopping rule:** a run stops at the first generation in which any genome is solved (all four outputs on the correct side of 0.5). "Generations to solve" is that generation number. Every genome is checked, not only the highest-fitness one, because a genome can have the top fitness with one output on the wrong side while a lower-fitness genome solves the task.

## Implementation notes

13. **The network is two plain functions, not a class.** `compile_genome(genome)` does the graph work once (Kahn's topological sort, per-node incoming lists). `run(net, x)` evaluates a batch of shape `(batch, num_inputs)`. The behaviour is the same as the guide's `FeedForwardNetwork`, with less code.

- **Immutable genes:** `NodeGene` and `ConnGene` are frozen dataclasses, changed only through `dataclasses.replace`. `Genome` is mutable so its fitness can be set. Every mutation and crossover builds new dicts, so children never share containers with parents.
- **Cycle check follows all connections**, enabled or disabled. The graph of all connections is therefore always acyclic, which makes enable-toggling safe.
- **Randomness:** one `np.random.Generator` per run, passed down explicitly. A test confirms that the same seed gives an identical run.
- **Checkpointing:** `Population.save` and `Population.load` pickle the whole population, including the tracker, species, rng state and generation counter.
- **Best-genome snapshots:** `run(..., snapshot_dir=...)` pickles each generation's champion to `gen_XXXX.pkl`, so the networks at generations 10, 50, 200 and so on are kept for the complexification figures.
- **Statistics per generation:** best-ever and champion fitness, mean fitness, species count, current threshold, mean hidden nodes and enabled connections, the champion's hidden nodes and enabled connections, and the count of each hidden-node activation across the population.

## XOR hyperparameters (all `NeatConfig` defaults)

| Parameter | Value |
|---|---|
| Population size | 150 |
| Weight limit `w_max` | 5.0 |
| Perturb probability / σ | 0.9 / 0.5 |
| Weight mutation rate | 0.8 |
| Add-connection rate | 0.1 |
| Add-node rate | 0.05 |
| Activation-change rate | 0.05 |
| Enable-toggle rate | 0.01 |
| Allowed hidden activations | tanh, sigmoid, sin, gauss, relu, abs |
| Output activation | sigmoid |
| Distance coefficients `c1, c2, c3` | 0.5, 0.5, 0.3 |
| Max stagnation | 15 |
| Survival threshold | 0.3 |
| Crossover rate | 0.75 |
| Elitism species size | > 5 |
| Minimum children per species | 2 |
| Dynamic threshold | start 3.0, target 8–12 species, step 0.1, minimum 0.3 |

## XOR results

The fitness is `(4 − Σ|y − ŷ|)²`, with a maximum of 16. A run stops at the first generation in which any genome is solved: all four outputs on the correct side of 0.5.

- **Solved in 10 of 10 seeds** (seeds 0–9) within 300 generations.
- **Generations to solve** (the first generation in which any genome is solved): mean 15.3 ± 9.5 (range 6–34).
- **Hidden nodes in the solutions:** 1–4.
- The first solutions are often borderline: some outputs are close to 0.5, with fitness between 7 and 14. Running longer sharpens them.
- **Comparison with the paper (about 32 generations):** the settings differ (per-node bias, different distance coefficients and mutation rates, a dynamic threshold, a different success definition), so this is a sanity check, not a like-for-like comparison.
- Per-generation statistics are in `experiments/results/xor_seed*.csv`.

Reproduce with `uv run python experiments/xor.py`.
