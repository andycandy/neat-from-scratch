# Unit 2 — Progress log

A record of what has been built so far, how it was checked, and what is left.
Design decisions and their reasons are in [DECISIONS.md](DECISIONS.md).

**Status (2026-10-22):** the NEAT core is complete, and XOR solves in 10 of 10 seeds. Backprop-NEAT (Unit 3) is complete apart from figures and the write-up: test accuracy averages circle 0.955, XOR 0.998, spiral 0.923, with 5–18 connections, and padded + vmap training is 5.9×–7.8× faster than per-genome compilation at identical accuracy. Slime Volleyball has had three runs, none of which has learned to score yet; the latest diagnosis points to evaluation noise (see the Slime runs section and DECISIONS.md).

## Setup

- **Environment:** a uv project (`pyproject.toml`, `.venv`). Dependencies are numpy and matplotlib, plus pytest as a dev dependency.
- **Run the tests:** `uv run python -m pytest -q tests`
- **Run the XOR benchmark:** `uv run python experiments/xor.py`

## What has been built, in build order

| # | Module | What it does | Tests |
|---|---|---|---|
| 1 | `neat/config.py` | `NeatConfig`: a frozen dataclass holding every hyperparameter (population, mutation rates, distance coefficients, threshold, stagnation, elitism, output activation). | — |
| 2 | `neat/genes.py` | `NodeType`, plus the frozen `NodeGene` and `ConnGene`. `Genome` is mutable and holds `nodes`, `conns` (keyed by innovation number), `conns_broken` (how many times each connection has been split) and `fitness`. | — |
| 3 | `neat/innovation.py` | `InnovationTracker`. `conn_innovation(src, dst)` uses one global `(src, dst)` registry. `split_node_id(conn, genome)` is keyed by `(conn, times split)`, so repeated splits match across genomes, and it has a collision fallback. | 3 |
| 4 | `neat/population.py` | `make_initial_population`: minimal genomes, every input connected to every output, with innovation numbers from the tracker so they are shared by all genomes. | 1 |
| 5 | `neat/activations.py` | A name → NumPy function table: identity, sigmoid, tanh, relu, sin, gauss, abs, step. Sigmoid and gauss clip their input for numerical safety. | — |
| 6 | `neat/network.py` | `topo_order` (Kahn's algorithm, which raises on a cycle). `compile_genome` builds the evaluation order and per-node incoming lists once. `run` evaluates a batch `(batch, n_in)` → `(batch, n_out)`. | 3 |
| 7 | `neat/mutation.py` | `mutate_weights` (perturb or replace, then clip), `reachable` (DFS over all connections), `mutate_add_connection` (with a cycle check), `add_node` (split a connection, increment `conns_broken`), `mutate_activation`, `toggle_enable`, and `mutate`, which applies each one with its own probability. All of them return a new genome and never change their input. | 8 |
| 8 | `neat/crossover.py` | Matching genes come from a random parent. Disjoint and excess genes come from the fitter parent. A gene disabled in either parent stays disabled with probability 0.75. Nodes come from the fitter parent, with shared nodes from a random parent. | 5 |
| 9 | `neat/species.py` | `compatibility_distance` (excess, disjoint, mean weight difference). The `Species` dataclass. `speciate` picks new representatives from the closest genomes, assigns the rest to the first species under the threshold, and drops empty species. | 6 |
| 10 | `neat/reproduction.py` | `allocate` (largest-remainder child counts). `species_scores` (mean fitness shifted to be non-negative). `update_and_cull` (stagnation, never removing the best species). `breed_species` (elitism, top-30% parent pool, crossover or clone, then mutate, with a new id per child). `reproduce` ties them together, guarantees each species 2 children, and returns the surviving species. | 8 |
| 11 | `neat/population.py` | The `Population` class. `run` loops: evaluate → track the best → snapshot the champion → stop check → speciate → log stats → adjust the threshold → reproduce. `adjust_threshold` implements the dynamic threshold. `save_stats` writes a CSV. `save` and `load` checkpoint the whole run with pickle. | 5 |
| 12 | `experiments/xor.py` | The XOR benchmark: the fitness `(4 − Σ|y − ŷ|)²`, stopping at the first solved generation, 10 seeds, and a stats CSV per seed in `experiments/results/`. | — |

## Notable fixes and changes along the way

- **Split-node ids:** the guide keys a split only by the connection it splits. That was extended to `(connection, times split)`, so the second and later splits of the same connection also match across genomes. See decision 12.
- **Network:** first written as a `FeedForwardNetwork` class, then simplified to two plain functions with the same behaviour and tests. See decision 13.
- **Cycle check:** an early version of `reachable` followed only enabled connections. It now follows all of them, because a disabled connection can be re-enabled later.
- **Crossover parent choice:** `rng.choice(a, b)` treated `b` as a sample size. It was replaced with a coin flip.
- **Fitter-parent ordering:** a misplaced parenthesis in the tie-break swapped parents whenever their fitness was not tied. It is fixed.
- **Clone aliasing:** cloned children are now copied before mutation, so setting a child's id cannot change its parent.
- **Stagnation filter:** the filter was inverted and kept only the stagnant species. It is fixed, and the best species is now always protected.
- **Child counts:** `allocate` repeated the list instead of scaling it. It was rewritten as the largest-remainder method, with an even split when all shares are 0.
- **Distance:** fixed the weight difference (it compared a genome with itself), `N` (it counted nodes instead of connections) and the excess/disjoint set arithmetic.
- **Output activation:** it is now a config field (`sigmoid` for XOR) instead of a hard-coded `tanh`.
- **Statistics:** each row now records the current generation, best-ever and champion fitness, mean fitness, species count, the threshold, mean and champion hidden-node and connection counts, and the count of each hidden activation. `save_stats` writes the union of all columns, so a column that first appears late cannot break the CSV.
- **Species resurrection:** stagnant species were culled inside `reproduce`, but `Population` kept the old list, so the next speciation brought them back. Stagnation removal was silently ineffective. `reproduce` now returns the surviving species.
- **Worst species always extinct:** after the fitness shift, the worst species scored 0 and got no children, including newborn species. Every surviving species is now guaranteed 2 children.
- **Generations to solve:** this used to be the generation where fitness reached 15. It is now the first generation in which any genome is solved (`run(..., stop=solved)`).
- **Dynamic threshold and snapshots:** added for Slime Volleyball, with tests.
- **Speciation fix (found in the first Slime run):** the species count cycled 1 → 19 → 1. There were three causes: (1) dividing the distance by N = 36 made new structure nearly invisible; (2) the dynamic threshold had no cap (windup); (3) old species claimed their closest genome regardless of distance, so the threshold could never shrink the species count. The fixes are N = 1, `threshold_max`, and Stanley's "join the first compatible species, empty species die" rule. Afterwards the species count held at 10–12.
- **Slime runner:** a vendored environment, parallel evaluation, a hall of fame (min-heap), atomic checkpoints and resume. Fixed the built-in AI's RNN memory leaking between games.

## Test suite (49 tests)

| File | Covers |
|---|---|
| `test_innovation.py` | Same `(src, dst)` gives the same number. The same split in two genomes gives the same node. A second split gives a new node that matches across genomes. |
| `test_population.py` | Initial genomes share innovation numbers but have different weights. The same seed gives an identical run. Save and load preserve the rng state and the tracker. The threshold rises with too many species, falls with too few, and is clamped. A run with a low starting threshold pulls the species count down. Snapshots are written every generation, and the stats CSV has the new columns. The stop check finds a passing genome that is not the fittest. |
| `test_network.py` | A hand-computed output. Batch results equal row-by-row results. A disabled connection has no effect. |
| `test_mutation.py` | The parent is unchanged. Weights stay clipped. Mutation is deterministic for a seed. `reachable` follows disabled edges. 1,000 add-connection mutations never create a cycle. Add-node structure is correct. 2,000 chained mutations stay acyclic and runnable, and never target an input. |
| `test_crossover.py` | Self-crossover returns the same genome. The child's innovations equal the fitter parent's. Every referenced node exists. Disabled genes are re-enabled about 25% of the time. Parents are unchanged. |
| `test_species.py` | `δ(g, g) = 0`. Distance is symmetric. A hand-computed distance. Identical genomes form 1 species. Distant genomes form 2. Every genome is in exactly one species. A higher threshold merges existing species into one. |
| `test_slime.py` | The random policy scores about −5. The same genome and seeds give the same fitness, even after other games. Parallel equals serial. The survival bonus never outranks a point. Resume matches an uninterrupted run. The hall of fame keeps the top k, sorted, without duplicates. |
| `test_reproduction.py` | Largest-remainder examples. Counts always sum to the total, including when all shares are 0. The child count equals the population size. Negative fitness still favours the better species. The elite survives unchanged. A culled species is not returned. The worst species still gets its minimum children. |

## XOR result

- **Solved in 10 of 10 seeds** (0–9) within 300 generations.
- **Generations to solve** (the first generation in which any genome has all four outputs on the correct side of 0.5): 18.1 ± 9.4 (range 6–40).
- **Hidden nodes in the solutions:** 1–6.
- The paper reports about 32 generations, but the settings differ, so this is a sanity check, not a like-for-like comparison.

## Remaining checklist

**Unit 2**
- [x] `neat/visualize.py`: `draw_genome` (depth layout, colour by activation, edge colour by sign and width by |w|, disabled edges dashed). Every XOR run saves `experiments/results/xor_seed*.png`.
- [x] `ruff check` passes (vendored `slime/slimevolley.py` excluded).
- [x] Remove the leftover scratch file `neat/test.py`.
- [ ] Add XOR results to the write-up doc.
- [x] Write the reason for the distance coefficients in DECISIONS.md, decision 8 (tested on XOR).

**Unit 4 (Slime Volleyball)**
- [x] Vendored environment (no gym). A random policy scores −4.8 (README: −4.87).
- [x] Observation ranges measured. No extra scaling (the env already divides by 10).
- [x] Fitness with shaping, shared seeds per generation, hall of fame (min-heap).
- [x] Parallel evaluation. Checkpoints with atomic writes, and resume (tested identical to an uninterrupted run).
- [x] `experiments/watch.py`: searches fresh seeds and saves a GIF of an agent's best game, plus button-usage statistics.
- [x] Decide on and launch the next run: `slime5` (self-play + lockstep batching).
- [x] Final evaluation (`experiments/eval_slime.py`): best of top-5 hall of fame on 200 games, then 1,000 new games: −0.852 ± 1.260, win 11.8%, draw 31.0%, loss 57.2%.
- [x] Figures (`experiments/figures_slime.py` -> `figures/slime/`): training curves, network of the reported agent and of champions at generations 10/50/200/786, input usage, slime5 vs slime6. GIF: `runs/slime5/gen00786_rank0_seed10014.gif`.
- [ ] Write-ups: skeletons in `writeup/` (Part 1A, Part 1B); prose to write.

## Slime runs

| Run | Generations | Change | Outcome |
|---|---|---|---|
| `runs/slime1` | 0–170 | first version | Species count cycled 1 → 19 → 1. That led to the speciation fixes (Stanley's rule, N = 1, threshold cap). |
| `runs/slime2` | 0–871 | speciation fixed | Species stable at 9–13, but the population converged on fence-camping. Fitness standard deviation was 0.086 (no signal), and there were 90 hidden nodes (bloat). Hall of fame −4.55. |
| `runs/slime3` | 0–256 | fitness v2: per-rally survival bonus, returns over the net, node penalty | Agents became active (moving, jumping) and then plateaued: returns about 0.6 per game, hall of fame −4.60. Diagnosis: with 4 games per genome, rankings on two different 4-game sets barely agree (Spearman +0.13), so selection is close to random. |
| `runs/slime4` | 0–243 | `slime3` + 16 games per genome, node penalty 0.01 (noise ablation) | Hall of fame −4.95 → −4.55 by generation 68, then frozen until it was stopped; returns 0.5–0.8; bloat controlled at first, 16 hidden nodes by the end. Noise was only part of the problem. Next idea: self-play. |
| `runs/slime5` | 0–1670 (stopped) | 12 self-play (mirror) + 4 built-in AI games per genome; lockstep batching (4× faster) | Hall of fame −4.65 → −0.70 by generation 786, then flat for ~770 generations. 100 fresh games: −1.14, 18.8 returns per game, best game +2. Final, 1,000 new games: −0.852 ± 1.260, win rate 11.8%. Leaderboard self-play: CMA-ES −0.071, GA +0.353. |
| `runs/slime6` | running | `slime5` + the paper's distance coefficients (1.0, 1.0, 0.4), threshold range doubled | Overnight ablation (see DECISIONS.md). |
