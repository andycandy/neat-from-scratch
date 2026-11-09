# Lab notebook

Design decisions, run-by-run notes and bugs found along the way. The [report](report.md) has the results. This file has the reasons behind them.

Raw outputs cited here are in [`results/logs/`](../results/logs/README.md).

**Contents**
1. [NEAT core](#1-neat-core)
2. [XOR experiments](#2-xor-experiments)
3. [Backprop-NEAT](#3-backprop-neat)
4. [Slime Volleyball](#4-slime-volleyball)
5. [Bugs found and fixed](#5-bugs-found-and-fixed)
6. [Test suite](#6-test-suite)

---

## 1. NEAT core

### Algorithm decisions

1. **Per-node bias**, not a bias input node. It is simpler and mutates exactly like a weight. The paper used a bias input.
2. **One global innovation registry for the whole run**, keyed by `(src, dst)`. The same structure gets the same number in every genome and generation. The paper reuses numbers only within a generation.
3. **Outputs are never connection sources.** Sources are inputs or hidden nodes. Targets are hidden or output nodes.
4. **Add-connection on an existing disabled pair re-enables it**, with the same innovation number and a fresh random weight. Only enabled duplicates are rejected.
5. **New hidden nodes start as `tanh` with bias 0.** This disturbs the network less than a random activation. Only hidden nodes change activation. Inputs have none. Outputs keep `cfg.output_activation`.
6. **Equal-fitness crossover takes one random parent's structure.** The paper takes the union of both parents' extra genes. That union can combine `4→5` from one parent with `5→4` from the other and make a cycle.
7. **`N` in the distance formula** is the larger genome's connection count, or 1 when both have fewer than 20 connections (the paper's rule). Disabled genes count.
8. **Distance coefficients** `c1 = 0.5, c2 = 0.5, c3 = 0.3` instead of the paper's 1.0 / 1.0 / 0.4. I did not tune them. I assumed the adaptive threshold would absorb the scale. **This turned out wrong for short runs** (see [Section 2](#distance-coefficients-tested)). I kept the defaults so all results stay comparable.
9. **Adaptive species threshold.** It starts at 3.0. After each speciation it rises by 0.1 if there are more than 12 species and falls by 0.1 (minimum 0.3) if there are fewer than 8. A fixed threshold can give 1 species or dozens, depending on the task.
10. **Negative fitness is shifted by the population minimum** before species shares are computed. The ranking is preserved.
11. **No interspecies crossover.** At the paper's 0.001 rate it would touch about 0.15 of 150 children per generation.
12. **Hidden-node ids are keyed by `(split connection, prior splits of that connection in this genome)`.** Each genome counts its splits in `conns_broken`. Keyed by the connection alone, a second split gets unshared ids in different genomes. With the count, the k-th split of a connection gets the same id everywhere, so repeated structure lines up in crossover and distance. `conns_broken` comes from the fitter parent. If the id is somehow already in the genome, a fresh id is used as a safety net.

### Other choices

- **Speciation rule (Stanley's, not neat-python's).** Each genome joins the *first* species whose previous representative is within the threshold, or founds a new one. Empty species die. The new representative is the member closest to the old one.
  - An earlier version followed neat-python: every old species first claims its closest genome, regardless of distance.
  - Under that rule a species never loses its last member through the threshold. So raising the threshold can't reduce the species count.
  - On Slime the count cycled 1 → 19 → 1. With Stanley's rule, a higher threshold lets older species absorb newer ones. Tested.
- **Threshold cap.** `threshold_max` clamps the threshold, as in neat-python. Without it, the controller kept raising the threshold while it had no effect (integral windup). That caused a long single-species phase.
- **Stagnation uses best-ever fitness** per species (the paper's rule; neat-python uses the mean). A species is removed after 15 generations without improvement. The species holding the best genome is never removed.
- **Child counts.** Every surviving species gets at least 2 children. The rest are shared by mean shifted fitness, rounded with the largest-remainder method, so the total is exactly the population size. Without the minimum, the worst species (and every newborn one) scores 0 after the shift and dies at once.
- **Extinction.** Species culled for stagnation or given no children are dropped from the species list. They are not carried into the next speciation.
- **Elitism.** A species with more than 5 members keeps its best genome unchanged.
- **Parent pool.** The top 30% of each species, at least 2 genomes.
- **Weights and biases** start from N(0, 1), clipped to ±5.
- **XOR stopping rule.** A run stops at the first generation in which any genome is solved. Every genome is checked, not only the fittest. A genome can have top fitness with one output on the wrong side while a weaker one solves the task.

### Implementation notes

- **The network is two plain functions.** `compile_genome(genome)` does the graph work once (Kahn's topological sort, per-node incoming lists). `run(net, x)` evaluates a batch of shape `(batch, num_inputs)`.
- **Immutable genes.** `NodeGene` and `ConnGene` are frozen dataclasses. Every mutation and crossover builds new dicts, so children never share containers with parents.
- **The cycle check follows all connections**, enabled or disabled. The full graph is always acyclic, so re-enabling a connection is always safe.
- **Randomness.** One `np.random.Generator` per run, passed down explicitly. The same seed gives an identical run (tested).
- **Checkpoints.** The Slime runner pickles the whole `Population` (tracker, species, rng state, generation) plus the hall of fame. Writes go to a temp file and are then renamed, so a crash can't leave a half-written file. The last 3 are kept.
- **Champion snapshots.** Each generation's champion is pickled to `best/gen_XXXX.pkl` for the complexification figures.
- **Per-generation statistics:** best-ever and champion fitness, mean fitness, species count, threshold, mean and champion hidden nodes and enabled connections, and the count of each hidden activation.

### XOR hyperparameters (all `NeatConfig` defaults)

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
| Adaptive threshold | start 3.0, target 8–12 species, step 0.1, minimum 0.3 |

---

## 2. XOR experiments

### Benchmark

Fitness is `(4 − Σ|y − ŷ|)²`, maximum 16. Solved means all four outputs on the correct side of 0.5.

- 10 of 10 seeds (0–9) solved within 300 generations.
- Generations to solve: 18.1 ± 9.4 (range 6–40). Per seed: 10, 18, 15, 19, 40, 21, 21, 7, 24, 6.
- Hidden nodes in the solutions: 1–6.
- First solutions are often borderline (some outputs near 0.5, fitness 7–14). Running longer sharpens them.
- The paper reports about 32 generations. Settings differ (per-node bias, coefficients, mutation rates, adaptive threshold, success definition), so this is a sanity check.

How the result moved as bugs were fixed:

| Version | Generations to solve | Log |
|---|---|---|
| First working core | 26.5 ± 17.2 | `xor_v1_2026-10-21.txt` |
| Survivor fix, 2-child minimum, adaptive threshold | 15.7 ± 9.2 | `xor_v2_2026-10-21.txt` |
| "Check every genome" stop rule | 15.3 ± 9.5 | `xor_v3_2026-10-22.txt` |
| Final code (after the Slime speciation fixes) | 18.1 ± 9.4 | `xor_final_2026-10-25.txt` |

### Distance coefficients tested

Same 10 seeds, everything else equal:

| Coefficients | Solved | Generations |
|---|---|---|
| Ours (0.5, 0.5, 0.3) | 10/10 | 18.1 ± 9.4 |
| Paper's (1.0, 1.0, 0.4) | 10/10 | **9.8 ± 3.2** |

Permutation test p = 0.015. The threshold moves only 0.1 per generation from 3.0. In a 10–20 generation run it can't make up for the smaller scale. The likely effect (not checked) is fewer species early on. Raw output: `decision8_coefficients_xor_2026-10-23.txt`.

### Mutation-rate sweep

One setting changed at a time, 20 seeds each, 300-generation limit. 480 runs in 30 s, all solved. Median generations to solve:

| Setting | Values → median generations | Note |
|---|---|---|
| `p_activation` | 0.0 → **29**, 0.05 (default) → 17, 0.2 → **13** | the only clear effect |
| `sigma` | 0.1 → 22, 0.5 → 17, 1.0 → 16.5, 2.0 → 15.5 | small trend, within noise |
| `p_weight` | 0.5 → 21, 0.8 → 17, 0.95 → 17.5 | within noise |
| `p_add_conn` | 0.03 → 21.5, 0.1 → 17, 0.3 → 18.5 | within noise |
| `p_add_node` | 0.01 → 19, 0.05 → 17, 0.1 → 18.5, 0.2 → 18 | no speed effect; solutions grow from 1.3 to 4.5 hidden nodes |
| `p_toggle` | 0.0 → 18.5, 0.01 → 17, 0.05 → 17.5 | no effect |
| `p_crossover` | 0.0 → 19, 0.25 → 20, 0.75 → 17, 1.0 → 18 | no effect |

Bootstrap CIs against the default are in `significance_2026-10-25.txt`. Only `p_activation` = 0 has a CI that excludes zero.

**Why activation mutation matters.** New nodes start as `tanh`. With `p_activation` = 0, all 77 hidden nodes in the solutions are `tanh`, and solutions need more nodes (3.9 on average). At 0.2, `sin` is the most common (20 of 44). A `sin` node is non-monotonic, so one node can put the two "different" inputs on one side of 0.5 and the two "same" inputs on the other. That is XOR's shape.

---

## 3. Backprop-NEAT

### Decisions

1. **Datasets:** exact port of `dataset.js` from hardmaru/backprop-neat-js (itself from TensorFlow Playground). 200 points per split, noise 0.5, same formulas. The JS makes only train and test, so I add a 200-point validation split from the same generator. Fitness must not see train or test data. Splits are fixed per seed.
2. **Inputs:** x and y only. No hand-made features.
3. **Output:** `identity` (a logit). The loss is `optax.sigmoid_binary_cross_entropy` on logits, which is numerically stable.
4. **Fitness:** validation accuracy − 0.002 × enabled connections. So 10 extra connections must buy more than 2 points of accuracy.
5. **Lamarckian inheritance:** trained weights are written back into the genome. Children start from trained weights.
6. **Weight mutation:** rate 0.8 → 0.2, σ 0.5 → 0.1. Gradients tune the weights now; a little exploration is kept. **c3** 0.3 → 0.1, so trained weights don't split species that share a structure.
7. **Activations:** identity, sigmoid, tanh, relu, sin, gauss, abs. No `step`: its gradient is 0 everywhere.
8. **Padding size 32** (`max_nodes = 32`; add-node is skipped at the cap). No size buckets.
9. **Optimizer:** Adam, learning rate 0.02, 200 full-batch steps in `lax.scan`, global-norm gradient clipping at 1.0.
10. **Runs:** population 150, 40 generations, seeds 0 and 1, for each dataset.

**No clipping of activation inputs.** An early version clipped every activation input to ±30. That changed identity, relu, abs and sin for large sums. The NumPy-equivalence test caught it (4 of 100 random genomes disagreed). Only the sigmoid can overflow, and `jax.nn.sigmoid` is overflow-safe.

**Hardware.** Native Windows JAX is CPU-only. A CUDA install in WSL2 failed on a broken `jax_cuda12_pjrt` wheel, so everything ran on the CPU (16 threads). The networks are small enough for that.

### Engineering improvement: padded + vmap vs one jit per genome

Same 20 genomes (3–15 nodes), same starting weights, data and optimizer:

| Version | Time | Compiles | Genomes/s |
|---|---|---|---|
| Naive: one jit per genome | 13.27 s | 20 | 1.5 |
| Padded + vmap, first call | 2.24 s | 1 | 8.9 |
| Padded + vmap, compiled | 1.71 s | 0 | 11.7 |

5.9× faster with the compile, 7.8× without. Validation accuracy is identical for every genome. The naive cost grows by one compile per genome (150 per generation). The padded version compiles once per run. Both were timed while the experiment grid was running, so both carry the same CPU load. Data: [`results/backprop/bench.json`](../results/backprop/bench.json).

### Results

Best genome by validation fitness, scored once on test.

| Dataset | Seed | Validation acc. | Test acc. | Hidden | Connections | Time |
|---|---|---|---|---|---|---|
| Circle | 0 | 0.995 | 0.980 | 3 | 8 | 532 s |
| Circle | 1 | 0.990 | 0.930 | 3 | 7 | 532 s |
| XOR | 0 | 1.000 | 1.000 | 2 | 5 | 591 s |
| XOR | 1 | 1.000 | 0.995 | 3 | 7 | 1014 s |
| Spiral | 0 | 0.980 | 0.950 | 6 | 18 | 866 s |
| Spiral | 1 | 0.940 | 0.895 | 5 | 16 | 532 s |

- **Validation–test gap.** Test is 0–6 points below validation in every run. The best genome is the top of about 6,000 on validation, so its validation score is optimistic.
- **Circle seed 1** has the largest gap (0.990 → 0.930). Its training accuracy is also 0.990, so it is not overfitting the training points. A fixed circle rule (radius 3.125) scores 0.975 on that test split. So about 2.5 points are label noise, and about 4.5 points are the evolved boundary generalizing poorly where data is sparse. Log: `circle_s1_rule_check_2026-10-25.txt`.
- Times vary because two runs shared the CPU for part of the grid.
- **Determinism.** The spiral runs were accidentally run twice in parallel and gave identical results. A later spiral seed-0 rerun matched too.

---

## 4. Slime Volleyball

### Setup decisions

1. **The game file is vendored** (`slime/slimevolley.py`). Old gym does not install on current Python. Only the `gym` dependency was removed (small stand-ins for `gym.Env` and `gym.spaces`; `env.seed` uses `np.random.RandomState`). `cv2` and pyglet are imported lazily. Physics, rules, observations and the built-in AI are unchanged. A random policy averages −4.8 over 20 games (README: −4.87).
2. **No input scaling.** The environment already divides raw values by 10. Over 30 random games every observation is within ±6, most within ±2.5.
3. **Outputs:** `tanh`. A button is pressed when its output is > 0.
4. **Hidden activations:** identity, sigmoid, tanh, relu, sin, gauss, abs, step. No gradients needed, so `step` is allowed.
5. **Shared game seeds.** Each generation draws new game seeds, and every genome plays the same games.
6. **Hall of fame.** Every 10 generations the top 5 genomes by fitness play 20 fresh games against the built-in AI. They are ranked by mean pure score. The best 5 ever are kept in a size-5 min-heap. Duplicates are stored once. It is saved in every checkpoint.
7. **NEAT changes from XOR:** 12 inputs, 3 outputs, `tanh` output, full activation set, `max_stagnation` 15 → 20 (noisy fitness).
8. **Speciation scale: N = 1 always** (as in Stanley's original code). Slime genomes start with 36 connections. Dividing by N made a whole new node worth about 0.03, below weight noise (about 0.4). With N = 1, weight-only differences average 0.33 (max 0.45), and one new node adds 1.0. The threshold starts at 3.0 and moves in steps of 0.3 within [1.0, 6.0]. The floor sits above weight noise, so weights alone never split species.
9. **Budget:** population 150, 15 worker processes, about 14.4k game steps/s per worker with a network attached.

**Bug found in the smoke test.** The built-in AI is a small RNN, and `env.reset()` does not reset its memory. Workers reuse one environment, so a game's result depended on the previous game. That broke "same genome + same seeds = same fitness" and resume. `play` now calls `env.policy.reset()` before every game. After the fix, a resumed run equals an uninterrupted one, and parallel equals serial. Both are tested.

### slime1: speciation failure

Direct play against the built-in AI. Fitness: mean of `score + 0.5 · T / 3000` over 4 games. Old speciation (neat-python rule, N = 36, no threshold cap). Generations 0–170.

- The species count cycled 1 → 17 → 1 → 19 → 1.
- Best score −3.37, hall of fame −4.50.
- Led to the speciation fixes in [Section 1](#other-choices) and the N = 1 scale above.

### slime2: speciation fixed, nothing learned

Generations 0–870. Hall of fame stayed at −4.55, near random play. Diagnosis of the final population (`slime2_diagnosis_2026-10-22.txt`, `slime2_watch_2026-10-22.txt`):

- **One strategy everywhere:** camp at the fence, press forward 90% of the time, ignore the ball. 135 of 150 genomes scored exactly −4.75 on the same 4 games.
- **Fitness could not rank genomes.** Standard deviation 0.086. Games lasted 540–758 steps, so the survival bonus varied by only about ±0.02. One lucky point (0.25 of a 4-game mean) was worth 10× more. Selection was picking seed luck.
- **Bloat:** 90 hidden nodes. Nothing selected against structure.

### Fitness v2

Fixed for the rest of the project:

`fitness = mean over games [ score + Σ_rallies A·(1 − e^(−t/τ)) + γ·returns ] − λ·hidden_nodes`

| Term | Value | Reason |
|---|---|---|
| Survival per rally | A = 0.75, τ = 300 steps (10 s) | Rewards keeping a rally alive, with diminishing returns, so stalling can't farm it. The clock resets every point. A < 1, so a rally is always worth less than a point. Campers lose a point every 4.3 s (bonus 0.26); a 30 s rally earns 0.71. |
| Returns | γ = 0.2 | Counted when our agent touches the ball and it then crosses to the AI's side. Rewards the outcome, not a hand-coded angle. A touch is required, so drifting serves don't count. |
| Bloat penalty | λ = 0.002 (0.01 from slime4) | Every hidden node must pay for itself. |
| Score | weight 1 | Still the main objective. The hall of fame and all reporting use the pure score. |

**Why the fitness never changes over time.** Annealed shaping is a real technique. But NEAT's stagnation check compares each species' best-ever fitness with its current one. A fitness whose scale changes every generation would break that comparison. The saturating bonus already caps stalling.

**Touch detection** reads the game state without changing physics. After a bounce the game pushes the ball just outside the agent, so a touch is the ball within agent radius + ball radius + 0.3.

### slime3: fitness v2, then a noise plateau

Generations 0–256. Agents went from fence-camping to moving across their half and jumping 87% of the time. Then the run stalled:

- Returns per game stayed at 0.55–0.61 after generation 40.
- Hall of fame stayed at −4.60 from generation 40.
- Hidden nodes still grew to 22.8.

Noise check on the final population (`slime3_noise_check_2026-10-22.txt`):

| Check | Result |
|---|---|
| Rank agreement, games A vs games B (Spearman), fitness | +0.13 |
| Same, pure score | −0.07 |
| Same, returns | +0.26 |
| Genomes in the top 15 of both A and B | 2 of 15 (chance ≈ 1.5) |
| 4-game vs 16-game fitness (Spearman) | +0.42 |

With 4 games per genome, seed luck decided the ranking. Selection was close to random. That explains the plateau, the frozen hall of fame and the bloat.

### slime4: 16 games per genome

Changes from slime3, nothing else: games 4 → 16, node penalty 0.002 → 0.01. Same seed.

**Prediction:** if noise is the problem, the hall of fame moves off −4.60 and returns keep rising.

**Result (stopped at 243):**
- Hall of fame −4.95 → −4.55 by generation 60. Returns rose above slime3's plateau (about 0.70 vs 0.58). Hidden nodes stayed at 3–5 for about 100 generations.
- Then it stalled too: −4.55 until the stop, returns 0.5–0.8, hidden nodes up to 16.

**Conclusion:** noise was part of the problem, not all of it. Almost every game against the built-in AI is lost 5–0, so there is little signal to climb.

### slime5: self-play (reported run)

**Why.** In the slimevolleygym authors' own results, self-play from scratch reached −0.071 (CMA-ES) and +0.353 (GA) against the built-in AI. Direct play here was stuck at −4.55.

**Design.**
- 16 games per genome per generation: **12 against a copy of itself**, **4 against the built-in AI**. The mix is fixed for the whole run.
- Self-play scores are left out of fitness (zero-sum against yourself). Rally length and returns are the signal there. Only AI games add score.
- The hall of fame is scored against the built-in AI only.
- The game mirrors each player's view, so one network plays either side.

**Lockstep batching.** A genome's 16 games step together, with one network call per step for every player (up to 28 rows). On 16 AI games (10,340 steps): 0.89 s one at a time, 0.22 s in lockstep. **4.0× faster, identical results** (tested; `lockstep_timing_2026-10-22.txt`). Generations take about 6–9 s, against 20–60 s for slime4.

**Progress** (hall of fame, from the run log): −4.85 → −4.35 (gen 170) → −3.85 (210) → −3.35 (260) → **−1.95** (330) → −1.75 (440) → −0.95 (570) → **−0.70** (620).

**Checks on fresh games during training:**
- Generation 380: −2.35 on 40 games, 14.1 returns per game, one game won outright (`slime5_fresh40_gen380_2026-10-22.txt`).
- Generation 786: −1.14 on 100 games, 18.8 returns per game, best game +2 with 22 returns (`slime5_fresh100_gen786_2026-10-22.txt`). Buttons: jump 92%, forward 55%, backward 43%.
- Generation 1555: same genome on top, identical numbers (`slime5_fresh100_gen1555_2026-10-23.txt`).

**Stop.** The hall of fame did not move for about 1,050 generations after −0.70. Best training fitness kept rising (6.59 → 7.24) and networks kept growing. The population got better at rallying against itself, but that stopped turning into points against the AI. Stopped at generation 1670.

**Final evaluation** (`experiments/eval_slime.py`):
- **Select:** the top 5 hall-of-fame agents each play the same 200 games (seeds 100,000+): −1.005, −1.675, −1.010, −1.205, −1.830. #0 wins.
- **Report:** #0 on 1,000 different games (seeds 200,000+): **−0.852 ± 1.260** (standard error 0.04). Win 11.8%, draw 31.0%, loss 57.2%.
- Training draws seeds from 0 to 2^31, so an overlap with these ranges is possible but about 1% likely.
- Data: [`results/slime/final_eval.json`](../results/slime/final_eval.json).

### slime6: the paper's distance coefficients (ablation)

slime5 with one change: coefficients (1.0, 1.0, 0.4). The threshold range is doubled with them (start 6.0, step 0.6, clamp 2.0–12.0), because distances roughly double. Same seed.

**Result (stopped at 681):**
- Hall of fame −3.80 at generation 350, then flat. slime5 was at −0.70 by 620.
- Champion on 40 fixed games: −4.6 to −3.9 (slime5: about −1.2 at generation 650).
- 20.3 hidden nodes on average at 681, against 6–9 for slime5.
- Speciation behaved as intended: threshold 2.05× slime5's over generations 0–374 (`slime5_vs_slime6_threshold_2026-10-23.txt`), 10 species on average, cap never reached.

**Reading.** With weight differences counting less, new structure founds its own species more easily. The population spends its effort on bigger networks that are slower to tune. One seed, so this is evidence, not proof. The paper's coefficients helped on XOR but not on Slime.

### slime7: more activation mutation (ablation)

slime5 with `p_activation` 0.05 → 0.2, the one clear lever on XOR. Same seed.

**Result (stopped at 647):**
- Hall of fame −2.55.
- On the 40 fixed games, level with or ahead of slime5 until about generation 300 (−3.1 vs −3.45). Then it stayed at −3.5 to −2.8 while slime5 reached −1.7 at 450 and −1.2 at 650.
- Best training fitness stuck at 5.71 from generation 300.

**Reading.** On XOR one `sin` node solves the task, so trying activations more often finds it sooner. Slime has no single node that does the job. Changing activations more often mostly breaks nodes that already work.

### Run summary

| Run | Generations | Change | Outcome |
|---|---|---|---|
| slime1 | 0–170 | first version | species cycled 1 → 19 → 1; HoF −4.50 |
| slime2 | 0–870 | speciation fixed | fence-camping, flat fitness; HoF −4.55 |
| slime3 | 0–256 | fitness v2 | agents active, then plateau; HoF −4.60 |
| slime4 | 0–243 | 16 games per genome | HoF −4.55 by gen 60, then frozen |
| **slime5** | 0–1670 | self-play + lockstep batching | HoF −0.70; **final −0.852 ± 1.260** |
| slime6 | 0–681 | + paper's coefficients | HoF −3.80; bigger networks |
| slime7 | 0–647 | + `p_activation` 0.2 | HoF −2.55 |

---

## 5. Bugs found and fixed

- **Cycle check** followed only enabled connections. A disabled one could be re-enabled into a cycle. It now follows all of them.
- **Crossover parent choice:** `rng.choice(a, b)` treated `b` as a sample size. Replaced with a coin flip.
- **Fitter-parent tie-break:** a misplaced parenthesis swapped parents whenever fitness was not tied.
- **Clone aliasing:** cloned children shared objects with their parent, so setting a child's id changed the parent. Children are now copied first.
- **Stagnation filter** was inverted and kept only stagnant species.
- **Child counts:** `allocate` repeated the list instead of scaling it. Rewritten as largest-remainder rounding.
- **Distance:** the weight term compared a genome with itself, `N` counted nodes instead of connections, and the excess/disjoint sets were wrong.
- **Species resurrection:** `reproduce` culled stagnant species, but `Population` kept the old list, so they came back next generation. Stagnation removal silently did nothing. `reproduce` now returns the survivors.
- **Worst species always extinct:** after the fitness shift it scored 0 and got no children, including every newborn species. Fixed with the 2-child minimum.
- **Stop rule** checked only the fittest genome. It now checks every genome.
- **Slime speciation** (found in slime1): species claiming genomes beyond the threshold, N = 36 hiding structure, and no threshold cap.
- **Built-in AI memory leak** between games (see [Section 4](#setup-decisions)).
- **Activation clipping** in the padded JAX network broke equality with NumPy (see [Section 3](#decisions)).

---

## 6. Test suite

63 tests. Run with `uv run python -m pytest -q tests`.

| File | Covers |
|---|---|
| `test_innovation.py` | Same `(src, dst)` gives the same number. The same split in two genomes gives the same node. A second split gives a new node that still matches across genomes. |
| `test_population.py` | Initial genomes share innovation numbers but have different weights. Same seed, identical run. Pickled populations round-trip. The threshold moves the right way and is clamped. The stats CSV has all columns. The stop check finds a solver that is not the fittest. |
| `test_network.py` | A hand-computed output. Batch equals row-by-row. A disabled connection has no effect. |
| `test_mutation.py` | Parent unchanged. Weights stay clipped. Deterministic per seed. `reachable` follows disabled edges. 1,000 add-connections never make a cycle. Add-node structure is correct. 2,000 chained mutations stay acyclic and runnable. |
| `test_crossover.py` | Self-crossover is the identity. The child has the fitter parent's innovations. Every referenced node exists. Disabled genes re-enable about 25% of the time. Parents unchanged. |
| `test_species.py` | δ(g, g) = 0. Distance is symmetric. A hand-computed distance. Identical genomes form 1 species, distant ones 2. Every genome is in exactly one species. A higher threshold merges species. |
| `test_reproduction.py` | Largest-remainder examples. Counts sum to the total. Negative fitness still favours the better species. The elite survives unchanged. Culled species are not returned. The worst species still gets its minimum. |
| `test_slime.py` | Random policy scores about −5. Same genome and seeds give the same fitness. Parallel equals serial. The survival bonus never outranks a point. Resume equals an uninterrupted run. The hall of fame keeps the top k without duplicates. Lockstep equals one game at a time. |
| `test_backprop_*.py` | Datasets match the JS port. The padded forward pass equals the NumPy network. Padding is inert. Masked gradients are zero. Whole-population training compiles once and lowers the loss. |
