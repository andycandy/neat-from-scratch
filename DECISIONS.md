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

- **Speciation rule (Stanley's, not neat-python's):** each genome joins the *first* species whose previous-generation representative is within the threshold, or founds a new species. Species with no members die. The new representative is the member closest to the old one. An earlier version followed neat-python, where every old species first claims its closest genome regardless of distance. Under that rule a species could never lose its last member through the threshold, so raising the threshold could not reduce the species count. The count only fell when whole cohorts stagnated together, and in the Slime run it cycled 1 → 19 → 1. With Stanley's rule, a higher threshold lets older species absorb newer ones. There is a test for this.
- **Threshold cap:** `threshold_max` clamps the dynamic threshold, as in neat-python. Without it, the controller kept raising the threshold while it had no effect (integral windup), which then caused a long single-species phase.

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
- **Generations to solve** (the first generation in which any genome is solved): mean 18.1 ± 9.4 (range 6–40).
- **Hidden nodes in the solutions:** 1–6.
- The first solutions are often borderline: some outputs are close to 0.5, with fitness between 7 and 14. Running longer sharpens them.
- **Comparison with the paper (about 32 generations):** the settings differ (per-node bias, different distance coefficients and mutation rates, a dynamic threshold, a different success definition), so this is a sanity check, not a like-for-like comparison.
- Per-generation statistics are in `experiments/results/xor_seed*.csv`.

Reproduce with `uv run python experiments/xor.py`.

---

# Unit 4 — Slime Volleyball decisions

Section numbers refer to the Unit 4 guide (`unit4_slimevolley.md`).

1. **Environment: the game file is vendored** (`slime/slimevolley.py`, guide Option 2). Old gym does not install on Python 3.14. Only the `gym` dependency was removed (tiny stand-ins for `gym.Env` and `gym.spaces`; `env.seed` uses `np.random.RandomState`), and `cv2` and pyglet are imported lazily. The game physics, rules, observations and built-in AI are unchanged. Check: a random policy averages −4.8 over 20 games (README: −4.87). (§3)
2. **Input scaling: none added** (`INPUT_SCALE = 1.0`). The environment already divides raw values by 10. Measured over 30 random games, every observation lies within ±6, and most within ±2.5 (the ball velocities are the widest). Values of order 1 do not saturate the activations. Any scaling here would be a fixed linear normalization, not feature engineering. (§4)
3. **Outputs:** `tanh`, and a button is pressed when its output is > 0. The output activation is fixed. Evolution changes only hidden activations. (§4)
4. **Hidden activations:** identity, sigmoid, tanh, relu, sin, gauss, abs and step. No gradients are needed, so `step` is allowed. (§4)
5. **Fitness:** the mean over E games of `score + β · T / 3000`, with **β = 0.5**. Because β < 1, one extra point always outranks any survival difference (there is a test for this). The bonus only breaks ties early, when nearly every genome loses 5–0. All reporting uses the pure score. (§5)
6. **Common random numbers:** each generation draws E = 4 game seeds from the run's rng, and every genome plays those same games. New seeds are drawn each generation. (§6.1)
7. **Hall of fame:** every 10 generations, the top 5 genomes by fitness are re-evaluated on 20 fresh games and ranked by mean pure score. The best 5 ever are kept, in a min-heap of size 5 (the weakest kept genome sits at the root, so checking and evicting it is cheap). Identical genomes, which elitism re-sends, are stored once. It is saved in every checkpoint. (§6.2)
8. **Training regime: direct play against the built-in AI**, not self-play. On the README leaderboard, the methods trained directly against the built-in AI scored clearly higher against it (CMA-ES 1.148, PPO 1.377) than the self-play versions (GA 0.353, CMA-ES −0.071, PPO −0.371). The deliverable is the score against the built-in AI. Self-play is future work. (§7)
9. **Budget:** population 150, E = 4, 15 worker processes (16 logical CPUs − 1), and 14.4k steps/s per worker with a network attached (measured). The worst case (every game 3,000 steps) is about 8 s of compute per generation in theory. Early generations measured 4–6 s. (§9)
10. **NEAT changes from the XOR config:** 12 inputs and 3 outputs, `tanh` output, the full activation set (+ identity, + step), and `max_stagnation` 15 → 20 (noisy fitness). **Speciation scale:** the distance is never divided by N (N = 1 always, as in Stanley's original code). Slime genomes start with 36 connections, so dividing by N made a whole new hidden node worth about 0.03, below the weight noise (≈ 0.4). Structure was then invisible to speciation. Measured with N = 1: weight-only differences average 0.33 (max 0.45), and one new node adds 1.0. The threshold starts at 3.0 and moves in steps of 0.3 (Stanley's values), within [1.0, 6.0]. The floor of 1.0 sits above weight noise, so weights alone never split species. In a 40-generation check, the species count stayed at 10–12 for over 25 generations, with the threshold steady at 3.3. (§10)

**Fix found during the smoke test:** the built-in AI is a small RNN, and `env.reset()` does not reset its memory. Because each worker reuses one environment, a game's result depended on the worker's previous game, which broke both "same genome + same seeds = same fitness" and resume. `play` now calls `env.policy.reset()` at the start of every game. After the fix, an uninterrupted run and a stopped-and-resumed run give identical statistics, and parallel evaluation equals serial. Both are tested.

**First overnight run (`runs/slime1`, kept as the "before" run):** generations 0–170 with the old speciation (neat-python rule, N = 36, no threshold cap). The species count cycled 1 → 17 → 1 → 19 → 1 instead of settling. That led to the speciation fixes above. The best score reached −3.37, and the hall of fame was at −4.50.

**Checkpoints:** one per generation, written to a temporary file and then renamed (so a crash cannot leave a half-written file). The last 3 are kept. Each holds the population (rng, tracker, species, stats) and the hall of fame. Each generation's champion is also pickled to `best/gen_XXXX.pkl` for the complexification figures.

## Slime fitness v2 (after the failed `runs/slime2`)

**What failed:** `runs/slime2` ran 871 generations with fitness = score + 0.5·T/3000. The hall of fame stayed at −4.55, about the same as random play (−4.87). A diagnosis of the final population (`experiments/watch.py` plus a population scan) found:
- **Every genome had converged on one strategy:** camp against the fence, press "forward" 90% of the time, and ignore the ball. 135 of 150 genomes scored exactly −4.75 on the same 4 games.
- **Fitness could not rank genomes:** the standard deviation was 0.086. Episode lengths were 540–758 steps, so the survival bonus differed by only about ±0.02. One lucky point (0.25 of a 4-game mean) outweighed it by more than 10×, so selection was picking seed luck.
- **Bloat:** 90 hidden nodes, all of them wired to the outputs. Nothing selected against structure.

**New fitness** (fixed for the whole run):

`fitness = mean over games [ score + Σ_rallies A·(1 − e^(−t/τ)) + γ·returns ] − λ·hidden_nodes`

| Term | Value | Reason |
|---|---|---|
| Survival per rally, `A·(1 − e^(−t/τ))` | A = 0.75, τ = 300 steps (10 s) | Rewards keeping each rally alive, with diminishing returns, so stalling can't farm unlimited reward. The clock resets at every point. **A < 1, so a whole rally's bonus is always worth less than one point.** τ = 10 s puts the steep part of the curve where improvement happens: campers lose a point every ~4.3 s (bonus 0.26), and a 30 s rally earns 0.71. A curve that saturates in ~4 s would give campers nearly the full bonus and no gradient. |
| Returns, `γ·returns` | γ = 0.2 | A return is counted when our agent touches the ball and the ball then crosses to the AI's side. This rewards the outcome ("the shot cleared the net") instead of a hand-coded angle, which would need a physics model. Requiring a touch first excludes serves that drift over on their own. Juggling on our own side earns nothing. A return (0.2) is worth less than a point (1.0). For scale: the built-in AI, driving our side, makes 22–27 returns per game. |
| Bloat penalty, `λ·hidden_nodes` | λ = 0.002 | Every hidden node must pay for itself. 50 nodes cost 0.1, less than one return, so useful structure can still grow. |
| Score | weight 1 | Still the primary objective. **All reporting and the hall of fame use the pure score.** |

**Why the fitness doesn't change across generations:** annealed shaping is a real technique (curriculum learning, OpenAI Five's "team spirit" weight annealed from 0 to 1). But NEAT's stagnation check compares each species' all-time best fitness with its current fitness, and "best ever" tracking does the same. A fitness whose scale changes every generation would make those comparisons meaningless, and species could be culled just because the reward shrank. The saturating per-rally bonus already caps stalling, which was the main reason to anneal.

**Touch detection** reads the game state without changing physics: after a bounce, the game pushes the ball just outside the agent, so a touch is the ball being within the agent radius + ball radius + 0.3.

**Watch item:** returns can add up to more than the score range (25 returns × 0.2 = 5). If agents learn long rallies without winning points, lower γ in a later run.

**Early signal:** in a 10-generation smoke test, mean returns per game rose from 0.21 to 0.92 by generation 7. The old fitness had no measurable gradient.

## Run `runs/slime3` (fitness v2): result and noise diagnosis

**Result:** stopped at generation 256. Fitness v2 changed the behaviour once: agents went from fence-camping to moving across the whole half and jumping 87% of the time. After that the run plateaued. Averaged over 40-generation windows, returns per game stayed at 0.55–0.61 after generation 40, the population's mean pure score stayed at about −4.85, and the hall of fame stayed at −4.60 from generation 40 onward. Hidden nodes still grew, to 22.8 (λ = 0.002 was too weak to stop it).

**Diagnosis:** the final population (generation 246, 150 genomes) was scored on two disjoint sets of 4 games, and on 16 games:

| Check | Result |
|---|---|
| Rank agreement, games A vs games B (Spearman), fitness | +0.13 |
| Same, pure score | −0.07 |
| Same, returns | +0.26 |
| Genomes in the top 15 of both A and B | 2 of 15 (random picking would give ≈ 1.5) |
| 4-game fitness vs 16-game fitness (Spearman) | +0.42 |

**Conclusion:** with 4 games per genome, seed luck dominates the ranking. Selection was close to random: each generation kept the genomes that happened to get easier games, and nothing real accumulated. That explains the plateau, the frozen hall of fame, and the unchecked bloat, since selection could not see whether structure was worth its cost. Even 16 games agree only partly with 4, so the real differences between genomes in this population are small as well.

**Candidate next changes (not yet decided):** more games per genome (E = 4 → 16, which roughly halves the ranking noise, at about 4× the time per generation) and a stronger node penalty (λ = 0.002 → 0.01) to stop bloat and keep generations fast.

## Run `runs/slime4`: noise ablation

**Changes from `slime3`, and nothing else:** `EPISODES` 4 → 16 and `NODE_LAMBDA` 0.002 → 0.01. The seed (0), the fitness formula, the NEAT config and the hall of fame are all unchanged, so `slime3` vs `slime4` is a one-variable ablation of the noise diagnosis (λ is changed alongside it to keep 4× longer generations from being slowed further by bloat).

**Prediction if the diagnosis is right:** rankings become meaningful, so the hall of fame moves off −4.60 and returns per game keep rising past about 0.6 instead of plateauing. If it stays flat even with 16 games, the remaining explanation is that the built-in AI is too strong for a from-scratch population to get any signal. In that case self-play is the next fix (on the README leaderboard, a GA trained by self-play reached a positive score against the built-in AI).
