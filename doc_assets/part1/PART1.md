# Part 1: NEAT from scratch, Backprop-NEAT in JAX, and Slime Volleyball

## 1. Summary

- I wrote a NEAT library from scratch (`neat/`, 63 tests) and used it for three tasks: XOR, Backprop-NEAT and Slime Volleyball. Slime needed task-specific settings and fixes to speciation (§4).
- **XOR** (rerun on the final code): 10 of 10 seeds solved, in 18.1 ± 9.4 generations (range 6–40), with 1–6 hidden nodes.
- **Backprop-NEAT (JAX):** NEAT evolves the architecture and gradient descent trains the weights.
  - Mean test accuracy over 2 seeds: circle 0.955, XOR 0.998, spiral 0.923.
  - The networks have 2–6 hidden nodes and 5–18 connections.
  - Engineering improvement: I pad genomes to a fixed shape and train the whole population in one `jit(vmap(...))`. This is 5.9× faster than compiling each genome separately (7.8× once compiled), with identical accuracy.
- **Slime Volleyball:** my own feed-forward NEAT, reading the raw 12-number state.
  - My best agent scores **−0.852 ± 1.260 over 1,000 fresh games** against the built-in AI: 11.8% wins, 31.0% draws, 57.2% losses.
  - That is **below all three self-play baselines** on the slimevolleygym leaderboard (CMA-ES −0.071, PPO −0.371, GA +0.353). It loses to the built-in AI on average.
  - Four runs failed before it, each for a different reason that I diagnosed from data.
- **Negative results:** two settings motivated by XOR (the paper's distance coefficients, and more activation mutation) made Slime worse.

## 2. NEAT core

**Design.** Standard NEAT (Stanley & Miikkulainen, 2002): innovation numbers, speciation by compatibility distance, fitness sharing with stagnation removal, crossover aligned by innovation number, and add-connection / add-node mutations.

**My choices, where they differ from the paper:**
- **Split-node ids keyed by (connection, times this genome has split it).**
  - If the id is keyed only by the split connection, a genome that splits the same connection twice gets a node id that no other genome shares.
  - With the count in the key, repeated splits line up across genomes for crossover and distance.
- **Per-node bias** instead of a bias input node.
- **Hidden-node activations mutate** within an allowed set. New nodes start as `tanh`.
- **Equal-fitness crossover takes one parent's structure,** because the union of both parents' genes can join `a→b` with `b→a` and create a cycle.
- **Strictly feed-forward networks.**
  - Add-connection checks reachability over all connections, including disabled ones, since they can be re-enabled.
  - Evaluation uses a topological sort, which raises an error on any cycle.
- **Speciation:**
  - Stanley's first-compatible-species rule.
  - An adaptive threshold targeting 8–12 species, clamped.
  - At least 2 children per surviving species.

**Tests (63).** They check:
- shared innovation numbers
- batch output equals row-by-row output
- 2,000 chained mutations stay acyclic
- crossover leaves its parents unchanged
- distance is symmetric
- child counts sum to the population size
- the same seed gives an identical run
- Slime evaluation: parallel equals serial, resume equals an uninterrupted run, batched games equal games played one at a time
- the padded JAX network equals the NumPy one

**XOR benchmark.**
- 10/10 seeds solved, in 18.1 ± 9.4 generations (range 6–40), with 1–6 hidden nodes. Population 150.
- "Solved" means the first generation in which any genome has all four outputs on the correct side of 0.5.
- The paper reports about 32 generations under different settings, so I treat XOR as a sanity check, not a comparison.

**Two checks of my own settings on XOR.**
- **Distance coefficients.**
  - I used (c1, c2, c3) = (0.5, 0.5, 0.3) untuned, against the paper's (1.0, 1.0, 0.4), assuming the adaptive threshold would absorb the scale.
  - It doesn't in a short run. On the same 10 seeds, the paper's values solved in 9.8 ± 3.2 generations against my 18.1 ± 9.4 (permutation test p = 0.015).
  - The threshold moves only 0.1 per generation, too slowly to adapt in 10–40 generations.
- **Mutation-rate sweep:** one setting at a time, 20 seeds per value, 480 runs, all solved.
  - 16 of 17 comparisons with the default have a 95% bootstrap CI that includes zero. These cover the weight rate and step size, add-connection, add-node, enable-toggle and crossover rates.
  - The one clear effect: turning activation mutation **off** (`p_activation` 0.05 → 0) slows solving by a median of 12 generations (CI [+3, +20.5]).
  - Raising it to 0.2 is not significant: −4 generations, CI [−10.5, +1.0].
  - With 17 uncorrected comparisons, one false positive is expected by chance.
  - With activation mutation off, all 77 hidden nodes in the solutions are `tanh`. At 0.2, `sin` is the most common (20 of 44).

[FIGURE 1: figures/fig01_xor_p_activation.png]
*Figure 1. Only activation mutation has a clear effect on XOR: switching it off slows solving from 17 to 29 median generations.*

## 3. Backprop-NEAT (JAX)

**Setup.**
- **Datasets:** circle, XOR and spiral, ported exactly from `backprop-neat-js` (200 points per split, noise 0.5).
  - I added a 200-point validation split from the same generator.
  - Train is used for gradients, validation for fitness, and test once, for the final genome.
- **Network:** inputs x and y only. The output is one logit, trained with sigmoid cross-entropy.
- **Training:** each genome gets 200 full-batch Adam steps (learning rate 0.02, gradient clipping at 1.0). The trained weights are written back into the genome (Lamarckian inheritance).
- **Complexity penalty:** fitness = validation accuracy − 0.002 × enabled connections.
- **Activations:** identity, sigmoid, tanh, relu, sin, gauss, abs.
- **Changes from the XOR settings:**
  - weight mutation rate 0.8 → 0.2 and step 0.5 → 0.1, because gradients do the fine-tuning
  - c3 0.3 → 0.1, so trained weights don't split species that share a structure
  - at most 32 nodes
- **Runs:** population 150, 40 generations, seeds 0 and 1, on CPU.

**Results.** The best genome per run is chosen on validation fitness, then scored once on test.

| Dataset | Seed | Validation acc. | Test acc. | Hidden | Connections | Hidden activations |
|---|---|---|---|---|---|---|
| Circle | 0 | 0.995 | 0.980 | 3 | 8 | tanh ×3 |
| Circle | 1 | 0.990 | 0.930 | 3 | 7 | tanh ×3 |
| XOR | 0 | 1.000 | 1.000 | 2 | 5 | abs, tanh |
| XOR | 1 | 1.000 | 0.995 | 3 | 7 | tanh ×3 |
| Spiral | 0 | 0.980 | 0.950 | 6 | 18 | sin, tanh ×5 |
| Spiral | 1 | 0.940 | 0.895 | 5 | 16 | sin, tanh ×4 |
| **Mean** | | circle 0.993, XOR 1.000, spiral 0.960 | **circle 0.955, XOR 0.998, spiral 0.923** | | | |

[FIGURE 2: figures/fig02_backprop_circle.png]
*Figure 2. Three tanh nodes in series draw a closed boundary around the disc (test 0.980); the red corner regions lie outside the data and are unconstrained.*

[FIGURE 3: figures/fig03_backprop_xor.png]
*Figure 3. XOR with 2 hidden nodes and 5 connections: an `abs` node fed by x and by a tanh of y forms the four quadrants (test 1.000).*

[FIGURE 4: figures/fig04_backprop_spiral.png]
*Figure 4. The spiral needs the most structure (6 hidden nodes, 18 connections, one `sin`), and most of its 10 test errors sit where the arms meet at the centre (test 0.950).*

**Engineering improvement: pad, stack, vmap.**
- **The problem:** each genome is a different graph, so a direct JAX implementation compiles a new program per genome, 150 per generation.
- **What I did:**
  - Pack each genome into fixed arrays: a 32×32 weight matrix and mask, biases, activation ids and output positions.
  - Positions follow topological order, so the matrix is strictly upper triangular and the forward pass is a `fori_loop`.
  - Stack the population and train it with one `jit(vmap(train_one))`.
  - Padding and masked weights get zero gradient (tested).
- **Benchmark:** the same 20 genomes (3–15 nodes), with the same starting weights, data, optimizer and step count.

| Version | Time | Compiles | Genomes/s |
|---|---|---|---|
| Naive: one jit per genome | 13.27 s | 20 | 1.5 |
| Padded + vmap, first call | 2.24 s | 1 | 8.9 |
| Padded + vmap, compiled | 1.71 s | 0 | 11.7 |

5.9× faster including the compile, 7.8× once compiled, with a maximum validation-accuracy difference of 0.0. Both versions were measured with other jobs sharing the CPU.

**Analysis.**
- **What the penalty trades off.** In the final spiral seed-0 population (150 genomes), the Pareto front is 15 connections → 0.950 validation accuracy, 16 → 0.975 and 18 → 0.980.
  - The 2 extra connections from 16 to 18 cost 2 × 0.002 = 0.4 points of fitness and earned 0.5 points of accuracy.
  - So the 18-connection genome wins by 0.001 (fitness 0.944 vs 0.943).
- **Same structure, worse weights.** Many genomes with a front-level connection count score far lower (down to about 0.5), so 200 steps from an inherited start do not always converge.
- **Validation–test gap.** Test is 0–6 points below validation in every run, because the reported genome is the best of about 6,000 judged on validation.
  - The largest gap is circle seed 1: 0.990 → 0.930, 6 points.
  - Comparison: a fixed circle rule (radius 3.125) scores 0.975 on both validation and test for that seed.
  - So the genome beat the rule by 1.5 points on validation (selection fitting validation noise), then fell 4.5 points below it on test (a boundary that generalizes worse than a circle where data is sparse).
- **Activations.** `tanh` dominates, partly because it is the default for new nodes. The only `sin` nodes appear in the spiral solutions, one per seed.
- **Reproducibility.** Reruns of spiral seed 0 gave identical results.

[FIGURE 5: figures/fig05_backprop_spiral_pareto.png]
*Figure 5. The penalty chooses along a nearly flat front: going from 16 to 18 connections buys only 0.005 validation accuracy.*

## 4. Slime Volleyball

**Setup.**
- slimevolleygym, vendored without `gym`; physics, rules and the built-in AI are unchanged. Random play scores −4.8 (README: −4.87).
- **Inputs:** the raw 12-number state (own, ball and opponent position and velocity), unscaled.
- **Outputs:** 3 `tanh` outputs (forward, backward, jump); a button is pressed when its output is above 0.
- **Hidden activations:** identity, sigmoid, tanh, relu, sin, gauss, abs, step.
- Population 150, evaluated on 15 CPU worker processes.

**Run history as a diagnosis chain.**
- Each run responded to a measured failure in the one before.
- HoF (hall of fame) = the best kept agent's mean pure score over 20 fresh games against the built-in AI, re-scored every 10 generations.

| Run | Change | Generations | What happened | Diagnosis → next change |
|---|---|---|---|---|
| slime1 | Direct play vs built-in AI | 0–170 | Species count cycled 1 → 19 → 1; HoF −4.50 | Old species kept their closest genome regardless of distance; dividing distance by N = 36 connections made a new node worth ~0.03, below weight noise (~0.4); no threshold cap. → Stanley's rule, N = 1 for Slime, threshold cap. |
| slime2 | Speciation fixed | 0–870 | Species stable; HoF −4.55; mean 90 hidden nodes | Everyone camped at the fence (forward pressed 90% of steps); fitness std 0.086; 135 of 150 genomes scored exactly −4.75. → New fitness. |
| slime3 | Fitness v2 (below) | 0–256 | Agents moved and jumped; HoF −4.60 from gen 40 | Rankings of one population on two different 4-game sets agreed at Spearman +0.13: selection was close to random. → More games. |
| slime4 | 16 games per genome (was 4); node penalty 0.002 → 0.01 | 0–243 | HoF −4.95 → −4.55 by gen 60, then frozen | Noise was only part of it: nearly every game vs the built-in AI is lost 5–0, leaving little signal. → Self-play. |
| slime5 | 12 self-play + 4 built-in-AI games per genome | 0–1669 | HoF −4.85 → −1.95 (gen 330) → −0.70 (gen 620), then flat | Reported run. |

[FIGURE 6: figures/fig06_slime_run_history.png]
*Figure 6. Fixing speciation, fitness and evaluation noise did not move the score; self-play did, from −4.85 to −0.70, still below the self-play baselines.*

**Fitness** (fixed for the whole run; reporting uses the pure score only):

`fitness = mean over games [ score_vs_AI + Σ_rallies 0.75·(1 − e^(−t/300)) + 0.2·returns ] − 0.01·hidden_nodes`

- **Saturating survival bonus per rally (my choice).**
  - Stalling can't farm it.
  - Its maximum (0.75) is below one point, so a point always outranks a rally (tested).
- **Returns:** a touch followed by the ball crossing the net. This rewards the outcome instead of a hand-coded shot angle.
- **No schedule.** The fitness is not annealed, because stagnation compares fitness across generations.
- **Self-play:**
  - 12 of each genome's 16 games are against a copy of itself, and only their survival and returns count; the score against yourself is zero-sum. This follows the rally-length idea of the CMA-ES self-play baseline.
  - The 4 games against the built-in AI count fully.
  - The environment mirrors each player's view, so one network plays either side.
- **Lockstep batching:** a genome's 16 games advance together, with one network call per step. 4.0× faster than one game at a time (0.89 s → 0.22 s on 16 games), with identical results.
- **Cost:** slime5 ran 1,670 generations in 4.4 hours on a 16-thread laptop CPU.

**Final evaluation.**
1. **Select:** the 5 hall-of-fame agents each play the same 200 fresh games (seeds 100,000+). Means: −1.005, −1.675, −1.010, −1.205, −1.830. I keep #0.
2. **Report:** that agent plays 1,000 different fresh games (seeds 200,000+), the leaderboard's episode count.
   - Selection and report seeds never overlap.
   - Training draws seeds from 0 to 2^31, so an overlap with training is possible but about 1% likely.

| Agent | Games | Mean score ± std | Win / draw / loss |
|---|---|---|---|
| **My NEAT, self-play** | 1,000 | **−0.852 ± 1.260** | 11.8% / 31.0% / 57.2% |
| Random policy | 1,000 | −4.866 ± 0.372 | — |
| PPO self-play | 1,000 | −0.371 ± 1.085 | — |
| CMA-ES self-play | 1,000 | −0.071 ± 0.827 | — |
| GA self-play | 1,000 | +0.353 ± 0.728 | — |
| CMA-ES, direct | 1,000 | +1.148 ± 1.071 | — |
| PPO, direct | 1,000 | +1.377 ± 1.133 | — |

- A draw means the game reached its 3,000-step limit with lives level.
- My agent is below all three self-play baselines, by margins far outside noise.
- Baselines are the slimevolleygym authors' published results (TRAINING.md and README), not retrained by me.
- **One agent, several numbers:**
  - −0.70 is its hall-of-fame score on 20 games.
  - −1.005 is the selection score on 200 games.
  - −1.14 is a 100-game check during training.
  - The 20-game score was optimistic. **Only the 1,000-game −0.852 is the result.** It is within about 2 standard errors of the smaller checks (1.7 for the 200-game −1.005, 2.3 for the 100-game −1.14).

**The reported agent's network.**
- 3 hidden nodes (2 `sin`, 1 `identity`) and 57 enabled connections.
- All 12 inputs are still connected, each with 3–5 connections. Evolution did not prune, for example, the opponent's state.
- Total |weight| leaving each input:

| Input | my x | my y | my vx | my vy | ball x | ball y | ball vx | ball vy | opp x | opp y | opp vx | opp vy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Σ\|w\| | 8.5 | 8.6 | 7.4 | 2.3 | 9.0 | 2.1 | 6.9 | 5.6 | 8.2 | 3.4 | 5.8 | 4.2 |

- These are raw weights, and the inputs differ in typical magnitude, so this is a rough guide.
- Strongest links: `my x → backward` (negative), `ball vx → forward` (negative), and `ball x → sin node → backward`.
- **Behaviour** over 100 fresh games: jump pressed on 92% of steps, forward 55%, backward 43%. It covers its whole half and makes 18.8 returns per game.

[FIGURE 7: figures/fig07_slime_best_network.png]
*Figure 7. The reported agent is mostly direct input-to-output wiring plus two `sin` nodes, and it still uses all 12 raw inputs.*

**Complexification.**
- Every genome starts with 0 hidden nodes and 36 connections.
- **Population average** (hidden nodes / connections): 6.6 / 54.4 at generation 200; 6.7 / 69.1 at generation 620, when the HoF reached −0.70; 13.8 / 96.8 at the end.
- **Generation champion:** 2 / 44, 3 / 57 and 7 / 74 at the same points.
- `sin` grows from 13% of hidden nodes at generation 200 to 27–29% later. `tanh`, the default, stays the largest share.
- **The champion is not a reliable player against the built-in AI.** It is picked on mostly-self-play fitness, and its score on 40 fixed games swings from −1.2 at generation 650 to −3.175 at 675. The AI-scored hall of fame is what keeps the good agents.

[FIGURE 8: figures/fig08_slime_complexification.png]
*Figure 8. Hidden nodes roughly doubled after the score stopped improving at generation 620, so the late growth is bloat.*

**Negative results: settings motivated by XOR, tried on Slime.**
- One seed per Slime setting, so each comparison is a single sample.
- **slime6, the paper's distance coefficients** (threshold range doubled, so its cap can't confound):
  - HoF stalled at −3.80 from generation 350 to the stop at 681, against −0.70 for slime5.
  - Networks grew to 20.3 hidden nodes on average.
  - Speciation behaved the same: threshold about 2× slime5's, never at the cap, 10.1 species on average.
- **slime7, `p_activation` 0.2:** level with or ahead of slime5 until about generation 300, then behind. HoF −2.55 at generation 647.
- **Reading:** both changes speed up *early* search (more species early, more activation tries). That is all that counts on XOR, which ends in about 20 generations. Slime's progress came after generation 200, where both runs fell behind.

[FIGURE 9: figures/fig09_slime_ablations.png]
*Figure 9. Both XOR-motivated changes fell behind the reported settings after about generation 300.*

**GIF.**
- A full 3,000-step game against the built-in AI. Our agent is orange, the built-in AI is blue.
- It is the reported agent: replaying this game's seed with it gives the same +2, 22 returns, 3,000 steps.
- This is its best of 100 fresh games, not a typical one (its mean over those 100 was −1.14).

[FIGURE 10 (GIF): figures/fig10_slime_game.gif]
*Figure 10. The agent's best game of 100: it shadows the ball across its half and jumps on most steps.*

## 5. Assumptions and engineering choices

| Area | Standard (paper / reference) | My choice and why |
|---|---|---|
| Innovation numbers | Shared within a generation | One registry for the whole run, so the same structure always gets the same number |
| Split-node id | Keyed by the split connection | Keyed by (connection, times split), so repeated splits match across genomes |
| Bias | Bias input node | Per-node bias, mutated like a weight |
| Equal-fitness crossover | Union of both parents' genes | One parent's structure, to avoid cycles |
| Interspecies crossover | Rate 0.001 | Dropped: at 0.001 it would affect about 0.15 of 150 children per generation, so I left it out for simplicity |
| Distance normalisation N | Larger genome's gene count (1 under 20 genes) | Kept for XOR; N = 1 for Slime, which starts at 36 connections |
| Species assignment | First compatible species | Same (an earlier "claim closest genome" rule caused the slime1 cycling) |
| Species threshold | Fixed (3.0) in the paper's XOR setup | Adaptive, 8–12 species, clamped |
| Child counts | Proportional to adjusted fitness | Largest-remainder rounding, at least 2 children per surviving species |
| Stagnation | Best-ever fitness, 15 generations | Same; 20 for Slime; the best species is never removed |
| Backprop-NEAT | — | Lamarckian write-back, validation fitness, 32-node padding |
| Slime training | slimevolleygym's strongest baselines (CMA-ES, PPO) train directly against the built-in AI | Self-play for 12 of 16 games, after 4 direct-play runs failed |
| Slime evaluation | 1,000 episodes against the built-in AI | Same, plus separate selection seeds and an AI-only hall of fame |

I did not tune the population size (150) or the Backprop-NEAT optimizer.

## 6. What was new to me

- Batching variable-topology graphs into one compiled program. Every genome is a different graph, so I padded each to a fixed 32-node shape, masked the missing connections so they get zero gradient, and used `vmap` over genomes so one `jit` trains the whole population. Getting the masking right, and proving it with equivalence and zero-gradient tests, was the new part.
- Evolution as an experiment loop. Most of my Slime time went into finding out why selection had no signal: speciation, a flat fitness, evaluation noise, then opponent strength. I fixed one cause per run, and checked each with a small measurement before changing anything.

## 7. Limitations and next steps

**Limitations:**
- One seed per Slime setting and two per Backprop-NEAT dataset, so run-to-run variance is not measured.
- The final agent loses to the built-in AI on average, and scores below all three self-play baselines.
- slime5's HoF did not improve for about 1,050 generations after −0.70, while the networks roughly doubled and training fitness kept rising. The self-play signal stopped turning into points against the AI.

**Next steps, most promising first:**
1. Self-play against a league of past champions, close to the GA tournament baseline, instead of an exact copy.
2. Raise the share or the weight of built-in-AI games once the population can return the ball.
3. 3–5 seeds per Slime setting.
4. Size-bucketed padding for Backprop-NEAT, so small genomes don't pay for 32 positions.

## 8. Artefacts

| What | Where |
|---|---|
| NEAT library and tests | `neat/`, `tests/` |
| XOR benchmark and sweep | `experiments/xor.py`, `experiments/sweep_xor.py` |
| Backprop-NEAT | `backprop/`, `experiments/run_backprop.py`, `experiments/bench_backprop.py` |
| Slime training, evaluation, GIF tool | `experiments/run_slime.py`, `experiments/eval_slime.py`, `experiments/watch.py` |
| Significance checks | `experiments/stats_part1.py` |
| Figures for this section | `experiments/figures_doc_part1.py` |
| Decision log and run history | `DECISIONS.md`, `PROGRESS.md`, `JOURNEY.md` |
