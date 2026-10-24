# Part 1: NEAT from scratch, Backprop-NEAT in JAX, and Slime Volleyball

## 1. Summary

- I wrote a NEAT library from scratch (`neat/`, 63 tests passing) and used the same code, unchanged, for three tasks: XOR, Backprop-NEAT and Slime Volleyball.
- **XOR:** solved in 10 of 10 seeds, in 18.1 ± 9.4 generations (range 6–40), with 1–6 hidden nodes.
- **Backprop-NEAT (JAX):** NEAT evolves the architecture, and gradient descent trains its weights. The mean test accuracy over 2 seeds is 0.955 on circle, 0.998 on XOR and 0.923 on spiral. The networks have 2–6 hidden nodes and 5–18 connections.
- **Engineering improvement:** I pad every genome to a fixed shape and train the whole population in one compiled `jit(vmap(...))` call. This is 5.9× faster than compiling each genome separately (7.8× once compiled), with identical accuracy on all 20 benchmark genomes.
- **Slime Volleyball:** the agent is my own feed-forward NEAT, reading the raw 12-number state. My best agent scores **−0.852 ± 1.260 over 1,000 fresh games** against the built-in AI: 11.8% wins, 31.0% draws and 57.2% losses.
  - This is **below** both self-play baselines on the slimevolleygym leaderboard: CMA-ES self-play −0.071 and GA self-play +0.353.
  - It does not beat the built-in AI on average.
  - It took five runs to get here. Four of them failed, each for a different reason that I diagnosed from data.
- **Null results:** two settings that clearly helped on XOR did not help on Slime (one seed each).

## 2. NEAT core

**Design.** The algorithm is standard NEAT (Stanley & Miikkulainen, 2002):
- historical markings (innovation numbers)
- speciation by compatibility distance
- fitness sharing, with stagnation removal
- crossover aligned by innovation number
- structural mutations (add connection, add node)

My own choices, where they differ from the paper:
- **Split-node ids keyed by (connection, number of times this genome has split it).**
  - In the paper and in my course guide, a node id is keyed only by the connection it splits. So when a genome splits the same connection a second time, two genomes doing this get different node ids.
  - With the count in the key, repeated structure still lines up for crossover and distance.
  - A collision fallback hands out a fresh, unshared id.
- **Per-node bias** instead of a bias input node.
- **Hidden-node activation can mutate.** The allowed set is a config option. New hidden nodes start as `tanh`.
- **Equal-fitness crossover takes one random parent's structure** instead of the union of both parents' genes. The union can join `a→b` from one parent with `b→a` from the other and create a cycle.
- **Strictly feed-forward networks.**
  - Every add-connection runs a reachability check over all connections, including disabled ones, because a disabled connection can be re-enabled later.
  - The forward pass uses a topological sort (Kahn's algorithm), which raises an error if it ever finds a cycle.
- **Speciation:**
  - Assignment follows Stanley's rule: each genome joins the first species whose previous representative is within the threshold.
  - The threshold adapts to keep 8–12 species, and it is clamped at both ends.
  - Each surviving species is guaranteed 2 children.
  - Section 4 explains why each of these was needed.

**Tests.** There are 63 tests in `tests/`. They cover:
- Innovation numbers are shared across genomes.
- A network's batch output equals its row-by-row output, and disabled connections have no effect.
- 2,000 chained mutations stay acyclic.
- Crossover leaves its parents unchanged.
- Distance is symmetric and zero for a genome compared with itself.
- Child counts always sum to the population size.
- The same seed gives an identical run.
- For Slime: parallel equals serial, a stopped-and-resumed run equals an uninterrupted one, and batched games equal games played one at a time.
- For Backprop-NEAT: the padded JAX forward pass equals the NumPy network on random genomes.

**XOR benchmark.**
- 10 of 10 seeds solved within 300 generations.
- Generations to solve: 18.1 ± 9.4 (range 6–40). "Solved" means the first generation in which any genome has all four outputs on the correct side of 0.5.
- Solutions have 1–6 hidden nodes. Population 150.
- The paper reports about 32 generations, but my settings and success definition differ, so I treat XOR as a sanity check, not a comparison.

**Two tests of my own settings on XOR.**
- **Distance coefficients (c1, c2, c3) = (0.5, 0.5, 0.3)**, versus the paper's (1.0, 1.0, 0.4).
  - I picked these without tuning, assuming the adaptive threshold would absorb any scale.
  - That assumption was wrong for short runs. On the same 10 seeds, the paper's values solved in 9.8 ± 3.2 generations, against 18.1 ± 9.4 for mine (permutation test p = 0.015).
  - Likely reason: the threshold moves only 0.1 per generation, too slowly to adapt within a 10–40 generation run.
- **Mutation-rate sweep:** one setting changed at a time, 20 seeds per value, 480 runs, all solved.
  - **Null first:** 16 of the 17 comparisons with the default setting have a 95% bootstrap confidence interval that includes zero. These cover the weight-mutation rate, weight step size, add-connection, add-node, enable-toggle and crossover rates.
  - The one clear effect: turning activation mutation **off** (`p_activation` 0.05 → 0) slows solving by a median of 12 generations (CI [+3, +20.5]).
  - Raising it to 0.2 is only **suggestive**: −4 generations, CI [−10.5, +1.0].
  - These are 17 comparisons without correction, so one false positive is expected by chance. Treat the single clear effect as moderate evidence.
  - With activation mutation off, all 77 hidden nodes in the solutions are `tanh`. At 0.2, `sin` is the most common (20 of 44).

[FIGURE 1: figures/fig01_xor_p_activation.png]
*Figure 1. Activation mutation is the only XOR setting with a clear effect: switching it off slows solving from 17 to 29 median generations (20 seeds per value).*

## 3. Backprop-NEAT (JAX)

**Setup.**
- **Datasets:** circle, XOR and spiral, ported exactly from `backprop-neat-js` `dataset.js` (200 points per split, noise 0.5).
  - I added a 200-point validation split, made with the same generator.
  - Splits are used like this: train for gradients, validation for fitness, test touched once for the final genome.
- **Inputs:** x and y only, with no hand-made features. The output is a single logit, and the loss is sigmoid cross-entropy on the logit.
- **Loop:**
  - NEAT proposes architectures.
  - Each genome's weights and biases get 200 full-batch Adam steps (learning rate 0.02, global-norm clipping at 1.0).
  - The trained weights are written back into the genome (Lamarckian inheritance), so children start from trained weights.
- **Fitness = validation accuracy − 0.002 × enabled connections.** This is the complexity penalty: 10 extra connections must buy more than 2 points of accuracy.
- **Activations:** identity, sigmoid, tanh, relu, sin, gauss, abs (all differentiable or almost everywhere differentiable).
- **NEAT changes from the XOR settings:**
  - weight mutation rate 0.8 → 0.2 and step size 0.5 → 0.1, because gradients now tune the weights
  - weight coefficient c3 0.3 → 0.1, so that trained weights don't split species that share a structure
  - at most 32 nodes
- **Runs:** population 150, 40 generations, seeds 0 and 1, all on CPU.

**Results.** The best genome per run is chosen by validation fitness, then scored once on test.

| Dataset | Seed | Validation acc. | Test acc. | Hidden nodes | Connections | Hidden activations |
|---|---|---|---|---|---|---|
| Circle | 0 | 0.995 | 0.980 | 3 | 8 | tanh ×3 |
| Circle | 1 | 0.990 | 0.930 | 3 | 7 | tanh ×3 |
| XOR | 0 | 1.000 | 1.000 | 2 | 5 | abs, tanh |
| XOR | 1 | 1.000 | 0.995 | 3 | 7 | tanh ×3 |
| Spiral | 0 | 0.980 | 0.950 | 6 | 18 | sin, tanh ×5 |
| Spiral | 1 | 0.940 | 0.895 | 5 | 16 | sin, tanh ×4 |
| **Mean** | | circle 0.993, XOR 1.000, spiral 0.960 | **circle 0.955, XOR 0.998, spiral 0.923** | | | |

[FIGURE 2: figures/fig02_backprop_circle.png]
*Figure 2. Three tanh nodes in series draw a closed boundary around the disc (test accuracy 0.980); the red regions in the corners lie outside the data and are unconstrained.*

[FIGURE 3: figures/fig03_backprop_xor.png]
*Figure 3. XOR is solved with 2 hidden nodes and 5 connections: an `abs` node fed by x and by a tanh of y produces the four quadrants (test accuracy 1.000).*

[FIGURE 4: figures/fig04_backprop_spiral.png]
*Figure 4. The spiral needs the most structure (6 hidden nodes, 18 connections, one of them `sin`), and most of its 10 errors sit near the centre, where the two arms meet (test accuracy 0.950).*

**Engineering improvement: pad, stack, vmap.**
- **The problem:** every genome has a different graph, so a straightforward JAX implementation has to compile a new program for each genome (150 per generation).
- **What I did:**
  - Pack each genome into fixed-size arrays: a 32×32 weight matrix, a mask, biases, activation ids and output positions.
  - Stack the population into one batch.
  - Train it with a single `jit(vmap(train_one))`.
- Positions follow topological order, so the weight matrix is strictly upper triangular. The forward pass is a `fori_loop` over positions.
- Padding positions and masked weights get zero gradient (tested).
- **Benchmark:** the same 20 genomes (3–15 nodes), the same starting weights, data, optimizer and step count.

| Version | Time | Compiles | Genomes/s |
|---|---|---|---|
| Naive: one jit per genome | 13.27 s | 20 | 1.5 |
| Padded + vmap, first call | 2.24 s | 1 | 8.9 |
| Padded + vmap, compiled | 1.71 s | 0 | 11.7 |

- That is 5.9× faster including the compile, and 7.8× once compiled. The maximum validation-accuracy difference between the two versions is 0.0.
- Both versions were measured while other experiments shared the CPU.

**Analysis.**
- **What the penalty trades off.** In the final spiral population (seed 0, 150 genomes), the Pareto front is 15 connections → 0.950 validation accuracy, 16 → 0.975 and 18 → 0.980.
  - After the penalty, the 18-connection genome wins by 0.001 fitness over the 16-connection one (0.944 vs 0.943).
  - So λ = 0.002 sits right at the point where 2 extra connections have to earn their 0.5 point.
- **Same structure, different weights.** Many genomes share a connection count but sit far below the front (down to about 0.5). That is the same architecture with worse trained weights, so 200 steps from an inherited start do not always converge.
- **Validation–test gap.** Test accuracy is 0–6 points below validation in every run. Part of this is selection: the reported genome is the best of about 6,000 evaluated on validation.
  - The largest gap is circle seed 1 (0.990 → 0.930). Its training accuracy is also 0.990, so it is not overfitting the training points.
  - A fixed circle rule (radius 3.125) scores 0.975 on that test split. So about 2.5 points are label noise, and about 4.5 points are the boundary generalizing poorly where the data is sparse.
  - With 2 seeds, I read this as seed variance.
- **Activations.**
  - `tanh` dominates, but it is also the default for new nodes, so its share partly reflects that default.
  - The only `sin` nodes appear in the spiral solutions (one per seed). That fits the spiral's periodic structure, but with 2 seeds this is an observation, not a finding.
- **Reproducibility:** the spiral runs were accidentally executed twice, and a later spiral seed-0 rerun gave identical results each time.

[FIGURE 5: figures/fig05_backprop_spiral_pareto.png]
*Figure 5. The complexity penalty picks from a nearly flat front: going from 16 to 18 connections buys only 0.005 validation accuracy.*

## 4. Slime Volleyball

**Setup.**
- The game is slimevolleygym, vendored into the repo without `gym`. The physics, rules and built-in AI are unchanged; random play scores −4.8 (README: −4.87).
- **Inputs:** the raw 12-number state (own, ball and opponent position and velocity), as the environment provides it, with no added features or scaling.
- **Outputs:** 3 outputs (forward, backward, jump) with `tanh`; a button is pressed when its output is above 0.
- **Hidden activations:** identity, sigmoid, tanh, relu, sin, gauss, abs, step.
- Population 150. Evaluation runs on 15 CPU worker processes.

**Run history as a diagnosis chain.**
- Each row changed one thing in response to a measured failure.
- "HoF" (hall of fame) is the best kept agent's mean pure score over 20 fresh games against the built-in AI. It is re-scored every 10 generations.

| Run | Change | Generations | What happened | Diagnosis → next change |
|---|---|---|---|---|
| slime1 | Direct play vs built-in AI | 0–170 | Species count cycled 1 → 19 → 1; HoF −4.50 | Three causes: old species kept their closest genome regardless of distance; distance divided by N = 36 connections, so a new node was worth ~0.03, below weight noise of ~0.4; the threshold had no cap. → Stanley's rule, N = 1, threshold cap. |
| slime2 | Speciation fixed | 0–870 | Species stable, but HoF −4.55; mean 90 hidden nodes | Everyone camped at the fence (forward pressed 90% of steps). Fitness std 0.086; 135 of 150 genomes scored exactly −4.75. Fitness gave no gradient. → New fitness. |
| slime3 | Fitness v2 (below) | 0–256 | Agents moved and jumped; HoF −4.60 from gen 40 | Two rankings of the same population on different 4-game sets agreed at Spearman +0.13. Selection was close to random. → More games. |
| slime4 | 16 games per genome (was 4), node penalty 0.01 | 0–243 | HoF −4.95 → −4.55 by gen 60, then frozen | Noise was part of it. Against the built-in AI nearly every game is lost 5–0, so there is almost no signal. → Self-play. |
| slime5 | 12 self-play + 4 built-in-AI games per genome | 0–1669 | HoF −4.85 → −1.95 (gen 330) → −0.70 (gen 620), then flat until stopped | Reported run. |

[FIGURE 6: figures/fig06_slime_run_history.png]
*Figure 6. Fixing speciation, fitness and evaluation noise did not move the score; only self-play did, rising from −4.85 to −0.70 before plateauing below the leaderboard's self-play baselines.*

**Fitness design** (fixed for the whole run; reporting always uses the pure score):

`fitness = mean over games [ score_vs_AI + Σ_rallies 0.75·(1 − e^(−t/300)) + 0.2·returns ] − 0.01·hidden_nodes`

- **Survival bonus per rally (my choice).** It saturates, so stalling can't farm reward. Its maximum (0.75) is below one point, so a point always outranks a rally (tested). Its time constant is 300 steps (10 s).
- **Returns, worth 0.2 each:** our agent touches the ball, and the ball then crosses the net. This rewards the outcome instead of checking the shot angle, which would need a physics model.
- **Node penalty** of 0.01 per hidden node.
- **No schedule.** I kept the fitness fixed instead of annealing it, because stagnation compares fitness across generations.
- **Self-play:**
  - 12 of each genome's 16 games are against a copy of itself. Their scores are left out of fitness, because against yourself they are zero-sum. Only survival and returns count there.
  - This follows the rally-length idea of the slimevolleygym CMA-ES self-play baseline.
  - The other 4 games are against the built-in AI and count fully.
  - The environment mirrors each player's view to the right-hand side, so one network plays either side.
- **Lockstep batching:** a genome's 16 games advance together, with one network call per step for every player I control. This is 4.0× faster than playing games one at a time (0.89 s → 0.22 s on 16 games), with identical results.
- **Cost:** slime5 ran 1,670 generations in 4.4 hours of wall-clock time on a 16-thread laptop CPU.

**Final evaluation protocol.**
1. **Select:** the 5 hall-of-fame agents each play the same 200 fresh games (seeds 100,000+). Their means are −1.005, −1.675, −1.010, −1.205 and −1.830. I keep #0.
2. **Report:** that one agent plays 1,000 *different* fresh games (seeds 200,000+), the leaderboard's episode count. Selection and report seeds never overlap.
   - Training draws random seeds from 0 to 2^31, so an overlap with training seeds is possible but about 1% likely.

| Agent | Games | Mean score ± std | Win / draw / loss |
|---|---|---|---|
| **My NEAT, self-play (slime5)** | 1,000 | **−0.852 ± 1.260** (standard error 0.040) | 11.8% / 31.0% / 57.2% |
| Random policy (leaderboard) | 1,000 | −4.866 ± 0.372 | — |
| CMA-ES self-play (TRAINING.md) | — | −0.071 | — |
| GA self-play (leaderboard) | 1,000 | +0.353 ± 0.728 | — |
| CMA-ES, direct (leaderboard) | 1,000 | +1.148 ± 1.071 | — |
| PPO (leaderboard) | 1,000 | +1.377 ± 1.133 | — |

- A draw means the game reached its 3,000-step limit with lives level.
- My agent is clearly below both self-play baselines: −0.852 against −0.071 is about 20 of my standard errors.
- The gap to random play is large, and the agent wins about 1 game in 8.
- I did not re-run the baselines. Their numbers are quoted from the slimevolleygym README and TRAINING.md.

**The reported agent's network.**
- It has 3 hidden nodes (2 `sin`, 1 `identity`) and 57 enabled connections.
- **All 12 inputs are still connected**, each with 3–5 enabled connections. Evolution did not prune away the opponent's state, for example.
- Total |weight| leaving each input:

| Input | my x | my y | my vx | my vy | ball x | ball y | ball vx | ball vy | opp x | opp y | opp vx | opp vy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Σ\|w\| | 8.5 | 8.6 | 7.4 | 2.3 | 9.0 | 2.1 | 6.9 | 5.6 | 8.2 | 3.4 | 5.8 | 4.2 |

- These are raw weights. The inputs have different typical magnitudes, so this is a rough guide, not an importance measure.
- The strongest single links are `my x → backward` (negative), `ball vx → forward` (negative), and `ball x → sin node → backward`.
- **Behaviour:** over 100 fresh games, it presses jump on 92% of steps, forward on 55% and backward on 43%. It covers its whole half of the court and makes 18.8 returns per game.

[FIGURE 7: figures/fig07_slime_best_network.png]
*Figure 7. The reported agent is mostly direct input-to-output wiring plus two `sin` nodes, and it still uses all 12 raw inputs.*

**Complexification.**
- Every genome starts with 0 hidden nodes and 36 connections (every input to every output).
- **Population average:** 6.6 hidden nodes and 54.4 connections at generation 200; 6.7 and 69.1 at generation 620, when the hall of fame reached −0.70; 13.8 and 96.8 at the end (generation 1669).
- **Generation champion:** 2 hidden nodes and 44 connections at generation 200; 3 and 57 at generation 620; 7 and 74 at the end.
- Network growth after generation 620 bought no score against the built-in AI.
- Among hidden nodes, `tanh` (the default) stays the largest share. `sin` grows from 13% at generation 200 to 27–29% later.
- **The generation champion is not a reliable player against the built-in AI.**
  - The champion is picked on a fitness that is mostly self-play. On 40 fixed games its score swings: −1.2 at generation 650, then −3.175 at 675.
  - The hall of fame, scored only against the built-in AI, is what keeps the good agents.

[FIGURE 8: figures/fig08_slime_complexification.png]
*Figure 8. The population roughly doubled its hidden nodes after the score stopped improving (generation 620), so the late growth is bloat, not progress.*

**GIF.** One full 3,000-step game against the built-in AI at generation 786: our agent in orange, the built-in AI in blue. The agent wins this game +2 with 22 returns. It is the best of 100 fresh games, so it is not typical: the mean over those 100 games is −1.14.

[FIGURE 10 (GIF): figures/fig10_slime_game.gif]
*Figure 10. The agent's best game of 100: it shadows the ball across its half and jumps on most steps.*

**Ablations: settings that helped on XOR, tried on Slime.** Null results, one seed each, so treat them as suggestive.
- **slime6, the paper's distance coefficients** (with the threshold range doubled, so its cap can't confound the comparison):
  - The hall of fame stalled at −3.80 from generation 350 until I stopped at 681. Networks grew to 20.3 hidden nodes on average.
  - Speciation behaved the same: the threshold was about 2× slime5's, never hit its cap, and averaged 10.1 species.
- **slime7, `p_activation` 0.2:**
  - Level with or ahead of slime5 until about generation 300, then behind. The hall of fame was −2.55 at 647.
  - The XOR evidence for 0.2 was itself only suggestive.
- **Takeaway:** tuning on XOR did not transfer to Slime. XOR is solved by one well-chosen node, and Slime is not.

[FIGURE 9: figures/fig09_slime_ablations.png]
*Figure 9. Both XOR-motivated changes fell behind the reported settings after about generation 300 (single seed each, so suggestive only).*

## 5. Assumptions and engineering choices

| Area | Standard (paper / reference code) | My choice and why |
|---|---|---|
| Innovation numbers | Shared within a generation | One registry for the whole run, so the same structure always gets the same number |
| Split-node id | Keyed by the split connection | Keyed by (connection, times this genome has split it), so repeated splits match across genomes |
| Bias | A bias input node | Per-node bias, mutated like a weight |
| Equal-fitness crossover | Union of both parents' genes | One random parent's structure, to avoid creating cycles |
| Interspecies crossover | Rate 0.001 | Dropped |
| Distance normalisation N | Larger genome's gene count (1 if under 20 genes) | Kept for XOR; N = 1 always for Slime, which starts at 36 connections |
| Species assignment | Stanley: first compatible species | Stanley's rule (an earlier neat-python-style "claim closest" rule caused the slime1 cycling) |
| Species threshold | Fixed (3.0) in the paper's XOR setup | Adaptive, targeting 8–12 species, clamped; step 0.1 for XOR, 0.3 for Slime |
| Child counts | Proportional to adjusted fitness | Largest-remainder rounding, plus at least 2 children per surviving species, so newborn species survive the fitness shift |
| Stagnation | Best-ever fitness, 15 generations | Same; 20 generations for noisy Slime; the best species is never removed |
| Backprop-NEAT weights | — | Lamarckian write-back; validation-based fitness; 32-node padding |
| Slime training | The guide recommended direct play | Self-play (12 of 16 games), after 4 direct-play runs failed |
| Slime evaluation | — | Common seeds within a generation; AI-only hall of fame; separate selection and report seeds |

- **Unexplored:** I did not tune the population size (150) or the Backprop-NEAT optimizer settings.

## 6. What was new to me

*(Draft for me to edit.)*
- JAX as function transforms: `grad` returns a gradient function, `jit` compiles, `vmap` batches. I also had to give up in-place updates (`.at[].set()`) and Python loops (`fori_loop`, `scan`).
- How much evaluation noise and opponent strength shape what evolution can learn, and diagnosing that with simple checks: rank agreement between game sets, and one-variable ablations.

## 7. Limitations and next steps

**Limitations:**
- **Few seeds:** one seed per Slime setting and two per Backprop-NEAT dataset. Run-to-run variance is not measured, and every Slime comparison is suggestive.
- **Below the self-play baselines:** the final agent scores −0.852, under CMA-ES self-play (−0.071) and GA self-play (+0.353), and it loses to the built-in AI on average.
- **Plateau:** slime5's hall of fame did not improve for about 1,050 generations after reaching −0.70. Meanwhile the population roughly doubled in size and training fitness kept rising. The self-play signal stopped turning into score against the built-in AI.
- **The XOR experiments are short runs.** The coefficient result does not carry over to long runs (slime6).

**Next steps, most promising first:**
1. Self-play against a league of past champions, close to the GA tournament baseline, instead of only an exact copy.
2. Schedule the built-in-AI share of games, or weight AI-game points more, once the population can return the ball.
3. 3–5 seeds per Slime setting, to turn the suggestive ablations into measured effects.
4. A size-bucketed padding for Backprop-NEAT, so small genomes don't pay for 32 positions.

## 8. Artefacts

| What | Where |
|---|---|
| NEAT library | `neat/` |
| Tests (63) | `tests/` |
| XOR benchmark and per-seed stats | `experiments/xor.py`, `experiments/results/xor_seed*.csv` |
| Backprop-NEAT (padding, JAX training) | `backprop/`, `experiments/run_backprop.py` |
| Backprop-NEAT results | `runs/backprop/*/result.json`, `runs/backprop/results.json` |
| Speed-up benchmark | `experiments/bench_backprop.py`, `runs/backprop_bench.json` |
| Slime runner, evaluation, watch/GIF tool | `experiments/run_slime.py`, `experiments/eval_slime.py`, `experiments/watch.py` |
| Slime final result | `runs/slime5/final_eval.json` |
| Slime run logs and per-generation stats | `runs/slime{1..5}.log`, `runs/slime6/log*.txt`, `runs/slime7/log.txt`, `runs/slime*/stats.csv` |
| Significance checks | `experiments/stats_part1.py`, `logs/stats_part1_2026-10-25.txt` |
| Raw outputs of one-off diagnostics | `logs/` (index in `logs/README.md`) |
| Decision log and run history | `DECISIONS.md`, `PROGRESS.md`, `JOURNEY.md` |
| Figures for this section | `experiments/figures_doc_part1.py` → `doc_assets/part1/figures/` |

- `runs/` is excluded from git (checkpoints are large). The small result files listed above should be committed or attached separately.
