# Raw outputs

Command + output pairs for numbers whose output was never saved to a file by a script. Each header gives the time the command ran (IST). Home-folder paths are redacted as `<home>`, and temporary script paths as `<scratch>`.

| File | Backs |
|---|---|
| `xor_v1_2026-10-21.txt` | First XOR result: 10/10, 26.5 ± 17.2 generations (before the reproduction fixes) |
| `xor_v2_2026-10-21.txt` | XOR after the survivor fix, 2-child minimum and adaptive threshold: 15.7 ± 9.2 |
| `xor_v3_2026-10-22.txt` | XOR after the "check every genome" stop fix: 15.3 ± 9.5 |
| `xor_final_2026-10-25.txt` | Rerun of `experiments/xor.py` on the final code: 10/10, 18.1 ± 9.4, 1–6 hidden nodes |
| `decision8_coefficients_xor_2026-10-23.txt` | Distance coefficients on XOR: ours 18.1 ± 9.4 vs the paper's 9.8 ± 3.2 generations |
| `significance_2026-10-25.txt` | Output of `experiments/significance.py`: coefficient permutation test, sweep bootstrap CIs, Slime standard error |
| `circle_s1_rule_check_2026-10-25.txt` | Circle seed 1: evolved genome vs a fixed circle rule on train/val/test (0.990/0.990/0.930 vs 1.000/0.975/0.975) |
| `slime2_diagnosis_2026-10-22.txt` | slime2 fence-camping: fitness std 0.086, 135/150 genomes at −4.75 |
| `slime2_watch_2026-10-22.txt` | slime2 best agent presses forward on 90% of steps |
| `slime3_noise_check_2026-10-22.txt` + `slime3_noise_check.py` | slime3 noise check: Spearman +0.13 (fitness), −0.07 (score), +0.26 (returns), 2/15 top-15 overlap, +0.42 (4 vs 16 games). The script targets an older `run_slime.py` and does not run on the current code. |
| `lockstep_timing_2026-10-22.txt` | Lockstep batching: 0.89 s → 0.22 s on 16 AI games, 4.0×, identical results |
| `slime5_fresh40_gen380_2026-10-22.txt` | slime5 best agent on 40 fresh games at gen 380: −2.35, 14.1 returns per game |
| `slime5_fresh100_gen786_2026-10-22.txt` | slime5 best agent on 100 fresh games at gen 786: −1.14, best game +2, 18.8 returns per game, jump 92% |
| `slime5_fresh100_gen1555_2026-10-23.txt` | Same genome at gen 1555: identical numbers (no hall-of-fame change) |
| `slime5_vs_slime6_threshold_2026-10-23.txt` | slime6 threshold 5.71 vs slime5 2.79 (2.05×), species 10.1 vs 10.3, generations 0–374 |
| `gif_agent_check_2026-10-25.txt` | The reported agent replays the GIF game (seed 10014) as +2, 22 returns, 3,000 steps |
| `leaderboard_trainingmd_fetch_2026-10-22.txt` | slimevolleygym TRAINING.md baselines: GA self-play 0.353 ± 0.728, CMA-ES self-play −0.071 ± 0.827, PPO self-play −0.371 ± 1.085, PPO 1.377 ± 1.133, CMA-ES 1.148 ± 1.071 (1,000 episodes each) |
