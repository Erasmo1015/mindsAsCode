# Ablation results — October 8, 2026

Metric: mean participant test log likelihood, using `overall_best_train.test_loglik` (train/validation-selected program); higher is better. Average gives equal weight to each dataset. Main scores use the official gate-selected track.

| Experiment | Bergert–Nosofsky | Choice13k | Guan stopping | Average |
| --- | ---: | ---: | ---: | ---: |
| Original main experiment | -0.343 | -0.492 | -0.527 | -0.454 |
| A — No transfer (314677) | -0.546 | -0.536 | -0.569 | -0.550 |
| B — No population (315011) | -0.537 | -0.532 | Pending (0/30) | -0.534† |
| C — No exploration (314993) | -0.413 | -0.425 | -0.541 | -0.459 |
| D — No fresh candidates (314994) | -0.648 | -0.457 | -0.466 | -0.524 |
| E — No uniform prompt (314995) | -0.289 | -0.480 | -0.577 | -0.449 |

† B average covers only Bergert and Choice13k (2/3 datasets), so it is not directly comparable with full averages. B Bergert reuses certified job 314726; Choice13k and Guan use job 315011. All numeric dataset cells cover 30/30 participants. Guan B has no final `results.json` files; in-progress candidate scores are excluded.

Slurm: 314677, 314993, 314994, 314995 COMPLETED; 315011 RUNNING.

Main source: `analysis_2026Sep/Sep30_pics_v4/results/official_gate/dataset_results.csv`, column `gate_selected_equal_participant_test`.
