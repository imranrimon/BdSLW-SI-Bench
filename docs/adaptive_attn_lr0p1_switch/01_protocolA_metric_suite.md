# 1 — Protocol A held-out TEST + metric suite (Table 7, tab:suite, efficiency, tracks)

Protocol A, BdSLW60 held-out TEST signers 2/13 (1,365 clips). New = the 3 existing lr-0.1 runs (`config/lrsweep/bdsl_<arch>_si_lr0p1.yaml`, best checkpoint on val U15) scored with `main.py --phase test --config config/bdsl_<arch>_si_test.yaml` -> `work_dir/bdsl_<arch>_si_lr0p1_seed{S}_test/eval_results/best_acc.pkl`. Old = the P2 runs (`work_dir/bdsl_<arch>_si_seed{S}_test`). ± = population SD over 3 seeds.

| arch | lr | per seed (TEST Top-1) | mean ± SD | best-val epoch / val Top-1 |
|---|---|---|--:|---|
| Adaptive GNN | 0.01 (old) | 77.88 / 80.00 / 79.34 | 79.07 ± 0.89 | |
| Adaptive GNN | **0.1 (new)** | -- / -- / -- | **--** | ep -- / -- / --; val 89.77 / 90.69 / 89.31 (mean 89.92) |
| Adaptive GNN | 0.1 recheck (side-effect-free scorer) | -- / -- / -- | -- | |
| Attention GNN | 0.01 (old) | 72.89 / 73.70 / 75.31 | 73.97 ± 1.01 | |
| Attention GNN | **0.1 (new)** | -- / -- / -- | **--** | ep -- / -- / --; val 86.26 / 86.87 / 83.97 (mean 85.70) |
| Attention GNN | 0.1 recheck (side-effect-free scorer) | -- / -- / -- | -- | |

## Metric suite (tab:suite, §metrics calibration/per-signer paragraphs)

`tools/p2_metric_suite.metrics_for` on the TEST score dumps: single-model 3-seed mean (± pop. SD for Top-1); ens = 3-seed softmax ensemble. Old Adaptive GNN in P2 used only seeds 0/1 (its seed-2 `_test/eval_results/` is empty); `old (3 seeds)` adds a side-effect-free seed-2 dump of the same checkpoint (model-89) the seed-2 `_test` run loaded.

| arch | variant | seeds | Top-1 ± SD | Top-1 ens | Top-5 | Macro-F1 | MRR | R@10 | ECE % | ECE ens % | NLL | Brier | per-signer ens U13 / U2 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| Adaptive GNN | old (P2) | 2 | 78.94 ± 1.06 | 80.00 | 93.37 | 78.09 | 0.855 | 97.55 | 12.16 | 7.75 | 1.253 | 0.329 | 73.8 / 85.2 |
| Adaptive GNN | old (3 seeds) | 2 | 78.94 ± 1.06 | 80.00 | 93.37 | 78.09 | 0.855 | 97.55 | 12.16 | 7.75 | 1.253 | 0.329 | 73.8 / 85.2 |
| Adaptive GNN | **new lr 0.1** | 0 | -- |
| Attention GNN | old (P2) | 3 | 73.97 ± 1.01 | 77.73 | 92.33 | 71.86 | 0.823 | 96.21 | 13.71 | 3.42 | 1.274 | 0.402 | 70.4 / 83.9 |
| Attention GNN | **new lr 0.1** | 0 | -- |
