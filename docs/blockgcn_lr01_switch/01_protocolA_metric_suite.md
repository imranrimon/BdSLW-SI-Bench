# 1 — Protocol A (BdSLW60, TEST signers 2/13): BlockGCN lr 0.01 vs lr 0.1

lr 0.01 = `config/bdsl_block_gcn_si_lr01.yaml` (identical to the shared recipe except base_lr; no clipping),
runs `work_dir/bdsl_block_gcn_si_lr01_scratch_seed{0,1,2}` (best checkpoint on val U15).
TEST score dumps written by `tools/lr01_switch_eval.py --sections protoA` to
`work_dir/bdsl_block_gcn_si_lr01_seed{S}_test/eval_results/best_acc.pkl` (consumable by
`tools/p2_metric_suite.py` / `tools/arch_significance.py`, stem `bdsl_block_gcn_si_lr01`).
Metrics from `tools/p2_metric_suite.metrics_for`. ± = population SD over seeds.

| | seed 0 | seed 1 | seed 2 | single-model mean ± SD | ensemble |
|---|--:|--:|--:|--:|--:|
| **TEST Top-1, lr 0.01 (new)** | 68.06 | 69.01 | 69.38 | **68.82 ± 0.56** | 71.43 |
| TEST Top-1, lr 0.1 (old) | 67.99 | 65.20 | 62.49 | 65.23 ± 2.24 | 70.62 |

Full suite (single-model 3-seed mean; ensemble in brackets):

| metric | lr 0.01 (new) | lr 0.1 (old) |
|---|--:|--:|
| Top-5 | 90.38 [91.28] | 90.06 [92.89] |
| Macro-F1 | 67.04 [69.67] | 64.18 [69.18] |
| MRR | 0.785 [0.804] | 0.761 [0.804] |
| R@10 | 95.19 [95.53] | 95.12 [97.14] |
| ECE % (equal-mass) | 16.69 [8.04] | 16.57 [2.73] |
| NLL | 1.621 [1.275] | 1.606 [1.092] |
| Brier | 0.480 [0.405] | 0.524 [0.409] |
| per-signer (ensemble) U13 / U2 | 66.7 / 75.4 | 67.3 / 73.4 |

Note: the §metrics exemplar table in P2 (ensemble 70.62, ECE 2.73%, NLL 1.09, Brier 0.41, Macro-F1 69.18,
R@10 97.14, Top-5 92.89) becomes 71.43 / 8.04% / 1.275 / 0.405 / 69.67 / 95.53 / 91.28 under lr 0.01.
Params/GFLOPs unchanged (1.4 M / 1.2).
