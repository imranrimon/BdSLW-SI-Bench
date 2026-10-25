# R3 — BlockGCN base-LR selection (P2 audit item 4)

Source: SLURM job (`scripts/hpc/slurm_audit_reruns_2026_10.sbatch`), side-effect-free `tools/eval_ckpt_on_npy.py` on each run's `*_model-best.pt` (best epoch on SI val U15). TEST = data/bdsl_si/test (signers 2/13, 1365 clips). Macro-F1 = mean per-class F1 over classes present (same rule as tools/p2_metric_suite.py). SD = population SD (paper convention).

## (a) SI TEST of the lr 0.01 runs vs the shared-recipe lr 0.1 headline

| run | lr | seed | val (ckpt sel.) | TEST Top-1 | TEST macro-F1 |
|---|--:|--:|--:|--:|--:|
| bdsl_block_gcn_si_lr01_scratch | 0.01 | 0 | 80.61 | 68.06 | 66.23 |
| bdsl_block_gcn_si_lr01_scratch | 0.01 | 1 | 77.56 | 69.01 | 67.19 |
| bdsl_block_gcn_si_lr01_scratch | 0.01 | 2 | 75.73 | 69.38 | 67.70 |
| **bdsl_block_gcn_si_lr01_scratch mean** | 0.01 | — | 77.96 ± 2.02 | **68.82 ± 0.56** | **67.04 ± 0.61** |
| bdsl_block_gcn_si (headline) | 0.1 | 0 | — | 67.99 | 66.80 |
| bdsl_block_gcn_si (headline) | 0.1 | 1 | — | 65.20 | 64.42 |
| bdsl_block_gcn_si (headline) | 0.1 | 2 | — | 62.49 | 61.30 |
| **bdsl_block_gcn_si (headline) mean** | 0.1 | — | — | **65.23 ± 2.24** | **64.17 ± 2.25** |
