# Table 7 panel (b) — BdSLW401, held-out test signers S04/S08

Re-run pose rows, 3 seeds each. ± is population SD over seeds, as in the paper.

## Filled now (no training)

**TEST Top-1.** The CSV rows `bdsl_<arch>_w401_seed<N>_test` were recomputed from the saved test predictions `work_dir/bdsl_<arch>_w401_seed<N>_test/eval_results/best_acc.pkl`. Those are best-val checkpoints (selected on the seen-signer val split) evaluated on the 7,833 S04/S08 test clips. All three rows reproduce the paper.

**Macro-F1** is computed from the same predictions with `_metrics_one` in `tools/p2_metric_suite.py`, as a single-model 3-seed mean.

**Params and GFLOPs** come from `tools/benchmark_metrics.efficiency` (thop, one clip, T=120, V=27, 401 classes).

| Model | TEST Top-1 per seed (s0/s1/s2) | TEST Top-1 | Macro-F1 per seed | Macro-F1 | Params (M) | GFLOPs |
|---|---|--:|---|--:|--:|--:|
| CTR-GCN (`model.adaptive_gnn`) | 87.51 / 87.03 / 86.39 | **86.98 ± 0.46** | 86.92 / 86.44 / 85.85 | **86.40 ± 0.44** | 5.30 | 5.29 |
| ST-GCN (`model.st_gcn_vanilla`) | 87.37 / 86.99 / 86.25 | 86.87 ± 0.47 | 86.76 / 86.27 / 85.41 | 86.15 ± 0.56 | 3.18 | 3.45 |
| BlockGCN (`model.block_gcn`) | 52.90 / 51.19 / 80.51 | 61.53 ± 13.43 | 51.60 / 50.13 / 79.63 | 60.46 ± 13.57 | 1.49 | 1.19 |

Sanity check: the same routine on the BdSLW60 configs gives 5.21/5.29, 1.40/1.19 and 3.09/3.45, which matches the paper's efficiency table. The W401 params are slightly higher only because of the 401-way classifier.

BlockGCN is bimodal: two seeds sit near 52 and one at 80.5. This is the known high-variance W401 BlockGCN; its val spread is ±7.5 as well.

## Mixed + SMG (DONE — jobs, both COMPLETED exit 0, 2026-10-09)

**Split** (`data/bdslw401_sd`, built by `preprocessing/make_w401_mixed_bundle.py`, seed 42):
- The clip pool is the SI bundle's train+val+test, 51,098 clips over 401 classes.
- It is re-split at random, stratified by class, to the SI proportions (38,876 / 4,389 / 7,833).
- Signers therefore appear on both sides, including S04/S08.
- This mirrors `generate_random_split_npy.py`, which built `data/bdsl_sd` for BdSLW60. The only difference is that it reads the existing bundles, so preprocessing is identical to the SI run.

**Runs.** Configs are `config/bdsl_<arch>_w401_sd.yaml`, identical to the SI configs except for the data paths:
- SGD at lr 0.1, 100 epochs, batch 32.
- Mirror off, no grad clipping (`grad_clip_norm` 0.0, as in the SI runs).
- Seeds 0/1/2.
- Launcher: `scripts/hpc/slurm_w401_mixed.sbatch`, using 2 jobs with the runs inside each job executing concurrently.

**Mixed metric** = best-epoch Top-1 on the mixed val set, max over the `results_final.csv` rows of `bdsl_<arch>_w401_sd_seed<N>`. This is exactly the procedure behind the BdSLW60 Mixed column (`tab:smg8`). As a secondary number, each best checkpoint is also evaluated on the mixed test set (`work_dir/<exp>/mixed_test_top1.txt`).

Split sizes are 38,890 / 4,352 / 7,856. They differ slightly from the SI split's 38,876 / 4,389 / 7,833 because of per-class rounding. Train and val are disjoint, all 18 signers appear in mixed val, and S04/S08 contribute 5,975 training clips.

| Model | Mixed per seed (s0/s1/s2) | **Mixed** | **SMG = Mixed − TEST** | mixed-TEST per seed (secondary) | mixed-TEST |
|---|---|--:|--:|---|--:|
| CTR-GCN | 98.51 / 98.32 / 98.62 | **98.48 ± 0.12** | **+11.51** | 98.71 / 98.77 / 98.87 | 98.78 ± 0.07 |
| ST-GCN | 98.18 / 98.55 / 98.39 | **98.37 ± 0.15** | **+11.50** | 98.29 / 98.46 / 98.52 | 98.42 ± 0.10 |
| BlockGCN | 96.14 / 95.29 / 79.30 | **90.24 ± 7.75** | **+28.71** | 96.36 / 95.29 / 79.05 | 90.23 ± 7.92 |

All 9 runs ran the full 100 epochs (20 eval rows each) with no requeues.

**BlockGCN seed 2 trained unstably**, though without NaN. It peaked at 79.3 at epoch 39, crashed to 14.3 at epoch 54, and recovered only to 76.7. The same high variance shows on the SI side, where the seeds score 52.9 / 51.2 / 80.5. With the other two seeds alone, Mixed would be 95.7, so the BlockGCN SMG is fragile and should carry the ± caveat.

The CTR-GCN and ST-GCN SMG of about +11.5 agrees with the independent seed-0 SMG(k) value of +12.2 at k=16.

**Seed-0 cross-check (consistent).** In the SMG(k) grid, CTR-GCN at k=16 (draw 0, seed 0) scores 98.43 on the seen-signer holdout and 86.19 on unseen S04/S08, a gap of +12.2 pp.
