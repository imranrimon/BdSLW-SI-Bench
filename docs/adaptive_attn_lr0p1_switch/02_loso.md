# 2 — LOSO, 11 held-out signers (Table 7 LOSO column, sec:archtable signer-clustered ranking)

Held-out-signer Top-1 per fold (each fold's best-on-fold-val checkpoint scored on the fold's held-out TEST signer, as `scripts/hpc/slurm_loso_heldout_allarch.sbatch`); fold value = mean over seeds; stats over the 11 folds; 95% CI = signer-clustered percentile bootstrap over folds (10k, seed 1234; the estimator of docs/baselines_rerun/item3_loso_all8.md and the BlockGCN "74.4±16.1 [64.7, 82.9]"); SD = sample SD across folds (Table 7 LOSO column = mean ± this SD). New per-fold CSVs: `docs/loso_heldout/{adaptive_gnn,gnn}_lr0p1.csv` (old: `{adaptive_gnn,gnn}.csv`, untouched).

| arch | lr | mean | 95% CI | min–max | SD (sample) | folds | seed-runs | Table 7 cell |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| Adaptive GNN | 0.01 (old) | 80.23 | [71.07, 88.03] | 49.6–94.4 | 14.94 | 11 | 33 | 80.2±14.9 |
| Adaptive GNN | **0.1 (new)** | -- | | | | | | |
| Attention GNN | 0.01 (old) | 75.74 | [65.80, 84.35] | 41.7–93.4 | 16.44 | 11 | 33 | 75.7±16.4 |
| Attention GNN | **0.1 (new)** | -- | | | | | | |

Per fold (seed 0/1/2), folds sorted by signer:

- Adaptive GNN 0.01 (old): U01: 52.0/48.8/48.1; U02: 81.2/81.0/79.3; U04: 77.2/82.9/78.2; U05: 94.5/94.3/94.3; U06: 91.2/88.5/92.8; U08: 90.9/89.5/87.6; U09: 91.5/92.2/90.8; U11: 88.7/88.1/87.4; U12: 57.2/55.9/57.0; U13: 72.1/73.8/70.8; U15: 90.1/89.8/89.9
- Attention GNN 0.01 (old): U01: 39.7/43.0/42.6; U02: 72.8/78.6/74.2; U04: 77.7/76.3/74.4; U05: 93.0/94.2/93.2; U06: 81.1/86.8/85.6; U08: 84.8/86.4/88.7; U09: 89.9/90.2/92.5; U11: 84.4/87.5/85.6; U12: 51.1/53.0/55.0; U13: 64.4/65.2/61.2; U15: 81.1/82.6/82.6
