# 4 — Matched random splits, Adaptive GNN (tab:smg, Fig. 1, abstract, conclusion)

Five matched random signer-mixed splits `data/bdsl_rand42..46`, Adaptive GNN, one model per split (seed 0). Old (P2) = lr 0.01, test feeder = the split's own test, best of the every-5-epoch evals (epochs 4, 9, …, 99). New = lr 0.1 (`config/lr0p1_switch/bdsl_adaptive_gnn_rand4X_lr0p1.yaml`), selected on the split's own val as in the BlockGCN lr-0.01 switch, with checkpoints saved at exactly the old eval epochs (save_interval 5), so the paper's estimator (best epoch on own test) is recomputed on the same epoch set.

| split | old lr 0.01 best-on-own-test | new lr 0.1 best-on-own-test (same 20 epochs) | new val-selected → test | new val best |
|---|--:|--:|--:|--:|
| rand42 | 99.19 | -- | -- | -- |
| rand43 | 99.34 | -- | -- | -- |
| rand44 | 98.61 | -- | -- | -- |
| rand45 | 98.83 | -- | -- | -- |
| rand46 | 98.90 | -- | -- | -- |
| **mean [min, max]** | 98.97 [98.6, 99.3] | **--** | -- | |

Headline Adaptive GNN SMG (tab:smg, Fig. 1, abstract): old 98.97 − 79.07 = +19.90 (P2: +19.90).
