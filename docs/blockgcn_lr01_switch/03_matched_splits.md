# 3 — Matched random splits (headline SMG, tab:smg, Fig. 1)

Five matched random signer-mixed splits `data/bdsl_rand42..46` (one model per split, seed 0). New runs select the checkpoint on each split's own val (`config/lr01_switch/bdsl_block_gcn_rand4X_lr01.yaml`). Old estimator = best epoch on the split's own test (old runs evaluated test every 5 epochs; for the new runs it is recomputed as the max over all saved checkpoints, a slightly larger set).

| split | old lr 0.1 best-on-own-test | new lr 0.01 best-on-own-test (all ckpts) | new lr 0.01 val-selected → test |
|---|--:|--:|--:|
| rand42 | 97.29 | 98.39 | 98.24 |
| rand43 | 94.51 | 98.32 | 98.17 |
| rand44 | 97.51 | 98.1 | 98.02 |
| rand45 | 90.18 | 98.17 | 98.17 |
| rand46 | 96.41 | 98.17 | 98.02 |
| **mean [range]** | 95.18 [90.2, 97.5] | 98.23 [98.1, 98.4] | **98.12 [98.0, 98.2]** |

Headline SMG (tab:smg, abstract): old 95.18 − 65.23 = +29.95.
New, same estimator as the paper: 98.23 − 68.82 = **+29.41**; val-selected: 98.12 − 68.82 = +29.30.
