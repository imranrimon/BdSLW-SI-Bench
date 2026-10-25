# 4 — LOSO, 11 held-out signers (tab:losofolds, Fig. 2, variance decomposition, Table 7 LOSO)

Held-out-signer Top-1 per fold (fold value = mean over seeds); stats over the 11 folds; CI = signer-clustered percentile bootstrap over folds (10k, seed 1234), as docs/baselines_rerun/item3_loso_all8.md. New per-fold CSV: docs/loso_heldout/block_gcn_lr01.csv.

| | mean | 95% CI | min–max | SD (sample) | folds | seed-runs |
|---|--:|--:|--:|--:|--:|--:|
| old lr 0.1 | 65.09 | [54.94, 74.31] | 28.8–86.6 | 17.17 | 11 | 33 |
| **new lr 0.01** | 74.37 | [64.72, 82.91] | 42.6–91.3 | 16.07 | 11 | 33 |

Per fold (seed 0/1/2):

- old lr 0.1: U01: 36.4/18.3/31.6; U02: 70.2/66.1/69.0; U04: 70.3/54.3/56.8; U05: 64.8/81.2/59.0; U06: 28.6/79.5/83.9; U08: 86.5/84.3/88.9; U09: 65.5/89.0/90.4; U11: 85.0/85.2/87.4; U12: 42.6/44.7/48.4; U13: 63.9/53.0/64.4; U15: 71.8/57.9/68.8
- **new lr 0.01**: U01: 41.5/44.5/41.9; U02: 71.8/74.5/76.6; U04: 60.9/72.4/74.4; U05: 91.8/90.7/91.3; U06: 86.1/81.0/86.6; U08: 88.1/85.8/85.8; U09: 84.0/89.0/90.8; U11: 90.8/85.5/79.6; U12: 50.1/53.3/49.9; U13: 61.2/67.3/62.6; U15: 79.4/78.5/86.4
