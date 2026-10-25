# 2 — Signer-mixed split (Mixed column, tab:smg8 / Table 7)

Signer-mixed `data/bdsl_sd`, BlockGCN. Current P2 protocol: checkpoint selected on sd val, scored on sd test (docs/audit_reruns_2026-10/r5_mixed_selection.md). Old lr-0.1 sd-test values from R5; old sd-val-best from CSV.

| | per seed | mean ± SD |
|---|---|--:|
| old lr 0.1, Mixed (sd test, R5) | 97.44 / 97.51 / 97.81 | 97.59 ± 0.16 |
| old lr 0.1, sd val best (pre-R5 estimator) | 99.08 / 97.71 / 98.63 | 98.47 ± 0.57 |
| **new lr 0.01, Mixed (sd test)** | 97.88 / 98.02 / 98.02 | **97.98 ± 0.07** |
| new lr 0.01, sd val best | 99.08 / 99.08 / 99.24 | 99.13 ± 0.07 |

SMG = Mixed − TEST: old 97.59 − 65.23 = +32.36; **new 97.98 − 68.82 = +29.16**.
