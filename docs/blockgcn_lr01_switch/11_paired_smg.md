# 11 — Paired-instance SMG (§smg (ii), fragility table)

Paired-instance SMG (tools/run_paired_smg.py): TEST Top-1 of each condition on the identical frozen clips of the focus signer; gap = seen_matched − unseen.

| | old lr 0.1 per seed | old mean ± SD | new lr 0.01 per seed | new mean ± SD |
|---|---|--:|---|--:|
| U02 unseen | 64.86 / 60.98 / 65.37 | 63.74 ± 1.96 | 69.25 / 68.99 / 73.39 | **70.54 ± 2.02** |
| U02 seen_matched | 76.49 / 93.80 / 90.96 | 87.08 ± 7.58 | 97.16 / 96.38 / 94.32 | **95.95 ± 1.20** |
| U02 seen_aug | 92.51 / 87.60 / 96.38 | 92.16 ± 3.59 | 96.12 / 96.38 / 96.12 | **96.21 ± 0.12** |
| U13 unseen | 61.18 / 52.80 / 58.70 | 57.56 ± 3.51 | 65.53 / 55.28 / 64.91 | **61.91 ± 4.69** |
| U13 seen_matched | 80.75 / 88.20 / 97.20 | 88.72 ± 6.73 | 97.83 / 95.96 / 97.83 | **97.21 ± 0.88** |
| U13 seen_aug | 95.96 / 95.65 / 97.83 | 96.48 ± 0.96 | 97.20 / 98.14 / 98.14 | **97.83 ± 0.44** |

old: paired gap U02 23.35 ± 8.80, U13 31.16 ± 8.29, overall 27.25 ± 9.40 (n=6)

new: paired gap U02 25.41 ± 3.17, U13 35.30 ± 3.81, overall 30.36 ± 6.06 (n=6)
