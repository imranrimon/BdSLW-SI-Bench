# 7 — Controlled cue perturbations (§cues table)

Final-epoch Top-1 on the held-out TEST signers 2/13 (the cue configs' test feeder), as P2's cue table. Control = unperturbed BlockGCN final-epoch checkpoint on TEST (old: 64.25/62.34, seed 0 final ckpt pruned).

| | old lr 0.1 per seed | old mean ± SD | new lr 0.01 per seed | new mean ± SD |
|---|---|--:|---|--:|
| none (control) | -- / 64.25 / 62.34 | 63.30 ± 0.96 | 68.13 / 68.50 / 69.96 | **68.86 ± 0.79** |
| static | 32.01 / 34.43 / 34.73 | 33.72 ± 1.22 | 44.25 / 43.81 / 44.32 | **44.13 ± 0.23** |
| speednorm | 24.32 / 46.45 / 48.06 | 39.61 ± 10.83 | 56.85 / 56.92 / 54.14 | **55.97 ± 1.29** |
| shuffle | 44.69 / 57.51 / 45.42 | 49.21 ± 5.88 | 59.12 / 58.90 / 62.86 | **60.29 ± 1.82** |
| bodyonly | 48.72 / 45.42 / 5.79 | 33.31 ± 19.51 | 47.62 / 47.40 / 49.96 | **48.33 ± 1.16** |
| handsonly | 8.50 / 57.66 / 65.86 | 44.01 ± 25.33 | 75.97 / 78.46 / 76.34 | **76.92 ± 1.10** |
| rootcenter | 64.62 / 69.82 / 68.50 | 67.65 ± 2.21 | 69.60 / 69.30 / 71.87 | **70.26 ± 1.15** |
