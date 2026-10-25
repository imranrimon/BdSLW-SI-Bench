# Item 2 — SPOTER-family baselines under Protocol A + signer-mixed split

Protocol A (train {1,4,5,6,8,9,11,12}, val U15, TEST {2,13}); 27-kpt pose, T=120, mirror OFF.
Shared-recipe rows: SGD+nesterov, wd 1e-4, warm-up 5, step [60,80], 100 ep, batch 16, **no grad-clip** (grad_clip_norm unset = 0.0, matching the headline Protocol-A/LOSO runs); base LR selected on val U15 at seed 0 from {0.1, 0.05, 0.01, 0.001}. Native row: BdSL-SPOTER's released recipe (tools/train_spoter_native.py; grad-clip 1.0 is part of that recipe). TEST = best-val-epoch checkpoint on held-out signers 2/13. Mixed = best-on-sd-val checkpoint scored on data/bdsl_sd/test (1,367 clips; P2 symmetric-selection protocol, docs/audit_reruns_2026-10/r5_mixed_selection.md). SMG = Mixed − TEST, per seed. Secondary: old Mixed = best-epoch Top-1 on sd val (selected on the reporting set) and its SMG. ± = population SD over 3 seeds. Macro-F1 = single-model mean (tools/p2_metric_suite.py estimator).

| Method | base LR | Params (M) | GFLOPs (thop) | TEST Top-1 | Macro-F1 | Mixed (sd test) | SMG (pp) | old Mixed (sd val) | old SMG |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| BdSL-SPOTER (shared recipe) | 0.001 | 0.70 | 0.055 | 59.22 ± 2.41 | 57.66 ± 2.17 | 96.44 ± 0.15 | 37.22 ± 2.56 | 97.00 ± 0.19 | 37.78 ± 2.42 |
| SPOTER, original (shared recipe) | 0.001 | 4.34 | 0.241 | 61.98 ± 2.31 | 60.56 ± 2.69 | 97.37 ± 0.10 | 35.39 ± 2.39 | 98.17 ± 0.12 | 36.19 ± 2.40 |
| BdSL-SPOTER (native released recipe) | AdamW 3e-4 / OneCycle max 1e-3 | 1.33 | 0.109 | 61.37 ± 2.22 | 59.49 ± 2.46 | 97.61 ± 0.54 | 36.24 ± 2.65 | 97.76 ± 0.50 | 36.39 ± 2.07 |

## Per-seed values (seeds 0/1/2)

- **BdSL-SPOTER (shared recipe)** — TEST Top-1 55.90/61.54/60.22; Macro-F1 54.73/59.90/58.34; Mixed (sd test) 96.63/96.27/96.42; SMG 40.74/34.73/36.20; old Mixed (sd val) 96.95/96.79/97.25; old SMG 41.05/35.26/37.03
- **SPOTER, original (shared recipe)** — TEST Top-1 59.63/61.17/65.13; Macro-F1 57.63/59.93/64.13; Mixed (sd test) 97.51/97.29/97.29; SMG 37.88/36.12/32.17; old Mixed (sd val) 98.17/98.32/98.02; old SMG 38.53/37.15/32.89
- **BdSL-SPOTER (native released recipe)** — TEST Top-1 60.59/64.40/59.12; Macro-F1 58.76/62.79/56.91; Mixed (sd test) 98.24/96.93/97.66; SMG 37.66/32.53/38.54; old Mixed (sd val) 98.32/97.86/97.10; old SMG 37.73/33.47/37.98
