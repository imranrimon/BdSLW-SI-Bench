# Item 2 — BdSLW401 SMG(k) on the signer-disjoint validation split

Old grid: `scripts/hpc/slurm_w401_diversity_grid.sbatch` (CTR-GCN `config/bdsl_ctr_gcn_w401.yaml`, lr 0.1, seed 0; 16-signer train pool; checkpoint selected on the official **seen-signer** val; data/bdslw401_si_div). New grid: `scripts/hpc/slurm_w401_sdval_smgk.sbatch` (job; first submission <job> cancelled after OOM from over-packing) + `tools/build_w401_sdval_divgrid.py` — identical recipe (`config/bdsl_ctr_gcn_w401_sdval.yaml`, only data paths differ; configs in `config/w401_sdval_div/`), but the training-signer pool is the 15 signers of `data/bdslw401_si_sdval` (S15 removed) and the best checkpoint is selected on **S15** (all 3,950 clips; held out from every subset). Unseen TEST = S04/S08 (7,833 clips, unchanged); seen TEST = the subset's 20% class-stratified holdout (same holdout seeding as before). SMG = seen − unseen Top-1 of the one best-val checkpoint. Bundles: `data/bdslw401_si_sdval_div/k<k>_d<r>`.

**Design changes forced by excluding S15** (document in the paper footnote):
- The largest k is **15** (full 15-signer pool), not 16 — k=16 is infeasible once S15 is the val signer.
- Draws are re-sampled from the 15-signer pool with the same `subset_for` seeding, so the per-k subsets differ from the old grid (the old pool included S15). Per-draw subsets are listed below.
- Each subset's clips now come from old train+val (the sdval train), so a signer contributes all its clips rather than its old-train share.

## SMG(k), mean ± SD over draws (pp)

| k (old / new) | old SMG | **new SMG** | Δ | old unseen TEST | new unseen TEST | old seen | new seen | old val (seen-signer) | new val (S15) | draws old/new |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| 2 / 2 | 83.23 ± 3.50 | **77.02 ± 8.70** | -6.20 | 14.30 ± 3.14 | 20.07 ± 9.57 | 97.52 ± 0.99 | 97.09 ± 1.63 | 21.19 ± 3.30 | 17.48 ± 7.27 | 5/5 |
| 4 / 4 | 57.39 ± 8.00 | **61.81 ± 6.53** | +4.42 | 40.19 ± 8.14 | 36.05 ± 6.58 | 97.58 ± 0.47 | 97.86 ± 0.22 | 51.47 ± 5.17 | 36.27 ± 8.79 | 5/5 |
| 8 / 8 | 32.92 ± 9.34 | **28.02 ± 5.27** | -4.89 | 65.57 ± 9.40 | 70.67 ± 5.34 | 98.49 ± 0.16 | 98.69 ± 0.11 | 76.57 ± 6.37 | 66.93 ± 4.75 | 5/5 |
| 12 / 12 | 17.09 ± 1.48 | **16.94 ± 1.25** | -0.15 | 81.50 ± 1.51 | 81.70 ± 1.28 | 98.59 ± 0.13 | 98.64 ± 0.19 | 92.73 ± 1.07 | 77.86 ± 1.79 | 5/5 |
| 16 / 15 | 12.24 | **14.31** | +2.07 | 86.19 | 84.34 | 98.43 | 98.65 | 99.04 | 82.61 | 1/1 |

Fig. 4b plot values (one decimal, as `tools/make_paper_figures.py` line ~176 expects: k list, mean SMG, SD):

- old: k=[2, 4, 8, 12, 16], SMG=[83.2, 57.4, 32.9, 17.1, 12.2], SD=[3.5, 8.0, 9.3, 1.5, 0.0]
- new: k=[2, 4, 8, 12, 15], SMG=[77.0, 61.8, 28.0, 16.9, 14.3], SD=[8.7, 6.5, 5.3, 1.3, 0.0]

## Per draw

| cell | old subset | old val | old seen | old unseen | old SMG | new subset | new val (S15) | best ep | new seen | new unseen | **new SMG** |
|---|---|--:|--:|--:|--:|---|--:|--:|--:|--:|--:|
| k2/2 d0 | [3, 16] | 16.09 | 98.55 | 10.19 | 88.36 | [3, 16] | 8.86 | 64 | 98.09 | 10.81 | **87.28** |
| k2/2 d1 | [11, 14] | 24.61 | 98.22 | 16.09 | 82.13 | [11, 13] | 26.89 | 74 | 98.85 | 28.99 | **69.86** |
| k2/2 d2 | [5, 10] | 20.14 | 95.94 | 17.30 | 78.64 | [5, 9] | 24.56 | 94 | 98.19 | 34.01 | **64.18** |
| k2/2 d3 | [11, 17] | 24.99 | 98.12 | 17.12 | 81.00 | [11, 17] | 16.71 | 44 | 94.64 | 14.89 | **79.75** |
| k2/2 d4 | [6, 10] | 20.12 | 96.79 | 10.79 | 86.00 | [6, 10] | 10.38 | 39 | 95.68 | 11.63 | **84.05** |
| k4/4 d0 | [1, 2, 7, 9] | 57.69 | 97.83 | 43.16 | 54.67 | [1, 2, 7, 9] | 44.84 | 74 | 98.15 | 43.87 | **54.28** |
| k4/4 d1 | [5, 10, 15, 16] | 47.71 | 97.50 | 41.87 | 55.63 | [5, 10, 14, 16] | 20.63 | 94 | 98.08 | 28.71 | **69.37** |
| k4/4 d2 | [2, 5, 9, 18] | 57.33 | 97.90 | 52.84 | 45.06 | [2, 5, 7, 18] | 41.14 | 94 | 97.68 | 41.50 | **56.18** |
| k4/4 d3 | [2, 12, 13, 18] | 49.74 | 96.70 | 33.78 | 62.92 | [2, 11, 12, 18] | 41.97 | 84 | 97.79 | 38.29 | **59.50** |
| k4/4 d4 | [2, 7, 14, 15] | 44.88 | 97.98 | 29.31 | 68.67 | [2, 7, 13, 14] | 32.76 | 99 | 97.58 | 27.87 | **69.71** |
| k8/8 d0 | [1, 5, 10, 11, 12, 13, 14, 17] | 82.07 | 98.46 | 72.37 | 26.09 | [1, 5, 9, 10, 11, 12, 13, 17] | 69.27 | 84 | 98.74 | 78.48 | **20.26** |
| k8/8 d1 | [2, 5, 11, 13, 15, 16, 17, 18] | 81.50 | 98.67 | 71.70 | 26.97 | [2, 3, 5, 11, 12, 14, 16, 17] | 69.54 | 74 | 98.59 | 70.61 | **27.98** |
| k8/8 d2 | [2, 3, 6, 9, 10, 12, 14, 16] | 77.53 | 98.23 | 64.34 | 33.89 | [2, 6, 7, 9, 11, 13, 16, 17] | 73.01 | 99 | 98.85 | 74.10 | **24.75** |
| k8/8 d3 | [3, 5, 6, 9, 10, 13, 15, 18] | 77.28 | 98.65 | 71.73 | 26.92 | [3, 5, 6, 7, 9, 12, 17, 18] | 62.53 | 99 | 98.73 | 66.99 | **31.74** |
| k8/8 d4 | [3, 6, 7, 10, 14, 15, 16, 17] | 64.46 | 98.45 | 47.73 | 50.72 | [3, 5, 6, 10, 13, 14, 16, 17] | 60.28 | 89 | 98.55 | 63.16 | **35.39** |
| k12/12 d0 | [1, 2, 3, 6, 7, 9, 10, 11, 13, 16, 17, 18] | 90.75 | 98.51 | 78.59 | 19.92 | [1, 2, 3, 6, 7, 9, 10, 11, 12, 14, 16, 17] | 76.58 | 99 | 98.82 | 79.96 | **18.86** |
| k12/12 d1 | [2, 3, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16] | 92.44 | 98.48 | 82.28 | 16.20 | [2, 3, 5, 6, 7, 10, 11, 12, 13, 14, 17, 18] | 77.27 | 99 | 98.40 | 80.62 | **17.78** |
| k12/12 d2 | [1, 2, 5, 6, 9, 11, 13, 14, 15, 16, 17, 18] | 93.44 | 98.62 | 82.92 | 15.70 | [1, 3, 5, 6, 7, 9, 11, 12, 13, 16, 17, 18] | 76.78 | 89 | 98.78 | 83.52 | **15.26** |
| k12/12 d3 | [1, 3, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16] | 93.57 | 98.83 | 81.94 | 16.89 | [1, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16] | 77.27 | 89 | 98.76 | 82.55 | **16.21** |
| k12/12 d4 | [1, 3, 5, 6, 11, 12, 13, 14, 15, 16, 17, 18] | 93.46 | 98.52 | 81.78 | 16.74 | [1, 2, 5, 6, 10, 11, 12, 13, 14, 16, 17, 18] | 81.39 | 74 | 98.42 | 81.83 | **16.59** |
| k16/15 d0 | [1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] | 99.04 | 98.43 | 86.19 | 12.24 | [1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17, 18] | 82.61 | 99 | 98.65 | 84.34 | **14.31** |

## Same-subset cells (isolates the val/selection change from the re-drawn subsets)

| cell | subset | old SMG | new SMG | Δ |
|---|---|--:|--:|--:|
| k2 d0 | [3, 16] | 88.36 | 87.28 | -1.08 |
| k2 d3 | [11, 17] | 81.00 | 79.75 | -1.25 |
| k2 d4 | [6, 10] | 86.00 | 84.05 | -1.95 |
| k4 d0 | [1, 2, 7, 9] | 54.67 | 54.28 | -0.39 |
| mean | | 77.51 | 76.34 | -1.17 |

## What changes in P2

- **Fig. 4b, BdSLW401 curve** (`tools/make_paper_figures.py` ~l.176): label `BdSLW401 (Bangla, 15)`; k `[2, 4, 8, 12, 15]`; SMG / SD from the 'new' line above.
- **tab:smgk, BdSLW401 row**: k range `2--15`; SMG at min-k and max-k = the new k=2 and k=15 means.
- **§div text** (\S near Fig. 4b): "83pp at k=2 ... with sixteen signers, drops to 12pp" → the new k=2 value and "with fifteen training signers, drops to" the new k=15 value. The footnote's "selects on the official VAL" no longer holds for BdSLW401: it selects on the held-out signer S15 (BdSLW60/AUTSL/LSFB/ASL-Citizen curves unchanged).
- Qualitative conclusion (monotone decay with signer count; BdSLW401 starts highest and ends near ~12–14pp) is unchanged; the k=4 point is now *higher* than old and k=2 lower, so the min-k endpoint shrinks by ~6pp and the max-k endpoint rises by ~2pp.
- Unseen-TEST accuracy at the full pool drops slightly (86.19 → 84.34) with one signer fewer and S15-based selection.
