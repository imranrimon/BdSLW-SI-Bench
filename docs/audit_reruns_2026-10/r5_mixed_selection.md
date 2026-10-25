# R5 — Symmetric selection for the signer-mixed ("Mixed") accuracy (P2 audit item 7)

Source: SLURM job. For each `work_dir/bdsl_<arch>_sd_seed<N>`, the `*_model-best.pt` checkpoint (best epoch on data/bdsl_sd/val, 655 clips) is scored on the held-out data/bdsl_sd/test (1367 clips, signers shared with train) — i.e. selected on val, reported on test, the same protocol as the SI TEST numbers. Old Mixed = best-epoch sd-val Top-1 (selected on the reporting set). SI TEST = paper tab:arch. SD = population.

| Arch | sd-val (re-eval of best ckpt) per seed | **Mixed (sd TEST) per seed** | Mixed (sd TEST) mean ± SD | macro-F1 (sd TEST) | old Mixed (sd val best) | SI TEST | old SMG | **new SMG** |
|---|---|---|--:|--:|--:|--:|--:|--:|
| BlockGCN | 99.08/97.71/98.63 | 97.44/97.51/97.81 | 97.59 ± 0.16 | 97.60 ± 0.15 | 98.47 | 65.23 | +33.24 | **+32.36** |
| Adaptive GNN | 98.93/98.93/98.93 | 98.90/98.39/98.98 | 98.76 ± 0.26 | 98.76 ± 0.26 | 98.93 | 79.07 | +19.86 | **+19.69** |
| ST-GCN | 98.93/98.78/99.08 | 98.76/98.76/98.68 | 98.73 ± 0.03 | 98.73 ± 0.04 | 98.93 | 74.51 | +24.42 | **+24.22** |
| Attention GNN | 98.02/97.56/98.78 | 97.95/98.17/98.24 | 98.12 ± 0.12 | 98.13 ± 0.12 | 98.12 | 73.97 | +24.15 | **+24.15** |
| SLGTFormer | 99.24/98.78/99.08 | 98.76/98.24/98.32 | 98.44 ± 0.23 | 98.46 ± 0.22 | 99.03 | 73.26 | +25.77 | **+25.18** |
| GNN+Bi-LSTM | 98.78/99.39/98.47 | 97.81/98.54/97.81 | 98.05 ± 0.34 | 98.06 ± 0.33 | 98.88 | 71.57 | +27.31 | **+26.48** |
| Pose-LSTM | 96.03/95.27/96.18 | 94.44/94.95/95.10 | 94.83 ± 0.28 | 94.88 ± 0.28 | 95.83 | 61.64 | +34.19 | **+33.19** |
| GNN+Transformer | 98.93/99.24/98.93 | 98.98/98.83/99.20 | 99.00 ± 0.15 | 98.99 ± 0.15 | 99.03 | 56.92 | +42.11 | **+42.08** |
