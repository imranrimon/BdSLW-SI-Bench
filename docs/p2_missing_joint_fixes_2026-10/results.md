# P2 missing-joint fixes — generated results (tools/p2_mjfix_report.py)

## A. Masked bone stream and joint+bone fusion

Bone = child − parent along graph.sign_27 inward edges; masked version zeroes a bone whose child OR parent is undetected (only the two hand-root bones change: body wrist → hand wrist, 10.0% of missing-joint entries in valid frames; median |old bone| there 1.17–1.31 vs 0.11 for real bones). Best-on-val checkpoint scored on held-out TEST (signers 2/13, 1,365 clips). Fusion s = s_joint + α·s_bone on logits; α=1 is P2's rule (fixed, not tuned); 'α val' picks α∈{0,0.05,…,3} maximizing the 3-seed-ensemble VAL (U15) accuracy and applies it to TEST.

| arch | joint TEST | bone TEST old (unmasked) | bone TEST new (masked) | P2 bone |
|---|--:|--:|--:|--:|
| BlockGCN (lr 0.01, P2 recipe) | 68.06 / 69.01 / 69.38 → **68.82 ± 0.56** | 67.69 / 65.20 / 63.74 → **65.54 ± 1.63** | 65.79 / 65.93 / 66.15 → **65.96 ± 0.15** | 65.54 ± 1.63 |
| Adaptive GNN (P2 recipe: lr 0.01, clip 5) | 77.88 / 80.00 / 79.34 → **79.07 ± 0.89** | 30.77 / 32.23 / 31.28 → **31.43 ± 0.61** | 76.85 / 78.17 / 73.63 → **76.21 ± 1.91** | 31.43 ± 0.60 |
| Adaptive GNN (lr 0.1, candidate recipe) | 77.73 / 77.66 / 75.75 → **77.05 ± 0.92** | 25.05 / 24.62 / 22.34 → **24.00 ± 1.19** | 79.63 / 81.03 / 79.41 → **80.02 ± 0.71** | (in flight) |

Fusion (3-seed ensembles, paired clip bootstrap over 1,365 TEST clips):

| arch | bone | joint ens | bone ens | fused α=1 | fused−joint α=1 | per-seed fused α=1 (mean ± SD) | α (val) | fused α_val TEST | fused−joint α_val |
|---|---|--:|--:|--:|---|--:|--:|--:|---|
| BlockGCN (lr 0.01, P2 recipe) | old (unmasked) | 71.72 | 69.16 | 78.68 | +6.96 pp [+5.05, +8.86], p=0.0000 | 77.05 ± 0.85 | 1.45 (val 88.40) | 78.02 | +6.30 pp [+4.32, +8.35], p=0.0000 |
| BlockGCN (lr 0.01, P2 recipe) | new (masked) | 71.72 | 70.26 | 78.10 | +6.37 pp [+4.40, +8.35], p=0.0000 | 75.85 ± 0.78 | 1.45 (val 88.40) | 77.07 | +5.35 pp [+3.30, +7.40], p=0.0000 |
| BlockGCN (lr 0.01, P2 recipe) | masked − unmasked bone (ensembles) | | | | +1.10 pp [-0.44, +2.64], p=0.1738 | | | | |
| Adaptive GNN (P2 recipe: lr 0.01, clip 5) | old (unmasked) | 81.90 | 32.23 | 67.11 | -14.80 pp [-16.92, -12.75], p=0.0000 | 64.74 ± 1.23 | 0.30 (val 90.23) | 79.12 | -2.78 pp [-4.10, -1.47], p=0.0000 |
| Adaptive GNN (P2 recipe: lr 0.01, clip 5) | new (masked) | 81.90 | 79.27 | 84.98 | +3.08 pp [+1.61, +4.54], p=0.0000 | 83.13 ± 0.86 | 0.50 (val 94.50) | 85.05 | +3.15 pp [+1.98, +4.40], p=0.0000 |
| Adaptive GNN (P2 recipe: lr 0.01, clip 5) | masked − unmasked bone (ensembles) | | | | +47.03 pp [+44.18, +49.89], p=0.0000 | | | | |
| Adaptive GNN (lr 0.1, candidate recipe) | old (unmasked) | 78.90 | 25.49 | 60.07 | -18.83 pp [-21.10, -16.63], p=0.0000 | 59.19 ± 0.63 | 0.10 (val 91.45) | 78.90 | +0.00 pp [-0.66, +0.66], p=1.0000 |
| Adaptive GNN (lr 0.1, candidate recipe) | new (masked) | 78.90 | 83.22 | 85.27 | +6.37 pp [+4.76, +8.06], p=0.0000 | 84.13 ± 1.36 | 0.50 (val 95.42) | 84.98 | +6.08 pp [+4.69, +7.55], p=0.0000 |
| Adaptive GNN (lr 0.1, candidate recipe) | masked − unmasked bone (ensembles) | | | | +57.73 pp [+54.95, +60.44], p=0.0000 | | | | |

P2 (BlockGCN): bone 65.54 ± 1.63; joint 71.72 → fused 78.75 ensemble, fused − joint +7.03 [+5.13, +8.94].

## B. Masked cue perturbations (BlockGCN lr 0.01, final-epoch held-out TEST, P2 estimator)

| cue | per seed | mean ± SD | Δ vs control | P2 |
|---|---|--:|--:|--:|
| none (control, final epoch) | 68.13 / 68.57 / 69.96 | 68.89 ± 0.78 | -- | 68.9 ± 0.8 |
| speed-normalized (P2, unmasked) | 56.85 / 56.92 / 54.14 | **55.97 ± 1.29** | -12.92 | 56.0 ± 1.3 |
| speed-normalized, MASKED | 70.92 / 71.72 / 69.52 | **70.72 ± 0.91** | +1.83 |  |
| root-centred (P2, unmasked) | 69.60 / 69.30 / 71.87 | **70.26 ± 1.15** | +1.37 | 70.3 ± 1.1 |
| root-centred, MASKED | 71.87 / 69.45 / 73.41 | **71.58 ± 1.63** | +2.69 |  |

Paired clip bootstrap on the 3-seed final-epoch ensembles (TEST):

- speed masked − speed unmasked: ensemble 75.31 vs 59.12: +16.19 pp [+13.92, +18.46], p=0.0000
- speed masked − control: ensemble 75.31 vs 71.43: +3.88 pp [+1.83, +5.93], p=0.0004
- speed unmasked − control: ensemble 59.12 vs 71.43: -12.31 pp [-14.73, -9.89], p=0.0000
- root masked − root unmasked: ensemble 74.21 vs 73.26: +0.95 pp [-1.10, +3.00], p=0.3822
- root masked − control: ensemble 74.21 vs 71.43: +2.78 pp [+0.59, +4.91], p=0.0120
- root unmasked − control: ensemble 73.26 vs 71.43: +1.83 pp [-0.07, +3.74], p=0.0632

## C. Signer-decodability probe v2 (tools/decodability_probe_v2.py)

v2 = feeder-processed test-time input (120-frame centre crop, root-normalised), held-out clips (SI TEST + labelled pretrain pool: n=2904, S=9 signers, none seen in any supervised training), recording-grouped StratifiedGroupKFold, mean of 3 CV seeds. v1 = the P2 probe protocol (raw 300-frame arrays of the first 4000 SI-TRAIN clips, ungrouped CV) re-run here for the same rows. wc is ungrouped by construction (one recording per signer×class). Δ = v2 cc minus the raw-input baseline (unmasked / masked joint means).

| encoder | flag | v2 cc | v2 marginal | v2 wc* | v2 cc ungrouped | Δcc vs raw unmasked / masked | v2+val cc | v1 cc / mrg / wc | P2 (cc / wc) |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| rawmean_unmasked | reference | 0.524 | 0.470 | 0.960 | 0.764 | +0.000 / -0.043 | 0.520 | 0.802 / 0.669 / 0.947 |  |
| rawmean_masked | reference | 0.567 | 0.502 | 0.976 | 0.823 | +0.043 / +0.000 | 0.571 | 0.788 / 0.725 / 0.996 |  |
| missrate_len | reference | 0.072 | 0.067 | 0.665 | 0.087 | -0.452 / -0.496 | 0.073 | 0.234 / 0.175 / 0.606 |  |
| randinit_blockgcn | reference | 0.307 | 0.224 | 0.930 | 0.657 | -0.217 / -0.261 | 0.297 | 0.559 / 0.456 / 0.839 |  |
| randinit_adaptive | reference | 0.411 | 0.347 | 0.959 | 0.827 | -0.113 / -0.156 | 0.444 | 0.798 / 0.687 / 0.953 |  |
| sup_blockgcn_lr01_s0 |  | 0.236 | 0.117 | 0.980 | 0.885 | -0.287 / -0.331 | 0.248 | 0.825 / 0.783 / 0.970 | 0.853 ± 0.021 / 0.975 ± 0.002 (3-seed) |
| sup_blockgcn_lr01_s1 |  | 0.276 | 0.145 | 0.985 | 0.890 | -0.248 / -0.291 | 0.282 | 0.869 / 0.833 / 0.978 | 0.853 ± 0.021 / 0.975 ± 0.002 (3-seed) |
| sup_blockgcn_lr01_s2 |  | 0.260 | 0.132 | 0.982 | 0.863 | -0.264 / -0.307 | 0.272 | 0.863 / 0.819 / 0.974 | 0.853 ± 0.021 / 0.975 ± 0.002 (3-seed) |
| sup_adaptive_lr0p01_s0_m75 | P2 row | 0.320 | 0.173 | 0.990 | 0.924 | -0.204 / -0.248 | 0.350 | 0.909 / 0.891 / 0.985 | 0.911 / 0.986 |
| sup_adaptive_lr0p01_s0_m59 |  | 0.320 | 0.173 | 0.988 | 0.926 | -0.203 / -0.247 | 0.352 | 0.908 / 0.891 / 0.985 |  |
| sup_adaptive_lr0p01_s1_m59 |  | 0.319 | 0.192 | 0.988 | 0.932 | -0.205 / -0.249 | 0.337 | 0.897 / 0.883 / 0.983 |  |
| sup_adaptive_lr0p01_s2_m89 |  | 0.316 | 0.169 | 0.988 | 0.927 | -0.208 / -0.251 | 0.306 | 0.892 / 0.869 / 0.981 |  |
| sup_adaptive_lr0p1_s0 |  | 0.355 | 0.208 | 0.989 | 0.930 | -0.168 / -0.212 | 0.358 | 0.889 / 0.878 / 0.986 |  |
| sup_adaptive_lr0p1_s1 |  | 0.348 | 0.219 | 0.987 | 0.935 | -0.176 / -0.219 | 0.317 | 0.908 / 0.895 / 0.987 |  |
| sup_adaptive_lr0p1_s2 |  | 0.353 | 0.201 | 0.989 | 0.923 | -0.171 / -0.215 | 0.356 | 0.888 / 0.871 / 0.982 |  |
| ssl_bdsl_only |  | 0.228 | 0.170 | 0.934 | 0.729 | -0.296 / -0.339 | 0.213 | 0.914 / 0.891 / 0.978 | 0.913 / 0.979 |
| grl_lr01_l01 |  | 0.212 | 0.114 | 0.985 | 0.829 | -0.312 / -0.355 | 0.237 | 0.798 / 0.774 / 0.970 | 0.804 (cc) |
| grl_lr01_l05 |  | 0.278 | 0.169 | 0.976 | 0.876 | -0.246 / -0.290 | 0.280 | 0.831 / 0.811 / 0.976 | 0.832 (cc) |
| grl_lr01_l10 |  | 0.242 | 0.141 | 0.984 | 0.872 | -0.281 / -0.325 | 0.253 | 0.834 / 0.829 / 0.976 | 0.832 (cc) |
| ssl_xlingual | FLAG: WLASL-scrambled pool | 0.258 | 0.214 | 0.959 | 0.842 | -0.266 / -0.309 | 0.262 | 0.876 / 0.838 / 0.965 | 0.871 / 0.967 |
| ssl_dgxl | FLAG: WLASL-scrambled pool | 0.358 | 0.266 | 0.964 | 0.868 | -0.166 / -0.209 | 0.345 | 0.821 / 0.778 / 0.957 | 0.824 / 0.957 |

Seed summaries (v2): supervised BlockGCN lr0.01 cc 0.258 ± 0.016, wc 0.982 ± 0.002; Adaptive lr0.01 best ckpts (m59/m59/m89) cc 0.318 ± 0.002; Adaptive lr0.1 cc 0.352 ± 0.003.

*wc: within-class signer probe; ungrouped by necessity (each signer has one recording per class), so it measures same-recording trial similarity, not identity transfer to new recordings.

## D. Validation-selected matched-split SMG (Adaptive GNN, lr 0.01 = P2 recipe, seed 0)

| split | P2: best epoch on own TEST | new run: val-selected → TEST | new run: best on own TEST (all ckpts) | new run: final epoch TEST | new run: best val (epoch) |
|---|--:|--:|--:|--:|--:|
| rand42 | 99.19 | 99.41 | 99.49 | 99.34 | 99.24 (ep 49) |
| rand43 | 99.34 | 99.05 | 99.27 | 99.19 | 99.85 (ep 73) |
| rand44 | 98.61 | 98.75 | 98.90 | 98.68 | 99.08 (ep 73) |
| rand45 | 98.83 | 98.83 | 98.83 | 98.68 | 99.24 (ep 55) |
| rand46 | 98.90 | 99.05 | 99.12 | 99.05 | 99.24 (ep 81) |
| **mean [min, max]** | 98.97 [98.61, 99.34] | **99.02 [98.75, 99.41] (n=5)** | 99.12 [98.83, 99.49] (n=5) | 98.99 [98.68, 99.34] (n=5) | |

SMG vs held-out TEST 79.07 (Adaptive GNN lr 0.01, 3-seed mean): P2 (test-selected) 98.97 − 79.07 = +19.90; **val-selected 99.02 − 79.07 = +19.95 pp**.

## E. Paired-instance SMG decomposition: signer familiarity vs same-recording familiarity

BlockGCN lr 0.01 (P2 paired protocol: select on U15, score best ckpt once on the frozen focus-signer clips; seeds 0/1/2). `seenother` = the focus signer's clips of 30 classes in training (matched swap, |train| = 5,748), tested on the frozen clips of the OTHER 30 classes; halves A and B together cover exactly the frozen set. Accuracies are on the identical frozen clips for all three conditions.

| focus | seed | unseen | seen via other classes | seen_matched (same class/recording) | signer fam. (other−unseen) | same-recording fam. (matched−other) | paired gap (matched−unseen) |
|---|---|--:|--:|--:|--:|--:|--:|
| U02 | 0 | 69.25 | 71.58 | 97.16 | +2.33 | +25.58 | +27.91 |
| U02 | 1 | 68.99 | 69.77 | 96.38 | +0.78 | +26.61 | +27.39 |
| U02 | 2 | 73.39 | 73.13 | 94.32 | -0.26 | +21.19 | +20.93 |
| **U02 mean ± SD** | | 70.54 ± 2.01 | 71.49 ± 1.37 | 95.95 ± 1.20 | 0.95 ± 1.06 | 24.46 ± 2.35 | 25.41 ± 3.17 |
| U13 | 0 | 65.53 | 62.73 | 97.83 | -2.80 | +35.09 | +32.30 |
| U13 | 1 | 55.28 | 63.98 | 95.96 | +8.70 | +31.99 | +40.68 |
| U13 | 2 | 64.91 | 63.98 | 97.83 | -0.93 | +33.85 | +32.92 |
| **U13 mean ± SD** | | 61.90 ± 4.69 | 63.56 ± 0.59 | 97.20 ± 0.88 | 1.66 ± 5.04 | 33.64 ± 1.28 | 35.30 ± 3.81 |

Per-clip paired bootstrap on the 3-seed-mean correctness of each frozen clip:

- U02: signer familiarity (other − unseen) +0.95 pp [-1.81, +3.53], p=0.5180; same-recording (matched − other) +24.46 pp [+20.84, +28.17], p=0.0000; total paired gap +25.41 pp [+21.62, +29.03], p=0.0000
- U13: signer familiarity (other − unseen) +1.66 pp [-1.04, +4.35], p=0.2374; same-recording (matched − other) +33.64 pp [+28.99, +38.30], p=0.0000; total paired gap +35.30 pp [+30.75, +39.86], p=0.0000
- pooled U02+U13 (709 clips): signer familiarity +1.27 pp [-0.61, +3.15], p=0.1974; same-recording +28.63 pp [+25.81, +31.55], p=0.0000; total +29.90 pp [+27.03, +32.82], p=0.0000; share of the paired gap explained by signer familiarity alone: 4% (clips pooled; per-signer means are the independent unit, n=2)

P2: paired gap +30.36 ± 6.06 (U02 +25.41 ± 3.17, U13 +35.30 ± 3.81).

