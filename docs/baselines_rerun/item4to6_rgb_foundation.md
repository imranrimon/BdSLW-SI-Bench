# Baseline items 4-6 — RGB + foundation baselines (P2 Table 7)

Protocol A held-out TEST signers {2,13}. **Mixed = the model selected on `data/bdsl_sd/val` scored on `data/bdsl_sd/test` (1,367 clips, signers shared with train)** — the P2 R5 protocol (docs/audit_reruns_2026-10/r5_mixed_selection.md). `Mixed (old)` = best-epoch sd-val Top-1, the superseded estimator, kept for reference. mean ± SD = population SD over seeds. SMG = Mixed − TEST (pp). Each model keeps its original recipe.

| Model | TEST Top-1 | Macro-F1 | Mixed (sd test) | SMG | Mixed (old, sd-val best) | seeds (TEST/Mixed) |
|---|--:|--:|--:|--:|--:|--:|
| VideoMAE | 80.93 ± 2.11 | 79.91 ± 1.63 | 99.27 ± 0.00 | +18.34 | 99.54 ± 0.12 | 3/3 |
| S3D | 76.46 ± 1.48 | 75.10 ± 1.49 | 99.78 ± 0.00 | +23.32 | 99.64 ± 0.07 | 3/3 |
| Uni-Sign | 83.61 ± 1.08 | 82.75 ± 1.32 | 99.07 ± 0.12 | +15.46 | 99.59 ± 0.07 | 3/3 |
| SHuBERT | 67.42 ± 0.19 | 66.42 ± 0.23 | 98.92 ± 0.03 | +31.50 | 99.54 ± 0.00 | 3/3 |

## Params / GFLOPs (fills Table 7 cost cells)

GFLOPs follow the paper's convention: raw `thop` count (multiply-accumulates of Linear/Conv layers; attention matmuls not counted), single clip.

| Model | Params (M) | GFLOPs | What is counted |
|---|--:|--:|---|
| S3D | 7.97 | 36.3 | full network, thop, 1×3×32×224² training clip |
| Uni-Sign (fine-tuned) | 587.7 | 9.9 | pose encoder 5.3 M + mT5-base 582.4 M (incl. 250k-vocab embeddings); thop, one teacher-forced forward on the median SI-test clip (39 frames); excludes RTMPose extraction and autoregressive decode steps |
| SHuBERT (frozen probe) | 87.4 | 3.3 | SHuBERT encoder at inference (87.33 M; pretraining heads excluded) + linear probe (0.05 M); analytic thop-equivalent at the median 38 feature frames; EXCLUDES the two frozen DINOv2 ViT-S/14 crop encoders (+43.2 M params, ≈+629 GFLOPs for 3 crops × 38 frames) and MediaPipe/YOLO |

## Provenance and caveats

- **VideoMAE SI is a full 3-seed re-run** (original recipe: 20 ep, bs 6, lr 5e-5; job) because the original runs saved no predictions. New per-seed TEST 78.10/83.15/81.54 = 80.93 ± 2.11 vs the paper's 78.76 ± 3.78 (82.64/73.63/80.00). Same seeds do NOT reproduce per-seed (non-deterministic decode/augment workers + cuDNN); the means agree within one seed-SD. Macro-F1 is from the re-run.
- **S3D: the paper's 77.11 ± 1.42 is a 2-seed mean** (seeds 0,1; Windows runs). Seed 2 never reached its TEST eval (training stopped near epoch 19 of 30), so it was re-trained in full on the cluster (`rgb_s3d_bdsl_si_seed2r`, same recipe, clip root `data/bdslw60_clips`) → 75.16. 3-seed = 76.46 ± 1.48.
- **Uni-Sign: the paper's 82.34 is one seed** (`--seed` default 42). Seeds 1, 2 added (identical low-data recipe, dev-selected on U15, job) → 3-seed 83.61 ± 1.08. NB: this now sits ~0.3 pp below Track-B SLGTFormer (83.91).
- **SHuBERT**: frozen features are fixed; seeds vary only the probe init. Seed 0 reproduces 67.60 exactly. Features exist for 9,264/9,307 clips → SI TEST n = 1,364/1,365; sd test n = 1,359/1,367 (Mixed computed on the clips that have features).
- **Mixed footing** = the pose `data/bdsl_sd` clip names (train 5,746 / val 655 / test 1,367; 11 signers shared across splits), mapped to the per-trial RGB clips / Uni-Sign poses by name. VideoMAE and S3D sd-test errors are 10 and 3 clips per seed respectively (different clips per seed, same count at ceiling).
- Mixed is near-ceiling for every model (98.9–99.8), so SMG differences are driven by the TEST column.


## Per-seed values

**VideoMAE**
- seed 0: top1 78.10, f1 78.08, mixed_val 99.54, mixed 99.27, n 1365
- seed 1: top1 83.15, f1 82.04, mixed_val 99.39, mixed 99.27, n 1365
- seed 2: top1 81.54, f1 79.61, mixed_val 99.69, mixed 99.27, n 1365

**S3D**
- seed 0: top1 78.53, f1 77.16, mixed_val 99.54, mixed 99.78, n 1365
- seed 1: top1 75.68, f1 74.46, mixed_val 99.69, mixed 99.78, n 1365
- seed 2: top1 75.16, f1 73.69, mixed_val 99.69, mixed 99.78, n 1365

**Uni-Sign**
- seed 42: top1 82.34, f1 81.29, mixed_val 99.69, mixed 98.90, n 1365
- seed 1: top1 83.52, f1 82.47, mixed_val 99.54, mixed 99.12, n 1365
- seed 2: top1 84.98, f1 84.49, mixed_val 99.54, mixed 99.20, n 1365

**SHuBERT**
- seed 0: top1 67.60, f1 66.60, mixed_val 99.54, mixed 98.97, n 1364
- seed 1: top1 67.52, f1 66.58, mixed_val 99.54, mixed 98.90, n 1364
- seed 2: top1 67.16, f1 66.09, mixed_val 99.54, mixed 98.90, n 1364

