# P2 analysis 2 -- RGB + pose late fusion (Protocol A TEST)

Script: `tools/p2_analysis/rgb_pose_fusion.py`; numbers: `docs/p2_analysis_2026-10/rgb_pose_fusion.json`.

**Protocol.** TEST = held-out signers U02+U13 (n = 1365 clips, `data/bdsl_si/test_label.pkl`); VAL = U15 (n = 655, `data/bdsl_si/val_label.pkl`). Fusion = softmax-probability averaging `w*p_A + (1-w)*p_B`; w picked on VAL only from {0, 0.05, ..., 1} (ties -> w nearest 0.5), separately per seed pair; seed i of A paired with seed i of B (S3D seeds 0r/1r/2r = cluster re-trains; Uni-Sign seed index 0 = its seed 42). mean +- SD = population SD over the 3 seed pairs. Bootstrap: paired clip-level, 10000 resamples, on per-clip correctness averaged over the 3 seed pairs; one-sided p (H1: fusion better); Holm over the 6 RGB x pose pairs. Clip-level CIs ignore within-signer correlation (only 2 TEST signers, so a signer-clustered CI is impossible) -- they are per-instance, not signer-generalisation, CIs.

## Alignment check (done first)

All pose (3 models x 3 seeds), VideoMAE and S3D TEST dumps contain exactly the 1,365 TEST clip names and all VAL dumps exactly the 655 U15 names (0 missing / 0 extra; rows aligned by clip name, not by position). Uni-Sign TEST predictions cover all 1,365 names and its reference labels match `test_label.pkl` for every clip. SHuBERT covers 1,364/1,365 (missing: U2W215F_trial_1_R); its class indices differ from `classes.json` (0/60 identical) but map by a consistent bijection derived from the clip names, applied before scoring.

Score types: VideoMAE dumps are already probabilities; pose and S3D dumps are logits (softmaxed here). Uni-Sign saves only decoded text labels (no logits) and no VAL predictions; SHuBERT saves TEST probabilities only (no VAL). Hence Uni-Sign and SHuBERT cannot enter val-tuned probability fusion (see the last two sections).

## Single models (TEST Top-1, per seed)

| Model | seed 0 | seed 1 | seed 2 | mean +- SD | VAL (U15) mean |
|---|--:|--:|--:|--:|--:|
| SLGTFormer (Track-B) | 82.34 | 85.79 | 83.59 | 83.91 +- 1.42 | 92.47 |
| Adaptive GNN | 77.88 | 80.00 | 79.34 | 79.07 +- 0.89 | 88.80 |
| BlockGCN (lr 0.01) | 68.06 | 69.01 | 69.38 | 68.82 +- 0.56 | 77.96 |
| VideoMAE | 78.10 | 83.15 | 81.54 | 80.93 +- 2.11 | 93.64 |
| S3D | 72.38 | 76.63 | 75.16 | 74.73 +- 1.76 | 88.60 |
| Uni-Sign | 82.34 | 83.52 | 84.98 | 83.61 +- 1.08 | n/a |
| SHuBERT | 67.60 | 67.52 | 67.16 | 67.42 +- 0.19 | n/a |

Best single model by TEST mean: **SLGTFormer (Track-B)**.

## Val-tuned fusion (TEST Top-1)

| Pair | kind | w (A) per seed | fused per seed | fused mean +- SD | equal-w mean | Delta vs stronger component [95% CI], p_Holm | Delta vs best single (SLGTFormer (Track-B)) [95% CI], p_Holm | oracle (either correct) | both wrong |
|---|---|---|---|--:|--:|---|---|--:|--:|
| VideoMAE + SLGTFormer (Track-B) | RGB+pose | 0.55/0.60/0.50 | 87.62/90.18/87.62 | 88.47 +- 1.21 | 88.50 | +4.57 [+3.54, +5.62], p=0.0000 (vs SLGTFormer (Track-B)) | +4.57 [+3.54, +5.59], p=0.0000 | 92.14 | 7.86 |
| VideoMAE + Adaptive GNN | RGB+pose | 0.60/0.60/0.60 | 85.86/87.47/85.57 | 86.30 +- 0.84 | 85.47 | +5.37 [+4.20, +6.59], p=0.0000 (vs VideoMAE) | +2.39 [+1.07, +3.69], p=0.0016 | 92.28 | 7.72 |
| VideoMAE + BlockGCN (lr 0.01) | RGB+pose | 0.65/0.60/0.65 | 83.22/84.18/83.74 | 83.71 +- 0.39 | 81.54 | +2.78 [+1.78, +3.81], p=0.0000 (vs VideoMAE) | -0.20 [-1.64, +1.29], p=1.0000 | 89.77 | 10.23 |
| S3D + SLGTFormer (Track-B) | RGB+pose | 0.50/0.50/0.45 | 83.59/86.89/86.74 | 85.74 +- 1.52 | 85.71 | +1.83 [+0.93, +2.78], p=0.0001 (vs SLGTFormer (Track-B)) | +1.83 [+0.93, +2.74], p=0.0000 | 90.31 | 9.69 |
| S3D + Adaptive GNN | RGB+pose | 0.50/0.50/0.50 | 84.10/86.01/85.93 | 85.35 +- 0.88 | 85.35 | +6.28 [+5.05, +7.55], p=0.0000 (vs Adaptive GNN) | +1.44 [+0.22, +2.64], p=0.0342 | 89.45 | 10.55 |
| S3D + BlockGCN (lr 0.01) | RGB+pose | 0.50/0.55/0.60 | 80.51/80.44/82.49 | 81.15 +- 0.95 | 81.88 | +6.42 [+5.32, +7.55], p=0.0000 (vs S3D) | -2.76 [-4.32, -1.20], p=1.0000 | 87.67 | 12.33 |
| SLGTFormer (Track-B) + Adaptive GNN | control: pose+pose | 0.70/0.40/0.75 | 83.44/83.15/83.81 | 83.47 +- 0.27 | 84.59 | -0.44 [-1.15, +0.27], p=0.8861 (raw) (vs SLGTFormer (Track-B)) | -0.44 [-1.15, +0.27], p=0.8937 (raw) | 87.99 | 12.01 |
| VideoMAE + S3D | control: RGB+RGB | 0.60/0.55/0.70 | 81.54/82.71/83.08 | 82.44 +- 0.66 | 82.05 | +1.51 [+0.56, +2.47], p=0.0008 (raw) (vs VideoMAE) | -1.47 [-3.13, +0.24], p=0.9557 (raw) | 88.94 | 11.06 |

Controls (pose+pose, RGB+RGB) use the identical val-tuned rule; their p-values are raw (not in the Holm family).
Oracle / both-wrong are TEST-only diagnostics of complementarity (an upper bound, never used for selection).

## Uni-Sign (hard labels only): parameter-free plurality vote

Members: Uni-Sign + VideoMAE + SLGTFormer (Track-B); when all three disagree, the val-tuned VideoMAE+SLGTFormer fusion decides (pre-declared). No parameter is fit on TEST.

| seed pair | Uni-Sign seed | vote TEST Top-1 | all-3-disagree % |
|---|--:|--:|--:|
| 0 | 42 | 89.52 | 7.69 |
| 1 | 1 | 91.14 | 7.47 |
| 2 | 2 | 89.30 | 8.06 |

Vote: 89.99 +- 0.82; vs best single +6.08 [+5.08, +7.16] (p=0.0000, raw); vs Uni-Sign +6.37 [+5.15, +7.62] (p=0.0000, raw). Oracle Uni-Sign|SLGTFormer = 91.33, Uni-Sign|VideoMAE = 92.72.

## SHuBERT: equal-weight fusion with SLGTFormer (NOT val-tuned; no SHuBERT VAL dumps), 1,364 clips

Per seed (SHuBERT / SLGTFormer / fused): s0 67.60 / 82.33 / 83.72; s1 67.52 / 85.78 / 87.61; s2 67.16 / 83.58 / 84.97. Fused mean 85.43 +- 1.62 vs SLGTFormer 83.90 on the same 1,364 clips.

## Source files

- SLGTFormer (Track-B)/seed0: TEST `work_dir/bdsl_slgtformer_si_bp_seed0_test/eval_results/best_acc.pkl`; VAL `work_dir/bdsl_slgtformer_si_bp_seed0/eval_results/best_acc.pkl`
- SLGTFormer (Track-B)/seed1: TEST `work_dir/bdsl_slgtformer_si_bp_seed1_test/eval_results/best_acc.pkl`; VAL `work_dir/bdsl_slgtformer_si_bp_seed1/eval_results/best_acc.pkl`
- SLGTFormer (Track-B)/seed2: TEST `work_dir/bdsl_slgtformer_si_bp_seed2_test/eval_results/best_acc.pkl`; VAL `work_dir/bdsl_slgtformer_si_bp_seed2/eval_results/best_acc.pkl`
- Adaptive GNN/seed0: TEST `work_dir/bdsl_adaptive_gnn_si_seed0_test/eval_results/best_acc.pkl`; VAL `work_dir/bdsl_adaptive_gnn_si_seed0/eval_results/best_acc.pkl`
- Adaptive GNN/seed1: TEST `work_dir/bdsl_adaptive_gnn_si_seed1_test/eval_results/best_acc.pkl`; VAL `work_dir/bdsl_adaptive_gnn_si_seed1/eval_results/best_acc.pkl`
- Adaptive GNN/seed2: TEST `work_dir/bdsl_adaptive_gnn_si_seed2_test/eval_results/best_acc.pkl`; VAL `work_dir/bdsl_adaptive_gnn_si_seed2/eval_results/best_acc.pkl`
- BlockGCN (lr 0.01)/seed0: TEST `work_dir/bdsl_block_gcn_si_lr01_seed0_test/eval_results/best_acc.pkl`; VAL `work_dir/bdsl_block_gcn_si_lr01_scratch_seed0/eval_results/best_acc.pkl`
- BlockGCN (lr 0.01)/seed1: TEST `work_dir/bdsl_block_gcn_si_lr01_seed1_test/eval_results/best_acc.pkl`; VAL `work_dir/bdsl_block_gcn_si_lr01_scratch_seed1/eval_results/best_acc.pkl`
- BlockGCN (lr 0.01)/seed2: TEST `work_dir/bdsl_block_gcn_si_lr01_seed2_test/eval_results/best_acc.pkl`; VAL `work_dir/bdsl_block_gcn_si_lr01_scratch_seed2/eval_results/best_acc.pkl`
- VideoMAE/seed0: TEST `work_dir/rgb_videomae_bdsl_si_rerun_seed0/test_scores.pkl`; VAL `work_dir/rgb_videomae_bdsl_si_rerun_seed0/val_best_scores.pkl`
- VideoMAE/seed1: TEST `work_dir/rgb_videomae_bdsl_si_rerun_seed1/test_scores.pkl`; VAL `work_dir/rgb_videomae_bdsl_si_rerun_seed1/val_best_scores.pkl`
- VideoMAE/seed2: TEST `work_dir/rgb_videomae_bdsl_si_rerun_seed2/test_scores.pkl`; VAL `work_dir/rgb_videomae_bdsl_si_rerun_seed2/val_best_scores.pkl`
- S3D/seed0: TEST `work_dir/rgb_s3d_bdsl_si_seed0r/eval_results/testset_scores.pkl`; VAL `work_dir/rgb_s3d_bdsl_si_seed0r/eval_results/best_acc.pkl`
- S3D/seed1: TEST `work_dir/rgb_s3d_bdsl_si_seed1r/eval_results/testset_scores.pkl`; VAL `work_dir/rgb_s3d_bdsl_si_seed1r/eval_results/best_acc.pkl`
- S3D/seed2: TEST `work_dir/rgb_s3d_bdsl_si_seed2r/eval_results/testset_scores.pkl`; VAL `work_dir/rgb_s3d_bdsl_si_seed2r/eval_results/best_acc.pkl`
- Uni-Sign/seed0: `external/Uni-Sign/out/bdslw60_si_ft_lowdata_eval`
- Uni-Sign/seed1: `external/Uni-Sign/out/bdslw60_si_ft_lowdata_seed1_eval`
- Uni-Sign/seed2: `external/Uni-Sign/out/bdslw60_si_ft_lowdata_seed2_eval`
- SHuBERT/seed0: `work_dir/shubert_probe_seg/si_test_pred_seed0.pkl`
- SHuBERT/seed1: `work_dir/shubert_probe_seg/si_test_pred_seed1.pkl`
- SHuBERT/seed2: `work_dir/shubert_probe_seg/si_test_pred_seed2.pkl`
- labels: `data/bdsl_si/test_label.pkl`, `data/bdsl_si/val_label.pkl`, classes `data/bdsl_si/classes.json`
