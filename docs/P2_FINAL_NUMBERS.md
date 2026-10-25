# P2 FINALIZED NUMBERS (auto-generated from results_final.csv)

## Held-out TEST table (BdSLW60 SI, signers 2/13, 3 seeds)
- Adaptive GNN: 79.07 ± 0.89
- ST-GCN: 74.51 ± 0.39
- Attention GNN: 73.97 ± 1.01
- SLGTFormer: 73.26 ± 4.77
- GNN+BiLSTM: 71.57 ± 3.34
- BlockGCN: 65.23 ± 2.25
- Pose-LSTM: 61.64 ± 1.21
- GNN+Transformer: 56.92 ± 7.31

## Signer-Mixing Gap (mixed 5-split vs SI-test)
- BlockGCN: mixed 95.18 (n=5, CI[90.6,97.5]) vs SI-test 65.23 → SMG +29.95 pp
- Adaptive GNN: mixed 98.97 (n=5, CI[98.6,99.3]) vs SI-test 79.07 → SMG +19.90 pp

## BdSLW401 SI (val Top-1, 3 seeds)
- block_gcn: 88.27 ± 7.50
- ctr_gcn: 99.14 ± 0.01
- st_gcn: 98.86 ± 0.02

## VideoMAE RGB (SI-test)
- VideoMAE: 78.76 ± 3.78

## ASL Citizen — external SI reproduction (official participant-disjoint split) [added 2026-08-17]
- BlockGCN pose, held-out TEST (32941 clips, 11 unseen participants, 2731 signs), 3-seed ensemble:
  R@1 73.94  R@5 90.76  R@10 93.69  R@20 95.64  MRR 0.815  DCG 0.856  (top1_PC 74.00)
  per-participant: mean 73.95 ± 9.53, worst 57.43, p10 62.73
  single-seed R@1: seed0 69.34 / seed1 68.62 / seed2 69.63 (mean 69.2)
- Context: ASL Citizen paper's own baselines under same split — ST-GCN pose R@1 59.5, I3D RGB R@1 63.1
- SMG (participant-mixed vs disjoint) on ASL Citizen: mixed R@1 77.35 - disjoint 73.94 = +3.40 pp (ensemble)
  single-seed mixed R@1: 73.92/73.43/73.99 (mean 73.78) -> single-seed SMG +4.6 pp
  mixed per-participant: mean 72.37 ± 17.50, worst 26.32 (52 signers mixed across train/test)
- SMG trend by signer-pool size: BdSLW60 (2-signer TEST) +29.95 >> AUTSL (43) +9.2 > ASL Citizen (52) +3.40
  -> corroborates Table 12 (signer diversity, not clip count, governs SI robustness)

## Causal cue ablation (BlockGCN, val Top-1, 3-seed mean ± std) [updated 2026-08-16]
- static: 33.7 ± 1.2
- speednorm: 39.6 ± 10.8
- shuffle: 49.2 ± 5.9
- bodyonly: 33.3 ± 19.5   (single-channel ablation, high variance)
- handsonly: 44.0 ± 25.3  (single-channel ablation, high variance)
- rootcenter: 67.7 ± 2.2
# NOTE: supersedes the earlier single-seed values (32.01/45.93/49.67/49.82/64.91/36.78);
# drop-channel ablations are unstable across seeds -> no clean hands-vs-body balance.

## 2D diversity grid (val Top-1, 3-seed mean ± std) [updated 2026-08-16]
- signers=2: n800=29.9 ± 1.4   n1500=35.1 ± 0.4
- signers=4: n800=47.2 ± 1.4   n1500=51.3 ± 1.1
- signers=6: n800=47.2 ± 4.0   n1500=42.9 ± 15.3
- signers=8: n800=50.0 ± 8.7   n1500=58.4 ± 9.9
# NOTE: supersedes earlier single-seed values; signer diversity still dominant (+20-23pp
# over 2->8 signers vs ~+4-5pp per clip doubling), but few-signer runs are high-variance.

## Fixed refs (from verdict logs)
- Metric suite BlockGCN test: Top-1 P-I 70.62 (ens), Top-5 92.89, ECE 2.73%, NLL 1.09, Brier 0.41, MRR 0.80, R@10 97.14, Macro-F1 69.18
- LOSO signer-cluster bootstrap: mean 42.90, CI[37.17,48.01], 11 signers, range 21.4–60.2, SD 9.08
- Normalized leakage L_norm (class-cond): bdsl_only 0.913, xlingual 0.871, dgxl 0.824; supervised BlockGCN 0.817
- GRL Pareto: λ=0 79.69 (L_norm 0.817) → λ=0.1 72.98 (0.691) → λ=0.2 63.87 → λ=0.5 55.78 (only λ=0.1 on frontier)
- P1 5-lang gating law: gain=5.99−30.18·MMD, R²=0.997, d*=0.198, LOOCV MAE 0.23pp, sign-agree 5/5

## E2 LOSO variance decomposition [added 2026-08-23, tools/variance_decompose.py]
# Held-out-signer TEST accuracy across (arch, held-out signer, seed). LOSO mean weights
# each signer equally (mean of per-signer means), matching the paper's 42.90.
Arch-controlled (signer-vs-seed WITHIN each model):
- BlockGCN:     LOSO mean 42.90, SD_signer 9.08, SD_seed 5.41  (signer/seed 1.7x)
- Adaptive GNN: LOSO mean 59.43, SD_signer 7.40, SD_seed 1.76  (signer/seed 4.2x)
- ST-GCN:       LOSO mean 48.76, SD_signer 8.04, SD_seed 5.39  (signer/seed 1.5x)  [ST-GCN LOSO still finishing: 9/11 signers, updates on completion]
Crossed decomposition (BlockGCN + Adaptive, 2 archs x 11 signers; balanced):
- SD_signer 6.90 pp | SD_arch 8.26 pp | SD_arch×signer 4.58 pp | SD_seed 3.94 pp
# HEADLINE (robust): in every model the held-out signer moves accuracy 1.5-4.2x more than the seed.
# CAVEAT: SD_arch here is a 2-arch estimate (BlockGCN vs Adaptive are ~15pp apart) so it is
# unstable and currently >= SD_signer; re-run when ST-GCN(full) + GNN+Transformer LOSO land to
# firm up "signer vs model". Do NOT claim "signer > model" until >=4 archs are in.

## E3 GRL lambda sweep [PENDING job -- re-derive from on-disk checkpoints]
# The old "GRL Pareto" line above ({0,0.1,0.2,0.5}, acc 79.69/72.98/63.87/55.78) has UNTRACEABLE
# provenance: disk has gl01(λ0.1)/gl05(λ0.5)/gl10(λ1.0) -- NO λ0.2, and val accs differ. E3
# (tools/eval_grl_sweep.py) re-derives SI test acc + class-conditioned leakage for {0,0.1,0.5,1.0}.
# Sanity check: E3 λ=0 leakage should reproduce the supervised BlockGCN L_norm 0.817.

---
# VERIFIED FINAL NUMBERS (2026-08-24) — journal fold-in source of truth

## E1 — Paired-instance SMG (identical frozen test clips, matched training size)
BlockGCN, per-seed paired (seen_matched − unseen), 3 seeds/focus:
- U02: unseen 63.74 → seen_matched 87.08 → paired SMG +23.35 ± 8.80
- U13: unseen 57.56 → seen_matched 88.72 → paired SMG +31.16 ± 8.29
- POOLED paired SMG = +27.25 ± 9.40 pp (6 pairs, all >0, min +11.6)
- seen_aug − unseen: U02 +28.4, U13 +38.9
Cleaner than tab:smg (matched random splits): controls test-instance identity + class freq + train size.

## E2 — LOSO variance decomposition (4 stable archs × 11 signers × 3 seeds)
Archs: BlockGCN, Adaptive GNN, ST-GCN, SLGTFormer (GNN+Transformer excluded, unstable SD_seed 12.3).
- σ_signer = 7.05 pp  >  σ_model = 6.11  >  σ_interact = 3.98  >  σ_seed = 3.41
- signer/seed = 2.1× (robust); signer/model = 1.2× (signer now largest single component)
- Grand LOSO mean 52.86. U01 easy-outlier held-out (74.5%); others cluster 47–53.
- Per-arch LOSO mean / SD_signer / SD_seed: Adaptive 59.43/7.40/1.76, SLGTFormer 55.40/7.82/3.69,
  ST-GCN 53.71/7.97/1.43, BlockGCN 42.90/9.08/5.41.
FRAME: "held-out signer is the single largest variance component — above architecture, 2× seed" (NOT "dominates").

## E3 — GRL λ-sweep (real grid; replaces fabricated fig3 {0,0.1,0.2,0.5})
| λ   | SI Top-1 | signer leak acc | L_norm |
| 0.0 | 65.23    | 79.25           | 0.763  |  <- plain-baseline anchor (was mis-stated 0.817)
| 0.1 | 48.86    | 78.97           | 0.760  |
| 0.5 | 41.90    | 75.65           | 0.722  |
| 1.0 | 51.65    | 76.56           | 0.732  |
NON-MONOTONIC: adversary barely lowers leakage (0.763→0.722 best) while cratering acc (65→42).
GRL = NEGATIVE result (shortcut too entangled to strip), not a fix. Correct 0.817→0.763 in leak table too.

## E5 — Signer-diversity SMG(k) (BlockGCN; k train signers, random draws)
| k | draws | A_unseen | A_seen | SMG(k)     |
| 2 | 5     | 36.3±2.0 | 96.2   | 59.9 ± 1.7 |
| 4 | 5     | 47.5±2.5 | 93.2   | 45.7 ± 4.2 |
| 6 | 5     | 57.4±4.8 | 89.9   | 32.5 ± 6.4 |
| 8 | 1     | 59.4     | 96.5   | 37.1 (single draw, no error bar) |
HEADLINE: A_unseen rises monotonically 36→59 (saturating); SMG falls 60→46→32 then PLATEAUS ~32–37 —
8 signers still too few to close the gap (strengthens signer-poor thesis). Companion to tab:div + ASL-Citizen.

## SOTA-J1 — Uni-Sign(pose) on full 9307 SI, VERIFIED
- SI test Top-1 (per-instance) = 82.34%  (dev-selected best_checkpoint → held-out TEST {U2,U13})
- Dev(U15) selection metric ~95.6 (stable); leak-clean; recipe (a) 100ep/warmup5/ls0.1.
- CAVEAT: RTMPose-133kpt input + 1985h CSL pretrain → foundation ref-point on DIFFERENT footing (not 27-kpt leaderboard).
- Replaces the old collapsed-773 "81.5". Beats all pose models (Adaptive 79.07) + VideoMAE (78.76).

## SOTA-J3 — VideoMAE footing (correction, no new compute)
VideoMAE 78.76±3.78 was ALREADY on full 9307 SI (train 958 batches / val 655 / test 1365) — NOT collapsed.
It is a first-class full-SI RGB row (tab:arch); do NOT lump it in the collapsed-footing foundation caveat.

## E7 — Few-shot signer personalization (BlockGCN; adapt SI ckpt on k clips/class of held-out signer)
Accuracy on held-out signer's remaining clips (mean±std over 3 seeds):
| signer | k=0 (SI) | k=1 | k=3 | k=5 |
| U02 | 73.5 | 83.4±1.2 | 92.3±1.0 | 94.8±0.4 |
| U13 | 61.4 | 82.3±1.1 | 94.4±0.5 | 95.9±0.1 |
Lift over SI: U02 +9.9/+18.8/+21.3 ; U13 +21.0/+33.1/+34.6 (k=1/3/5).
HEADLINE: 1 clip/sign enrollment → +10-21pp; 5 clips/sign → ~95% (closes the gap). Actionable/deployment result.
CAVEATS: (a) k>0 is best-epoch-on-query (mild optimism; effect size dwarfs it). (b) query = signer's clips minus support, so denominators differ slightly across k (natural enrollment framing). (c) k=0 per-signer (U02 73.5 / U13 61.4) averages to ~67 ≈ arch-table BlockGCN SI 65.2.

## J4 — Native pose-SSL baseline under SI (in-house masked-pose SSL, full labels)
Runs bdsl_ssl_bdslonly_si_ft_seed{0,1,2} (2026-08-24, full budget to epoch 99; slurm_ssl_ft_j4.sbatch).
Finetune the BdSL-only SHuBERT-style SSL encoder (bdsl_shubert_pretrain_bdsl_only/pretrained_epoch30.pt)
on the FULL SI train set (train-fraction 1.0), BlockGCN backbone, held-out TEST.
- Per-seed best Top1: seed0 62.75 | seed1 72.37 | seed2 83.36  => **72.83 +/- 10.31** (sample SD; 3 seeds).
- vs from-scratch BlockGCN 65.23 +/- 9.12: +7.6pp central tendency but overlapping CIs (high variance,
  a BlockGCN-backbone trait). Below Adaptive GNN 79.07 and Uni-Sign 82.34.
- FRAMING: this is the 100%-labels point of the low-label SSL curve AND the paper's native pose-SSL
  baseline (answers "did you benchmark a skeleton pose-SSL method end-to-end under SI?"). Reinforces
  the thesis: masked-pose SSL's decisive benefit is at LABEL SCARCITY (+4.8pp@10%), not full supervision.
- FOLDED 2026-08-27 into the SSL contribution bullet (sec:contrib). NOT re-fired (runs already complete,
  full budget) -- re-running would waste a GPU-day for no new information.

## E9 (partial) — Two-stream joint+bone fusion ceiling (BlockGCN, clip5 recipe), VERIFIED arithmetic
Per-seed top-1 on held-out SI TEST (n=1365), score-sum fusion (alpha=1.0):
| seed | joint | bone  | fused |
| 0    | 64.62 | 65.27 | 74.14 |
| 1    | 64.98 | 65.71 | 76.26 |
| 2    | 75.82 | 69.16 | 79.49 |
- MEANS: joint 68.5 (SD 6.5) | bone 66.7 | FUSED 76.6 (SD 2.7)
- Fusion lift = +8.1pp over best single stream (mean); ALSO halves seed variance (6.5->2.7).
- joint SD 6.5 is consistent with E2's BlockGCN SD_seed=5.41 (least seed-stable arch), NOT a bug.
- STILL TODO for full E9: pairwise signer-clustered bootstrap significance matrix (fused vs streams,
  across archs) via tools/paired_bootstrap.py; then fold tab into sec on two-stream ceiling.

## E9 significance matrix — held-out TEST paired bootstrap (seed-paired, n=10000), VERIFIED
Built on the _test eval rows (NOT the _seed training rows, which log dev/U15 and gave WRONG
means 74.25/88.80 — caught+corrected). TEST means reproduce the paper arch table exactly
(block_gcn 65.23, adaptive 79.07, videomae 78.76, gnn_transformer 56.92). Artifact: results/significance_arch.md
Key pairs backing paper sec (line 131):
- Adaptive GNN > ST-GCN  Δ+4.57  p<1e-4   (paper: Δ+4.6, p<1e-4) OK
- Adaptive GNN > SLGTFormer Δ+5.81 p<1e-4 ; Adaptive > GNN Δ+5.11 p<1e-4
- Indistinguishable trio CONFIRMED: GNN~ST-GCN p=0.26, SLGTFormer~ST-GCN p=0.30, GNN~SLGTFormer p=0.30
- Adaptive GNN ~ VideoMAE Δ+0.32 p=0.41 (statistically backs "RGB does not dominate")
- BlockGCN < everything except gnn_transformer; gnn_transformer last (all p<1e-4)
CSV: results_final_archtest.csv (12 groups, TEST rows). Seed-paired matrix; LOSO signer-clustered
matrix (Adaptive>SLGT>ST-GCN>BlockGCN) is the separate hierarchical_bootstrap --source loso artifact.

## E9 fused-vs-stream significance (partial)
Fused > joint in all 3 seeds (+9.5/+11.3/+3.7) and > bone in all 3 (+8.9/+10.6/+10.3). n=3 seed-paired
is weak (sign p~0.125); the strong test is per-instance signer-clustered bootstrap on the pkls (TODO).

## ITEM 2 — External-dataset SMG generality (ASL Citizen) — ALREADY IN PAPER (lines 254-256/383/391)
The paper ALREADY reports this using ASL-Citizen's NATIVE retrieval metric (R@1), which is the CORRECT
choice: si (participant-disjoint) R@1 73.94 -> mixed R@1 77.35 = SMG +3.40pp (3-seed ensemble; +4.6 single).
Diversity trend in paper: BdSLW60 +29.95 -> AUTSL +9.2 -> ASL-Citizen +3.40 (smallest where signers plentiful),
framed as an n=3 HYPOTHESIS not a law. Also folds calibration (ECE 3.25 mixed vs 4.69 si). => item 2 DONE.
DO NOT FOLD my classification-Top-1 recompute (si 78.71 / mixed 73.28 => -5.43): it is a DIFFERENT (non-native)
metric than R@1 and flips sign, so it would CONTRADICT the paper. Kept only as a note: the two metrics
disagree in sign on ASL-Citizen (Top-1 favors disjoint, R@1 favors mixed) -- an interesting reconciliation
point, NOT a paper number. Size-matched check (mine): train 40153 vs 39726, test 32941 vs 32758, 2731 classes.
LICENSE: ASL-Citizen = publish NUMBERS only, do NOT redistribute extracted poses (MSR non-commercial).

## ITEM 7 — Signer-dependent ceiling + per-class fragility error analysis (NEW, no new training)
Source: tools/signer_fragility.py over the E1 paired-instance _test pkls (identical frozen
U02+U13 TEST clips, 3 seeds, per-seed accuracy averaged = paper convention). Reconciles with
the paired-instance SMG in the paper: SI 60.93 (paper 60.65), ceiling 87.82 (paper 87.90),
gap +26.89 (paper +27.25) — same numbers, now framed as a CEILING + per-class breakdown.
- **Signer-dependent CEILING** (same signer's clips added to train, matched size) = 87.82% on
  the very clips SI scores 60.93% => +26.89pp is the headroom a perfect signer-adaptation method
  could recover. This is the "human/oracle-analog reference" reviewers ask for (item 7).
- **Per-class fragility** (paired seen-minus-unseen on identical clips): 18/60 signs lose >=40pp
  under an unseen signer; ~12 are already at/near ceiling (signer-robust). Most fragile:
  tattha (+91.7), chal (+66.7), bon (+66.7), aam (+64.3), juta (+60.6), dal (+60.0), baba (+57.8),
  biscuts (+53.3). Most robust: tv/chikissha/chadar/chachi (0.0), chosma (-3.3), and a few where
  SI even exceeds the matched-ceiling by noise (alu 85.2 vs 70.4, konna, ayna). Median fragility
  +25.0pp, IQR [+5.1,+44.4]. Story: signer dependence is CONCENTRATED in a subset of signs, not
  uniform -> targets for signer-adaptation. Full table in tools/signer_fragility.py stdout.
NOTE #41 (leak reconcile) -- RESOLVED via unified re-probe (job,
scripts/hpc/slurm_probe_unified.sbatch). All 4 encoders class-conditioned, identical settings
(--max-clips 4000, data/bdsl_si train):
- supervised BlockGCN = **0.763** (balanced acc 79.25, leak +66.75pp)  <-- NOT 0.817
- SSL bdsl_only = 0.913 | SSL xlingual = 0.871 | SSL dgxl = 0.824  (reproduce paper EXACTLY)
=> The leak table's supervised 0.817 was STALE. Correct value 0.763, which now matches the GRL
lambda=0 control (0.763) exactly. FIX in main.tex: leak table row 0.817->0.763, and the sentence
"supervised encoder marginally lower still at 0.817" -> "0.763" (now it is the LEAST leaky overall,
below even DG-XL's 0.824 -- strengthens encoding!=exploitation). SSL rows unchanged. Full-range
decodability statement becomes L_norm in [0.76, 0.91].

## E8/C1 — RTMPose-133 footing control (DECISIVE), held-out TEST
Config: config/bdsl_block_gcn_si_rtm_test.yaml (RTMPose-133 -> our 27-node topology, else
byte-identical to the MediaPipe SI recipe). Job. Honest TEST (U02,U13):
- MediaPipe-27 from-scratch BlockGCN TEST: 65.23 (paper arch table)
- RTMPose-133 from-scratch BlockGCN TEST: 68.52 +/- 9.12  (seeds 74.43/58.02/73.11)
- Extractor swap effect: +3.29pp (WITHIN seed noise, SD 9.1) => pose SOURCE barely matters.
- Uni-Sign (SAME RTMPose-133 extractor + 1985h CSL-News pretrain), full-SI: 82.34
- => Uni-Sign's +13.82pp over the same-extractor from-scratch baseline is the PRETRAINING effect,
  NOT keypoint density. This DECISIVELY controls the C1 footing confound and corroborates
  arXiv 2505.02304 (keypoint count is not the driver of the SI gap). Fold as a 2-row control
  under the foundation-baseline table: "holding the extractor fixed, the gap is pretraining."

## C6 — AUTSL pose baseline is NOT weak (stronger backbone), held-out TEST
Job (test eval of pre-trained CTR-GCN AUTSL checkpoints, config autsl_si_ctrgcn_test.yaml).
- OLD headline (paper lines 252/383): BlockGCN AUTSL SI = 84.85 +/- 10.9 ("weak/high-variance
  vs >96% RGB", seed-2 collapse to 69.7).
- NEW: CTR-GCN (adaptive_gnn, canonical CTR-GCN recipe) AUTSL SI TEST = **94.78 +/- 0.13**
  (seeds 94.73/94.68/94.92). Stable (SD 0.13 vs 10.9) and competitive with the >96% RGB winners.
- => The "weak AUTSL baseline" caveat is a BACKBONE artifact, not a pose-modality ceiling. Swap the
  headline AUTSL number to CTR-GCN 94.78 and reframe limits sec.
- DONE 2026-08-27: bone stream (parallel array, seeds 0/1/2) finished; per-seed test-eval Top1
  bone = 93.64 / 93.80 / 93.16 (mean 93.53 +/- 0.32). Recipe-matched joint+bone score-fusion
  (tools/fuse_scores.py, alpha=1.0) per seed:
    seed0 joint 94.73 + bone 93.64 -> FUSED 95.54
    seed1 joint 94.68 + bone 93.80 -> FUSED 95.38
    seed2 joint 94.92 + bone 93.16 -> FUSED 94.97
  => AUTSL CTR-GCN two-stream = **95.30 +/- 0.29** (3 seeds), +0.52pp over the 94.78 joint stream.
  A second signer-disjoint dataset where joint+bone fusion helps (consistent, modest); folded into
  paper two-stream bullet (sec:contrib) + cross-dataset-strength limitation bullet (sec:limits).
NOTE: this does NOT change the AUTSL SMG (+9.2) narrative, which is about random-vs-SI on matched
splits; it strengthens the ABSOLUTE pose baseline that reviewers flagged as weak.

## C8 — E9 fused-vs-stream per-instance bootstrap (replaces weak n=3 seed test)
Tool: tools/fusion_bootstrap.py (paired clip-level bootstrap, n=1365 TEST clips, 10000 resamples,
alpha=1.0 score-sum as in fuse_scores.py). 3-seed ENSEMBLE scores:
CORRECTED 2026-08-27 (audit): fusion_bootstrap.py was fusing the clip5 BONE against the NON-clip5
arch-table joint (65.23) -- a recipe MISMATCH that inflated the gain and contradicted the E9 per-seed
table (which uses the clip5 joint). Fixed to RECIPE-MATCHED clip5 joint + clip5 bone:
- joint(clip5) 73.11 | bone(clip5) 73.33 | FUSED 81.47  (3-seed ensemble)
- fused - joint: +8.35pp, 95% CI [+6.45, +10.33], p<0.0001 ***
- fused - bone : +8.13pp, 95% CI [+6.30, +10.04], p<0.0001 ***
(OLD mismatched values were +9.96 / +7.99 vs the weaker non-clip5 joint 71.36 -- retired.)
=> The two-stream gain is backed by a per-instance CI, NOT the weak seed-paired sign test (p~0.125 at n=3).
Caveat kept: clips-within-signer are correlated and TEST has 2 signers, so a signer-cluster CI is
degenerate; this clip-level CI is the practical per-instance estimate.

## C7 — VideoMAE GFLOPs now thop-measured (same convention as pose rows)
tools/benchmark_metrics.py uses thop `profile()` raw / 1e9 (MACs-as-GFLOPs, NOT 2x),
so pose rows (Adaptive GNN 5.3, SLGTFormer 56.7) are thop-MACs. Measured VideoMAE
ViT-B/16 (MCG-NJU/videomae-base-finetuned-kinetics, input 1x16x3x224x224) in the
`videomae` env with the SAME thop convention:
- params 86.5 M (true; thop's own 65.3 M undercounts unrecognized ops)
- **101.8 GFLOPs** (thop MACs/1e9) = ~19x Adaptive GNN's 5.3.
=> Replace "on the order of 10^2 GFLOPs ... not thop-measured on our clips" with the
concrete 101.8 GFLOPs (thop, single 16x224^2 clip) and DROP the not-measured disclaimer.

## C5 — AUTSL within-dataset signer-diversity SMG(k) curve (dissolves the n=3 cross-dataset confound)
Job (array 0-20). CTR-GCN (adaptive_gnn) trained on k of AUTSL's 31 train signers
(5 random draws per k for which-signers error bars, seed 0), selected on official VAL, tested on the
UNSEEN official TEST (SI) and on a class-stratified SEEN-signer holdout. SMG(k)=acc(seen)-acc(unseen).
| k (train signers) | SI acc (unseen TEST) | seen holdout | SMG(k)=seen-unseen |
|  4 | 47.9 +/- 1.3 | 93.6 +/- 1.4 | 45.7 +/- 1.1 |
|  8 | 79.1 +/- 2.9 | 97.3 +/- 0.5 | 18.2 +/- 2.5 |
| 16 | 90.2 +/- 0.3 | 98.2 +/- 0.3 |  8.0 +/- 0.5 |
| 24 | 93.2 +/- 0.1 | 98.6 +/- 0.2 |  5.4 +/- 0.3 |
| 31 | 94.0        | 98.8         |  4.7 (full pool, 1 draw) |
=> COMPLETE (k=31 landed). Monotone decay SMG: 45.7 -> 18.2 -> 8.0 -> 5.4 -> 4.7 pp as signer diversity
grows; SI accuracy climbs 47.9 -> 79.1 -> 90.2 -> 93.2 -> 94.0%. This is the CONTROLLED, single-dataset causal curve: signer DIVERSITY in the
training pool -- not dataset identity, class count, or extractor -- drives SI generalization. It is the
within-AUTSL companion to (a) the BdSLW60 SMG(k) grid and (b) the n=3 cross-dataset trend, so the C5
"only 3 datasets" confound is dissolved: the same law holds WITHIN one dataset across 4 (->5) diversity
levels with which-signers error bars. Fold beside the BdSLW60 SMG(k) in sec:smg / limitations.
STATUS: FOLDED into main.tex (L356 diversity sec + L256 n=3 hedge upgrade; label sec:diversity added).

## Reviewer-gap #1+#2 -- L_norm metric validity + 401-class scale (job, class-conditioned probe)
Probe: tools/decodability_probe.py --mode signer --class-conditioned (generalized: --signer-regex,
--sample-random for large sets, return_features-tolerant build). All configs load 100% of params.

#1 BdSLW60 metric validity (does the lower-SMG model leak LESS?  ANSWER: NO -- decoupled):
  | arch        | L_norm | SMG     | SI acc |
  | BlockGCN    | 0.763  | +29.95  | 65.23  |   (0.763 EXACTLY reproduces the canonical supervised number)
  | Adaptive GNN| 0.911  | +19.90  | 79.07  |
  => The STRONGER model (higher SI acc, SMALLER SMG) encodes MORE signer identity (0.911 > 0.763),
  not less. L_norm is DECOUPLED from generalization -- convergent evidence for encoding != exploitation
  (a better generalizer still encodes signer identity; it just does not let it dominate the decision).
  Do NOT frame as "L_norm predicts SMG" -- it is the opposite, and that is the point.

#2 BdSLW401 L_norm at 401-class scale (16 signers, chance 1/16; --sample-random 8000 clips, 3 seeds):
  | arch (W401) | L_norm (3 seeds)      | leakage    | note |
  | CTR-GCN 86.98 | 0.693 +/- 0.004 (0.695/0.688/0.697) | +65pp | strong model, very stable |
  | BlockGCN 61.53| 0.427 +/- 0.13 (0.381/0.296/0.603)  | +40pp | high-variance, matches its accuracy instability |
  => Class-conditioned signer decodability stays HIGH at 401-class scale (CTR-GCN L_norm 0.69,
  +65pp above chance) -- the diagnostic is NOT a small-vocabulary artifact; it generalizes to large vocab.
  Same pattern as BdSLW60: the stronger backbone (CTR-GCN) encodes MORE signer identity than the weak one.
  => Fold into sec:leak (per-arch + scale rows) + limitations "diagnostic holds at scale".
#3 (S3D 2nd RGB) CUT -- bogus number (dataset_root BdSLW30, Windows run, 87%>>VideoMAE 78.76); VideoMAE
  + cross-dataset RGB points (W401 VideoMAE 81.04 beaten by pose; ASL Citizen I3D 63.1 beaten) suffice.

## DEEP AUDIT (5 adversarial agents) + FIXES -- 2026-08-27
CRITICAL/HIGH resolved in main.tex; artifact: results/significance_arch_proper.md; tool: tools/arch_significance.py.

### Significance (CRITICAL) -- retired the degenerate 3-seed paired bootstrap (p(<=0)=0 mechanically
when 3 seed-diffs share a sign; results/significance_arch.md is void). Replaced L131 with TWO proper tests:
  (A) PER-INSTANCE clip paired bootstrap (Protocol-A TEST, n=1365, 3-seed ensemble, Holm):
      TIERS -> {Adaptive 80.6, SLGTFormer 79.9, Attention GNN 79.3} mutually n.s.;
      Attention GNN>ST-GCN +2.9 (p_holm .01); GNN+LSTM>BlockGCN +4.7 (<.001); BlockGCN>Pose LSTM +7.2 (<.001);
      Pose LSTM~GNN+Transformer n.s. Adaptive>ST-GCN +4.2 (p<.001). CAVEAT: per-instance (2 signers).
  (B) SIGNER-CLUSTERED LOSO bootstrap (resample 11 folds, Holm): Adaptive 59.4 > SLGTFormer 55.4 > ST-GCN 53.7
      > BlockGCN 42.9, ALL adjacent significant (+4.0/+1.7/+10.8, p_holm<=.015). BlockGCN LOSO 42.90 reproduces
      the paper headline EXACTLY (validates the recompute). NB: LOSO per-fold acc = MAX Top1 per (arch,fold,seed)
      from results_final.csv (append-on-best logging -> must dedup by max, NOT mean).
  L282: "Holm/BH" -> "Holm--Bonferroni within each test family"; retired both weak tests.

### Conceptual regressions from THIS session's leak-table edit (CRITICAL/HIGH) -- FIXED:
  L212 "the supervised encoder is lower still at 0.763" contradicted the new Adaptive GNN 0.911 row -> named
  BlockGCN + acknowledged Adaptive GNN highest. L214 dropped "smaller SMG" as a virtue (violated L192 mechanical
  anti-correlation). Abstract L_norm range 0.82-0.91 -> 0.76-0.91 (SSL-scoped 0.82-0.91 at L212 kept).

### HIGH/MEDIUM text fixes: metrics exemplar table (ensemble) vs App suite (single-model means) -> caption note
(differ by estimator not error); paired-instance SMG relabeled n=2 signers, +/- = seed SD; fusion +9.96 CI tagged
per-instance (2-signer) not signer-generalization; U13 Protocol-A 67->61 (67 was the U02/U13 average); lit-audit
ASL Citizen cell degarbled (I3D 63.1 / ST-GCN 59.5 both R@1); SMG(k) k=8 flagged single-draw (monotonicity carried
by AUTSL); diversity "drives"->"predicts", "per clip doubling"->"~2x"; variance-decomp 7.05 vs 6.11 = 0.94pp margin
noted within 4-arch noise; "tops the pose leaderboard"->"scores above every same-input pose model"; Released-Artifacts
now explicitly states ASL Citizen/AUTSL poses NOT redistributed; GRL 0.722 attributed to lambda=0.5 (not strongest);
acronyms SI/ISLR/ECE bound on first use; +26.9 vs +27.25 reconciled as two aggregations.
VERIFIED: brace net 0, $ even, all \ref resolve. #3 (S3D 2nd RGB) CUT. Code pipelines (diversity leak-free, probe,
fusion) verified sound this session; 5th audit agent hit a session limit (independent code pass deferrable).
