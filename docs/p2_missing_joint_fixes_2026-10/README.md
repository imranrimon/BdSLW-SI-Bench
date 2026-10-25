# P2 missing-joint fixes (2026-10-22)

Fixes for audit findings 1–5 (MediaPipe stores an undetected joint as (x, y, conf) = (0, 0, 0)). No paper `.tex`, `main.py`,
`feeders/feeder.py`, `preprocessing/generate_bone_data.py` or existing data was modified; everything below is new,
versioned scripts, data dirs and configs. Full generated tables: `results.md` (`python tools/p2_mjfix_report.py`).
Job IDs: `jobids.txt`. **GPU use: 49.8 GPU-h allocated** (see the end of this file).

New code: `preprocessing/generate_bone_data_masked.py`, `preprocessing/make_cue_bundle_masked.py`,
`preprocessing/generate_paired_smg_otherclass_bundle.py`, `tools/decodability_probe_v2.py`, `tools/p2_mjfix_setup.py`
(configs `config/p2_mjfix/` + task lists `scripts/hpc/p2_mjfix/`), `tools/p2_mjfix_score.py` (per-clip score dumps,
gated on the final-epoch checkpoint), `tools/p2_mjfix_report.py`, `scripts/hpc/slurm_p2_mjfix.sbatch` (copy of
`slurm_lr0p1_switch.sbatch`), `scripts/hpc/p2_mjfix/{submit.sh,submit_all.sh,dataprep.sh}`.
New data: `data/bdsl_si_bonemask/`, `data/bdsl_cue_speednorm_masked/`, `data/bdsl_cue_rootcenter_masked/`,
`data/bdsl_si_pairsmg_seenother/U{02,13}_seenother_{A,B}/`. The data was built and checked on a compute node
(`slurm_logs/p2_mjfix_dataprep.log`). The check confirmed that the old bone file equals `generate_bone_data(joint)`,
that masking changes only joints 7 and 17, and that the cue bundles re-derive with 0 mismatches.
Do not use `data/bdsl_si_bone_masked/` or `data/bdsl_si_pairsmg_otherclass/`. They are partial files from writes that
hung on the login node (Lustre `osc_extent_wait`, PIDs 1827793/1832904 still hung). They are referenced nowhere. Delete
them once those processes clear.

Sanity: the re-scoring code reproduces P2's per-seed values. BlockGCN joint gives 68.06/69.01/69.38, bone
67.69/65.20/63.74 and Adaptive bone 30.77/32.23/31.28. Paired unseen/seen_matched reproduce exactly (gap 25.41 / 35.30).
Probe v1 reproduces P2's BlockGCN cc 0.825/0.869/0.863 and Adaptive 0.909. The fused BlockGCN ensemble re-scores at
78.68 vs P2 78.75, a 1-clip difference: per-seed fused values are identical and the gap is a float tie.

---------------------------------------------------------------------------------------------------------------------
## A. Masked bone stream (finding 1)

**What changed.** The bone is zeroed where the child OR the parent joint is undetected. Only the two hand-root bones
(body wrist → hand wrist, joints 7/17) are affected. Whenever a hand is undetected, the old bone there was −(body wrist):
10.0% of missing-joint entries in valid frames, median |bone| 1.17–1.31 vs 0.11 for real bones. Each arch was retrained
× 3 seeds with its configs copied unchanged except for the data path: BlockGCN `bone_vec_lr01` (P2 recipe); Adaptive GNN
at the P2 bone recipe (= `bdsl_adaptive_gnn_bone_clip5`: lr 0.01, **grad-clip 5.0**, which does not match its unclipped
joint stream) and at lr 0.1. Best-on-val checkpoint → held-out TEST.

| bone stream TEST (3 seeds) | P2 / unmasked | **masked** |
|---|--:|--:|
| BlockGCN (lr 0.01) | 65.54 ± 1.63 | **65.96 ± 0.15** |
| Adaptive GNN, P2 recipe (lr 0.01, clip 5) | 31.43 ± 0.60 | **76.21 ± 1.91** |
| Adaptive GNN, lr 0.1 (joint 77.05 ± 0.92) | 24.00 ± 1.19 (`bdsl_adaptive_gnn_bone_lr0p1`, now finished) | **80.02 ± 0.71** |

Fusion: 3-seed ensembles with a paired clip bootstrap (10k) over 1,365 TEST clips. **P2's fusion weight is fixed at
α = 1 and is not chosen on VAL** (App. "Two-stream fusion": s = s_joint + α·s_bone, α = 1). The α=1 numbers are the P2
protocol. The VAL-selected α (grid 0–3 on the U15 ensemble) is a sensitivity check.

| arch, bone | joint ens → fused (α=1) | fused − joint, α=1 | α_val → fused − joint |
|---|--:|--:|--:|
| BlockGCN, unmasked (P2) | 71.72 → 78.68 | +6.96 [+5.05, +8.86] (P2 +7.03 [+5.13, +8.94]) | 1.45 → +6.30 [+4.32, +8.35] |
| BlockGCN, **masked** | 71.72 → 78.10 | **+6.37 [+4.40, +8.35]** | 1.45 → +5.35 [+3.30, +7.40] |
| Adaptive (P2 recipe), unmasked | 81.90 → 67.11 | −14.80 [−16.92, −12.75] | 0.30 → −2.78 |
| Adaptive (P2 recipe), **masked** | 81.90 → 84.98 | **+3.08 [+1.61, +4.54]** | 0.50 → +3.15 [+1.98, +4.40] |
| Adaptive lr 0.1, unmasked | 78.90 → 60.07 | −18.83 | 0.10 → +0.00 |
| Adaptive lr 0.1, **masked** | 78.90 → 85.27 | **+6.37 [+4.76, +8.06]** | 0.50 → +6.08 [+4.69, +7.55] |

Per-seed fused means at α=1 (masked): BlockGCN 75.85 ± 0.78, Adaptive P2 recipe 83.13 ± 0.86, Adaptive lr 0.1
84.13 ± 1.36. Masked − unmasked bone ensemble: BlockGCN +1.10 [−0.44, +2.64] (n.s.); Adaptive +47.0 / +57.7 pp.

**Claims that change.**
- "the bone stream ... collapses for Adaptive GNN (31.43)", "fusion can only help the former", "arch-dependent",
  "can hurt when the bone stream is weak". **This claim is false.** The collapse was the missing-joint artifact. With
  masked bones, Adaptive's bone stream (76.2; 80.0 at lr 0.1) matches or beats its joint stream, and fusion helps both
  architectures.
- BlockGCN: +7.03 → +6.37 [+4.40, +8.35]; 78.8 → 78.1 ensemble; bone 65.54 → 65.96 ± 0.15. The qualitative claim holds.
- The best fused pose number is now Adaptive joint+bone, 85.0–85.3 TEST (3-seed ensemble; 83.1–84.1 single-seed mean).
  It sits above VideoMAE 80.93. Only per-instance CIs are available (2 TEST signers).
- Suggested wording (App. bone): "With missing joints masked (a bone is zero when either endpoint is undetected), the
  bone stream is competitive with the joint stream for both BlockGCN (65.96 ± 0.15 vs 68.82) and Adaptive GNN
  (76.21 ± 1.91 vs 79.07), and joint+bone fusion helps both (+6.37 pp [+4.40, +8.35] and +3.08 pp [+1.61, +4.54], paired
  clip bootstrap, 3-seed ensembles)." Drop "collapses" and "arch-dependent". Update the generate_bone_data description to
  say missing endpoints are masked. Update `tab:bone` (SPEC `bone=` → `bdsl_*_bonemask_*_test_seed{s}`).

## B. Masked cue perturbations (finding 2)

**What changed.** speednorm_masked computes the mean speed only over (joint, t→t+1) pairs where the joint is detected in
both frames. On TEST, the 0↔coordinate jumps were a median 69% of the old speed sum (77% train, 84% val), and the old
speed estimate was inflated a median 2.2–3.7×. rootcenter_masked: missing joints stay (0,0) instead of −root. Configs
`config/p2_mjfix/bdsl_cue_*_masked_lr01.yaml` are copies of the P2 lr01 cue configs. Seeds 0/1/2, final-epoch held-out
TEST (P2's estimator).

| cue (BlockGCN, final-epoch TEST) | P2 | **masked** | Δ vs control (68.89) |
|---|--:|--:|--:|
| none (control) | 68.9 ± 0.8 | — | — |
| speed-normalized | 56.0 ± 1.3 (−12.9) | **70.72 ± 0.91** | **+1.83** |
| root-centred | 70.3 ± 1.1 | **71.58 ± 1.63** | +2.69 |

Ensemble paired clip bootstrap: speed masked − unmasked +16.19 [+13.92, +18.46]; speed masked − control +3.88
[+1.83, +5.93]; root masked − unmasked +0.95 [−1.10, +3.00] (n.s.); root masked − control +2.78 [+0.59, +4.91].

**Claims that change.**
- "speed-normalizing (56.0 ± 1.3, −12.9 pp) destroys much of it", "signing speed/timing carries sign info", and "Sign
  information is carried primarily by motion and articulation timing" (as far as it rests on speednorm). **These claims
  are false.** The −12.9 pp came from the undetected-joint jumps. Correctly masked, speed normalization does not hurt:
  70.7 ± 0.9, ≥ control. The motion claim must rest on static (−24.7) and shuffle (−8.6) alone.
- Root-centring is unaffected in direction ("absolute position is not load-bearing"; 71.6 ± 1.6).
- Suggested wording: "...collapsing to a static frame (44.1 ± 0.2, −24.7 pp) destroys much of it and shuffling frame
  order costs 8.6 pp, whereas normalizing overall speed/amplitude does not (70.7 ± 0.9 vs 68.9 ± 0.8 once undetected
  joints are excluded from the speed estimate)". Table row: `speed-normalized & 70.7±0.9 & tempo/amplitude normalization
  is harmless`. In App. cues, state that missing joints are excluded from the speed estimate and kept missing under
  root-centring.

## C. Leak probe v2 (finding 3)

**What changed** (`tools/decodability_probe_v2.py`):
- Input is the clip exactly as the test feeder gives it: 120-frame centre crop/pad and root normalisation.
- Probe clips are held out, never seen by any encoder: SI TEST (U02, U13) plus the labelled pretrain pool (U03, 07, 10,
  14, 16, 17, 18). That is n = 2904 clips and S = 9 signers. A `+val` variant adds U15.
- Cross-validation is StratifiedGroupKFold grouped by recording stem, so sibling trials never straddle folds. The scaler
  is fit inside each fold, and results are the mean of 3 CV seeds.
- The same code with `--protocol v1` re-runs P2's protocol.
- xlingual/dgxl are flagged (WLASL-scrambled pool) and excluded from conclusions.

| encoder | **v2 cc** | v2 marginal | v2 wc* | v2 cc, ungrouped | Δcc vs raw-input mean (unmasked / masked) | v1 cc (P2 protocol) | P2 |
|---|--:|--:|--:|--:|--:|--:|--:|
| raw joint means, unmasked (reference) | 0.524 | 0.470 | 0.960 | 0.764 | — | 0.802 | — |
| raw joint means, masked (reference) | 0.567 | 0.502 | 0.976 | 0.823 | — | 0.788 | — |
| missing-rate + clip length (reference) | 0.072 | 0.067 | 0.665 | 0.087 | −0.45 / −0.50 | 0.234 | — |
| random-init BlockGCN | 0.307 | 0.224 | 0.930 | 0.657 | −0.22 / −0.26 | 0.559 | — |
| random-init Adaptive GNN | 0.411 | 0.347 | 0.959 | 0.827 | −0.11 / −0.16 | 0.798 | — |
| sup. BlockGCN lr 0.01 (3 seeds) | **0.258 ± 0.016** | 0.131 ± 0.011 | 0.982 ± 0.002 | 0.879 ± 0.012 | −0.27 / −0.31 | 0.852 ± 0.020 | 0.853 ± 0.021 |
| sup. Adaptive GNN (P2 row, s0 m75) | **0.320** | 0.173 | 0.990 | 0.924 | −0.20 / −0.25 | 0.909 | 0.911 |
| sup. Adaptive lr 0.01 best ckpts (3 seeds) | 0.318 ± 0.002 | — | 0.988 | — | −0.21 / −0.25 | 0.899 | — |
| sup. Adaptive lr 0.1 (3 seeds) | 0.352 ± 0.003 | 0.209 | 0.988 | 0.929 | −0.17 / −0.22 | 0.895 | — |
| SSL BdSL-only | **0.228** | 0.170 | 0.934 | 0.729 | −0.30 / −0.34 | 0.914 | 0.913 |
| GRL λ=0.1 / 0.5 / 1.0 (seed 0) | 0.212 / 0.278 / 0.242 | | | | | 0.798 / 0.831 / 0.834 | 0.804 / 0.832 / 0.832 |
| *SSL xlingual (FLAG)* | *0.258* | | | | | *0.876* | *0.871* |
| *SSL DG-XL (FLAG)* | *0.358* | | | | | *0.821* | *0.824* |

*wc cannot be recording-grouped: each signer has exactly one recording per class. It therefore measures
same-recording trial similarity, and the raw-input means already reach 0.96–0.98 on it.

Reading: P2's 0.82–0.91 is almost entirely an artefact of sibling-trial leakage across CV folds. Ungrouped v2 cc stays
at 0.73–0.93, but grouped cc falls to 0.21–0.36. Every learned encoder sits **below** the untrained raw-input baseline
(0.52–0.57) and below a random-init network of the same architecture. The leak table's ordering also changes:
- BdSL-only SSL (0.228) is the least leaky, not the most.
- DG-XL is the most leaky of the SSL rows (flagged anyway). "Distance-gating reshapes the SSL representation away from
  signer features" is false.
- Adaptive > BlockGCN survives (0.32 vs 0.26; seed SDs ≤ 0.016). "The stronger model encodes more identity" holds in
  direction, at a far lower level.
- GRL: λ=0.1 0.212 vs seed-0 control 0.236, λ=0.5 0.278, λ=1.0 0.242. "L_norm does not fall" still holds: no monotone
  reduction, all inside or near the control's 0.236–0.276 seed range.

Missing-rate + length alone: v1 0.23 → v2 0.07.

**Claims that change.**
- Abstract / C2 / §leak: "Signer identity stays highly decodable from learned representations (L_norm = 0.82–0.91)" and
  "even from self-supervised skeleton encoders" fall, and "within each sign class it is near-perfect (0.96–0.99)" is
  uninformative (raw inputs reach the same). Suggested wording: "Under a recording-grouped probe on held-out signers,
  signer identity is only moderately decodable from frozen encoders (class-conditioned L_norm 0.21–0.36). This is below
  the 0.52 obtained from the raw per-joint means of the same inputs, so the encoders reduce rather than add linearly
  available identity. Ungrouped probes that let trials of one recording straddle folds inflate this to 0.73–0.93."
- Delete the DG-XL "least leaky" sentence. Keep "Adaptive > BlockGCN (0.32 vs 0.26)" for the encoding ≠ generalization
  point. Update App. probe (input, clip set, CV). The W401 numbers (0.63–0.72) use the v1 protocol and need the same
  rerun before they are quoted; this was not done here.

## D. Validation-selected matched-split SMG (finding 4)

**What changed.** The 5 Adaptive GNN rand-split runs were retrained (`config/p2_mjfix/bdsl_adaptive_gnn_rand4X_valsel.yaml`).
They use an identical recipe (lr 0.01, seed 0) with the split's VAL as the eval feeder. The best-on-val checkpoint was
scored once on the rand TEST.

| | rand42 | 43 | 44 | 45 | 46 | mean [min, max] | SMG vs 79.07 |
|---|--:|--:|--:|--:|--:|--:|--:|
| P2 (best epoch on own TEST) | 99.19 | 99.34 | 98.61 | 98.83 | 98.90 | 98.97 [98.6, 99.3] | +19.90 |
| **val-selected → TEST** | 99.41 | 99.05 | 98.75 | 98.83 | 99.05 | **99.02 [98.75, 99.41]** | **+19.95** |
| same new runs, best-on-TEST over all ckpts | 99.49 | 99.27 | 98.90 | 98.83 | 99.12 | 99.12 | +20.05 |
| same new runs, final epoch | | | | | | 98.99 | +19.92 |

**Claim holds.** The headline +19.90 becomes +19.95 under validation selection; the test-selection optimism here is
≈0.1 pp. Suggested change: report 99.02 → +19.95 and drop "scored at its best epoch on that split's own test set, an
optimistic estimator" in favour of "checkpoint selected on that split's validation set". Caveat (from E): the rand splits
also put sibling trials of one recording on both sides, so this SMG includes same-recording familiarity.

## E. Paired SMG decomposition (finding 5)

**What changed.** A new condition, `seenother`. The 60 classes are split into fixed halves A/B (seeded). The focus
signer's clips of the A classes are swapped into training (matched |train| = 5,748; 285–381 focus clips injected vs
299–357 in seen_matched). The model is tested on the frozen focus clips of the B classes, and vice versa. The A∪B test
set is exactly P2's frozen set, verified name-for-name. This uses BlockGCN lr 0.01, the P2 paired protocol (select on
U15, score once), and seeds 0/1/2. The unseen and seen_matched results are the existing P2 runs.

| focus | unseen | seen only via OTHER classes | seen_matched (same class = same recording) | signer familiarity | same-recording familiarity | total (P2) |
|---|--:|--:|--:|--:|--:|--:|
| U02 | 70.54 ± 2.01 | 71.49 ± 1.37 | 95.95 ± 1.20 | +0.95 [−1.81, +3.53] | +24.46 [+20.84, +28.17] | +25.41 |
| U13 | 61.90 ± 4.69 | 63.56 ± 0.59 | 97.20 ± 0.88 | +1.66 [−1.04, +4.35] | +33.64 [+28.99, +38.30] | +35.30 |
| pooled (709 clips) | | | | **+1.27 [−0.61, +3.15]**, p = 0.20 | **+28.63 [+25.81, +31.55]** | +29.90 |

The brackets are a per-clip paired bootstrap on the 3-seed-mean correctness. The independent unit is the signer (n = 2).
**About 96% of the paired gap is same-recording familiarity.** Having seen the test signer only on other signs gives
+1.3 pp (n.s.).

**Claims that change.** "the gap is pure signer familiarity: +30.36 pp" (§smg ii, C2, and the "signer-dependent
ceiling" in §pers) is false as worded. In BdSLW60 a signer × sign is one recording, so seen_matched trains on sibling
trials of the test recording. Suggested wording: "...on the identical frozen clips the gap is +30.36 pp, but it is
almost entirely same-recording familiarity. When the test signer is seen in training only through other signs, accuracy
rises by just +1.3 pp (95% CI [−0.6, +3.2]; U02 +0.9, U13 +1.7), so training on a signer's other signs confers little.
The large gains arise when trials of the very recording being tested are in training." The same caveat applies to the
fragility table and the "+29.9 ceiling" (both are seen_matched), and to the matched-split SMG (random clip splits
separate sibling trials).

---------------------------------------------------------------------------------------------------------------------
## Related issue (not fixed, flag only)

`feeders/feeder.py` normalization (`is_vector: False`) subtracts the clip's mean root x/y from **every** entry, including
undetected (0,0) joints. In every joint-stream run, a missing hand is therefore encoded as all 10 joints at −mean(root),
a signer/position-dependent phantom location, and `random_shift` then shifts it as a real joint. This is the same class
of artefact as findings 1–2 and affects all P2 joint-stream numbers. Fixing it needs a new feeder option and full
retraining, which is out of scope here.

## GPU-hours (sacct, allocated GPU × elapsed): 49.8 GPU-h

| job | content | GPU-h |
|---|---|--:|
| <job> J5 | A: Adaptive masked bone, 6 runs | 13.07 |
| <job> J1 | A: BlockGCN masked bone (3 runs) + B: speednorm_masked (3) | 5.89 |
| <job> J2 | B: rootcenter_masked (3) + E (3) | 5.11 |
| <job> J3 | E (6); 3.63 GPU-h lost to a NODE_FAIL on one node, auto-requeued and resumed from epoch 10 | 8.91 |
| <job> J4 | E (3) | 2.55 |
| <job> J6 | D: 5 rand val-selected runs | 13.06 |
| <job> | C: probe v2/v1 | 0.96 |
| <job> | score dumps + rand checkpoint scan | 0.2 |

By task: A ≈ 16.0, B ≈ 5.5, C ≈ 1.0, D ≈ 13.2, E ≈ 14.1 (incl. 3.6 lost). This is over the ~30–40 estimate, mainly
because the Adaptive runs took ~6.5 h at 3 per GPU and because of the node failure.
