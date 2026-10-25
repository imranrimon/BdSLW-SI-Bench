# P2 analysis 1: per-signer difficulty on held-out LOSO (exploratory, n = 11 signers)

## Scripts (run in this order)

1. `tools/p2_analysis/signer_covariates.py` computes per-clip and per-signer pose covariates and the MMD. It writes `signer_covariates.json` and `signer_covariates_per_clip.csv`.
2. `tools/p2_analysis/signer_difficulty.py` writes `signer_difficulty.json`. It produces:
   - the per-signer × arch table,
   - Spearman correlations,
   - MixedLM fits,
   - the clip-level within-signer GLM.
3. `tools/p2_analysis/make_mirrored_test.py` + `tools/eval_ckpt_on_npy.py --dump`, then `tools/p2_analysis/mirror_intervention.py`. This is the test-time mirroring intervention plus the Protocol-A mirror-augmentation decomposition. It writes `handedness_mirror.json`.

All JSON files are in `docs/p2_analysis_2026-10/`.

## Data and IDs

- **Accuracy source:** held-out-signer Top-1 from `docs/loso_heldout/<file>.csv`, using the paper's file choice: BlockGCN = `block_gcn_lr01`, GNN+Transformer = `gnn_transformer_clip5`. There are 11 folds U01, 02, 04, 05, 06, 08, 09, 11, 12, 13, 15, with 3 seeds each.
- **Fold → clip mapping:** `data/bdsl_si_loso/test_U<XX>_val_U<YY>/test_label.pkl`.
  - Every held-out clip name starts with `U<XX>`; the script asserts this.
  - Fold U01 validates on U02; every other fold validates on U01.
  - Fold train = the remaining 9 signers.
- **Covariates** are computed from exactly those 7,768 held-out clips, using:
  - the MediaPipe cache `data/bdsl_cache/<class>/<clip>.npz`, shape (3, T, 27): x, y, visibility. Joints 0–6 are body, 7–16 are MediaPipe `right_hand_landmarks`, 17–26 are `left_hand_landmarks`. Hand landmarks carry no confidence, so a missing hand is all-zero;
  - the raw video headers in `data/bdslw60_clips` (fps, resolution).
  - All clips are 30 fps. Resolutions mix 1280×720, 1920×1080, 960×720 and 640×480 within signers. Distances are aspect-corrected.
- **Clip-level held-out predictions did not exist** (the `work_dir/*loso*/eval_results/best_acc.pkl` are validation-signer). I regenerated them on CPU with `tools/eval_ckpt_on_npy.py --dump` for:
  - Adaptive GNN seed 0 and BlockGCN lr 0.01 seed 0, all 11 folds;
  - output in `work_dir/p2_analysis_tmp/loso_dumps/`.
  - All 22 runs reproduce the `docs/loso_heldout` Top-1 exactly (e.g., Adaptive U01 52.00, BlockGCN U01 41.54).

## (0) Per-signer table (held-out Top-1, 3-seed mean; `per_signer_accuracy` in JSON)

| Signer | Adaptive | ST-GCN | SLGTF | GNN+T‡ | AttnGNN | BlockGCN | GNN+LSTM | PoseLSTM | **mean** | SD across arch |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| U01 | 49.6 | 44.0 | 48.2 | 47.5 | 41.7 | 42.6 | 45.1 | 39.8 | **44.8** | 3.4 |
| U12 | 56.7 | 59.0 | 49.0 | 52.6 | 53.0 | 51.1 | 45.8 | 44.0 | **51.4** | 5.1 |
| U13 | 72.2 | 66.5 | 71.6 | 60.6 | 63.6 | 63.7 | 58.6 | 56.1 | **64.1** | 5.8 |
| U02 | 80.5 | 76.5 | 77.1 | 74.3 | 75.2 | 74.3 | 75.9 | 68.3 | **75.3** | 3.4 |
| U04 | 79.4 | 81.6 | 78.1 | 79.9 | 76.1 | 69.2 | 73.7 | 66.1 | **75.5** | 5.5 |
| U15 | 89.9 | 87.0 | 89.3 | 86.2 | 82.1 | 81.4 | 78.8 | 77.4 | **84.0** | 4.7 |
| U11 | 88.1 | 86.4 | 84.5 | 85.6 | 85.9 | 85.3 | 87.1 | 81.5 | **85.6** | 2.0 |
| U06 | 90.8 | 86.8 | 89.3 | 90.0 | 84.5 | 84.6 | 84.4 | 77.1 | **85.9** | 4.4 |
| U08 | 89.4 | 89.6 | 91.4 | 88.5 | 86.6 | 86.5 | 83.5 | 78.1 | **86.7** | 4.2 |
| U09 | 91.5 | 93.1 | 90.4 | 89.0 | 90.9 | 88.0 | 85.2 | 83.1 | **88.9** | 3.4 |
| U05 | 94.4 | 94.4 | 94.9 | 93.9 | 93.4 | 91.3 | 92.6 | 84.3 | **92.4** | 3.5 |

- Signer difficulty is shared across architectures. Kendall's W for agreement of the arch ranking across signers is 0.71. The SD across archs within a signer (2.0–5.8 pp) is small next to the 47.6 pp spread of signer means.
- The BlockGCN column reproduces `tab:losofolds` (42.6 … 91.3).

## (a) Signer-level covariates and Spearman correlation with mean accuracy

Settings: n = 11, permutation p with 100k permutations, BH and Holm across the 13-covariate family. Values are from `spearman` and `signer_covariates` in `signer_difficulty.json`.

| Covariate | ρ | p_perm | q_BH | p_Holm | LOO ρ range |
|---|--:|--:|--:|--:|---|
| Shoulder width (camera framing, fraction of frame width) | −0.76 | 0.008 | 0.083 | 0.11 | [−0.87, −0.69] |
| **Fraction of clips tagged `_L` (left-hand production)** | **−0.74** | 0.013 | 0.083 | 0.15 | [−0.79, −0.65] |
| Left-wrist share of body-wrist motion | −0.66 | 0.030 | 0.13 | 0.33 | [−0.83, −0.55] |
| Hand size / shoulder width | −0.62 | 0.048 | 0.13 | 0.48 | [−0.75, −0.52] |
| Fraction of clips where MediaPipe tracks the left hand more | −0.61 | 0.052 | 0.13 | 0.48 | [−0.78, −0.48] |
| Wrist low-visibility rate | −0.59 | 0.061 | 0.13 | 0.49 | [−0.70, −0.49] |
| Pose-motion MMD to the fold's 9 train signers | −0.41 | 0.21 | 0.40 | 1.0 | [−0.60, −0.21] |
| Arm length / shoulder width | −0.35 | 0.30 | 0.48 | 1.0 | |
| Mean joint speed | −0.16 | 0.63 | 0.87 | 1.0 | |
| Fraction of clips at 4:3 | −0.14 | 0.69 | 0.87 | 1.0 | |
| Signing-hand missing rate | −0.12 | 0.73 | 0.87 | 1.0 | |
| Both-hands-missing frame rate | −0.08 | 0.82 | 0.89 | 1.0 | |
| Clip length T | +0.01 | 0.99 | 0.99 | 1.0 | |

- **No covariate survives multiplicity correction** (best q_BH = 0.083).
- With n = 11, |ρ| must exceed about 0.62 for nominal p < 0.05.
- Hand dropout looks null at signer level only because it is concentrated in a single signer (U12). See (c).

**MMD details.**

- Descriptor: the `tools/p2_analysis/pose_mmd.py` clip descriptor with RBF MMD at the median-heuristic bandwidth. n = 400 clips per side, mean of 5 draws.
- U01 is by far the most distant signer (0.098 ± 0.010). The others are 0.006–0.051. U08 is second-most distant (0.051 ± 0.025, noisy) but easy (86.7).
- MMD captures U01's handedness, because the descriptor includes per-hand speeds. It is not a general predictor of difficulty.

**Handedness metadata check.**

- The filename hand tag (`_L`/`_R`) agrees with "which hand MediaPipe tracks more" on 80% of clips.
- It agrees with body-wrist motion dominance on only 63%. The body-wrist rule is the weakest handedness proxy.
- U01 is the only majority-left signer: 67% `_L` and 62% tracked-left. U13 (30%) and U02 (21%) come next.

## (b) Mixed models on the 88 signer × arch cell means (`mixed_models` in JSON)

| Covariate (per +1 SD) | Ma: acc ~ cov + (1\|arch), β ± SE, p_Holm | Mb: + (1\|signer), β ± SE, p_Holm |
|---|---|---|
| Fraction `_L` | −11.7 ± 1.2, <1e-4 | −11.7 ± 3.6, 0.012 |
| Fraction tracked-left | −11.1 ± 1.3, <1e-4 | −11.1 ± 3.8, 0.038 |
| Left-wrist motion share | −12.9 ± 1.1, <1e-4 | −12.9 ± 3.1, <0.001 |
| Wrist low-visibility | −9.8 ± 1.4, <1e-4 | −9.8 ± 4.2, 0.18 |
| MMD to fold-train | −9.0 ± 1.5, <1e-4 | −9.0 ± 4.4, 0.36 |
| Shoulder width | −8.2 ± 1.5, <1e-4 | −8.2 ± 4.5, 0.55 |
| Both-hands-missing | −6.7 ± 1.6, <1e-4 | −6.7 ± 4.8, 0.96 |
| Clip length | +0.3 ± 1.8, 0.87 | +0.3 ± 5.3, 1.0 |

- **Ma (the requested model) is pseudo-replicated.** It treats the 8 arch means of a signer as independent observations of a signer-level covariate, so almost everything comes out "significant". Its SEs are about 3× too small.
- Mb adds a signer random intercept. Then only the handedness covariates survive Holm. These are Wald z on 11 signers, so they are still optimistic.
- **Post-hoc two-predictor model:** fraction `_L` + both-hands-missing rate, chosen after seeing that U01 is the left-handed outlier and U12 the hand-dropout outlier.
  - OLS on the 11 signer means: R² = 0.885 (adjusted 0.857; LOO R² 0.81–0.95).
  - Coefficients: −13.6 pp/SD (p = 1e-4) and −9.4 pp/SD (p = 0.001), df = 8.
  - In the crossed MixedLM it explains 86% of the signer variance component (251.7 → 35.5).
  - This is a hypothesis generator, not a confirmed model.

## (c) Clip-level, within-signer evidence (the strong result)

**Model.** Binomial GLM: correct ~ left_tag + C(signer) + C(class) (+ covariates), with signer-clustered SEs and 11 clusters. Data: 7,768 clips per arch, Adaptive GNN seed 0 and BlockGCN seed 0 (`clip_level.by_arch` in `signer_difficulty.json`).

| | Adaptive GNN | BlockGCN lr 0.01 |
|---|---|---|
| `_L` clip, OR (signer + class FE) | **0.091** [0.033, 0.25] | **0.22** [0.12, 0.41] |
| `_L` clip, OR (+ hand-missing, speed, log T) | 0.075 [0.033, 0.17] | 0.20 [0.12, 0.34] |
| Both-hands-missing frame rate, OR per SD | **0.42** [0.31, 0.57] | **0.43** [0.36, 0.53] |
| Speed, OR per SD | 0.85 [0.67, 1.09] | 0.88 [0.75, 1.04] |
| log T, OR per SD | 1.08 [0.88, 1.32] | 1.24 [1.05, 1.45] |
| Tracked-left (metadata-free proxy), OR | 0.41 [0.23, 0.75] | 0.51 [0.30, 0.87] |

**Within-signer accuracy, `_L` vs `_R` clips (Adaptive GNN seed 0):**

| Signer | `_L` | `_R` |
|---|---|---|
| U01 | 35.3 (n = 654) | 86.0 (n = 321) |
| U02 | 45.9 (159) | 90.8 |
| U04 | 52.4 (105) | 82.1 |
| U11 | 45.7 (92) | 94.6 |
| U13 | 48.4 (186) | 82.3 |

- The `_L` penalty is **heterogeneous**. In U05, U06, U08 and U12 the few `_L` clips (28–38) are recognised as well as or better than `_R` clips. The tag may not always mean a left-dominant production.
- Restricted to `_R` clips, U01 is an ordinary signer (Adaptive 86.0 vs a median of about 90). The remaining hard signer is **U12** (56.6 on `_R` clips), the hand-dropout outlier: no hand detected in 57% of frames vs 6–34% for the others.
- Dropping `_L` clips cuts the Adaptive per-signer SD from 14.6 to 10.8. Without U12, the `_R`-only range is 82.1–94.6.

## (d) Causal check: test-time mirroring (`handedness_mirror.json`)

Same checkpoints (Adaptive GNN seed 0, each fold's model-best). The `_L`-tagged clips are mirrored exactly as the training feeder's `random_mirror` does (`flip_index` + x → 1 − x).

| Fold | Base | Mirror `_L` clips | `_L` acc base → mirrored | Clips fixed / broken (McNemar p) |
|---|--:|--:|---|---|
| U01 | 52.00 | **76.41** | 35.3 → 71.7 | +270 / −32 (p < 1e-40) |
| U02 | 81.18 | **90.73** | 45.9 → 90.6 | +77 / −6 |
| U04 | 77.24 | 81.48 | 52.4 → 78.1 | +33 / −6 |
| U11 | 88.72 | 92.74 | 45.7 → 79.3 | +42 / −11 |
| U13 | 72.14 | **82.13** | 48.4 → 81.7 | +64 / −2 |

- **Control:** mirroring *all* U01 clips drops the `_R` clips from 86.0 to 69.2. The effect is handedness-specific, not a generic gain from mirroring.
- **Metadata-free rule:** mirror a clip if MediaPipe tracks the left hand in more frames. Evaluated on all 11 folds:
  - U01 52.0 → 71.2, U02 81.2 → 86.2, U13 72.1 → 75.0, U11 88.7 → 91.2.
  - It costs 1.4–3.7 pp on the 6 right-dominant signers (U05, 06, 08, 09, 12, 15).
  - Net LOSO mean: +1.25 pp, CI [−1.8, +5.4] (signer bootstrap); 4 folds better, 6 worse (sign test p = 0.75).
  - Per-signer SD falls from 14.6 to 11.4 and the worst signer rises from 52.0 (U01) to 54.5 (U12).
  - **This crude rule is not a general fix.** A reliable handedness detector, or train-time mirroring, is needed.

**Train-time mirror augmentation, decomposed on Protocol-A TEST (paper `tab:mirroraug`; 3 seeds; 345 `_L` + 1,020 `_R` clips):**

| | Overall | `_L` clips | `_R` clips | Share of gain from `_L` |
|---|---|---|---|---|
| BlockGCN | +7.57 (68.82 → 76.39) | +15.8 (61.2 → 77.0) | +4.8 | 53% (`_L` are 25% of clips) |
| SLGTFormer | +7.96 (73.26 → 81.22) | +19.2 (65.1 → 84.3) | +4.2 | 61% |

This gives a mechanism for the paper's mirror-augmentation finding: most of the gain is left-hand productions becoming recognisable. With mirroring on, the `_L`/`_R` accuracies equalise (77.0/76.2 and 84.3/80.2).

## Reading

- Per-signer difficulty is largely explained by two concrete, measurable factors, not by "signer style" in general:
  1. **Left-handed productions.** U01 signs about two thirds of clips left-handed. U02, U04, U11 and U13 also carry a 21–30% `_L` share, and those clips fail at 35–52%.
  2. **MediaPipe hand dropout** (U12).
- Speed, clip length, body proportions and camera aspect are null or weak at n = 11.
- The signer-level associations are exploratory. The **within-signer clip-level effect and the mirroring intervention are not underpowered** (thousands of clips, signer and class fixed effects, counterfactual re-evaluation of the same checkpoints).

## Caveats

- n = 11 signers for every signer-level statistic.
- Clip-level results use one seed of two archs. They reproduce exactly across both archs in direction, with magnitude 0.09–0.22 OR.
- The `_L`/`_R` filename tag is dataset metadata whose semantics are undocumented in this repo. It is inferred here as the producing hand: 80% agreement with MediaPipe tracking, and the mirror intervention confirms it behaviourally for 5 signers.
- The MMD at n = 400 has draw SD up to 0.025 (U08).
- A covariate the model never sees directly (e.g. shoulder width) may proxy camera framing. The shoulder-width ρ = −0.76 is the top raw correlate but is not corrected-significant and has no mechanism tested here.
