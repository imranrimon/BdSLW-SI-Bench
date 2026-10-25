# Item 1 — BdSLW60 dataset-paper baselines under Protocol A

Protocol A: train {1,4,5,6,8,9,11,12} / val 15 / TEST {2,13}; 27-keypoint pose, T=120. Mixed = signer-dependent split `data/bdsl_sd`: checkpoint (or SVM C) selected on sd val (655), scored on sd TEST (1,367) — P2's symmetric protocol; 'old mixed' = best-epoch sd-val Top-1 (previous rule). SMG = Mixed − TEST. ± = population SD over seeds. grad_clip_norm unset (0.0).

## Attention Bi-LSTM + RQ — shared recipe (Track A; chosen base_lr (val U15, seed 0) = 0.01)

| seed | val Top-1 (U15) | TEST Top-1 | macro-F1 | Mixed (sd TEST) | SMG | old mixed (sd val best) | old SMG |
|---|--:|--:|--:|--:|--:|--:|--:|
| 0 | 6.56 | 5.20 | 1.62 | 6.95 | +1.75 | 7.02 | +1.82 |
| 1 | 5.80 | 5.64 | 2.10 | 7.17 | +1.53 | 7.48 | +1.84 |
| 2 | 6.26 | 5.93 | 1.98 | 7.32 | +1.38 | 6.87 | +0.94 |
| **mean ± SD** | 6.21 ± 0.31 | **5.59 ± 0.30** | 1.90 ± 0.21 | 7.14 ± 0.15 | +1.55 | 7.12 ± 0.26 | +1.53 |

## Attention Bi-LSTM + RQ — authors' recipe (Adam 3e-5, bs64, 200 ep)

| seed | val Top-1 (U15) | TEST Top-1 | macro-F1 | Mixed (sd TEST) | SMG | old mixed (sd val best) | old SMG |
|---|--:|--:|--:|--:|--:|--:|--:|
| 0 | 70.69 | 59.27 | 58.22 | 94.81 | +35.54 | 94.20 | +34.93 |
| 1 | 69.16 | 55.46 | 53.20 | 94.00 | +38.54 | 93.89 | +38.44 |
| 2 | 65.19 | 44.54 | 43.83 | 93.42 | +48.87 | 95.42 | +50.88 |
| **mean ± SD** | 68.35 ± 2.32 | **53.09 ± 6.24** | 51.75 ± 5.97 | 94.07 ± 0.57 | +40.99 | 94.50 ± 0.66 | +41.41 |

## Linear SVM — original

Deterministic (sklearn SVC linear): one value, no seeds.

| C | SI val Top-1 | SI TEST Top-1 | sd val | sd TEST |
|--:|--:|--:|--:|--:|
| 0.001 | 69.01 | 60.44 | 96.34 | 96.42 |
| 0.01 | 69.47 | 59.19 | 96.95 | 97.07 |
| 0.1 | 69.47 | 59.12 | 97.25 | 97.15 |
| 1.0 | 69.47 | 59.12 | 97.25 | 97.15 |

C on SI val = 0.01: **TEST Top-1 59.19**, macro-F1 57.55. C on sd val = 0.1: **Mixed (sd TEST) 97.15** (macro-F1 97.18), **SMG +37.95**. Old rule (max sd-val over C) 97.25, old SMG +38.06.

## Linear SVM — flipped (authors' hand-dominance correction; uses _L/_R clip metadata at test time)

Deterministic (sklearn SVC linear): one value, no seeds.

| C | SI val Top-1 | SI TEST Top-1 | sd val | sd TEST |
|--:|--:|--:|--:|--:|
| 0.001 | 68.09 | 67.18 | 97.25 | 96.78 |
| 0.01 | 67.79 | 66.74 | 97.56 | 97.15 |
| 0.1 | 68.09 | 66.74 | 97.86 | 97.29 |
| 1.0 | 68.09 | 66.74 | 97.86 | 97.29 |

C on SI val = 0.001: **TEST Top-1 67.18**, macro-F1 65.52. C on sd val = 0.1: **Mixed (sd TEST) 97.29** (macro-F1 97.35), **SMG +30.11**. Old rule (max sd-val over C) 97.86, old SMG +30.68.

## Reading / caveats

- **Shared-recipe Bi-LSTM does not train.** Under SGD+nesterov at every grid LR (val U15 seed 0: 0.1 → 4.73, 0.05 → 5.80, 0.01 → 6.56) the train loss plateaus at ~3.75 (chance ln60 = 4.09); the identical model reaches train loss ~0.12 under the authors' Adam recipe. This is an optimizer/recipe incompatibility of the authors' design (raw ×100 inputs, no normalisation layer, 30k-dim flatten head), not a code bug. Its SMG is meaningless (both splits at floor). **Use the authors'-recipe row as the Bi-LSTM baseline** (RECIPE_CONTROL item 3: "author recipe" rows are allowed when clearly labelled).
- **Headline numbers to quote:** Bi-LSTM (authors' recipe) TEST **53.09 ± 6.24**, macro-F1 51.75, Mixed 94.07, SMG **+40.99**. Linear SVM TEST **59.19** (original) / **67.18** (flipped), Mixed 97.15 / 97.29, SMG **+37.95 / +30.11**.
- Our flipped SVM (67.18) reproduces the published SVM 67.6 almost exactly; our Bi-LSTM (53.1) is far below the published 75.1. The published 75.1 is the **best of 10 cross-validation-fold models selected on the test set** (authors' notebook: `bestTestAccuracy` over folds), on different test signers (U4/U8), with 75 MediaPipe landmarks incl. depth. Their own per-fold test accuracies for the RQ model were 69–72 (notebook outputs).
- Bi-LSTM seed 2 is low (44.5 TEST): the authors' recipe is high-variance across seeds, consistent with the published architecture having no normalisation.
- The flipped SVM uses each test clip's `_L/_R` handedness annotation at inference — metadata the pose models do not get. Treat **original** as the like-for-like row; flipped as the faithful reproduction of the authors' number.

## Deviations from the original paper (all forced by the shared footing or fixing leaks)

| Aspect | Authors | Here |
|---|---|---|
| Input | 75 MediaPipe landmarks (33 pose + 2×21 hands) × (x, y, depth) | 27 keypoints (7 body + 2×10 hand) × (x, y, confidence) from data/bdsl_si |
| Split | test U4/U8, train all others, 10-fold CV | Protocol A (train {1,4,5,6,8,9,11,12}, val U15, TEST U02/U13) |
| Model selection | best of 10 CV folds **on test** | best epoch / C on val U15; TEST scored once |
| Sequence length | zero-pad (Bi-LSTM) / stretch (SVM) to 164 | window-120 crop (feeder) then zero-pad (Bi-LSTM) / stretch to 120 (SVM) |
| RQ min-max stats | fit on train+test | fit on train only (data/bdsl_si_rq, data/bdsl_sd_rq) |
| RQ 3rd channel | depth: relative-translated, quantised 5/3 levels | confidence: not translated, quantised 5/3 levels |
| Calibration reference | first frame of the whole video | first valid frame of each clip |
| Bi-LSTM training | Keras, early stopping on train loss (patience 5) | PyTorch port; no early stopping, best-val checkpoint |
| Bi-LSTM augmentation | none | shared temporal crop only (random_choose, window 120); mirror/shift/normalisation OFF |
| SVM scaling | StandardScaler refit on the test set | fit on train only |
| SVM C | fixed 1.0 | grid {0.001, 0.01, 0.1, 1} on val (ties → smaller C) |
| Hand-dominance flip | FLIPPED for headline | both variants reported |
| grad clipping | n/a | unset (0.0), matching the headline Protocol-A/LOSO runs |

## Provenance

- Jobs: <job> (`abl_shared`: LR sweep + 3 seeds SI/TEST + 3 seeds mixed), <job> (`abl_author`), <job> (SVM, calibrated; supersedes <job>, whose uncalibrated outputs are archived in `work_dir/bdsl_svm_baseline_*_nocalib_oldmixed`).
- Code: `model/attn_bilstm.py`, `preprocessing/make_rq_bundle.py`, `tools/run_svm_baseline.py`, `tools/run_resumable_seed.py`, `tools/summarize_item1_baselines.py`, `scripts/hpc/slurm_attn_bilstm.sbatch`, `scripts/hpc/slurm_svm_baseline.sbatch`, configs `config/bdsl_attn_bilstm{,_author}_{si,si_test,sd}.yaml`.
- Authors' code: github.com/hasanssl/BdSLW60_Code (pipeline_LSTM_225_FLIPPED_ST5_B64_D5MAX.ipynb, pipeline_SVM_POSE_HAND.ipynb, PREPROCESSING_RQ/*).
