# P2 add-ons (2026-10-22): label-free test-time adaptation (k=0 enrollment) + GRL sweep at 3 seeds

Nothing here edits the paper `.tex`, `main.py` or `feeders/feeder.py`, and nothing was committed. Job IDs: `jobids.txt`.
GPU use: **4.34 GPU-h** in total (GRL training + probes 1.94, TTA evaluation 2.40). This includes one cancelled job: <job> was allocated across 2 nodes, so only 1 of its 2 GPUs was usable. It was resubmitted as two 1-GPU jobs, and the runs resumed from their epoch-2 checkpoints.

Full auto-generated tables: `tta_tables.md` (from `tools/tta_report.py`) and `grl_3seed.md` (from `tools/grl_3seed_report.py`). Raw per-clip records: `raw/`.

---------------------------------------------------------------------------------------------------

## TASK 1: label-free test-time adaptation to an unseen signer

**Protocol (transductive).** Each method sees **all unlabeled clips of one target signer** (statistics, or one adaptation epoch) and never sees a label. Signers are adapted **separately**: Protocol-A TEST signers U02 and U13 each get their own adaptation, and every LOSO fold adapts to its held-out signer. Every (run, signer, method) unit starts from a fresh copy of the SI checkpoint. This is the k=0 point of the enrollment curve (tab:pers): no labeled support clips.

**Checkpoints (eval only, no retraining).**
- **BlockGCN, lr 0.01** (the paper's current model). Protocol A: `bdsl_block_gcn_si_lr01_scratch_seed{0,1,2}` best-on-val. LOSO: `bdsl_block_gcn_si_lr01_loso_test*_seed*` best-on-fold-val, as in `docs/loso_heldout/block_gcn_lr01.csv`. All 36 runs are complete.
- **Adaptive GNN, lr 0.01** (the current Table-7 checkpoints). Protocol A uses the checkpoints scored by `bdsl_adaptive_gnn_si_seed*_test`. LOSO uses `docs/loso_heldout/adaptive_gnn.csv` runs.
- **SLGTFormer** (Table-7 checkpoints).
- The no-adaptation baselines reproduce the paper exactly:
  - BlockGCN TEST 68.82 ± 0.56 and LOSO 74.37 ± 16.07.
  - Adaptive GNN TEST 79.07 ± 0.89.
- **Truncated runs are excluded.** A run counts as complete only if `log.txt` reaches `Eval epoch: num_epoch`, per the coordinator's audit. The excluded set was fixed at the moment of evaluation:
  - Adaptive GNN: LOSO 08/s2 and 09/s2 (31 of 33 runs kept). The baseline LOSO becomes 80.34 ± 15.02 instead of the paper's 80.2 ± 14.9.
  - SLGTFormer: Protocol-A seed 2 (stopped at epoch 25), so TEST uses **2 seeds**: 76.63 ± 0.29, versus the paper's 73.26 ± 4.77, which includes the truncated seed. LOSO loses 9 of 33 runs (01/s2, 02/s2, 06/s0, 08/s0, 09/s0, 11/s0, 12/s2, 13/s2, 15/s2), leaving 24 runs and every fold covered.
  - Treat **SLGTFormer as secondary**.

**Methods** (`tools/tta_eval.py`; `tools/tta_ssl_eval.py` for (iv)):
- **(i) AdaBN.** Reset every BatchNorm's running stats, then re-estimate them as a cumulative average from one no-grad pass over the signer's clips.
- **(ii) Tent** (Wang et al., ICLR'21; reference configuration):
  - BN layers use test-batch statistics; only BN affine parameters train.
  - Adam, entropy loss, 1 epoch, seeded shuffle.
  - lr ∈ {0, 1e-5, 1e-4, 1e-3, 1e-2}. lr 0 means test-batch BN statistics only.
  - Batch size = the model's training batch size (BlockGCN 32, Adaptive GNN and SLGTFormer 16).
  - Reported accuracy is **online** (the reference protocol). An offline second pass is in `tta_tables.md`.
  - `1e-3` is pre-declared as the Tent default. No lr can be picked label-free, so the grid is a sensitivity analysis, not a selection.
- **(iii) Per-signer input standardization (CMVN analogue).** On the model input (after the feeder's crop and node-0 centring), z = (x − μ_signer)/σ_signer is mapped back to the training distribution (z·σ_train + μ_train). μ_train and σ_train come from the run's own training bundle. Only valid entries are used. Variants:
  - `cmvnxy`: per (channel, joint).
  - `cmvnxy_c`: per channel.
  - `cmvnxy_adabn`: `cmvnxy` followed by AdaBN.
  - **Only x and y are standardized.** Channel 2 is MediaPipe *visibility* (body) or a constant 1.0 (hands), not a coordinate. A first run that also standardized it collapsed per-channel CMVN to about 2% accuracy. Those records (`cmvn*`) are superseded and ignored by the report.
- **(iv) Masked-pose SSL pass** (BlockGCN only; run because (i)–(iii) gained on TEST):
  - Objective: SHuBERT-style masked k-means-code prediction (K = 64, `pose_motion` features fitted on the signer's own clips).
  - The masking and targets sit at the backbone's strided output resolution (one output step = an 8-frame window).
  - Schedule: head warm-up for 1 epoch, then 3 epochs over all backbone parameters with Adam at lr ∈ {1e-4, 1e-3}. BN keeps source statistics. The classifier is frozen.
  - Reported as `ssl_lr*`, plus `ssl_lr*_adabn` with BN re-estimated afterwards.

**Statistics.**
- TEST: per seed, pooled Top-1 over both signers' 1,365 clips (each signer adapted separately). Mean ± population SD over seeds, the paper's Protocol-A convention.
  - Paired bootstrap: per-clip correctness averaged over seeds, clips resampled within signer, 10k draws.
- LOSO: fold value = mean over the fold's complete seeds. Mean ± sample SD over the 11 folds (the Table-7 convention).
  - **Signer-clustered** paired bootstrap over folds of the fold-level Δ (10k draws).
- p values are two-sided.

### Results (Δ = method − none, in percentage points)

| model | method | TEST Top-1 | Δ TEST [95% CI], p | LOSO mean ± SD | Δ LOSO [95% CI], p | folds ↑ |
|---|---|--:|--:|--:|--:|--:|
| **BlockGCN** | none | 68.82 ± 0.56 | – | 74.37 ± 16.07 | – | – |
| | AdaBN | **73.11 ± 1.01** | **+4.30** [+2.88, +5.74], <0.001 | 73.10 ± 19.24 | −1.27 [−3.85, +0.76], 0.27 | 5/11 |
| | CMVN-xy (joint) | 68.18 ± 0.45 | −0.63 [−2.10, +0.85], 0.42 | 72.65 ± 18.45 | −1.71 [−3.39, −0.19], 0.026 | 3/11 |
| | CMVN-xy (channel) | 70.21 ± 0.56 | +1.39 [+0.15, +2.64], 0.028 | 73.96 ± 17.46 | −0.41 [−1.62, +0.76], 0.49 | 6/11 |
| | CMVN-xy + AdaBN | 71.75 ± 0.31 | +2.93 [+1.37, +4.49], <0.001 | 72.82 ± 19.69 | −1.55 [−4.22, +0.65], 0.21 | 5/11 |
| | Tent lr 1e-3 | 71.43 ± 1.56 | +2.61 [+1.17, +4.10], <0.001 | 72.83 ± 18.73 | −1.53 [−3.47, −0.02], 0.045 | 5/11 |
| | Tent lr 0 (batch-stat BN) | 70.62 ± 1.71 | +1.81, 0.017 | 72.34 ± 18.99 | −2.02, 0.006 | 2/11 |
| | Tent lr 1e-2 | 71.04 ± 2.29 | +2.22, 0.005 | 72.67 ± 18.26 | −1.69, 0.007 | 2/11 |
| | SSL lr 1e-4 | 46.18 ± 6.11 | −22.64, <0.001 | 49.19 ± 13.75 | −25.18, <0.001 | 0/11 |
| | SSL lr 1e-4 + AdaBN | 69.52 ± 0.84 | +0.71 [−0.76, +2.17], 0.35 | 70.86 ± 19.10 | −3.51 [−5.81, −1.57], <0.001 | 2/11 |
| | SSL lr 1e-3 (± AdaBN) | 3.10 / 27.86 | −65.7 / −41.0 | 2.60 / 29.87 | −71.8 / −44.5 | 0/11 |
| **Adaptive GNN** | none | 79.07 ± 0.89 | – | 80.34 ± 15.02 (31 runs) | – | – |
| | AdaBN | **81.20 ± 1.16** | **+2.12** [+1.15, +3.13], <0.001 | 80.45 ± 16.51 | +0.11 [−1.12, +1.22], 0.83 | 8/11 |
| | CMVN-xy (joint) | 77.39 ± 1.59 | −1.68 [−2.81, −0.54], 0.005 | 79.87 ± 16.32 | −0.48, 0.37 | 6/11 |
| | CMVN-xy (channel) | 79.07 ± 1.55 | 0.00, 1.0 | 80.25 ± 15.72 | −0.10, 0.72 | 6/11 |
| | CMVN-xy + AdaBN | 79.78 ± 1.04 | +0.71, 0.19 | 80.30 ± 16.58 | −0.04, 1.0 | 6/11 |
| | Tent lr 1e-3 | 78.88 ± 0.91 | −0.20 [−1.39, +0.98], 0.73 | 79.34 ± 16.53 | −1.01 [−2.01, −0.16], 0.011 | 2/11 |
| | Tent lr 1e-2 | 73.19 ± 2.46 | −5.89, <0.001 | 74.01 ± 20.23 | −6.33, <0.001 | 0/11 |
| **SLGTFormer** (secondary; TEST 2 seeds, LOSO 24/33 runs) | none | 76.63 ± 0.29 | – | 80.32 ± 15.78 | – | – |
| | AdaBN | 80.62 ± 0.04 | +3.99 [+2.82, +5.24], <0.001 | 79.97 ± 17.01 | −0.34 [−1.53, +0.74], 0.58 | 5/11 |
| | CMVN-xy + AdaBN | 79.30 ± 0.26 | +2.67, <0.001 | 79.83 ± 17.27 | −0.48, 0.45 | 6/11 |
| | Tent lr 1e-3 | 76.56 ± 0.00 | −0.07, 0.93 | 79.23 ± 16.54 | −1.08 [−1.73, −0.50], <0.001 | 2/11 |
| | Tent lr 1e-2 | 74.47 ± 0.92 | −2.16, 0.004 | 77.14 ± 17.42 | −3.17, <0.001 | 1/11 |

Remaining rows (all Tent lrs, CMVN variants for SLGTFormer, per-signer U02/U13 columns, per-fold Δ) are in `tta_tables.md`.

### Reading

1. **Label-free TTA gives a real but small gain on the two canonical TEST signers, and the gain does not generalize to unseen signers.**
   - On Protocol-A TEST, AdaBN raises accuracy for all three models: BlockGCN +4.3 pp (68.8 → 73.1), Adaptive GNN +2.1, SLGTFormer +4.0. All three gains are significant on the paired clip bootstrap.
   - On the 11 held-out LOSO signers, the same procedure is null or negative: BlockGCN −1.3 pp (CI [−3.9, +0.8]), Adaptive GNN +0.1, SLGTFormer −0.3.
   - The gain on U02/U13 is therefore partly a property of those two signers. Even for them, the LOSO models of folds U02/U13 gain only +0.4 / +2.2 pp (BlockGCN).
   - This reproduces the paper's own caution that the canonical split is a two-signer point estimate.
2. **The hardest signers are hurt most.**
   - BlockGCN AdaBN loses 9.3 pp on U01 and 9.2 pp on U12, the two weakest folds (about 43% and 51% unadapted). Tent and CMVN+AdaBN show the same pattern.
   - Label-free adaptation therefore widens the across-signer spread (LOSO SD 16.1 → 19.2) rather than narrowing it.
   - The across-fold rank correlation between baseline accuracy and Δ is weak (AdaBN ρ = 0.08 for BlockGCN, 0.25 for Adaptive GNN), so this is not a clean "hurts the weak" law. The two worst folds, however, are consistently the biggest losers.
3. **Per-signer input standardization (CMVN)** helps neither model on LOSO.
   - The per-joint version removes signer-specific mean posture that partly carries sign information. It is negative on BlockGCN LOSO (−1.7, p = 0.026).
   - The per-channel version is roughly neutral.
4. **Tent adds nothing over plain batch-statistic normalization.**
   - At lr ≤ 1e-4 Tent is indistinguishable from lr 0 (test-batch BN).
   - Batch-statistic BN is itself *below the unadapted model* on LOSO (BlockGCN −2.0, Adaptive GNN −1.5, SLGTFormer −1.4 vs none; all p < 0.01), and below full-signer AdaBN.
   - Tent at lr 1e-3 is the best Tent setting and is still ≤ AdaBN everywhere.
5. **(iv) A masked-pose SSL pass is harmful.**
   - The objective is barely learnable in a short pass (masked-code accuracy 0.05–0.45), and updating the backbone degrades the frozen classifier (BlockGCN TEST −22.6 pp at lr 1e-4, −65.7 at 1e-3).
   - Re-estimating BN afterwards recovers most of the loss, but it ends below plain AdaBN (TEST +0.7, LOSO −3.5).
   - This is one pre-declared configuration, not a tuned one. It does not show that SSL-TTA cannot work; it shows that the naive variant does not.

**Tent collapse notes.** Diagnostics are first versus last quarter of the 1-epoch trajectory, over all adapted units.
- lr ≤ 1e-4: no movement. Entropy stays roughly flat (BlockGCN 0.42 → 0.43), and 54–56 of 60 classes are still predicted.
- lr 1e-3: mild entropy decrease (BlockGCN 0.41 → 0.39; Adaptive GNN 0.29 → 0.23), no collapse. The minimum is 51 predicted classes, and the largest single-class share is ≤ 0.12.
- **lr 1e-2: onset of collapse.**
  - Entropy roughly halves (Adaptive GNN 0.25 → 0.11; SLGTFormer 0.28 → 0.16).
  - Online batch accuracy declines within the epoch (Adaptive GNN 76.5 → 71.1).
  - The offline pass predicts as few as 25–28 distinct classes, with up to 42% of a signer's clips assigned to one class (SLGTFormer 0.418, Adaptive GNN 0.381).
  - Accuracy drops: Adaptive GNN −6.3 pp on LOSO, SLGTFormer −3.2.
  - BlockGCN degrades less (offline minimum 40 classes, share 0.16).

  Tent is therefore usable only at lr ≤ 1e-3, and there it gives no gain.

---------------------------------------------------------------------------------------------------

## TASK 2: GRL signer-adversarial sweep, 3 seeds per λ

**Recipe.** The paper's current GRL recipe, exactly: `config/lr01_switch/bdsl_block_gcn_si_signeradv_lr01.yaml`, BlockGCN lr 0.01, SGD with Nesterov momentum, warm-up of 5 epochs, step decay at 60 and 80, 100 epochs, **no gradient clipping** (`grad_clip_norm: 0.0`).
- The task brief said "GRL runs clip at 5.0". That was the **old lr-0.1 sweep**.
- The paper's tab:grl numbers (50.70 / 43.81 / 45.27) come from the lr-0.01, unclipped runs. Seed 0 reproduces them **exactly** with this config: 50.70 / 0.804, 43.81 / 0.832, 45.27 / 0.832.

**New runs.** Seeds 1 and 2 for λ ∈ {0.1, 0.5, 1.0}, trained by `tools/main_signer_adv_resumable.py`.
- This file is a resume-safe twin of `main_signer_adv.py`. It imports that file's `_lr_at`, `evaluate`, `_init_seed` and `_import_class`, keeps the training loop line-for-line, and adds only per-epoch resume checkpoints that also save the RNG state.

**Evaluation.** `tools/grl_3seed_report.py` applies the paper's `tools/eval_grl_sweep.py` procedure:
- The val-selected `model-best.pt` is scored on TEST signers 2 and 13.
- \Lnorm{} is the class-conditioned probe on the first 4,000 TRAIN clips (the tab:grl metric).
- The within-class \Lnorm^{wc} (n = 6,000; the appendix's headline probe) is computed as well.

| λ | seeds | TEST Top-1 mean ± SD (per seed) | \Lnorm{} (class-cond.) mean ± SD (per seed) | \Lnorm^{wc} | paper (seed 0 only) |
|---|--:|---|---|---|---|
| 0 (control, headline BlockGCN) | 3 | 68.82 ± 0.56 (68.06 / 69.01 / 69.38) | 0.853 ± 0.021 (0.824 / 0.871 / 0.863) | 0.975 ± 0.002 | 68.82 / 0.853 |
| 0.1 | 3 | **46.76 ± 3.01** (50.70 / 46.23 / 43.37) | **0.791 ± 0.009** (0.804 / 0.786 / 0.784) | 0.973 ± 0.003 | 50.70 / 0.804 |
| 0.5 | 3 | **44.35 ± 3.31** (43.81 / 40.59 / 48.64) | **0.822 ± 0.011** (0.832 / 0.828 / 0.807) | 0.976 ± 0.003 | 43.81 / 0.832 |
| 1.0 | 3 | **46.01 ± 1.93** (45.27 / 44.10 / 48.64) | **0.838 ± 0.008** (0.832 / 0.849 / 0.833) | 0.978 ± 0.002 | 45.27 / 0.832 |

SD is the population SD over seeds, the convention of the paper's control row (0.853 ± 0.021). Sample SDs are in `grl_3seed.md`.

Welch t-tests against the control, n = 3 per arm (indicative only):

| λ | Top-1 vs control | class-conditioned \Lnorm{} vs control |
|---|---|---|
| 0.1 | p = 0.008 | p = 0.036 |
| 0.5 | p = 0.008 | p = 0.16 |
| 1.0 | p = 0.002 | p = 0.43 |

\Lnorm^{wc} does not differ between any arm and the control (0.973–0.978 vs 0.975).

### Claim check against the current paper (sec:grl, tab:grl, Fig. 3 caption, C3 bullet)

| claim in paper | 3-seed status |
|---|---|
| Recognition craters, 68.82 → 43.81 at λ = 0.5 | **Holds.** Every λ falls to 44–47 (−22 to −24 pp, p ≤ 0.008). Restate as 68.82 → 44.35 ± 3.31 (λ = 0.5). |
| "non-monotonic (50.70 at λ = 0.1, 45.27 at λ = 1.0)" | **CHANGES.** The 3-seed means (46.76 / 44.35 / 46.01) lie within about one seed SD (2–3 pp) of each other. Accuracy is flat in λ, not non-monotonic; the 50.70 was a favourable seed 0. |
| "\Lnorm{} does not fall"; "only λ = 0.1 is lower (0.804, 0.02 below the seed-0 control)" | **CHANGES (direction-preserving).** λ = 0.1 now gives 0.791 ± 0.009, and all three seeds (0.784–0.804) lie *below* the control's seed range (0.824–0.871). That is −0.062 vs the control mean (Welch p = 0.036, n = 3), a small but consistent reduction. λ = 0.5 (0.822) sits at the lower edge of the control range and λ = 1.0 (0.838) inside it. The reduction *weakens* as λ grows. |
| "The strongest penalties (λ = 0.5, 1.0) give 0.832, slightly above the seed-matched control (0.824)" | **CHANGES.** The 3-seed values are 0.822 ± 0.011 and 0.838 ± 0.008, both within the control seed range. Say "within the control's seed range" and drop "slightly above". |
| Fig. 3 caption "0.804–0.832 vs control 0.853" | Becomes **0.791–0.838 (3-seed means)**. Error bars can now be shown for every arm, not only the control. |
| Table caption "Each GRL arm is a single run (seed 0)" | Becomes "mean ± SD over 3 seeds". |
| Headline "GRL does not remove the shortcut / wrecks accuracy" (C3) | **Holds, and is sharper.** The penalty buys at most a 0.06 drop in class-conditioned \Lnorm{} for a 22 pp accuracy loss. Within-class decodability (\Lnorm^{wc} 0.973–0.978) is untouched at every λ, so within a sign signer identity stays near-perfectly decodable. |

---------------------------------------------------------------------------------------------------

## Recommended paper text changes (not inserted)

1. **New short paragraph in §sec:pers (or Limitations), label-free k = 0 adaptation.** Suggested text:

   "Label-free test-time adaptation does not close the gap. Using only a new signer's unlabeled clips (transductive; each signer adapted separately), AdaBN re-estimation of BatchNorm statistics lifts BlockGCN on the two canonical TEST signers from 68.8 to 73.1 (+4.3 pp, paired bootstrap p < 0.001; Adaptive GNN +2.1). Over the 11 held-out LOSO signers, however, the same procedure is null (BlockGCN −1.3 pp, signer-clustered 95% CI [−3.9, +0.8]; Adaptive GNN +0.1), and it hurts the two hardest signers by about 9 pp. Tent (entropy minimisation of BN affine parameters, 1 epoch) adds nothing beyond batch-statistic normalisation at lr ≤ 10⁻³ and begins to collapse at 10⁻² (offline predictions concentrate on as few as 25 classes). Per-signer input standardisation and a short masked-pose SSL pass do not help either. Closing the gap requires labeled enrollment (k ≥ 1)."

   Add the table above (BlockGCN and Adaptive GNN rows: none, AdaBN, CMVN-xy+AdaBN, Tent 1e-3) as a k = 0 "label-free" row or column next to tab:pers. Strengthens C3: "few-shot enrollment closes the gap; label-free adaptation does not".
2. **tab:grl.** Replace the single-seed GRL rows with 3-seed mean ± SD: 46.76 ± 3.01 / 0.791 ± 0.009; 44.35 ± 3.31 / 0.822 ± 0.011; 46.01 ± 1.93 / 0.838 ± 0.008. Change the caption from "Each GRL arm is a single run (seed 0)…" to "all arms mean ± SD over 3 seeds"; keep the control row.
3. **sec:grl text.**
   - Replace "Meanwhile recognition craters (68.82 → 43.81 at λ = 0.5) and is non-monotonic (50.70 at λ = 0.1, 45.27 at λ = 1.0)" with "recognition falls by 22–24 pp at every λ (44.4–46.8, flat in λ within seed noise)".
   - Replace the \Lnorm{} sentences with: "only the weakest penalty (λ = 0.1) lowers class-conditioned \Lnorm{}, modestly (0.791 ± 0.009, all seeds below the control range 0.824–0.871), while λ = 0.5 and 1.0 remain within the control range (0.822, 0.838); within-class \Lnorm^{wc} is unchanged at every λ (0.973–0.978 vs 0.975)."
   - Keep the conclusion: no useful accuracy–leakage frontier.
4. **Fig. 3.** Regenerate with 3-seed means and SD error bars for every arm. The caption range becomes 0.791–0.838. The headline "GRL does not reduce signer leakage" becomes "GRL barely reduces signer leakage (≤ 0.06, only at the weakest λ) while wrecking accuracy".
5. **Appendix "Signer-adversarial training".** Already correctly says "no gradient clipping". Add "3 seeds (0/1/2)" to the sweep sentence.
6. **(Separate from this task, flagged by the truncation audit.)** SLGTFormer's Protocol-A row (73.26 ± 4.77) includes the truncated seed 2 (stopped at epoch 25). On the two complete seeds it is 76.63 ± 0.29. The paper's "widest seed band ±4.77" claim depends on that truncated run.

## Files

- **Tools (new):**
  - `tools/tta_eval.py`: AdaBN / Tent / CMVN; resume-safe; skips truncated runs.
  - `tools/tta_ssl_eval.py`: (iv) SSL-TTA.
  - `tools/tta_report.py`: tables and bootstraps.
  - `tools/main_signer_adv_resumable.py`: resume-safe GRL trainer.
  - `tools/grl_3seed_report.py`: GRL 3-seed Top-1, \Lnorm{} and \Lnorm^{wc}.
- **SLURM:** `scripts/hpc/slurm_tta_grl_2026_10.sbatch` (8-h chunks with USR1 requeue, scratch/GPU guards, ≤3 runs per GPU, `-N 1`); task lists in `scripts/hpc/tta_grl_2026_10/`.
- **Results:**
  - `docs/tta_grl_2026-10/tta_tables.md` and `grl_3seed.md`.
  - `raw/tta_*.jsonl`: per-clip correctness, Tent trajectories, SSL losses.
  - `raw/grl_3seed.jsonl` and `raw/tta_summary.json`.
- **New checkpoints:** `work_dir/bdsl_block_gcn_si_signeradv_lr01_gl{01,05,10}_seed{1,2}/`. Each contains `model-best.pt`, `model-99.pt` and `_resume.pt`.
  - The `_resume.pt` files can be pruned under the checkpoint-retention rule.
  - Their val rows were appended to `results_final.csv`, as `main_signer_adv.py` does.
