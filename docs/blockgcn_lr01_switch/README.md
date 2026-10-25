# P2 BlockGCN lr-0.01 switch: every BdSLW60 BlockGCN experiment re-run at base LR 0.01

**Decision (user, 2026-10-07): full switch.** The shared-recipe BlockGCN (base LR 0.1) did not use the
validation-selected learning rate. `config/bdsl_block_gcn_si_lr01.yaml` is identical except
`base_lr: 0.01`, with no gradient clipping, mirror off and 100 epochs. It scores higher on val (77.96 vs
74.25; docs/audit_reruns_2026-10/r3_blockgcn_lr.md). Every BdSLW60 BlockGCN experiment in P2 was re-run
at lr 0.01 with the protocol otherwise unchanged.

Nothing here edits the paper `.tex` and nothing is committed. lr-0.1 configs and work_dirs are untouched:
new configs are in `config/lr01_switch/`, and every new experiment name carries `lr01`.

Statistics: ± is the **population** SD over seeds (n=3 unless stated). Some old P2 values are sample SDs,
so an old-column SD here can be smaller than the paper's for the same runs (e.g. RTMPose 7.44 here vs
9.12 in P2). The old columns are recomputed by the same code from results_final.csv and saved scores, and
they reproduce every P2 value exactly. This doubles as a check that the aggregation is correct.

## Summary: old (lr 0.1) → new (lr 0.01)

| # | experiment (P2 location) | old | new | conclusion |
|---|---|--:|--:|---|
| 1 | Protocol A held-out TEST (tab:arch, abstract) | 65.23 ± 2.24 | **68.82 ± 0.56** | SD shrinks 4× |
| 2 | Mixed `bdsl_sd`, SMG col. (Table 7) | 97.59 / SMG +32.36 | **97.98 / SMG +29.16** | holds |
| 3 | matched random splits, SMG headline | 95.18 [90.2, 97.5] → **+29.95** | 98.23 [98.1, 98.4] → **+29.41** (val-selected 98.12 → +29.30) | **holds**: headline SMG moves 0.5 pp |
| 4 | LOSO, 11 folds (sec:loso, Fig. 2, tab:losofolds) | 65.09, CI [54.94, 74.31], range 28.8–86.6, SD 17.17 | **74.37, CI [64.72, 82.91], range 42.6–91.3, SD 16.07** | +9.3 pp mean; signer spread survives |
| 5 | GRL sweep, λ = 0.1 / 0.5 / 1.0 (tab:grl) | 48.86 / 41.90 / 51.65; L_norm 0.760 / 0.722 / 0.732 vs control 68.47 / 0.790 | 50.70 / 43.81 / 45.27; L_norm 0.804 / 0.832 / 0.832 vs control 68.82 / 0.824 (s0) | **holds, stronger**: GRL no longer reduces L_norm at all |
| 6 | leak table, supervised BlockGCN (sec:leak) | cc 0.763, wc 0.968 (1 seed) | **cc 0.853 ± 0.021** (0.824 / 0.871 / 0.863), wc 0.975 | **wording changes** (see below) |
| 7 | cue perturbations (sec:cues) | control 63.3; hands-only 44.0 ± 25.3, body-only 33.3 ± 19.5 | control 68.86; all cells SD ≤ 1.8 | **interpretations change** (see below) |
| 8 | few-shot enrollment, U02/U13 (tab:pers) | k=0 73.5 / 61.4; k=1 83.4 / 82.3; k=5 94.8 / 95.9 | k=0 69.8 / 66.0; k=1 89.4 / 90.6; k=5 96.5 / 97.8 | holds, slightly stronger |
| 9 | 2D diversity grid (tab:div, Fig. 4a) | 2→8 signers +20.1 / +23.3 pp | 2→8 signers +27.7 / +27.3 pp; **not monotonic** | **text changes** (see below) |
| 10 | BdSLW60 SMG(k), k = 2→8 (Fig. 4b, tab:smgk) | 59.9 → 37.1 | **61.6 → 30.4** | holds: decay is steeper |
| 11 | paired-instance SMG + fragility (sec:smg, app:fragility) | gap 27.25 ± 9.40; ceiling 60.9 → 87.8 (+26.9) | **gap 30.36 ± 6.06; ceiling 66.6 → 96.5 (+29.9)** | holds |
| 12 | keypoint drop (tab:kpdrop) | hands-only 61.88 (−3.35), body-only 48.94 (−16.29) | **hands-only 73.33 (+4.51), body-only 54.04 (−14.78)** | **sign flips for hands-only** |
| 13 | mirror augmentation (tab:mirroraug) | 71.86 ± 6.23 (7 seeds, bimodal); Δ +6.63 | **76.38 ± 1.99** (3 seeds, not bimodal); Δ +7.56 | holds; bimodality caveat goes |
| 14 | two-stream joint + bone (Interventions, tab:bone) | bone 66.71 ± 1.74; fused − joint +8.35 pp, CI [+6.5, +10.3] (73.1 → 81.5 ensemble) | **bone 65.54 ± 1.63; fused − joint +7.03 pp, CI [+5.13, +8.94]** (71.7 → 78.8 ensemble) | holds |
| 15 | RTMPose-133 BlockGCN (sec:eff) | 68.52 ± 9.12, "within seed noise" of MediaPipe | **78.10 ± 0.60, +9.28 pp over MediaPipe 68.82 ± 0.56** | **conclusion flips** |
| 16 | native pose-SSL init, full labels (Interventions) | "72.8 ± 10.3 held-out TEST" | **TEST 65.35 ± 2.89**, below scratch 68.82 | **conclusion flips**, and the old number was val |

## What changes a P2 conclusion

1. **RTMPose (§eff).** P2 says BlockGCN on RTMPose-133 reaches "only 68.52 ± 9.12, within seed noise of
   MediaPipe-27 (65.23)", so the extractor is not why Uni-Sign wins. At lr 0.01 both are tight:
   RTMPose 78.10 ± 0.60 vs MediaPipe 68.82 ± 0.56. That is a +9.3 pp extractor effect, far outside seed
   noise, and it explains over half of the Uni-Sign gap (83.61 − 68.82 = 14.8 pp, of which 9.3 pp is the
   extractor). The sentence and its footnote must be rewritten.
2. **Native pose-SSL at full labels (Interventions).** P2's "72.8 ± 10.3 (BlockGCN backbone, held-out
   TEST)" is **validation (U15) accuracy**: the J4 runs' test feeder was `val_data.npy`, and no TEST eval
   existed. On TEST, the same lr-0.1 checkpoints score 62.83 ± 6.30. At lr 0.01, SSL init gives TEST
   65.35 ± 2.89 (val 80.15 ± 1.07), **3.5 pp below** from-scratch 68.82. "Above from-scratch in central
   tendency" is false on TEST at either LR. This is a provenance error independent of the LR switch.
   The low-label SSL claim (+4.8 pp at 10 %) was not re-run here; check whether it was also val.
3. **Keypoint drop (tab:kpdrop) and cue table (sec:cues).** Hands-only now **beats** the full skeleton
   (73.33 ± 0.49 vs 68.82, +4.5 pp; cue-table hands-only 76.92 ± 1.10 vs control 68.86). The old reading
   "removing the body costs 3.35 pp" reverses: the 7 body nodes hurt unseen-signer accuracy, consistent
   with body pose carrying signer identity. The cue table's "single-channel drop: high variance"
   interpretations (±19.5 / ±25.3) no longer apply, since every cell now has SD ≤ 1.8. Body-only 48.33,
   static 44.13 (−24.7), speed-norm 55.97 (−12.9), shuffle 60.29 (−8.6), root-centered 70.26 (+1.4 over
   control). Motion is essential and absolute position is not load-bearing: both still hold.
4. **Leak table (sec:leak).** The supervised BlockGCN is no longer "lower still at 0.763" than every SSL
   encoder: it is 0.853 ± 0.021 (seed 0: 0.824, equal to the distance-gated SSL encoder's 0.824). So "the
   two supervised backbones bracket the SSL pools" fails. The decoupling claim survives in weaker form:
   Adaptive GNN generalizes better (79.07 vs 68.82) and still encodes more (0.911 vs 0.853). The abstract's
   "L_norm = 0.76–0.91" becomes about 0.82–0.91.
5. **Diversity grid (tab:div).** The lr-0.01 recipe gives lower val on small signer pools: 5 of 6 cells at
   2/4/6 signers drop 3–13 pp, while 6/1500 rises +7.9 (42.9 → 50.7). 8 signers is similar (54.6 / 57.6 vs
   50.0 / 58.4). Final training loss is about
   0.02, so this is not underfitting: lr 0.01, selected on the full pool, generalizes worse from tiny pools.
   - Diversity beats clip count still holds (2→8 signers +27.7 / +27.3 pp).
   - "Monotonic in signers except 6/1500" fails: 4 > 6 signers at n = 800 (40.9 vs 37.8).
   - Doubling clips no longer always helps: 4 signers, 800 → 1500 gives 40.9 → 38.5.
   - "6–8-signer cells are high-variance (±4–15)" fails: SDs are now 0.4–3.6.
6. **Mirror (tab:mirroraug).** BlockGCN mirror is now 76.38 ± 1.99 with no seed collapse, so the
   bimodality caption and 7-seed justification go. Δ vs mirror-off is +7.56 pp (was +6.63).
7. **SMG(k) BdSLW60 endpoint** (tab:smgk, Fig. 4b): 59.9 → 37.1 becomes 61.6 → 30.4.
8. **LOSO**: mean 65.09 → 74.37 everywhere it is quoted (abstract, conclusion, tab:losofolds, Fig. 2,
   Table 7 LOSO column, app:losovar). The variance decomposition (tab:losovar) mixes 8 archs; its BlockGCN
   slice changes and must be recomputed if BlockGCN becomes lr 0.01 there.

## What does NOT change

- **Headline SMG**: +29.95 → +29.41 (paper estimator). The abstract's "signer mixing inflates BlockGCN
  accuracy from X to Y" becomes 68.82 → 98.23. The random-split range collapses from [90.2, 97.5] to
  [98.1, 98.4].
- **LOSO is a distribution, not a point**: 11-fold range 42.6–91.3, SD 16.07. U01 and U12 are still the
  hard signers.
- **GRL negative result**: accuracy still craters (68.82 → 43.8–50.7). Leakage now rises slightly vs the
  seed-0 control, so "barely moves leakage" is, if anything, an understatement.
- **Enrollment**: k = 1 gives +19.6 / +24.6 pp; k = 5 reaches 96–98 %.
- **Paired ceiling / fragility**: gap +29.9 pp (median Δ +25.3, IQR [+8.3, +49.3], 23/60 signs lose ≥ 40 pp).

## NON-BdSLW60 BlockGCN numbers in P2 that used lr 0.1 (not re-run here)

| P2 number | config (lr) | lr-0.01 runs already on disk |
|---|---|---|
| BdSLW401 BlockGCN 61.53 ± 13.44 TEST (tab:arch (b), W401 table: val 88.27); L_norm 0.43 ± 0.13 (sec:leak) | `bdsl_block_gcn_w401.yaml` (0.1), `_w401_sd` (0.1) | `bdsl_block_gcn_w401_lr01_seed{0,1,2}`, `bdsl_block_gcn_w401_sd_lr01_seed{0,1,2}` |
| AUTSL BlockGCN 84.85 (SI) / 94.08 (random); tab:xds 84.29 ± 10.6 / 94.11 ± 3.2, SMG +9.8 | `autsl_si.yaml`, `autsl_random.yaml` (0.1) | `autsl_si_lr01_scratch_seed*`, `autsl_random_lr01_seed*` |
| ASL Citizen BlockGCN (R@1 73.94, SMG +3.40) | `aslcitizen_si.yaml`, `aslcitizen_mixed.yaml` (0.1) | `aslcitizen_si_lr01_scratch_seed*` (no lr-0.01 mixed run) |

Also BlockGCN-based and lr 0.1, out of scope: the SSL low-label curve fine-tunes (J4 frac < 1).

## Files

One `.md` per experiment (01–16), each with per-seed values, mean ± SD, provenance, and the old number
next to the new one. Raw evals are in `raw/`. LOSO per-fold CSV: `docs/loso_heldout/block_gcn_lr01.csv`.

Tooling (new, uncommitted):
- `tools/lr01_switch_tasks.py`: configs + packed task lists.
- `tools/lr01_switch_eval.py`: side-effect-free TEST scoring.
- `tools/lr01_switch_report.py`: these tables.
- `scripts/hpc/slurm_lr01_switch.sbatch`: N concurrent runs per GPU.
- `scripts/hpc/lr01_switch/`: task lists and probe/fusion tails.
- `tools/run_resumable_seed.py`: gained `--init-weights`, used for the SSL init.

Jobs: `jobs.txt` (<job> and <job>).
