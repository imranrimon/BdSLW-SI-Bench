# Item 3: §featiso FlatTemporal pose vs DINOv2 under the symmetric rule

Source: SLURM job (`scripts/hpc/slurm_improve_featiso_symmetric.sbatch`, ran 2 min on one node). The job only evaluated existing checkpoints; nothing was retrained. Score dumps are in `docs/improvement_runs_2026-10/raw/i3_*.pkl`. Population SD over 3 seeds throughout.

## What the old runs selected on, and whether a usable checkpoint exists

| Runs | Training-time eval set (from log `test_feeder_args`) | `*_model-best.pt` |
|---|---|---|
| `bdsl_pose_temporal_sd_seed{0,1,2}` | `data/bdsl_sd/val` (655 clips): the sd **val** split | present. ckpt `acc` = log max val (93.28 / 94.81 / 96.03 at epochs 69 / 69 / 94) |
| `bdsl_dino_temporal_sd_seed{0,1,2}` | `data/bdsl_sd_dino/val` (655): sd **val** | present. 99.24 / 99.08 / 99.08 at epochs 34 / 79 / 29 |
| `bdsl_pose_temporal_si_seed{0,1,2}` | `data/bdsl_si/val` (signer 15) | present |
| `bdsl_dino_temporal_si_fix_seed{0,1,2}` | `data/bdsl_si_dino/val` (signer 15) | present. The `_fix` runs are the NaN-guarded rerun; the non-fix `bdsl_dino_temporal_si_seed*` runs are dead at 1.53% (chance) |

Every run already selected on its own **validation** split, never on test, so a valid best-on-val checkpoint exists for all 12 runs. Re-evaluating the 6 sd checkpoints on sd TEST was enough; no retraining was needed.

The problem with the old §featiso numbers is not *which* checkpoint was used. It is *which set was reported*. Both arms reported the selection-set (val) accuracy: sd-val for Mixed and SI **val** (signer 15) for the disjoint arm. The val re-evals in this job reproduce the logged best values exactly. Clipping matches across all FlatTemporal runs (all unclipped). The sd and sd_dino TEST sets hold the same 1,367 clips; only the `.mp4` name suffix and the order differ, and the class counts are identical.

## Per-seed results (symmetric: select on val, score once on test)

| Representation | Seed | Mixed: sd TEST (best on sd val) | sd val (selection) | SI TEST (best on SI val) | SI val (selection) | per-seed gap (sd TEST − SI TEST) |
|---|--:|--:|--:|--:|--:|--:|
| MediaPipe pose | 0 | 92.32 | 93.28 | 58.83 | 66.11 | +33.49 |
| | 1 | 93.34 | 94.81 | 60.66 | 69.77 | +32.68 |
| | 2 | 95.98 | 96.03 | 60.95 | 70.84 | +35.02 |
| | **mean ± SD** | **93.88 ± 1.54** | 94.71 ± 1.12 | **60.15 ± 0.94** | 68.91 ± 2.03 | **+33.73** |
| DINOv2 crops | 0 | 98.76 | 99.24 | 64.69 | 72.52 | +34.07 |
| | 1 | 99.27 | 99.08 | 62.05 | 71.76 | +37.22 |
| | 2 | 98.46 | 99.08 | 61.25 | 73.44 | +37.22 |
| | **mean ± SD** | **98.83 ± 0.33** | 99.13 ± 0.07 | **62.66 ± 1.47** | 72.57 ± 0.69 | **+36.17** |

Macro-F1 (3-seed mean):

| Representation | sd TEST | SI TEST |
|---|--:|--:|
| pose | 93.95 | 58.48 |
| DINOv2 | 98.82 | 60.66 |

Clip counts: sd TEST n = 1,367 and SI TEST n = 1,365 for both feature types.

## Gaps: old vs new

| Representation | OLD (P2): sd-val − SI-val | Mixed only made symmetric: sd TEST − SI val | **NEW, fully symmetric: sd TEST − SI TEST** |
|---|--:|--:|--:|
| MediaPipe pose | +25.8 (94.71 − 68.91) | +24.97 | **+33.7** (93.88 − 60.15) |
| DINOv2 crops | +26.6 (99.13 − 72.57) | +26.26 | **+36.2** (98.83 − 62.66) |
| DINOv2 − pose | +0.8 | +1.3 | **+2.4** |

The old +25.8 / +26.6 reproduce exactly from the val numbers above, which confirms that P2 computed them on val/val. Under the protocol used everywhere else in P2 (Table 1: disjoint = held-out TEST signers 2/13, Mixed = sd TEST after selection on sd val), **both gaps grow by roughly 8–10 pp**. They also become comparable to the pose-architecture SMGs in Table 1 (+19.7 to +33.2).

DINOv2's gap now exceeds pose's by 2.4 pp. That is within seed noise:
- per-seed gaps are pose 32.7–35.0 and DINOv2 34.1–37.2;
- a Welch t-test on the three per-seed gaps gives t ≈ 1.95, so the difference is not significant at n = 3.

So the qualitative reading still holds: removing appearance does not close the gap. "Essentially the same" should be softened to "similar (DINOv2 slightly larger, within seed noise)".

## Exact P2 text that would change (overleaf L300–301 and L306; ieee L305–306 and L311)

**Table rows (tab in §featiso)**
- OLD: `MediaPipe pose (kinematic) & $+25.8\pp$ & large gap with \emph{no} appearance channel \\`
- NEW: `MediaPipe pose (kinematic) & $+33.7\pp$ & large gap with \emph{no} appearance channel \\`
- OLD: `DINOv2 crops (appearance)  & $+26.6\pp$ & essentially the same gap \\`
- NEW: `DINOv2 crops (appearance)  & $+36.2\pp$ & similar gap (within seed noise) \\`

Suggested caption addition: "(Mixed = signer-dependent test after selection on its validation set; disjoint = held-out TEST signers 2/13 after selection on signer 15; 3 seeds: pose $93.88\to60.15$, DINOv2 $98.83\to62.66$.)"

**Paragraph (L306 / L311)**
- OLD: "The two gaps are nearly identical ($+25.8$ vs $+26.6\pp$): removing appearance entirely (pose-only) does not close the signer-generalization gap, and a pure-appearance representation does not enlarge it."
- NEW: "The two gaps are similar ($+33.7$ vs $+36.2\pp$; per-seed ranges $32.7$--$35.0$ vs $34.1$--$37.2$): removing appearance entirely (pose-only) does not close the signer-generalization gap, and a pure-appearance representation enlarges it only marginally, within seed noise."

The remaining sentences ("pervasive across representation types", "rules out the common assumption that pose input alone confers signer-robustness") still hold.

Also note `docs/PAPER_P2_BENCHMARK.md` L93–95: "SI pose 68.9 ± 2.0, DINOv2 72.6 ± 0.7" are SI **val** numbers. The held-out TEST values are 60.15 ± 0.94 and 62.66 ± 1.47.
