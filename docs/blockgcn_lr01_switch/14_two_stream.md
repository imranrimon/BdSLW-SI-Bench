# 14 — Two-stream joint + bone (Interventions, tab:bone)

Joint (lr-0.01 Protocol A dumps) + bone (lr 0.01, no clipping) score-sum fusion, 3-seed ensemble, paired clip bootstrap (tools/fusion_bootstrap.py with lr-0.01 paths).

Old (clip-5 recipe, both streams): joint 73.1 → fused 81.5 ensemble, fused − joint +8.35 pp, CI [+6.5, +10.3]; single-stream TEST joint 68.47 / bone 66.71 ± 1.74 (tab:bone used joint 65.23 / bone 66.71).

| | old lr 0.1 per seed | old mean ± SD | new lr 0.01 per seed | new mean ± SD |
|---|---|--:|---|--:|
| bone stream TEST | 65.27 / 65.71 / 69.16 | 66.71 ± 1.74 | 67.69 / 65.20 / 63.74 | **65.54 ± 1.63** |

Bone runs: `config/lr01_switch/bdsl_block_gcn_si_bone_vec_lr01.yaml` (feeder `is_vector: true`, as the old clip5 bone training runs). A first attempt, `bdsl_block_gcn_si_bone_lr01_seed*`, inherited `is_vector: false` from the current `config/bdsl_block_gcn_si_bone.yaml` (root-centring + random shift applied to bone vectors) and reached only 28.84 ± 0.93 TEST; it is a config error, discarded (raw/fusion_lr01.md keeps its fusion for the record).

## E9 two-stream fusion — per-instance (clip) bootstrap, 3-seed ensemble

n = 1365 held-out TEST clips, 10000 paired bootstrap resamples.

- joint  ensemble acc = 71.72%
- bone   ensemble acc = 69.16%
- FUSED  ensemble acc = 78.75%

- fused - joint: Delta = +7.03pp  95% CI [+5.13, +8.94]  p(one-sided) = 0.0000  ***
- fused - bone: Delta = +9.60pp  95% CI [+7.77, +11.50]  p(one-sided) = 0.0000  ***

Per-seed fused acc (for reference):
  seed0: joint 68.06  fused 76.85  (+8.79)
  seed1: joint 69.01  fused 78.17  (+9.16)
  seed2: joint 69.38  fused 76.12  (+6.74)

