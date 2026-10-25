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
