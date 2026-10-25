## E9 two-stream fusion — per-instance (clip) bootstrap, 3-seed ensemble

n = 1365 held-out TEST clips, 10000 paired bootstrap resamples.

- joint  ensemble acc = 71.72%
- bone   ensemble acc = 30.62%
- FUSED  ensemble acc = 65.49%

- fused - joint: Delta = +-6.23pp  95% CI [-8.21, -4.32]  p(one-sided) = 1.0000  n.s.
- fused - bone: Delta = +34.87pp  95% CI [+32.31, +37.51]  p(one-sided) = 0.0000  ***

Per-seed fused acc (for reference):
  seed0: joint 68.06  fused 64.03  (+-4.03)
  seed1: joint 69.01  fused 61.83  (+-7.18)
  seed2: joint 69.38  fused 62.05  (+-7.33)
