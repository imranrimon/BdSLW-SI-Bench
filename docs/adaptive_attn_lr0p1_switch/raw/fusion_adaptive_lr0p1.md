## E9 two-stream fusion — per-instance (clip) bootstrap, 3-seed ensemble

n = 1365 held-out TEST clips, 10000 paired bootstrap resamples.

- joint  ensemble acc = 78.90%
- bone   ensemble acc = 25.49%
- FUSED  ensemble acc = 60.07%

- fused - joint: Delta = +-18.83pp  95% CI [-21.10, -16.63]  p(one-sided) = 1.0000  n.s.
- fused - bone: Delta = +34.58pp  95% CI [+32.01, +37.07]  p(one-sided) = 0.0000  ***

Per-seed fused acc (for reference):
  seed0: joint 77.73  fused 59.27  (+-18.46)
  seed1: joint 77.66  fused 59.93  (+-17.73)
  seed2: joint 75.75  fused 58.39  (+-17.36)
