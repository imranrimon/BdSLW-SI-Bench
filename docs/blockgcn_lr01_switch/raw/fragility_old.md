## Signer-dependent ceiling + per-class fragility (E1 paired, identical clips)

Clips pooled: U02+U13 held-out TEST (709 clips), 3 seeds (per-seed accuracy averaged, paper convention).

- **SI (signer unseen):** 60.93%
- **Signer-dependent ceiling (same signer's clips in train, matched size):** 87.82%
- **Ceiling gap (paired, identical clips):** +26.89 pp  — the headroom a perfect signer-adaptation method could recover.

### Most signer-FRAGILE signs (largest seen−unseen drop)

| sign | SI acc | ceiling acc | Δ (fragility) | n clips |
|---|---|---|---|---|
| tattha | 0.0 | 91.7 | +91.7 | 12 |
| chal | 5.6 | 72.2 | +66.7 | 12 |
| bon | 27.3 | 93.9 | +66.7 | 11 |
| aam | 26.2 | 90.5 | +64.3 | 14 |
| juta | 30.3 | 90.9 | +60.6 | 11 |
| dal | 13.3 | 73.3 | +60.0 | 5 |
| baba | 11.1 | 68.9 | +57.8 | 15 |
| biscuts | 37.8 | 91.1 | +53.3 | 15 |
| boroi | 28.2 | 79.5 | +51.3 | 13 |
| chacha | 27.8 | 75.0 | +47.2 | 12 |
| chini | 27.8 | 75.0 | +47.2 | 12 |
| chocolate | 39.4 | 84.8 | +45.5 | 11 |

### Most signer-ROBUST signs (generalize across signers)

| sign | SI acc | ceiling acc | Δ | n clips |
|---|---|---|---|---|
| alu | 85.2 | 70.4 | -14.8 | 9 |
| konna | 88.9 | 83.3 | -5.6 | 12 |
| ayna | 82.1 | 76.9 | -5.1 | 13 |
| chosma | 100.0 | 96.7 | -3.3 | 10 |
| tv | 100.0 | 100.0 | +0.0 | 12 |
| chikissha | 100.0 | 100.0 | +0.0 | 5 |
| chadar | 100.0 | 100.0 | +0.0 | 12 |
| chachi | 92.9 | 92.9 | +0.0 | 14 |

Fragility spread: median Δ +25.0 pp, IQR [+5.1, +44.4]; 12/60 signs already at/near ceiling under SI (signer-robust), 18 signs lose ≥40 pp.
