## Signer-dependent ceiling + per-class fragility (E1 paired, identical clips)

Clips pooled: U02+U13 held-out TEST (709 clips), 3 seeds (per-seed accuracy averaged, paper convention).

- **SI (signer unseen):** 66.62%
- **Signer-dependent ceiling (same signer's clips in train, matched size):** 96.52%
- **Ceiling gap (paired, identical clips):** +29.90 pp  — the headroom a perfect signer-adaptation method could recover.

### Most signer-FRAGILE signs (largest seen−unseen drop)

| sign | SI acc | ceiling acc | Δ (fragility) | n clips |
|---|---|---|---|---|
| tattha | 2.8 | 94.4 | +91.7 | 12 |
| dal | 0.0 | 80.0 | +80.0 | 5 |
| chokh utha | 9.1 | 84.8 | +75.8 | 11 |
| boroi | 23.1 | 94.9 | +71.8 | 13 |
| cake | 27.8 | 97.2 | +69.4 | 12 |
| tshirt | 38.5 | 100.0 | +61.5 | 13 |
| bhai | 38.9 | 100.0 | +61.1 | 12 |
| baba | 26.7 | 86.7 | +60.0 | 15 |
| chal | 33.3 | 91.7 | +58.3 | 12 |
| chini | 41.7 | 100.0 | +58.3 | 12 |
| denadar | 40.7 | 96.3 | +55.6 | 9 |
| aam | 42.9 | 97.6 | +54.8 | 14 |

### Most signer-ROBUST signs (generalize across signers)

| sign | SI acc | ceiling acc | Δ | n clips |
|---|---|---|---|---|
| tv | 100.0 | 100.0 | +0.0 | 12 |
| chikissha | 100.0 | 100.0 | +0.0 | 5 |
| chadar | 100.0 | 100.0 | +0.0 | 12 |
| chachi | 92.9 | 92.9 | +0.0 | 14 |
| apartment | 100.0 | 100.0 | +0.0 | 12 |
| anaros | 100.0 | 100.0 | +0.0 | 11 |
| alu | 100.0 | 100.0 | +0.0 | 9 |
| aids | 100.0 | 100.0 | +0.0 | 6 |

Fragility spread: median Δ +25.3 pp, IQR [+8.3, +49.3]; 9/60 signs already at/near ceiling under SI (signer-robust), 23 signs lose ≥40 pp.
