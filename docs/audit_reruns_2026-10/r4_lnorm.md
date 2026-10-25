# R4 — Signer-decodability L_norm under three definitions (cross-paper item 5)

Source: SLURM jobs (clip5 control, W401 cc@8000 + full-set within-class; `scripts/hpc/slurm_audit_reruns_2026_10_clip5.sbatch`); current committed `tools/decodability_probe.py`, frozen-backbone features (mean-pooled over T,V,M), 5-fold LogReg, chance-normalised balanced signer accuracy.
- **cc** = `--class-conditioned` (subtract per-class mean feature, then ONE global signer probe) — P2's definition; n=4000 first clips (BdSLW60), as in P2's scripts.
- **mrg** = marginal (no conditioning) — P1 tab:probe; n=6000 (all 5748 BdSLW60 train clips).
- **wc** = `--within-class` L_norm^cond (signer probe fit WITHIN each sign class, averaged) — P1's primary measure; n=6000.
- BdSLW401: signer regex `S(\d+)` (16 train signers). cc and marginal on a fixed-seed random 8000-clip sample (P2's setting); within-class on ALL 38,876 train clips (an 8000-clip sample leaves <3 clips/signer in every class -> all 401 classes skipped).

| Encoder (ckpt) | cc (P2 def.) | marginal | within-class | P2 reports | P1 reports (marg / wc) |
|---|--:|--:|--:|--:|--:|
| sup_blockgcn_m69 | 0.763 | 0.625 | 0.968 | 0.763 | 0.625 / 0.968 |
| ssl_bdsl_only | 0.913 | 0.847 | 0.979 | 0.913 | 0.847 / 0.979 |
| ssl_xlingual | 0.871 | 0.788 | 0.967 | 0.871 | 0.788 / 0.967 |
| ssl_dgxl | 0.824 | 0.727 | 0.957 | 0.824 | 0.727 / 0.957 |
| sup_adaptive_m75 | 0.911 | 0.843 | 0.986 | 0.911 | — / — |
| sup_adaptive_m59 | 0.910 | 0.845 | 0.986 | 0.911 (P2 used latest periodic ckpt = m75) | — / — |
| grl_l01 | 0.760 | 0.631 | 0.963 | 0.760 | — / — |
| grl_l05 | 0.722 | 0.594 | 0.956 | 0.722 | — / — |
| grl_l10 | 0.732 | 0.597 | 0.958 | 0.732 | — / — |
| sup_blockgcn_clip5 (λ=0 GRL control, 3 seeds, best ckpt) | 0.79±0.02 (0.780/0.812/0.778) | 0.67±0.02 (0.664/0.694/0.652) | 0.97±0.00 (0.968/0.976/0.965) | — (P2 uses unclipped 0.763) | — |
| W401 ctrgcn (3 seeds) | 0.69±0.00 (0.695/0.688/0.697) | 0.69±0.01 (0.691/0.668/0.697) | 0.99±0.00 (0.993/0.992/0.992) | 0.69 ± 0.00 | — |
| W401 blockgcn (3 seeds) | 0.43±0.13 (0.381/0.296/0.603) | 0.41±0.13 (0.362/0.288/0.592) | 0.95±0.03 (0.951/0.921/0.988) | 0.43 ± 0.13 | — |

Within-class usable classes (used/skipped): sup_blockgcn_m69: 60 used/0 skipped; ssl_bdsl_only: 60 used/0 skipped; ssl_xlingual: 60 used/0 skipped; ssl_dgxl: 60 used/0 skipped; sup_adaptive_m75: 60 used/0 skipped; sup_adaptive_m59: 60 used/0 skipped; grl_l01: 60 used/0 skipped; grl_l05: 60 used/0 skipped; grl_l10: 60 used/0 skipped; w401_ctrgcn_s0: 400 used/1 skipped; w401_blockgcn_s0: 400 used/1 skipped; w401_ctrgcn_s1: 400 used/1 skipped; w401_blockgcn_s1: 400 used/1 skipped; w401_ctrgcn_s2: 400 used/1 skipped; w401_blockgcn_s2: 400 used/1 skipped; sup_blockgcn_clip5_s0: 60 used/0 skipped; sup_blockgcn_clip5_s1: 60 used/0 skipped; sup_blockgcn_clip5_s2: 60 used/0 skipped

## eval_grl_sweep.py rerun (P2's GRL-table tool, verbatim)

| $\lambda$ | SI Top-1 (%) | signer leak acc (%) | $L_{norm}$ |
|---|---|---|---|
| 0.1 | 48.86 | 78.97 | 0.760 |
| 0.5 | 41.90 | 75.65 | 0.722 |
| 1 | 51.65 | 76.56 | 0.732 |

