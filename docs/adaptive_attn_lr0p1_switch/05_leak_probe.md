# 5 — Signer decodability of the supervised Adaptive GNN (leak table, sec:leak decoupling)

Same probe settings as docs/audit_reruns_2026-10/r4_lnorm.md and the BlockGCN lr-0.01 switch (cc = `--class-conditioned` n=4000, the P2 leak-table \Lnorm; marginal n=6000; within-class n=6000, the P2 \Lnorm^wc column). `config/probe_adaptive_gnn_si.yaml`; new = best-on-val checkpoints of the lr-0.1 runs. P2's row (0.911 / 0.986) is the single old seed-0 periodic checkpoint m75.

| encoder | cc (P2 \Lnorm) | marginal | within-class (P2 \Lnorm^wc) |
|---|--:|--:|--:|
| old P2 row: lr 0.01 seed 0 m75 (r4) | 0.911 | 0.843 | 0.986 |
| sup_adaptive_lr0p01_s0_m59 | 0.910 | 0.845 | 0.986 |
| sup_adaptive_lr0p01_s1_m59 | 0.896 | 0.832 | 0.984 |
| sup_adaptive_lr0p01_s2_m89 | 0.895 | 0.810 | 0.984 |
| sup_adaptive_lr0p1_s0 | 0.890 | 0.811 | 0.990 |
| sup_adaptive_lr0p1_s1 | 0.906 | 0.829 | 0.990 |
| sup_adaptive_lr0p1_s2 | 0.884 | 0.820 | 0.986 |
| old lr 0.01, 3 seeds (best ckpts m59/m59/m89) mean ± SD | 0.900 ± 0.007 | 0.829 ± 0.014 | 0.985 ± 0.001 |
| **new lr 0.1, 3 seeds** mean ± SD | 0.893 ± 0.009 | 0.820 ± 0.007 | 0.989 ± 0.002 |

Raw log: docs/adaptive_attn_lr0p1_switch/raw/probes.log
