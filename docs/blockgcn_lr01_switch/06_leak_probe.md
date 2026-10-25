# 6 — Signer decodability of the lr-0.01 BlockGCN (leak table) + GRL probes

Same probe settings as docs/audit_reruns_2026-10/r4_lnorm.md (cc = class-conditioned n=4000; marginal n=6000; within-class n=6000). Old supervised BlockGCN = `bdsl_block_gcn_si_seed0_model-69.pt`.

| encoder | cc (P2 table) | marginal | within-class |
|---|--:|--:|--:|
| old supervised BlockGCN lr 0.1 (seed 0) | 0.763 | 0.625 | 0.968 |
| grl_lr01_l01 | 0.804 | 0.694 | 0.972 |
| grl_lr01_l05 | 0.832 | 0.742 | 0.979 |
| grl_lr01_l10 | 0.832 | 0.742 | 0.977 |
| sup_blockgcn_lr01_s0 | 0.824 | 0.725 | 0.972 |
| sup_blockgcn_lr01_s1 | 0.871 | 0.766 | 0.978 |
| sup_blockgcn_lr01_s2 | 0.863 | 0.737 | 0.976 |
| **supervised BlockGCN lr 0.01, mean ± SD (3 seeds)** | **0.853 ± 0.021** | **0.743 ± 0.017** | **0.975 ± 0.002** |

Raw log: docs/blockgcn_lr01_switch/raw/probes.log
