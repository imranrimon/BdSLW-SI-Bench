# 5 — Signer-adversarial (GRL) sweep (tab:grl, Fig. 3)

GRL sweep at base LR 0.01, no clipping (`config/lr01_switch/bdsl_block_gcn_si_signeradv_lr01.yaml`), seed 0 as the original sweep; Top-1 on TEST and class-conditioned L_norm from `tools/eval_grl_sweep.py`. λ=0 control = the lr-0.01 headline (TEST 68.82; its L_norm in 06_leak_probe.md).

Old (lr 0.1, GRL clip 5.0): λ=0 (clipped control) 68.47 / 0.790; 0.1 → 48.86 / 0.760; 0.5 → 41.90 / 0.722; 1.0 → 51.65 / 0.732.

New:

| $\lambda$ | SI Top-1 (%) | signer leak acc (%) | $L_{norm}$ |
|---|---|---|---|
| 0.1 | 50.70 | 82.82 | 0.804 |
| 0.5 | 43.81 | 85.26 | 0.832 |
| 1 | 45.27 | 85.31 | 0.832 |

