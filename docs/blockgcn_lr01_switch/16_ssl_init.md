# 16 — Native pose-SSL init at full labels (Interventions bullet)

Native pose-SSL init (BdSL-only SHuBERT-style encoder `bdsl_shubert_pretrain_bdsl_only/pretrained_epoch30.pt`), full labels, BlockGCN backbone. **The paper's 72.83 ± 10.31 (sample SD) is best-epoch VALIDATION (U15) accuracy, not TEST**: the J4 runs' test feeder was val_data and no TEST eval existed. Old TEST below = their best-val checkpoints scored on TEST (tools/lr01_switch_eval.py --sections sslold).

| | per seed | mean ± SD |
|---|---|--:|
| old lr 0.1, val (as in paper) | 62.75 / 72.37 / 83.36 | 72.83 ± 8.42 |
| old lr 0.1, TEST | 54.14 / 65.49 / 68.86 | 62.83 ± 6.30 |
| new lr 0.01, val | 81.53 / 78.93 / 80.00 | 80.15 ± 1.07 |
| **new lr 0.01, TEST** | 68.64 / 61.61 / 65.79 | **65.35 ± 2.89** |

Comparison point: from-scratch BlockGCN TEST old 65.23 ± 2.24 / new 68.82 ± 0.56.
