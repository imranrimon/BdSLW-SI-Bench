# 3 — Signer-mixed split, Mixed/SMG columns (Table 7, tab:smg8)

Signer-mixed `data/bdsl_sd`, unclipped, symmetric rule: checkpoint selected on sd val, scored once on sd test (1,367 clips) via `tools/run_resumable_seed.py --test-config` (`config/lr0p1_switch/bdsl_<arch>_sd_noclip_lr0p1{,_test}.yaml`). Old = `bdsl_<arch>_sd_noclip_test_seed*` (lr 0.01). SMG = Mixed − Protocol-A TEST mean (same lr on both sides).

| arch | lr | Mixed per seed | Mixed mean ± SD | recheck (side-effect-free) | sd-val best | TEST mean | **SMG** |
|---|---|---|--:|---|---|--:|--:|
| Adaptive GNN | 0.01 (old) | 98.76 / 98.46 / 98.98 | 98.73 ± 0.21 |  | 99.39 / 99.24 / 99.08 | 79.07 | +19.66 |
| Adaptive GNN | **0.1 (new)** | -- / -- / -- | -- | -- / -- / -- | -- / -- / -- | None | -- |
| Attention GNN | 0.01 (old) | 98.46 / 97.95 / 98.32 | 98.24 ± 0.22 |  | 98.17 / 98.47 / 98.47 | 73.97 | +24.28 |
| Attention GNN | **0.1 (new)** | -- / -- / -- | -- | -- / -- / -- | -- / -- / -- | None | -- |
