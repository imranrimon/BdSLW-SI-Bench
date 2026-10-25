# 6 — Two-stream: Adaptive GNN bone (tab:bone, app:bone)

Adaptive GNN bone stream, held-out TEST (tab:bone). Old = `bdsl_adaptive_gnn_bone_clip5` (lr 0.01, grad clip 5.0). New = `config/lr0p1_switch/bdsl_adaptive_gnn_bone_lr0p1.yaml` (lr 0.1, no clipping = the new joint recipe; is_vector False as before), best-on-val checkpoint, `--test-config ..._bone_lr0p1_test.yaml`.

| | per seed | mean ± SD |
|---|---|--:|
| old bone (lr 0.01, clip 5) | 30.77 / 32.23 / 31.28 | 31.43 ± 0.60 |
| **new bone (lr 0.1, no clip)** | -- / -- / -- | **--** |
| new bone val best | -- / -- / -- | -- |
| joint, new (Protocol A lr 0.1) | -- / -- / -- | -- |

