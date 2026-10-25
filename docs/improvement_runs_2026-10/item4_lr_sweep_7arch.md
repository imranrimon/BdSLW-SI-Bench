# Item 4 — base-LR check for the 7 non-BlockGCN architectures (BdSLW60 Protocol A)

Configs `config/lrsweep/<reported>_lr0p1.yaml` / `_lr0p001.yaml` = the reported `config/bdsl_<arch>_si.yaml` (GNN+Transformer: `config/bdsl_gnn_transformer_si_clip5.yaml`, grad clip 5.0 kept) with only `base_lr` changed; seed 0, 100 epochs, no grad clipping except GNN-T. Job (`scripts/hpc/slurm_lrsweep_7arch.sbatch`). Selection metric = best-epoch **validation** Top-1 on signer 15 (max over the every-5-epoch evals logged in results_final.csv); lr 0.01 = the existing reported seed-0 run (`<reported>_seed0`). No TEST scoring.

| Architecture | val @ lr 0.1 | val @ lr 0.01 (reported) | val @ lr 0.001 | val argmax | 0.01 − best | flags |
|---|--:|--:|--:|---|--:|---|
| Adaptive GNN (CTR-GCN-style) | 89.77 | 87.48 | 87.94 | 0.1 **≠ 0.01** | -2.29 | — |
| ST-GCN (vanilla) | 85.95 | 87.18 | 84.43 | 0.01 | +0.00 | — |
| SLGTFormer | 68.09 | 86.87 | 75.88 | 0.01 | +0.00 | lr 0.1: only 6/20 evals |
| Attention GNN | 86.26 | 83.51 | 84.73 | 0.1 **≠ 0.01** | -2.75 | — |
| GNN+LSTM | 80.76 | 85.80 | 78.78 | 0.01 | +0.00 | — |
| Pose LSTM | 73.74 | 73.13 | 64.43 | 0.1 **≠ 0.01** | -0.61 | — |
| GNN+Transformer (clip 5.0) | 4.43 | 87.33 | 82.60 | 0.01 | +0.00 | lr 0.1: chance-level (4.43) |

## Archs where 0.01 is not the validation argmax

- **Adaptive GNN (CTR-GCN-style)**: val argmax is lr 0.1 (89.77) vs lr 0.01 (87.48), 2.29 pp better.
- **Attention GNN**: val argmax is lr 0.1 (86.26) vs lr 0.01 (83.51), 2.75 pp better.
- **Pose LSTM**: val argmax is lr 0.1 (73.74) vs lr 0.01 (73.13), 0.61 pp better.

## Seed-noise calibration (lr 0.01, existing reported runs)

The 0.1-vs-0.01 comparison is single-seed. Val best Top-1 of the reported lr-0.01 runs over seeds 0/1/2 (val = 655 S15 clips, 1 clip = 0.15 pp); a gap smaller than the lr-0.01 seed spread is not a decision signal.

| Architecture | lr 0.01 val seeds 0/1/2 | mean ± SD | lr-0.1 (seed 0) − lr-0.01 3-seed mean | final-epoch val 0.1 / 0.01 / 0.001 |
|---|---|--:|--:|---|
| Adaptive GNN (CTR-GCN-style) | 87.48 / 89.62 / 89.31 | 88.80 ± 0.94 | +0.97 | 89.47 / 86.41 / 86.87 |
| ST-GCN (vanilla) | 87.18 / 83.97 / 83.05 | 84.73 ± 1.77 | +1.22 | 84.58 / 87.18 / 83.66 |
| SLGTFormer | 86.87 / 86.41 / -- | 86.64 ± 0.23 | -18.55 | -- / 85.65 / 75.42 |
| Attention GNN | 83.51 / 85.80 / 86.26 | 85.19 ± 1.20 | +1.07 | 86.26 / 82.44 / 84.73 |
| GNN+LSTM | 85.80 / 82.29 / 81.07 | 83.05 ± 2.01 | -2.29 | 80.31 / 84.58 / 77.56 |
| Pose LSTM | 73.13 / 74.96 / 74.50 | 74.20 ± 0.78 | -0.46 | 72.67 / 70.08 / 62.75 |
| GNN+Transformer (clip 5.0) | 87.33 / 88.09 / 87.94 | 87.79 ± 0.33 | -83.36 | 1.68 / 86.87 / 82.60 |

## Divergence / NaN

- **SLGTFormer lr 0.1 diverged.** Val 47.2 (ep 4) → 68.1 (ep 9) → 53.3 → collapse to 5.3 at ep 19 and stays at chance (9.5, 4.6); weight max-abs grows from 1.5e3 (ep-9 ckpt) to 4.5e13 (ep-19/29 ckpts). At epoch 35 the run crashed with a CUDA device-side assert in `torch.bernoulli` (`0 <= p <= 1`, i.e. NaN/inf probabilities in the DropT/DropSke masks) — a NaN failure, not an infrastructure fault (job_0 FAILED after 4h21m; all 6 other runs of that task finished). It was **not** resumed: the latest checkpoint (ep 29) is already blown up (|w| ~ 4e13, chance-level val), so finishing the remaining 65 epochs cannot change the decision (best pre-divergence val 68.09 vs 86.87 at lr 0.01). Resume command if a full-budget row is wanted anyway: `python tools/run_resumable_seed.py --config config/lrsweep/bdsl_slgtformer_si_lr0p1.yaml --seed 0 --exp bdsl_slgtformer_si_lr0p1_seed0`.
- **GNN+Transformer lr 0.1 (grad clip 5.0) collapsed** to chance: best val 4.43 (ep 4), 1.5–1.7 thereafter; train loss stays at ~ln 60 = 4.09 through epoch 100. No NaN in the log (clipping keeps it finite), but it never learns.
- No NaN/divergence in any other run (all 13 others completed 100 epochs).

Best-epoch per run (epoch index, 0-based, of the max logged val Top-1):

- Adaptive GNN (CTR-GCN-style): lr 0.1: ep 74, lr 0.01: ep 59, lr 0.001: ep 64
- ST-GCN (vanilla): lr 0.1: ep 74, lr 0.01: ep 99, lr 0.001: ep 69
- SLGTFormer: lr 0.1: ep 9, lr 0.01: ep 94, lr 0.001: ep 94
- Attention GNN: lr 0.1: ep 74, lr 0.01: ep 19, lr 0.001: ep 99
- GNN+LSTM: lr 0.1: ep 84, lr 0.01: ep 84, lr 0.001: ep 79
- Pose LSTM: lr 0.1: ep 79, lr 0.01: ep 89, lr 0.001: ep 74
- GNN+Transformer (clip 5.0): lr 0.1: ep 4, lr 0.01: ep 74, lr 0.001: ep 99
