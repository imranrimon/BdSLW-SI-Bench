# Item 1: Clipped GNN+Transformer on the signer-mixed split

**Bottom line: no new training was needed, and the P2 numbers are already right. The paper text has the wrong reason.** The existing mixed-split runs `work_dir/bdsl_gnn_transformer_sd_seed{0,1,2}` were already trained with `grad_clip_norm: 5.0`. So the reported Mixed 99.00 ± 0.15 and SMG +23.45 already pair the clipped mixed run with the clipped Protocol-A run. The footnote saying the mixed run is unclipped is wrong. While checking this, a larger recipe mismatch turned up for the other 7 architectures (see the last section).

## Evidence that the existing sd runs are clipped

- Each run's training-log `Parameters:` dump (one start per run, never resumed) shows `'config': 'config/bdsl_gnn_transformer_sd.yaml'` and `'grad_clip_norm': 5.0`. The saved `work_dir/bdsl_gnn_transformer_sd_seed*/config.yaml` also has `grad_clip_norm: 5.0`. The runs were trained on 2026-08-06, launched by SLURM job (`slurm_logs/sd_gnn_transformer_124705.out`: "=== bdsl_gnn_transformer_sd_seed0 (config/bdsl_gnn_transformer_sd.yaml) --grad_clip_norm 5.0 ===", the `slurm_multiseed_one.sbatch` EXTRA_ARGS passthrough). The knob was added in commit 2a405ac on 2026-08-05.
- Diffing the saved run configs `bdsl_gnn_transformer_si_clip5_seed0/config.yaml` and `bdsl_gnn_transformer_sd_seed0/config.yaml` shows that only the experiment name, the config path and the train/val data paths differ.
- Diffing the config files themselves:
  - `bdsl_gnn_transformer_si.yaml` vs `bdsl_gnn_transformer_si_clip5.yaml`: the only change is `grad_clip_norm: 5.0`.
  - `bdsl_gnn_transformer_si.yaml` vs `bdsl_gnn_transformer_sd.yaml`: only the name and the train/val data paths differ.
- Because of this, I created `config/bdsl_gnn_transformer_sd_clip5.yaml` (`bdsl_gnn_transformer_sd.yaml` plus `grad_clip_norm: 5.0`, with a comment) to record the existing runs' effective recipe. **I did not retrain it**: three new runs would duplicate the existing ones exactly.

## Symmetric Mixed for GNN+Transformer (clipped)

Rule: take the checkpoint that is best on the sd VALIDATION split (655 clips) and score it once on the sd TEST split (1,367 clips). These numbers come from the R5 score dumps (`docs/audit_reruns_2026-10/raw/r5_gnn_transformer_sd{test,val}_seed*.pkl`, job), which already evaluated these same checkpoints.

| Seed | best epoch (sd val) | sd-val (best ckpt) | **Mixed = sd TEST** | macro-F1 (sd TEST) | SI TEST (clip5) | per-seed SMG |
|---|--:|--:|--:|--:|--:|--:|
| 0 | 69 | 98.93 | 98.98 | 98.97 | 76.48 | +22.50 |
| 1 | 74 | 99.24 | 98.83 | 98.82 | 73.77 | +25.06 |
| 2 | 94 | 98.93 | 99.20 | 99.19 | 76.41 | +22.79 |
| **mean ± pop. SD** | | 99.03 | **99.00 ± 0.15** | 98.99 ± 0.15 | 75.55 ± 1.26 | **+23.45** |

The SI TEST per-seed values are rescored from `work_dir/bdsl_gnn_transformer_si_clip5_seed*_test/eval_results/best_acc.pkl` on 1,365 clips: 76.48 / 73.77 / 76.41.

- New SMG = 99.00 − 75.55 = **+23.45 pp**. That is what P2 Table 1 already prints (`99.00$\pm$0.15 & $+23.45$`).
- The brief's "99.03 → +23.5" is the old best-on-val number (98.93 / 99.24 / 98.93, mean 99.03). Table 1 already moved off it under R5.

## P2 text that would change (both `paper_p2_overleaf/main.tex` L442 and `paper_p2_ieee/main.tex` L447, tab:smg8 footnote)

- OLD: "For GNN+Transformer the disjoint TEST is the clipped run and the mixed run is unclipped (it did not diverge on the mixed split)."
- NEW (if only GNN+T is addressed): "For GNN+Transformer both the disjoint-TEST and the mixed runs use grad-norm clipping at 5.0."
- Table 1 / tab:smg8 cells for GNN+Transformer: **no change** (99.00 ± 0.15, +23.45).

## Larger finding: the Mixed column for the 7 other architectures does not match their recipe

The saved config and training log of every `work_dir/bdsl_<arch>_sd_seed{0,1,2}` run, for all 8 Table-1 architectures, show `grad_clip_norm: 5.0`. The architectures are block_gcn, adaptive_gnn, st_gcn_vanilla, gnn, slgtformer, gnn_lstm, pose_lstm and gnn_transformer; all were launched 2026-08-06.

The Protocol-A SI headline runs `bdsl_<arch>_si_seed*` behave differently. They were trained before the knob existed (no `grad_clip_norm` in their parameter dump), so they are **unclipped**. This agrees with the memory note that the headline runs used no clipping.

As a result, for **Adaptive GNN, ST-GCN, Attention GNN, SLGTFormer, GNN+Bi-LSTM, BlockGCN and Pose-LSTM**:
- the Table 1 "Mixed" and SMG values pair a **clipped** mixed run with an **unclipped** disjoint-TEST run;
- this contradicts the §Shared-recipe sentence (L74): "One exception: GNN+Transformer … is trained with grad-norm clipping at 5.0".

Two runs are **not** affected:
- The multi-split random controls of tab:smg (`bdsl_{block_gcn,adaptive_gnn}_rand4{2..6}_seed0`, BlockGCN +29.95 / Adaptive +19.90) are unclipped, so they are recipe-matched.
- The §featiso FlatTemporal sd/SI runs are also unclipped and matched (item 3).

Clipped SI TEST runs (`bdsl_<arch>_si_clip5_seed*_test`) exist at least for BlockGCN (64.62 / 64.98 / 75.82) and Adaptive GNN (78.90 / 81.17 / 78.75).

**Fix options for the caller to choose from** (none were launched here):
- (a) Retrain the 7 archs × 3 seeds on `bdsl_sd` unclipped. Use the reported `config/bdsl_<arch>_sd.yaml` as-is (grad_clip_norm defaults to 0.0), through `tools/run_resumable_seed.py`, then score best-on-sd-val on sd TEST. Cost is about 30 GPU-h; SLGTFormer alone takes ~8 h/seed. I drafted this as a 2-task packed array, but submitting it was outside the scope I was given and was blocked, so it was not run.
- (b) Disclose instead. Change the tab:smg8 footnote and the L74 / Table-1 caption to say the Mixed runs use clipping at 5.0 for all architectures. Mixed accuracy sits at 94.8–99.0, so clipping probably moves it by well under 1 pp.
