# Recipe-control policy for the 11-architecture SI benchmark

Why this document exists: training recipe alone moves ISLR numbers by
+4 to +10 pp with the backbone held fixed (arXiv 2412.11553: +6.54 WLASL,
+3.93 AUTSL, +10.12 Slovo for MViTv2-S from augmentation/sampling/loss
changes only). An architecture-comparison table that silently mixes recipes
is a known review attack. This is the policy the main paper states and the
sweep implements.

## Policy (what the paper's experimental-setup section says)

1. **Shared, fixed recipe across all skeleton models.** Every model in
   `experiments_si.yaml` trains with the identical pipeline:
   - feeder: `random_choose` (window 120), `random_shift`, `normalization`;
     `random_mirror` OFF everywhere (BdSL hand dominance is
     class-informative);
   - optimizer: SGD + nesterov, weight decay 1e-4;
   - schedule: 5-epoch linear warm-up, step decay ×0.1 at [60, 80],
     100 epochs, batch 32;
   - **no gradient clipping** (`grad_clip_norm` 0.0, the `main.py` default) for
     every headline architecture. **Exception:** GNN+Transformer diverges to NaN on
     some seeds without clipping, so it is reported with global grad-norm clipping
     at 5.0 (`config/bdsl_gnn_transformer_si_clip5.yaml`) on *both* the signer-
     independent and the signer-mixed side, as a labelled recipe deviation;
   - 3 seeds (0/1/2), best-val-Top-1 checkpoint selection, Top-5 reported
     from the same epoch (`same_epoch_as_logged_top1`).

   *Correction (2026-10).* An earlier version of this file said every
   architecture was trained with clipping at 5.0. That was the plan, not what the
   reported runs did: the headline SI runs are unclipped, and only the original
   signer-mixed (`bdsl_<arch>_sd`) runs were clipped. Those mixed runs were redone
   unclipped (`config/bdsl_<arch>_sd_noclip.yaml`) so both sides of every SMG use
   the same recipe. History of the clip study (1.0 vs 5.0, run on val): clip 1.0
   acts as a regulariser (cost SLGTFormer ~10 pp); 5.0 bounds the GNN+Transformer
   explosion while leaving other models near their unclipped accuracy.
2. **One controlled degree of freedom: base LR.** Architectures differ in
   LR sensitivity (transformers vs GCNs). As run: every architecture uses
   base_lr 0.01. BlockGCN was originally run at 0.1 and was switched to 0.01
   because 0.01 wins on U15 validation (77.97 vs 74.25); the same val-selection
   picks 0.01 for BlockGCN on AUTSL and BdSLW401 (S15 val) and keeps 0.1 on ASL
   Citizen. The other seven were swept over {0.1, 0.01, 0.001} at seed 0
   (`docs/improvement_runs_2026-10/item4_lr_sweep_7arch.md`); a 3-seed check of
   0.1 vs 0.01 for Adaptive GNN, Attention GNN and Pose-LSTM is in progress.
3. **No per-model augmentation, schedule, epoch, or batch tuning.** If a
   model's published recipe demands something unusual, run it as an extra
   clearly-labeled row ("author recipe"), never as the main-table row.
4. **Foundation/RGB rows are a separate track.** Kinetics-pretrained RGB
   (path4) and large-scale-pretrained models (Uni-Sign, SHuBERT probes) use
   their native recipes and appear under a "pretrained, external data"
   table block — they are context rows, not recipe-controlled comparisons.
5. **Comparison hygiene against the literature.** When citing WLASL-2000
   numbers: NLA-SLR's 61.05 headline is RGB+pose fusion; its keypoint-only
   ablation is 49.10 (Table 4). Compare pose-only against pose-only:
   SignBERT+ 48.85, BEST 46.25, MASA 49.06, Uni-Sign pose-only 63.13
   (per-instance Top-1). BdSLW60/BdSLW401 prior-work numbers are WER-based
   in the RQE paper — convert or annotate the metric when tabulating.

## Implementation state

- Items 1 and the fixed parts of 2–3 are already what the `_si` configs
  encode (all copied from `config/bdsl_block_gcn_si.yaml`'s optim block).
- The LR grid selection (item 2) is a cheap pre-pass:
  `python tools/run_multiseed.py --single <cfg> --seeds 0` with base_lr
  overridden per grid value; pick by val Top-1; record in the config
  comment and appendix.
- Statement for the paper: "All skeleton models share one training recipe
  (augmentation, schedule, batch, epochs, seeds); the only per-model degree
  of freedom is the base learning rate, selected on validation at seed 0
  from {0.1, 0.05, 0.01}. RGB and externally-pretrained baselines follow
  their native recipes and are reported in a separate block."
