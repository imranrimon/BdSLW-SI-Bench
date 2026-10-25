# Documentation index

## Dataset documentation

| File | Content |
|---|---|
| [`DATASHEET.md`](DATASHEET.md) | Datasheet (Gebru et al.) for the derived BdSLW60-SI and BdSLW401-SI releases |
| [`../metadata/croissant_bdslw60_si.json`](../metadata/croissant_bdslw60_si.json) | Croissant 1.0 + Responsible-AI metadata, BdSLW60-SI |
| [`../metadata/croissant_bdslw401_si.json`](../metadata/croissant_bdslw401_si.json) | Croissant 1.0 + Responsible-AI metadata, BdSLW401-SI |
| [`AUDIT_FIX_9_KEYPOINT_JUSTIFICATION.md`](AUDIT_FIX_9_KEYPOINT_JUSTIFICATION.md) | Why the 27-keypoint skeleton (7 body + 10 per hand) |
| [`RECIPE_CONTROL.md`](RECIPE_CONTROL.md) | Shared training recipe for the architecture comparison |

The Croissant files are built and validated by `tools/build_p2_release.py` and
`tools/check_croissant_neurips.py` (both need the bundled data; see the main README).

## Results provenance

Every number in the generated paper tables traces to a file through the declarative `SPEC`
in `tools/p2_make_tables.py`:

| Location | What it holds |
|---|---|
| `../results/results_final.csv` | The run log: one row per evaluation (`Timestamp, Experiment, Epoch, Top1_Acc, Top5_Acc, Top5_Policy, WorkDir`). Top-5 is taken at the epoch of the best Top-1. The latest row of an experiment is the one read. |
| `../results/run_manifest.csv` | For every run the tables read: phase, seed, config file, the checkpoint a TEST run scored, and any argument that differs from the config file. |
| `../results/tables/resolved_sources.json` | Every source value the generator resolved for the published tables (per-seed values or scalar, with its source). Used when the per-clip score dumps under `work_dir/` or the label files under `data/` are not present. |
| `../results/tables/reference/` | The table fragments as published (`--check` diffs against them). |
| `loso_heldout/` | Held-out-signer accuracy per LOSO fold and seed (`<arch>.csv`); `derived/` holds the significance test output (`arch_significance_final_2026-10-22.txt`) and the variance decomposition (`variance_decomp_final_2026-10-22.txt`). |
| `p2_tables/manual_values.yaml` | Values with no machine-readable source (model cost, literature rows), each with a comment citing its origin. |
| `baselines_rerun/` | Prior-method re-runs (Attention Bi-LSTM, SVM, SPOTER family), RGB and foundation-model rows. |

## Program notes

Each directory documents one analysis program: what was run, with which configs and task
lists, and the resulting numbers. They are working notes kept for provenance; references to
the paper's LaTeX sources point to files that are not part of this repository.

| Directory | Program |
|---|---|
| `blockgcn_lr01_switch/` | BlockGCN at the validation-selected base LR 0.01: Protocol A, Mixed, matched random splits, LOSO, GRL, leak probe, cues, enrollment, diversity grid, paired SMG, keypoint subsets, mirror, two-stream, RTMPose, SSL init |
| `adaptive_attn_lr0p1_switch/` | Adaptive GNN and Attention GNN at base LR 0.1 (same set of analyses) |
| `p2_rerun_fixes_2026-10/` | Completeness gate, best-on-validation TEST re-scoring (`bestck`), S3D cluster seeds |
| `p2_missing_joint_fixes_2026-10/` | Missing-joint masking for bones, cues and probes |
| `signernorm_2026-10/` | Signer-normalized input recipe (Stage 1 ablation, Stage 2 on all architectures, LOSO) |
| `tta_grl_2026-10/` | Test-time adaptation (AdaBN, SSL pass) and gradient-reversal (GRL) signer adversary |
| `p2_analysis_2026-10/` | Signer-aware leaderboard, SMG ceiling, signer difficulty, handedness mirror, RGB + pose fusion |
| `audit_reruns_2026-10/` | Audit re-runs (AUTSL leak-free distances, BlockGCN LR, L_norm, Mixed selection rule) |
| `improvement_runs_2026-10/` | GNN+Transformer clipping, W401 SMG(k), feature isolation, LR sweep |
| `w401_sdval/` | BdSLW401 with a signer-disjoint validation signer (S15) |
| `P2_FINAL_NUMBERS.md`, `P2_METRIC_SUITE.*` | Consolidated number ledger and the metric-suite output |
