# P2 Adaptive GNN + Attention GNN lr-0.1 switch (IN PROGRESS: training blocked by the submit cap)

**Decision (user, 2026-10-11).** Under strict 3-seed validation argmax (the rule that moved BlockGCN 0.1 → 0.01),
Adaptive GNN (`config/bdsl_adaptive_gnn_si.yaml`) and Attention GNN (`config/bdsl_gnn_si.yaml`) move from base LR
0.01 to **0.1** (U15 val, 3 seeds: Adaptive 89.92 vs 88.80; Attention 85.70 vs 85.19). Pose-LSTM stays at 0.01.
Every P2 number that comes from a trained Adaptive GNN or Attention GNN is re-derived at lr 0.1, with the protocol
otherwise unchanged, mirroring `docs/blockgcn_lr01_switch/`.

Nothing here edits `.tex`, regenerates figures into the paper dirs, or commits. lr-0.01 configs and work_dirs are
untouched: new configs are in `config/lr0p1_switch/`, and every new experiment name carries `lr0p1`.

## Status (2026-10-12 07:00)

- **Training (80 runs) submitted 2026-10-13 03:19 as jobs (T0), <job> (T1), <job> (T2); pending for GPUs.** Previously: The account has been at or above its 100-job submit cap since
  2026-10-11 13:37 (101 pending + 4 running jobs of other projects; most were pending with estimated starts around
  2026-10-14). A detached submitter (`scripts/hpc/lr0p1_switch/submit_all.sh`, log
  `slurm_logs/lr0p1_submit_all.log`) waits for the account-wide count (`squeue -A "$SLURM_ACCOUNT" -r`) to drop below 100,
  then calls `sbatch`, backing off 120 s on any rejection. Job IDs land in `jobs.txt` here. The jobs then need
  ~33 h of GPU time each, as 8-h requeued chunks.
- **Done without training:** leak probes (item 5; on the login-node CPU, `raw/probes.log`) and an early
  side-effect-free check of Protocol-A TEST (CPU). Job T0 re-does Protocol A with `main.py --phase test`,
  writing the official score dumps.

## Numbers so far: old (lr 0.01) → new (lr 0.1)

| # | item (P2 location) | old | new | status |
|---|---|--:|--:|---|
| 1 | Protocol-A TEST, Adaptive GNN (Table 7, tab:suite, efficiency table, tab:tracks, related-work audit row, abstract/§eff comparisons) | 79.07 ± 0.89 (77.88 / 80.00 / 79.34) | **77.05 ± 0.92** (77.73 / 77.66 / 75.75) | early CPU recheck; official `--phase test` dumps pending (job T0 PRE) |
| 1 | Protocol-A TEST, Attention GNN (Table 7, tab:suite, efficiency table) | 73.97 ± 1.01 | **73.92 ± 0.39** (73.99 / 74.36 / 73.41) | same |
| 5 | Leak table "supervised Adaptive GNN" \Lnorm / \Lnorm^wc (sec:leak, decoupling para) | 0.911 / 0.986 (one encoder, seed-0 m75) | **0.893 ± 0.009 / 0.989 ± 0.002** (3 seeds: 0.890 / 0.906 / 0.884) | done (`05_leak_probe.md`); old 3-seed best-ckpt mean 0.900 ± 0.007 |
| 2 | LOSO (Table 7 LOSO column, signer-clustered ranking) | Adaptive 80.23 [71.07, 88.03], 49.6–94.4, SD 14.94; Attention 75.74 [65.80, 84.35], 41.7–93.4, SD 16.44 | pending | 66 runs queued |
| 3 | Mixed / SMG (Table 7, tab:smg8) | Adaptive 98.73 ± 0.21 / +19.66; Attention 98.24 ± 0.22 / +24.27 | pending | 6 runs |
| 4 | Matched random-split SMG (tab:smg, Fig. 1, abstract, conclusion, app:smg8 footnote) | 98.97 [98.6, 99.3] − 79.07 = +19.90 | pending | 5 runs |
| 6 | Bone stream (tab:bone, app:bone) | 31.43 ± 0.60 (lr 0.01, clip 5) | pending | 3 runs |
| 7 | Significance (tab:sig, §archtable) | see P2 | pending (needs LOSO + TEST pkls) | `tools/arch_significance.py --lr0p1` |
| 8 | Variance decomposition (sec:loso, tab:losovar) | 15.14 / 3.42 / 2.07 / 2.54 | pending (needs LOSO) | |

### Claims, on the early Protocol-A numbers (to be confirmed)

- **Leaderboard top:** Adaptive GNN still leads (77.05 vs GNN+Transformer 75.55, ST-GCN 74.51), so it is still
  Pareto-optimal in §eff. Attention GNN stays 4th (73.92 < ST-GCN 74.51). The top-to-bottom spread becomes
  77.05 → 61.64 = 15.4 pp (P2: "~17 pp").
- **"VideoMAE 80.93 ± 2.11 … within seed noise of Adaptive GNN" no longer holds:** the gap grows from 1.86 to
  3.88 pp (Adaptive SD 0.92). RGB VideoMAE is now the best re-run non-foundation model on the split. The abstract
  and conclusion wording "at the level of the best pose model" and §eff "does not beat Adaptive GNN" need rewording.
- **Uni-Sign 83.61:** still above every Track-A pose model; its lead over the best one grows from 4.5 to 6.6 pp.
- **§eff extractor comparison:** BlockGCN → Adaptive GNN gap becomes +8.2 pp (P2 +10.3), still "about" the
  RTMPose extractor gain (+9.3).
- **Track B (tab:tracks):** Adaptive GNN Δ becomes 77.49 − 77.05 = +0.44 (P2 −1.58). Note that the Track-B Adaptive
  config already used base LR 0.1 (batch 64, 65 epochs). "Does not benefit" still reads approximately true.
- **Leak decoupling (sec:leak):** holds. Adaptive GNN encodes more signer identity (0.893 ± 0.009, every seed
  ≥ 0.884) than BlockGCN (0.853 ± 0.021, top seed 0.871), while generalizing better (77.05 vs 68.82). The text
  "0.911 … above BlockGCN's highest seed" becomes "0.893 ± 0.009 (3 seeds)". The abstract range "0.82–0.91" is
  unchanged (the SSL BdSL-only pool is 0.913).
- Limitations paragraph: "Adaptive GNN ±0.89" → ±0.92.

## Tooling (new, uncommitted)

- `tools/lr0p1_switch_tasks.py`: configs in `config/lr0p1_switch/**` and packed task lists
  `scripts/hpc/lr0p1_switch/tasks_T{0,1,2}.txt` (80 runs: 66 LOSO, 5 matched random splits, 6 signer-mixed,
  3 Adaptive bone). Checkpoints every 5 epochs.
- `scripts/hpc/slurm_lr0p1_switch.sbatch`: 2 GPUs × 3 concurrent runs per GPU. Runs as 8-h chunks: USR1 →
  `scontrol requeue`, and a requeued chunk resumes every unfinished run from its last checkpoint through
  `tools/run_resumable_seed.py`. Excludes <node> only. 2-GPU jobs were kept because submit slots, not GPUs, are
  the binding constraint (3 slots instead of 6).
- `tools/lr0p1_switch_eval.py`: protoA (`main.py --phase test`), oldpkl (old Adaptive seed-2 TEST dump; P2's suite
  and significance used only seeds 0/1), rand, sd, loso (→ `docs/loso_heldout/{adaptive_gnn,gnn}_lr0p1.csv`), and
  bone + fusion. Idempotent and cached, so chunk reruns stay cheap. T0 runs `protoA oldpkl` + `probes.sh` as PRE.
  Every job's TAIL (final chunk) runs `rand sd loso bone`.
- `scripts/hpc/lr0p1_switch/probes.sh` (idempotent), `submit.sh`, `submit_all.sh`.
- `tools/lr0p1_switch_report.py`: writes `01`–`09_*.md` (old column recomputed by the same code; it reproduces
  the P2 values) plus `raw/summary.json`.
- `tools/arch_significance.py`: new `--lr0p1` flag; the default output is unchanged and reproduces tab:sig.

## To finish (after the jobs complete)

```
python tools/lr0p1_switch_eval.py --sections protoA oldpkl rand sd loso bone   # idempotent; any GPU or CPU
python tools/lr0p1_switch_report.py
```
Then update `tools/make_paper_figures.py` `fig_smg` (Adaptive GNN si / mixed / range / gap) from `04_matched_splits.md`.
After the evals, prune checkpoints to best + last per run (keep all of the rand runs' checkpoints until
`rand` has been evaluated).
