# Cluster launchers (SLURM)

Generic SLURM scripts for the experiment programs. Before submitting, set the partition on the
`#SBATCH -p <PARTITION>` line (`sinfo -s`) and, if your cluster needs it, an account
(`#SBATCH -A ...`). Every script `source`s `~/miniconda3/etc/profile.d/conda.sh`, activates the
`bdsl_graph` environment and runs from `$SLURM_SUBMIT_DIR`, so submit from the repository root.

| Script | Purpose |
|---|---|
| `slurm_si_sweep.sbatch` | Three-seed SI sweep over `experiments_si_main.yaml` + SLGTFormer ablations |
| `slurm_loso_array.sbatch` | 11-fold leave-one-signer-out sweep as a job array (`CONFIG=...` selects the model) |
| `slurm_ssl_pretrain.sbatch` | Masked pose SSL pre-training (`CONFIG=...`, default BdSL-only pool) |
| `slurm_lr01_switch.sbatch`, `slurm_lr0p1_switch.sbatch`, `slurm_p2_mjfix.sbatch`, `slurm_p2_rerun_fixes.sbatch`, `slurm_signernorm.sbatch`, `slurm_tta_grl_2026_10.sbatch` | Packed task-list runners: `TASKS=<list>` is dealt round-robin over the allocated GPUs (`MAXPAR` runs per GPU) |

The task-list runners work in 8-hour chunks: SLURM sends `USR1` 15 minutes before the limit,
the job requeues itself, and every training command goes through `tools/run_resumable_seed.py`,
which resumes from the latest checkpoint and skips finished runs.

```bash
sbatch --export=ALL,TASKS=scripts/hpc/lr01_switch/tasks_J1.txt -J lr01_J1 scripts/hpc/slurm_lr01_switch.sbatch
```

## Task lists

`<program>/tasks_*.txt` hold one shell command per line: the exact commands that produced each
program's runs (see the matching `docs/<program>/README.md`). They also run without SLURM:

```bash
bash scripts/run_tasks_local.sh scripts/hpc/lr01_switch/tasks_J1.txt
```

| Directory | Program notes |
|---|---|
| `lr01_switch/` | `docs/blockgcn_lr01_switch/` |
| `lr0p1_switch/` | `docs/adaptive_attn_lr0p1_switch/` |
| `p2_rerun_fixes/` | `docs/p2_rerun_fixes_2026-10/` |
| `p2_mjfix/` | `docs/p2_missing_joint_fixes_2026-10/` |
| `signernorm/` | `docs/signernorm_2026-10/` |

The `*.sh` files in those directories are the analysis steps run after training (probes,
reports, table generation); they `cd` to the repository root themselves.
