#!/usr/bin/env python
"""Generate configs + packed task lists for the P2 Adaptive GNN / Attention GNN lr-0.1 switch.

Under the strict 3-seed validation-argmax rule (the rule that moved BlockGCN 0.1 -> 0.01),
Adaptive GNN (config/bdsl_adaptive_gnn_si.yaml) and Attention GNN (config/bdsl_gnn_si.yaml)
move from base_lr 0.01 to 0.1 (U15 val, 3 seeds: Adaptive 89.92 vs 88.80, Attention 85.70 vs
85.19). The Protocol-A lr-0.1 runs already exist (config/lrsweep/bdsl_{adaptive_gnn,gnn}_si_lr0p1.yaml,
work_dir/bdsl_{adaptive_gnn,gnn}_si_lr0p1_seed{0,1,2}). This script derives every other trained
experiment P2 reports for these two architectures, at base_lr 0.1, protocol otherwise unchanged:

  LOSO     11 held-out-signer folds x 3 seeds x 2 archs (fold train/val bundles, select on fold val)
  rand     Adaptive GNN on the 5 matched random splits (select on own val; save every 5 epochs so the
           old best-on-own-test estimator is recomputable on exactly the old eval epochs)
  sd       signer-mixed data/bdsl_sd, unclipped, select on sd val, score once on sd test (2 archs)
  bone     Adaptive GNN bone stream (tab:bone), same recipe as the new joint stream (lr 0.1, no clip)

lr-0.01 configs and work_dirs are never touched: new configs live in config/lr0p1_switch/ and
every new experiment name carries `lr0p1`.

Writes config/lr0p1_switch/**.yaml and scripts/hpc/lr0p1_switch/tasks_T<n>.txt (one shell command
per line; run 3-concurrent per GPU on 2 GPUs by scripts/hpc/slurm_lr0p1_switch.sbatch).
Usage: python tools/lr0p1_switch_tasks.py [--ntasks 5 --gpus 2]
"""
import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = ROOT / "config" / "lr0p1_switch"
TASK_DIR = ROOT / "scripts" / "hpc" / "lr0p1_switch"
RR = "python -u tools/run_resumable_seed.py"
SEEDS = [0, 1, 2]
ARCH = {"adaptive_gnn": "config/lrsweep/bdsl_adaptive_gnn_si_lr0p1.yaml",
        "gnn": "config/lrsweep/bdsl_gnn_si_lr0p1.yaml"}
# rough wall-hours per run at 3 runs/GPU (used only to balance the packed task lists)
HOURS = {"adaptive_gnn": 7.3, "gnn": 6.0}   # lr-0.1 Protocol-A runs at 3/GPU took 7.25 h / 6.0 h


def load(p):
    return yaml.safe_load(open(ROOT / p))


def dump(cfg, rel):
    out = CFG_DIR / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(out, "w"), default_flow_style=False, sort_keys=False)
    return str(out.relative_to(ROOT))


def derive(src, name, save_interval=5, **over):
    """config/<src>.yaml -> config/lr0p1_switch/<name>.yaml with base_lr 0.1, no clip."""
    cfg = load(f"config/{src}.yaml")
    cfg["Experiment_name"] = name
    cfg["base_lr"] = 0.1
    cfg["grad_clip_norm"] = 0.0
    cfg["save_interval"] = save_interval   # checkpoint cadence only (resume granularity in 8-h chunks); no effect on training
    cfg.update(over)
    return dump(cfg, f"{name}.yaml")


def rr(cfg, seed, exp, test_cfg=None, test_exp=None):
    cmd = f"{RR} --config {cfg} --seed {seed} --exp {exp}"
    if test_cfg:
        cmd += f" --test-config {test_cfg} --test-exp {test_exp}"
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntasks", type=int, default=5)
    ap.add_argument("--gpus", type=int, default=2, help="GPUs per job (3 runs each)")
    args = ap.parse_args(); n_tasks = args.ntasks
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    TASK_DIR.mkdir(parents=True, exist_ok=True)
    runs = []   # (hours, cmd)

    # LOSO: per-fold configs = the lr-0.1 Protocol-A config with the fold's train/val bundles.
    folds = sorted(p.name for p in (ROOT / "data" / "bdsl_si_loso").glob("test_U*_val_U*"))
    assert len(folds) == 11, folds
    for arch, base_p in ARCH.items():
        base = load(base_p)
        for fd in folds:
            ts = int(fd.split("_")[1][1:])
            cfg = dict(base)
            cfg["train_feeder_args"] = {**base["train_feeder_args"],
                                        "data_path": f"data/bdsl_si_loso/{fd}/train_data.npy",
                                        "label_path": f"data/bdsl_si_loso/{fd}/train_label.pkl"}
            cfg["test_feeder_args"] = {**base["test_feeder_args"],
                                       "data_path": f"data/bdsl_si_loso/{fd}/val_data.npy",
                                       "label_path": f"data/bdsl_si_loso/{fd}/val_label.pkl"}
            name = f"bdsl_{arch}_si_lr0p1_loso_test{ts:02d}"
            cfg["Experiment_name"] = name
            cfg["grad_clip_norm"] = 0.0
            cfg["save_interval"] = 5   # 8-h chunked jobs: lose at most 5 epochs per requeue
            p = dump(cfg, f"loso/{name}.yaml")
            for s in SEEDS:
                runs.append((HOURS[arch] * 1.15, rr(p, s, f"{name}_seed{s}")))

    # Matched random splits (Adaptive GNN only; tab:smg). Select on the split's own val, as the
    # BlockGCN lr-0.01 switch did; save_interval 5 == the old every-5-epoch test evals.
    for r in range(42, 47):
        src = load(f"config/bdsl_adaptive_gnn_rand{r}.yaml")
        c = derive(f"bdsl_adaptive_gnn_rand{r}", f"bdsl_adaptive_gnn_rand{r}_lr0p1", save_interval=5,
                   test_feeder_args={**src["test_feeder_args"],
                                     "data_path": f"./data/bdsl_rand{r}/val_data.npy",
                                     "label_path": f"./data/bdsl_rand{r}/val_label.pkl"})
        runs.append((HOURS["adaptive_gnn"], rr(c, 0, f"bdsl_adaptive_gnn_rand{r}_lr0p1_seed0")))

    # Signer-mixed bdsl_sd, unclipped, symmetric rule (select on sd val, score once on sd test).
    for arch in ARCH:
        c = derive(f"bdsl_{arch}_sd_noclip", f"bdsl_{arch}_sd_noclip_lr0p1")
        t = derive(f"bdsl_{arch}_sd_noclip_test", f"bdsl_{arch}_sd_noclip_lr0p1_test")
        for s in SEEDS:
            runs.append((HOURS[arch], rr(c, s, f"bdsl_{arch}_sd_noclip_lr0p1_seed{s}",
                                         t, f"bdsl_{arch}_sd_noclip_lr0p1_test_seed{s}")))

    # Adaptive GNN bone stream (tab:bone). Old row = bdsl_adaptive_gnn_bone_clip5 (lr 0.01, clip 5.0);
    # new = lr 0.1, no clipping (= the new joint recipe), is_vector False as before.
    c = derive("bdsl_adaptive_gnn_bone", "bdsl_adaptive_gnn_bone_lr0p1")
    t = derive("bdsl_adaptive_gnn_bone_test", "bdsl_adaptive_gnn_bone_lr0p1_test")
    for s in SEEDS:
        runs.append((HOURS["adaptive_gnn"], rr(c, s, f"bdsl_adaptive_gnn_bone_lr0p1_seed{s}",
                                               t, f"bdsl_adaptive_gnn_bone_lr0p1_test_seed{s}")))

    # Greedy longest-first packing into n_tasks lists (each list runs 3-concurrent on one GPU).
    tasks = [[0.0, []] for _ in range(n_tasks)]
    for h, cmd in sorted(runs, key=lambda x: -x[0]):
        t = min(tasks, key=lambda x: x[0])
        t[0] += h; t[1].append(cmd)
    for old in TASK_DIR.glob("tasks_T*.txt"):
        old.unlink()
    for i, (h, cmds) in enumerate(tasks):
        (TASK_DIR / f"tasks_T{i}.txt").write_text("\n".join(cmds) + "\n")
        print(f"T{i}: {len(cmds)} runs, ~{h / (3 * args.gpus):.1f} h wall at 3 runs/GPU x {args.gpus} GPUs")
    print(f"total {len(runs)} runs")


if __name__ == "__main__":
    main()
