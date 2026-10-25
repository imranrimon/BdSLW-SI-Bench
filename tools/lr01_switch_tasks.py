#!/usr/bin/env python
"""Generate configs + packed task lists for the P2 BlockGCN lr-0.01 switch.

The shared-recipe BlockGCN (config/bdsl_block_gcn_si.yaml, base_lr 0.1) was not the
validation-selected learning rate: config/bdsl_block_gcn_si_lr01.yaml (identical except
base_lr 0.01) scores higher on val (77.96 vs 74.25). Every BdSLW60 BlockGCN experiment in
P2 is re-run at base_lr 0.01, no gradient clipping, otherwise unchanged. lr-0.1 configs and
work_dirs are never touched: every new config lives in config/lr01_switch/ and every new
experiment name carries `lr01`.

Writes:
  config/lr01_switch/*.yaml                 derived configs (base_lr 0.01, grad_clip_norm 0.0)
  scripts/hpc/lr01_switch/tasks_J<n>.txt    one shell command per line, run concurrently by
                                            scripts/hpc/slurm_lr01_switch.sbatch
Usage: python tools/lr01_switch_tasks.py
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = ROOT / "config" / "lr01_switch"
TASK_DIR = ROOT / "scripts" / "hpc" / "lr01_switch"
RR = "python -u tools/run_resumable_seed.py"
SEEDS = [0, 1, 2]
LR01 = "config/bdsl_block_gcn_si_lr01.yaml"   # Protocol-A lr-0.01 config (already run)
SI_INIT = "work_dir/bdsl_block_gcn_si_lr01_scratch_seed0/bdsl_block_gcn_si_lr01_scratch_seed0_model-best.pt"
SSL_BB = "work_dir/bdsl_shubert_pretrain_bdsl_only/pretrained_epoch30.pt"


def derive(src, name, **over):
    """Copy config/<src>.yaml -> config/lr01_switch/<name>.yaml with base_lr 0.01, no clip."""
    cfg = yaml.safe_load(open(ROOT / "config" / f"{src}.yaml"))
    cfg["Experiment_name"] = name
    cfg["base_lr"] = 0.01
    cfg["grad_clip_norm"] = 0.0
    for k, v in over.items():
        cfg[k] = v
    out = CFG_DIR / f"{name}.yaml"
    yaml.safe_dump(cfg, open(out, "w"), default_flow_style=False, sort_keys=False)
    return str(out.relative_to(ROOT))


def rr(cfg, seed, exp, test_cfg=None, init=None):
    cmd = f"{RR} --config {cfg} --seed {seed} --exp {exp}"
    if test_cfg:
        cmd += f" --test-config {test_cfg} --test-exp {exp.replace(f'_seed{seed}', '')}_test_seed{seed}"
    if init:
        cmd += f" --init-weights {init}"
    return cmd


def main():
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    TASK_DIR.mkdir(parents=True, exist_ok=True)
    J = {i: [] for i in range(1, 7)}

    # J1 (headline): signer-mixed bdsl_sd + 5 matched random splits.
    sd = derive("bdsl_block_gcn_sd", "bdsl_block_gcn_sd_lr01")
    for s in SEEDS:
        J[1].append(rr(sd, s, f"bdsl_block_gcn_sd_lr01_seed{s}"))
    for r in range(42, 47):
        # select on the split's own val (old runs selected on their own test set);
        # the old estimator is recomputed from saved checkpoints at eval time.
        c = derive(f"bdsl_block_gcn_rand{r}", f"bdsl_block_gcn_rand{r}_lr01",
                   test_feeder_args={**yaml.safe_load(open(ROOT / f"config/bdsl_block_gcn_rand{r}.yaml"))["test_feeder_args"],
                                     "data_path": f"data/bdsl_rand{r}/val_data.npy",
                                     "label_path": f"data/bdsl_rand{r}/val_label.pkl"})
        J[1].append(rr(c, 0, f"bdsl_block_gcn_rand{r}_lr01_seed0"))

    # J2/J3: LOSO 11 folds x 3 seeds, per-fold configs (train/val of the fold bundle).
    folds = sorted(p.name for p in (ROOT / "data" / "bdsl_si_loso").glob("test_U*_val_U*"))
    base = yaml.safe_load(open(ROOT / LR01))
    (CFG_DIR / "loso").mkdir(exist_ok=True)
    loso = []
    for fd in folds:
        ts = int(fd.split("_")[1][1:])
        cfg = dict(base)
        cfg["train_feeder_args"] = {**base["train_feeder_args"],
                                    "data_path": f"data/bdsl_si_loso/{fd}/train_data.npy",
                                    "label_path": f"data/bdsl_si_loso/{fd}/train_label.pkl"}
        cfg["test_feeder_args"] = {**base["test_feeder_args"],
                                   "data_path": f"data/bdsl_si_loso/{fd}/val_data.npy",
                                   "label_path": f"data/bdsl_si_loso/{fd}/val_label.pkl"}
        name = f"bdsl_block_gcn_si_lr01_loso_test{ts:02d}"
        cfg["Experiment_name"] = name
        p = CFG_DIR / "loso" / f"{name}.yaml"
        yaml.safe_dump(cfg, open(p, "w"), default_flow_style=False, sort_keys=False)
        for s in SEEDS:
            loso.append(rr(str(p.relative_to(ROOT)), s, f"{name}_seed{s}"))
    J[2], J[3] = loso[:17], loso[17:]

    # J4: GRL sweep, keypoint drop, mirror, bone, RTMPose, SSL init (+ TEST evals).
    grl = derive("bdsl_block_gcn_si_signeradv", "bdsl_block_gcn_si_signeradv_lr01")
    for lam, tag in [(0.1, "01"), (0.5, "05"), (1.0, "10")]:
        J[4].append(f"python -u main_signer_adv.py --config {grl} --seed 0 --grl-lambda {lam} "
                    f"-Experiment_name bdsl_block_gcn_si_signeradv_lr01_gl{tag}_seed0")
    for v in ["handsonly", "bodyonly", "mirror", "bone", "rtm"]:
        c = derive(f"bdsl_block_gcn_si_{v}", f"bdsl_block_gcn_si_{v}_lr01")
        for s in SEEDS:
            J[4].append(rr(c, s, f"bdsl_block_gcn_si_{v}_lr01_seed{s}",
                           test_cfg=f"config/bdsl_block_gcn_si_{v}_test.yaml"))
    for s in SEEDS:
        J[4].append(rr(LR01, s, f"bdsl_ssl_bdslonly_si_ft_lr01_seed{s}",
                       test_cfg="config/bdsl_block_gcn_si_test.yaml", init=SSL_BB))

    # J5: cue perturbations (final-epoch TEST via the cue test feeder) + paired-instance SMG.
    for cue in ["static", "speednorm", "shuffle", "bodyonly", "handsonly", "rootcenter"]:
        c = derive(f"bdsl_block_gcn_cue_{cue}", f"bdsl_cue_{cue}_lr01")
        for s in SEEDS:
            J[5].append(rr(c, s, f"bdsl_cue_{cue}_lr01_seed{s}"))
    for f in [2, 13]:
        for cond in ["unseen", "seen_matched", "seen_aug"]:
            for s in SEEDS:
                J[5].append(f"python -u tools/run_paired_smg.py --config {LR01} --focus-signers {f} "
                            f"--conditions {cond} --seeds {s} --skip-existing")

    # J6: 2D diversity grid, BdSLW60 SMG(k), few-shot enrollment.
    for sg in [2, 4, 6, 8]:
        for n in [800, 1500]:
            c = derive(f"bdsl_block_gcn_grid_s{sg}_n{n}", f"bdsl_grid_s{sg}_n{n}_lr01")
            for s in SEEDS:
                J[6].append(rr(c, s, f"bdsl_grid_s{sg}_n{n}_lr01_seed{s}"))
    for k, draws in [(2, range(5)), (4, range(5)), (6, range(5)), (8, [0])]:
        for d in draws:
            J[6].append(f"python -u tools/run_diversity_grid.py --config {LR01} --k {k} --draw {d} --seed 0 --skip-existing")
    for f in [2, 13]:
        for k in [0, 1, 3, 5]:
            for s in SEEDS:
                J[6].append(f"python -u tools/run_personalization.py --config {LR01} --signer {f} --k {k} "
                            f"--seed {s} --init-weights {SI_INIT} --skip-existing")

    for i, cmds in J.items():
        (TASK_DIR / f"tasks_J{i}.txt").write_text("\n".join(cmds) + "\n")
        print(f"J{i}: {len(cmds)} tasks")


if __name__ == "__main__":
    main()
