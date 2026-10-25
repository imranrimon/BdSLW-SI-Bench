#!/usr/bin/env python
"""Task lists for the P2 truncated-run fixes (docs/p2_rerun_fixes_2026-10/).

Two kinds of work, both through tools/run_resumable_seed.py (resume from the latest checkpoint,
never from epoch 0 when one exists; finished runs are skipped on requeue):

  RESUME   incomplete runs that P2 uses (found by tools/check_run_complete.py --p2-sweep):
           SLGTFormer Protocol-A seed 2, SLGTFormer clip-1 seed 1, 9 SLGTFormer LOSO cells,
           2 GNN+LSTM LOSO cells. Each run resumes with its OWN saved config
           (work_dir/<run>/config.yaml -> config/p2_rerun_fixes/<run>.yaml, old /users clone
           prefix stripped, run-specific keys dropped), so the recipe is exactly the original one.
  BESTCK   Protocol-A TEST dumps that scored a retained per-epoch checkpoint instead of the
           best-on-val epoch (main.py did not save <run>_model-best.pt yet). Affected seeds are
           re-trained from scratch under a new name `bdsl_<arch>_si_bestck_seed<S>` (old dirs kept);
           unaffected seeds (scored checkpoint == best-val epoch) are re-scored into
           `bdsl_<arch>_si_bestck_seed<S>_test` from the original checkpoint, so `bdsl_<arch>_si_bestck`
           is a complete 3-seed stem for tools/p2_make_tables.py / tools/p2_metric_suite.py.

Adaptive GNN / Attention GNN are excluded (moving to lr 0.1, re-run by jobs lr0p1_S00..S25).

Writes config/p2_rerun_fixes/*.yaml and scripts/hpc/p2_rerun_fixes/tasks_<name>.txt
(one shell command per line; the lane runner deals them round-robin to the job's GPUs).
"""
import os
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import check_run_complete as C  # noqa: E402

CFG = ROOT / "config" / "p2_rerun_fixes"
TASKS = ROOT / "scripts" / "hpc" / "p2_rerun_fixes"
RR = "python -u tools/run_resumable_seed.py"
DROP = {"Experiment_name", "config", "work_dir", "model_saved_name", "phase", "weights", "start_epoch",
        "seed", "ignore_weights"}


def strip_prefix(o):
    if isinstance(o, dict):
        return {k: strip_prefix(v) for k, v in o.items()}
    if isinstance(o, list):
        return [strip_prefix(v) for v in o]
    if isinstance(o, str):
        # strip an absolute repository prefix (".../<repo>/work_dir/..." -> "work_dir/...")
        return re.sub(r"^/.*?/(?=(?:work_dir|data|config)/)", "", o)
    return o


def own_config(run):
    """work_dir/<run>/config.yaml (the run's resolved args) -> config/p2_rerun_fixes/<run>.yaml."""
    cfg = yaml.safe_load(open(ROOT / "work_dir" / run / "config.yaml"))
    cfg = {k: v for k, v in strip_prefix(cfg).items() if k not in DROP}
    for k in ("train_feeder_args", "test_feeder_args"):
        p = cfg[k]["data_path"]
        assert (ROOT / p).is_file(), f"{run}: {p} missing"
    CFG.mkdir(parents=True, exist_ok=True)
    out = CFG / f"{run}.yaml"
    yaml.safe_dump(cfg, open(out, "w"), default_flow_style=False, sort_keys=False)
    return str(out.relative_to(ROOT))


def seed_of(run):
    return int(re.search(r"_seed(\d+)$", run).group(1))


def resume(run, test_cfg=None, test_exp=None):
    st = C.train_status(ROOT / "work_dir" / run)
    assert st["status"] in ("INCOMPLETE", "IN_PROGRESS") and st["ckpt_epochs"], (run, st["status"])
    cmd = f"{RR} --config {own_config(run)} --seed {seed_of(run)} --exp {run}"
    if test_cfg:
        cmd += f" --test-config {test_cfg} --test-exp {test_exp}"
    return cmd, f"{run}: resume from epoch {st['resume_from']} (ckpt {max(st['ckpt_epochs'])})"


def testonly(weights, test_cfg, exp, seed):
    """Idempotent TEST re-score (skipped once its score dump is newer than the weights)."""
    pk = f"work_dir/{exp}/eval_results/best_acc.pkl"
    return (f"{{ [ -f {pk} ] && [ {pk} -nt {weights} ]; }} || python -u main.py --config {test_cfg} "
            f"--phase test --weights {weights} -Experiment_name {exp} --seed {seed}")


def bestck(arch):
    """(train/test commands, notes) giving a complete best-on-val 3-seed stem bdsl_<arch>_si_bestck."""
    cmds, notes = [], []
    for s in (0, 1, 2):
        st = C.train_status(ROOT / "work_dir" / f"bdsl_{arch}_si_seed{s}_test")
        if arch == "slgtformer" and s == 2:   # resumed run: its own best ckpt, re-scored after training
            w = "work_dir/bdsl_slgtformer_si_seed2/bdsl_slgtformer_si_seed2_model-best.pt"
            cmds.append(("chain", testonly(w, f"config/bdsl_{arch}_si_test.yaml", f"bdsl_{arch}_si_bestck_seed2_test", 2)))
            notes.append(f"bdsl_{arch}_si_bestck_seed2_test: re-score of the resumed seed-2 model-best.pt")
        elif st.get("uses_best"):
            w = os.path.relpath(C.local_dir(st["weights"]), ROOT) + "/" + os.path.basename(st["weights"].replace("\\", "/"))
            cmds.append(("light", testonly(w, f"config/bdsl_{arch}_si_test.yaml", f"bdsl_{arch}_si_bestck_seed{s}_test", s)))
            notes.append(f"bdsl_{arch}_si_bestck_seed{s}_test: re-score of {os.path.basename(w)} (= best-val epoch)")
        else:
            exp = f"bdsl_{arch}_si_bestck_seed{s}"
            cmds.append(("train", f"{RR} --config config/bdsl_{arch}_si.yaml --seed {s} --exp {exp} "
                                  f"--test-config config/bdsl_{arch}_si_test.yaml --test-exp {exp}_test"))
            notes.append(f"{exp}: FRESH retrain (old TEST scored ep{st.get('scored_epoch')} "
                         f"val {st.get('scored_val')}; best ep{st.get('best_epoch')} val {st.get('best_val')}; "
                         f"best-epoch weights were never saved)")
    return cmds, notes


def main():
    TASKS.mkdir(parents=True, exist_ok=True)
    notes = []
    # ---- SLGTFormer lanes (1 run per GPU). Each inner list = one GPU lane, run sequentially.
    c, n = resume("bdsl_slgtformer_si_seed2", "config/bdsl_slgtformer_si_test.yaml", "bdsl_slgtformer_si_seed2_test")
    notes.append(n)
    slgt_bestck, nb = bestck("slgtformer")
    notes += nb
    lane_seed2 = [c] + [x for k, x in slgt_bestck if k == "chain"]
    lanes = [lane_seed2]
    c, n = resume("bdsl_slgtformer_si_clip1_seed1")
    notes.append(n)
    lanes.append([c])
    loso = {}
    for f_, s in [("12", 2), ("06", 0), ("08", 0), ("09", 0), ("11", 0), ("01", 2), ("02", 2), ("13", 2), ("15", 2)]:
        c, n = resume(f"bdsl_slgtformer_si_loso_test{f_}_seed{s}")
        loso[(f_, s)] = c
        notes.append(n)
    lanes += [[loso[("12", 2)]], [loso[("06", 0)]], [loso[("08", 0)]], [loso[("09", 0)]], [loso[("11", 0)]],
              [loso[("01", 2)], loso[("02", 2)]], [loso[("13", 2)], loso[("15", 2)]]]
    lanes += [[x] for k, x in slgt_bestck if k == "train"]
    # pack 2 lanes per job: g0 gets odd lines, g1 even lines (lane runner deals round-robin)
    jobs = [lanes[i:i + 2] for i in range(0, len(lanes), 2)]
    for j, pair in enumerate(jobs):
        lines = []
        for k in range(max(len(l) for l in pair)):
            for l in pair:
                lines.append(l[k] if k < len(l) else "true")   # placeholder keeps the round-robin aligned
        (TASKS / f"tasks_S{j}.txt").write_text("\n".join(lines) + "\n")
        print(f"tasks_S{j}.txt: {len(pair)} GPU(s), {len(lines)} lines")
    # ---- light lane (<=3 per GPU): GNN+LSTM LOSO resumes + best-ckpt Protocol-A fixes
    light = []
    for run in ("bdsl_gnn_lstm_si_loso_test01_seed2", "bdsl_gnn_lstm_si_loso_test04_seed2"):
        c, n = resume(run)
        light.append(c)
        notes.append(n)
    for arch in ("gnn_lstm", "pose_lstm", "st_gcn_vanilla"):
        cmds, nb = bestck(arch)
        notes += nb
        light += [x for k, x in cmds if k == "train"]
        light += [x for k, x in cmds if k == "light"]
    (TASKS / "tasks_L0.txt").write_text("\n".join(light) + "\n")
    print(f"tasks_L0.txt: {len(light)} lines")
    (TASKS / "PLAN.txt").write_text("\n".join(notes) + "\n")
    print("\n".join(notes))


if __name__ == "__main__":
    main()
