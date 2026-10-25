#!/usr/bin/env python
"""Held-out-signer LOSO scoring WITH a completeness gate (v2 of
scripts/hpc/slurm_loso_heldout_allarch.sbatch).

For docs/loso_heldout/<name>.csv (name = arch[_suffix], e.g. slgtformer, gnn_lstm,
gnn_transformer_clip5) this scores each fold run's `<run>_model-best.pt` on that fold's
HELD-OUT test signer (data/bdsl_si_loso/test_U<FF>_val_U*/test_*), exactly as the v1 script
did (tools/eval_ckpt_on_npy.py, same CSV format and row order), but it REFUSES to write when:
  * any fold run is incomplete (tools/check_run_complete.train_status: max "Eval epoch" <
    num_epoch of the run's saved config and no final-epoch checkpoint), or
  * a run has no model-best.pt, a fold bundle is missing, the checkpoint loads with missing /
    unexpected keys, or fewer than --expect cells exist.
v1 scored every model-best.pt it found, so killed runs (e.g. SLGTFormer U12 seed 2 stopped at
epoch 15 -> 37.1) entered the CSVs; see docs/p2_rerun_fixes_2026-10/.

The previous CSV is kept once as <name>.pre_fix_2026-10-22.csv (never overwritten).

  python tools/loso_heldout_eval.py slgtformer gnn_lstm [--dry-run] [--expect 33]
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import check_run_complete as C  # noqa: E402

BACKUP_SUFFIX = ".pre_fix_2026-10-22.csv"


def runs_for(name):
    stem = C.loso_run_stem(name)
    pat = re.compile("^" + re.escape(stem).replace(r"\{f\}", r"(\d+)").replace(r"\{s\}", r"(\d+)") + "$")
    out = []
    for d in sorted(os.listdir(os.path.join(ROOT, "work_dir"))):   # same order as the v1 bash glob
        m = pat.match(d)
        if m:
            out.append((m.group(1), int(m.group(2)), d))
    return out


def score(run, fold):
    wd = os.path.join(ROOT, "work_dir", run)
    ck = os.path.join(wd, f"{run}_model-best.pt")
    if not os.path.isfile(ck):
        return None, f"no {os.path.basename(ck)}"
    fds = sorted(glob.glob(os.path.join(ROOT, f"data/bdsl_si_loso/test_U{fold}_val_U*")))
    if len(fds) != 1:
        return None, f"fold bundle for U{fold}: {len(fds)} candidates"
    fd = fds[0]
    r = subprocess.run([sys.executable, "tools/eval_ckpt_on_npy.py", "--config", os.path.join(wd, "config.yaml"),
                        "--weights", ck, "--data", os.path.join(fd, "test_data.npy"),
                        "--label", os.path.join(fd, "test_label.pkl")],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        return None, "eval failed: " + r.stderr.strip()[-300:]
    if "[warn] missing=" in r.stderr:
        return None, "state_dict mismatch: " + r.stderr.strip()[-200:]
    m = re.search(r"top1=([0-9.]+)", r.stdout)
    return (m.group(1), "") if m else (None, "no top1 in output")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+")
    ap.add_argument("--expect", type=int, default=33, help="required cells per CSV (11 folds x 3 seeds)")
    ap.add_argument("--dry-run", action="store_true", help="gate only: report completeness, score nothing")
    a = ap.parse_args()
    rc = 0
    for name in a.names:
        runs = runs_for(name)
        bad = []
        for fold, seed, run in runs:
            st = C.train_status(os.path.join(ROOT, "work_dir", run))
            if not st.get("complete"):
                bad.append(f"{run}: {st['status']} (eval {st.get('last_eval')}/{st.get('num_epoch')})")
        if len(runs) < a.expect:
            bad.append(f"only {len(runs)}/{a.expect} fold runs on disk")
        if bad:
            print(f"[REFUSE] {name}: {len(bad)} problem(s); docs/loso_heldout/{name}.csv left untouched")
            for b in bad:
                print("   ", b)
            rc = 2
            continue
        print(f"[gate ok] {name}: {len(runs)} complete fold runs")
        if a.dry_run:
            continue
        rows, errs = ["fold,seed,heldout_signer,test_acc"], []
        for fold, seed, run in runs:
            acc, err = score(run, fold)
            if acc is None:
                errs.append(f"{run}: {err}")
                continue
            rows.append(f"{fold},{seed},U{fold},{acc}")
            print(f"  {name} test{fold} seed{seed} -> {acc}", flush=True)
        if errs:
            print(f"[REFUSE] {name}: scoring errors; CSV left untouched")
            for e in errs:
                print("   ", e)
            rc = 2
            continue
        out = os.path.join(ROOT, "docs/loso_heldout", f"{name}.csv")
        bak = out[:-4] + BACKUP_SUFFIX
        old = {}
        if os.path.isfile(out):
            if not os.path.isfile(bak):
                shutil.copy2(out, bak)
                print(f"  backup -> {os.path.relpath(bak, ROOT)}")
            for line in open(bak).read().splitlines()[1:]:
                f_, s_, _, v = line.split(",")
                old[(f_, s_)] = v
        tmp = out + ".tmp"
        open(tmp, "w").write("\n".join(rows) + "\n")
        os.replace(tmp, out)
        print(f"  wrote {os.path.relpath(out, ROOT)}")
        for line in rows[1:]:
            f_, s_, _, v = line.split(",")
            if old.get((f_, s_)) != v:
                print(f"    changed test{f_} seed{s_}: {old.get((f_, s_))} -> {v}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
