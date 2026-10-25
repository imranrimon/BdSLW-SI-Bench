#!/usr/bin/env python
"""Aggregate the BdSLW401 signer-disjoint-validation (S15) rerun into
docs/w401_sdval/README.md: old (seen-signer val) vs new (S15 val) val Top-1,
TEST Top-1 + macro-F1 per seed, the BlockGCN LR decision on the new val, and
the signer-decodability probes. Population SD throughout (paper convention).

Usage: python tools/w401_sdval_report.py
"""
import csv
import glob
import os
import pickle
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.p2_metric_suite import _metrics_one, to_prob  # noqa: E402

TEST_LABEL = os.path.join(ROOT, "data/bdslw401_si/test_label.pkl")
MODELS = [  # display, new stem, old stem (seen-signer val)
    ("CTR-GCN (lr 0.1)", "bdsl_ctr_gcn_w401_sdval", "bdsl_ctr_gcn_w401"),
    ("ST-GCN (lr 0.1)", "bdsl_st_gcn_w401_sdval", "bdsl_st_gcn_w401"),
    ("BlockGCN (lr 0.1)", "bdsl_block_gcn_w401_sdval", "bdsl_block_gcn_w401"),
    ("BlockGCN (lr 0.01)", "bdsl_block_gcn_w401_lr01_sdval", "bdsl_block_gcn_w401_lr01"),
]
PAPER = {"bdsl_ctr_gcn_w401": "86.98±0.46", "bdsl_st_gcn_w401": "86.87±0.47",
         "bdsl_block_gcn_w401": "61.53±13.44", "bdsl_block_gcn_w401_lr01": "— (job)"}
OLD_LNORM = {"bdsl_ctr_gcn_w401": "cc 0.693±0.004 / wc 0.992", "bdsl_block_gcn_w401": "cc 0.427±0.127 / wc 0.953"}


def val_best():
    best = {}
    with open(os.path.join(ROOT, "results_final.csv")) as f:
        for r in csv.reader(f):
            if len(r) < 4:
                continue
            m = re.fullmatch(r"(.+)_seed(\d)", r[1])
            if not m:
                continue
            try:
                v = float(r[3]) * 100
            except ValueError:
                continue
            k = (m.group(1), int(m.group(2)))
            best[k] = max(best.get(k, -1), v)
    return best


def test_metrics(stem, seed, lab):
    for p in (f"work_dir/{stem}_test_seed{seed}/eval_results/best_acc.pkl",
              f"work_dir/{stem}_seed{seed}_test/eval_results/best_acc.pkl"):
        p = os.path.join(ROOT, p)
        if os.path.isfile(p):
            sc = {k: to_prob(v) for k, v in pickle.load(open(p, "rb")).items()}
            common = sorted(set(sc) & set(lab))
            m = _metrics_one(np.stack([sc[c] for c in common]), np.array([lab[c] for c in common]))
            return m["top1"], m["macrof1"], len(common)
    return None


def fmt(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "pending"
    return f"{np.mean(xs):.2f} ± {np.std(xs):.2f}" + ("" if len(xs) == 3 else f" (n={len(xs)})")


def probes():
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "docs/w401_sdval/raw/probe_seed*.txt"))):
        for line in open(f):
            m = re.search(r"\[(.+?)_s(\d)\|(cc8k|wcall)\]", line)
            if not m:
                continue
            v = re.search(r"L_norm\^cond = ([0-9.]+)", line) if m.group(3) == "wcall" \
                else re.search(r"L_norm = ([0-9.]+)", line)
            if v:
                out.setdefault((m.group(1), m.group(3)), {})[int(m.group(2))] = float(v.group(1))
    return out


def main():
    names, labels = pickle.load(open(TEST_LABEL, "rb"))
    lab = dict(zip(names, labels))
    vb = val_best()
    pr = probes()
    L = ["# BdSLW401 with signer-disjoint validation (held-out signer S15)", "",
         "Bundle `data/bdslw401_si_sdval` (`preprocessing/make_w401_sdval_bundle.py`, `splits/bdslw401_sdval.json`): "
         "train = old train+val minus S15 (15 signers, 39,315 clips); val = all S15 clips (3,950; 400/401 classes); "
         "test = unchanged S04/S08 (7,833). Configs `config/*_w401*_sdval.yaml` differ from the reported ones only in "
         "data paths. Jobs: `docs/w401_sdval/jobids.txt` (`scripts/hpc/slurm_w401_sdval_{build,train}.sbatch`). "
         "TEST = best-val checkpoint scored once on S04/S08. ± = population SD over seeds.", "",
         "## Recognition: old (seen-signer val) vs new (S15 val)", "",
         "| Model | old val per seed | new val (S15) per seed | old TEST per seed | **new TEST per seed** | old TEST | **new TEST** | new macro-F1 | paper |",
         "|---|---|---|---|---|--:|--:|--:|--:|"]
    newval = {}
    for disp, new, old in MODELS:
        ov = [vb.get((old, s)) for s in range(3)]
        nv = [vb.get((new, s)) for s in range(3)]
        ot = [test_metrics(old, s, lab) for s in range(3)]
        nt = [test_metrics(new, s, lab) for s in range(3)]
        newval[disp] = nv
        j = lambda xs: " / ".join("—" if x is None else f"{x:.2f}" for x in xs)
        L.append(f"| {disp} | {j(ov)} | {j(nv)} | {j([t[0] if t else None for t in ot])} | "
                 f"{j([t[0] if t else None for t in nt])} | {fmt([t[0] if t else None for t in ot])} | "
                 f"**{fmt([t[0] if t else None for t in nt])}** | {fmt([t[1] if t else None for t in nt])} | {PAPER[old]} |")
    a = [x for x in newval["BlockGCN (lr 0.1)"] if x is not None]
    b = [x for x in newval["BlockGCN (lr 0.01)"] if x is not None]
    L += ["", "## BlockGCN LR decision on the signer-disjoint val (supersedes the seen-signer check, job)", ""]
    if len(a) == 3 and len(b) == 3:
        win = "0.01" if np.mean(b) > np.mean(a) else "0.1"
        L.append(f"S15-val mean: lr 0.1 = {np.mean(a):.2f}, lr 0.01 = {np.mean(b):.2f} → **select lr {win}**.")
    else:
        L.append("pending (needs all 3 seeds of both LRs)")
    L += ["", "## Signer decodability on the new checkpoints (probe on each model's own 15-signer training clips)", "",
          "cc = class-conditioned (per-class mean removed), random 8000-clip sample, chance 1/15; wc = within-class on all "
          "training clips. Not directly comparable to the old 16-signer values (different signer set / chance).", "",
          "| Model | cc per seed | cc | wc per seed | wc | old (16 signers, seen-signer-val ckpt) |", "|---|---|--:|---|--:|---|"]
    for disp, new, old in MODELS:
        cc = pr.get((new, "cc8k"), {}); wc = pr.get((new, "wcall"), {})
        j = lambda d: " / ".join(f"{d[s]:.3f}" if s in d else "—" for s in range(3))
        f3 = lambda d: (f"{np.mean(list(d.values())):.3f} ± {np.std(list(d.values())):.3f}" if d else "pending")
        L.append(f"| {disp} | {j(cc)} | {f3(cc)} | {j(wc)} | {f3(wc)} | {OLD_LNORM.get(old, '—')} |")
    L += ["", "## Other W401 numbers in P2 that used the seen-signer val (not re-run here)", "",
          "- SMG(k) BdSLW401 curve in Fig. 4b and tab:smgk (`work_dir/bdsl_ctr_gcn_w401_divk*`, `data/bdslw401_si_div`): "
          "selects on the official (seen-signer) val.",
          "- The W401 decodability numbers in P2 §leak and App. xds (0.69 / 0.43) come from the seen-signer-val checkpoints.",
          "- The W401 mixed split (jobs) is a separate signer-mixed protocol and stays as is.",
          "- VideoMAE 81.04 (W401 S04/S08, cited from the dataset paper) is not re-run; under S15-val selection CTR-GCN/ST-GCN still beat it, "
          "but val-selected BlockGCN (lr 0.01, 80.86) no longer does."]
    os.makedirs(os.path.join(ROOT, "docs/w401_sdval"), exist_ok=True)
    open(os.path.join(ROOT, "docs/w401_sdval/README.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
