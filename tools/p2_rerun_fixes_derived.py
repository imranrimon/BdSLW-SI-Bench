#!/usr/bin/env python
"""Recompute the P2 LOSO-derived files after the truncated-run fixes (docs/p2_rerun_fixes_2026-10/).

  1. docs/loso_heldout/derived/heldout_as_results_blockgcn_lr01.csv  (variance_decompose input:
     the 8 P2 LOSO CSVs, BlockGCN = block_gcn_lr01, GNN+Transformer = gnn_transformer_clip5)
  2. docs/loso_heldout/derived/variance_decomp_blockgcn_lr01.txt     (all 8 / 4-arch / 2-arch blocks)
  3. docs/loso_heldout/derived/arch_significance_blockgcn_lr01.txt   (default output, parsed by
     tools/p2_make_tables.py)
  4. docs/loso_heldout/derived/arch_significance_blockgcn_lr01_twosided.txt  (+ two-sided p, Holm)
  5. docs/loso_heldout/derived/arch_significance_bestck_twosided.txt  (best-on-val Protocol-A dumps
     for SLGTFormer/GNN+LSTM/Pose-LSTM/ST-GCN, if all bestck dumps exist)

Every file that already exists is first copied once to <name>.pre_fix_2026-10-22.<ext>.
Run on a compute node (reads work_dir score dumps).  python tools/p2_rerun_fixes_derived.py
"""
import csv
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DER = os.path.join(ROOT, "docs/loso_heldout/derived")
TAG = ".pre_fix_2026-10-22"
FILES = {"adaptive_gnn": "adaptive_gnn", "block_gcn": "block_gcn_lr01", "st_gcn_vanilla": "st_gcn_vanilla",
         "slgtformer": "slgtformer", "gnn": "gnn", "gnn_lstm": "gnn_lstm", "pose_lstm": "pose_lstm",
         "gnn_transformer": "gnn_transformer_clip5"}
BLOCKS = [("all 8", []),
          ("--archs bdsl_block_gcn_si bdsl_adaptive_gnn_si bdsl_st_gcn_vanilla_si bdsl_slgtformer_si",
           ["--archs", "bdsl_block_gcn_si", "bdsl_adaptive_gnn_si", "bdsl_st_gcn_vanilla_si", "bdsl_slgtformer_si"]),
          ("--archs bdsl_block_gcn_si bdsl_adaptive_gnn_si", ["--archs", "bdsl_block_gcn_si", "bdsl_adaptive_gnn_si"])]


def backup(path):
    root, ext = os.path.splitext(path)
    if os.path.isfile(path) and not os.path.isfile(root + TAG + ext):
        shutil.copy2(path, root + TAG + ext)


def write(path, text):
    backup(path)
    open(path, "w").write(text)
    print("wrote", os.path.relpath(path, ROOT))


def run(args):
    r = subprocess.run([sys.executable] + args, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"{args} failed:\n{r.stderr[-1500:]}")
    return r.stdout


def main():
    rows = ["Timestamp,Experiment,Epoch,Top1_Acc,Top5_Acc,Top5_Policy,WorkDir"]
    for arch, f in FILES.items():
        for r in csv.DictReader(open(os.path.join(ROOT, "docs/loso_heldout", f + ".csv"))):
            if r["test_acc"]:
                rows.append(f"heldout,bdsl_{arch}_si_loso_test{r['fold'].zfill(2)}_seed{r['seed']},0,"
                            f"{float(r['test_acc']) / 100.0},,,")
    src = os.path.join(DER, "heldout_as_results_blockgcn_lr01.csv")
    write(src, "\n".join(rows) + "\n")
    txt = ""
    for tag, args in BLOCKS:
        txt += f"######## archs: {tag}\n" + run(["tools/variance_decompose.py", "--csv", src] + args) + "\n"
    write(os.path.join(DER, "variance_decomp_blockgcn_lr01.txt"), txt)
    write(os.path.join(DER, "arch_significance_blockgcn_lr01.txt"), run(["tools/arch_significance.py"]))
    write(os.path.join(DER, "arch_significance_blockgcn_lr01_twosided.txt"),
          run(["tools/arch_significance.py", "--two-sided"]))
    need = [f"work_dir/bdsl_{a}_si_bestck_seed{s}_test/eval_results/best_acc.pkl"
            for a in ("slgtformer", "gnn_lstm", "pose_lstm", "st_gcn_vanilla") for s in (0, 1, 2)]
    missing = [p for p in need if not os.path.isfile(os.path.join(ROOT, p))]
    if missing:
        print("bestck variant skipped; missing:", *missing, sep="\n  ")
    else:
        write(os.path.join(DER, "arch_significance_bestck_twosided.txt"),
              run(["tools/arch_significance.py", "--two-sided", "--bestck"]))


if __name__ == "__main__":
    main()
