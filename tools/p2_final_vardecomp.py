#!/usr/bin/env python3
"""Final P2 LOSO variance decomposition (tab:losovar) on the completeness-gated held-out CSVs.

Builds docs/loso_heldout/derived/heldout_as_results_final_2026-10-22.csv (results_final.csv format, one
row per arch x held-out signer x seed) from the final per-arch held-out LOSO files and runs
tools/variance_decompose.py on it for the 8-, 4- and 2-arch designs. The arch -> file choice is the
paper's final one (tools/arch_significance.py --lr0p1 conventions):
  Adaptive GNN, Attention GNN : *_lr0p1.csv     (val-selected base LR 0.1)
  BlockGCN                    : block_gcn_lr01.csv
  GNN+Transformer             : gnn_transformer_clip5.csv
  others                      : <arch>.csv
Refuses to run unless every file has 33 cells (11 folds x 3 seeds).

  python tools/p2_final_vardecomp.py
"""
import csv
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = {"adaptive_gnn": "adaptive_gnn_lr0p1", "gnn": "gnn_lr0p1", "block_gcn": "block_gcn_lr01",
         "gnn_transformer": "gnn_transformer_clip5", "st_gcn_vanilla": "st_gcn_vanilla",
         "slgtformer": "slgtformer", "gnn_lstm": "gnn_lstm", "pose_lstm": "pose_lstm"}
DER = os.path.join(ROOT, "docs/loso_heldout/derived")
OUT_CSV = os.path.join(DER, "heldout_as_results_final_2026-10-22.csv")
OUT_TXT = os.path.join(DER, "variance_decomp_final_2026-10-22.txt")


def main():
    rows = ["Timestamp,Experiment,Epoch,Top1_Acc,Top5_Acc,Top5_Policy,WorkDir\n"]
    for arch, name in FILES.items():
        cells = list(csv.DictReader(open(os.path.join(ROOT, "docs/loso_heldout", name + ".csv"))))
        cells = [c for c in cells if c["test_acc"]]
        if len(cells) != 33:
            sys.exit(f"{name}.csv has {len(cells)} cells, expected 33")
        for c in cells:
            rows.append(f"heldout,bdsl_{arch}_si_loso_test{c['fold']}_seed{c['seed']},0,"
                        f"{float(c['test_acc']) / 100.0},,,\n")
    open(OUT_CSV, "w").write("".join(rows))
    txt = ""
    for tag, args in [("all 8", []),
                      ("--archs bdsl_block_gcn_si bdsl_adaptive_gnn_si bdsl_st_gcn_vanilla_si bdsl_slgtformer_si",
                       ["--archs", "bdsl_block_gcn_si", "bdsl_adaptive_gnn_si", "bdsl_st_gcn_vanilla_si",
                        "bdsl_slgtformer_si"]),
                      ("--archs bdsl_block_gcn_si bdsl_adaptive_gnn_si",
                       ["--archs", "bdsl_block_gcn_si", "bdsl_adaptive_gnn_si"])]:
        r = subprocess.run([sys.executable, "tools/variance_decompose.py", "--csv", OUT_CSV] + args,
                           capture_output=True, text=True, cwd=ROOT, check=True)
        txt += f"######## archs: {tag}\n" + r.stdout + "\n"
    open(OUT_TXT, "w").write(
        "# inputs: " + ", ".join(f"{a}={n}.csv" for a, n in FILES.items()) + " (docs/loso_heldout/)\n" + txt)
    print(txt)


if __name__ == "__main__":
    main()
