#!/usr/bin/env python3
"""Generate the P2 results-table bodies from the result files.

Every number in the generated tables traces to a result file through the
declarative SPEC below (row -> source file / experiment stem -> metric).
Values with no machine-readable source (model cost, literature rows) live in
docs/p2_tables/manual_values.yaml, each with a comment citing its origin.

Output: one LaTeX fragment (table rows only) per table block, written to
  results/tables/generated/<name>.tex        (override with --out-dir)
The paper's table environment keeps captions, headers and \\multicolumn group
labels and pulls the rows in with \\input{generated/<name>}. The fragments as
published are kept in results/tables/reference/ for --check.

Sources and the resolved-value cache
  Raw sources are the run log (results/results_final.csv, override with
  --results-csv), the result files under docs/, per-clip score dumps under
  work_dir/ and the label files under data/. Score dumps and data are not
  distributed, so every source value the generator resolved for the published
  tables is recorded in results/tables/resolved_sources.json. By default a
  source that cannot be read raw falls back to that cache (reported per run);
  --no-cache requires every source raw, --cache-only ignores raw sources.

Usage
  python tools/p2_make_tables.py                 # regenerate all fragments
  python tools/p2_make_tables.py --check         # diff reference vs generated, cell by cell
  python tools/p2_make_tables.py --check --no-cache    # same, from raw sources only (needs work_dir/, data/)
  python tools/p2_make_tables.py --check --paper-tex path/to/main.tex   # diff a LaTeX source instead
  python tools/p2_make_tables.py --check --rerun-sig   # also re-run tools/arch_significance.py (needs data)
  python tools/p2_make_tables.py --check --live        # ignore pins (preview a data refresh)
  python tools/p2_make_tables.py --check --use-recipe adaptive_gnn=lr0p1   # preview a recipe switch
  python tools/p2_make_tables.py --dry-run       # print fragments, write nothing
  python tools/p2_make_tables.py --provenance    # print the source of every generated cell

--check exit status: 0 when the reference matches the generator (pinned
"data newer than paper" cells are reported but are not failures), 1 otherwise.

Conventions (match the paper)
  * Per-seed accuracies are taken at results_final.csv precision (2 dp in %)
    before the seed mean / population SD; Macro-F1 and the metric suite use the
    exact single-model mean of tools/p2_metric_suite.metrics_for.
  * SMG = mean(Mixed) - mean(TEST), from the unrounded means.
  * LOSO = held-out-signer accuracy from docs/loso_heldout/<file>.csv (fold value
    = mean over seeds; mean and *sample* SD across the 11 folds), NOT the
    val-signer LOSO rows of results_final.csv.
  * Mixed = best-on-sd-val checkpoint scored on the sd TEST split (symmetric rule).
"""
import argparse
import collections
import csv
import json
import os
import pickle
import re
import subprocess
import sys
from decimal import Decimal, ROUND_HALF_UP

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import arch_significance  # noqa: E402,F401  (aliases numpy._core for numpy>=2 score pkls)
import p2_metric_suite as suite  # noqa: E402

OUT_DIR = "results/tables/generated"
REF_DIR = "results/tables/reference"
CACHE_FILE = "results/tables/resolved_sources.json"
MANUAL_FILE = "docs/p2_tables/manual_values.yaml"
RESULTS_CSV = "results/results_final.csv"
CACHE_MODE = "fallback"   # "fallback" (raw, else cache) | "off" (raw only) | "only" (cache only)
SEEDS = (0, 1, 2)
BDSL_TEST_LABELS = "data/bdsl_si/test_label.pkl"
BDSL_SD_TEST_LABELS = "data/bdsl_sd/test_label.pkl"
W401_TEST_LABELS = "data/bdslw401_si_sdval/test_label.pkl"
# Final P2 inputs (2026-10-22 consolidated rewrite): Adaptive/Attention GNN at lr 0.1, best-on-val TEST dumps
# (bestck) for SLGTFormer/GNN+LSTM/Pose-LSTM/ST-GCN, completeness-gated held-out LOSO CSVs, two-sided p + Holm.
#   python tools/arch_significance.py --lr0p1 --bestck --two-sided > SIG_FILE
#   tools/variance_decompose.py on docs/loso_heldout/derived/heldout_as_results_final_2026-10-22.csv > VARDECOMP_FILE
SIG_FILE = "docs/loso_heldout/derived/arch_significance_final_2026-10-22.txt"
VARDECOMP_FILE = "docs/loso_heldout/derived/variance_decomp_final_2026-10-22.txt"
SIG_ARGS = ["--lr0p1", "--bestck", "--two-sided"]   # the variant that produced SIG_FILE
# Masked bone streams (a bone is zero when either endpoint is undetected): per-clip TEST score dumps of the
# best-on-val checkpoints, docs/p2_missing_joint_fixes_2026-10/ (README section A).
_BONEMASK = "docs/p2_missing_joint_fixes_2026-10/raw/scores/{key}_new_s{{s}}_test.pkl"


# =============================================================================
# Source constructors (declarative). Each returns a plain dict; resolve() reads it.
# =============================================================================
def CSV(exp, seeds=SEEDS):
    """results_final.csv, Top1_Acc of the LATEST row of each experiment (one per seed).
    `exp` contains `{s}` for the seed."""
    return {"kind": "csv", "exp": exp, "seeds": list(seeds)}


def SUITE(stem):
    """BdSLW60 held-out TEST score dumps work_dir/<stem>_seed<S>_test/eval_results/best_acc.pkl
    via tools/p2_metric_suite.metrics_for (single-model 3-seed mean; ens = softmax ensemble)."""
    return {"kind": "suite", "stem": stem}


def PKL(files, labels, seeds=SEEDS):
    """Per-seed score pkls (`files` contains `{s}`) scored against a label pkl with
    tools/p2_metric_suite._metrics_one."""
    return {"kind": "pkl", "files": files, "labels": labels, "seeds": list(seeds)}


def JSONL(path, key, seeds=SEEDS):
    """One JSON object per line with a 'seed' field; value = obj[key]."""
    return {"kind": "jsonl", "path": path, "key": key, "seeds": list(seeds)}


def JSON(path, keys, field=None):
    """Walk `keys` into a JSON file. A list of numbers -> per-seed values; a number ->
    one deterministic value; a list/dict of objects -> [obj[field] ...] per seed."""
    return {"kind": "json", "path": path, "keys": list(keys), "field": field}


def TXT(files, seeds=SEEDS):
    """Per-seed text files containing 'top1=<value>' (e.g. work_dir/<exp>/mixed_test_top1.txt)."""
    return {"kind": "txt", "files": files, "seeds": list(seeds)}


def LOSO(name):
    """docs/loso_heldout/<name>.csv (held-out-signer Top-1, job convention)."""
    return {"kind": "loso", "file": name}


# =============================================================================
# SPEC — the single place that says where every number comes from.
# =============================================================================
# Pose architectures, shared by tab:arch (a), tab:smg8, the efficiency table,
# tab:suite and tab:bone.
#   src       : active sources (test / suite / mixed / loso / bone)
#   recipes   : alternative sources; select one with "recipe": "<name>" (or
#               --use-recipe arch=name) — only the keys given are overridden
#   pins      : {metric: (rendered value, reason)} keeps the paper's current
#               value while the data has moved on; --check lists every pin
#               whose live value differs as "data newer than paper".
#   comments  : {table: trailing LaTeX comment for that row}
NEWER, ROUNDING = "newer", "rounding"


def _bestck(arch):
    """Protocol-A TEST re-done with the best-on-val checkpoint (docs/p2_rerun_fixes_2026-10/): the
    original TEST dumps of some seeds scored a retained per-epoch checkpoint, not the best-val epoch."""
    return dict(test=CSV(f"bdsl_{arch}_si_bestck_seed{{s}}_test"), suite=SUITE(f"bdsl_{arch}_si_bestck"))   # pin kinds: data moved on / paper rounding slip

POSE = collections.OrderedDict([
    ("adaptive_gnn", dict(
        names=dict(arch="Adaptive GNN (CTR-GCN-style)", smg8="Adaptive GNN", eff="Adaptive GNN",
                   suite="Adaptive GNN", bone="Adaptive GNN"),
        src=dict(test=CSV("bdsl_adaptive_gnn_si_seed{s}_test"),
                 suite=SUITE("bdsl_adaptive_gnn_si"),
                 mixed=CSV("bdsl_adaptive_gnn_sd_noclip_test_seed{s}"),
                 loso=LOSO("adaptive_gnn"),
                 bone=CSV("bdsl_adaptive_gnn_bone_clip5_seed{s}_test")),
        recipes={"lr0p1": dict(  # docs/adaptive_attn_lr0p1_switch/ (runs in progress)
            test=CSV("bdsl_adaptive_gnn_si_lr0p1_seed{s}_test"),
            suite=SUITE("bdsl_adaptive_gnn_si_lr0p1"),
            mixed=CSV("bdsl_adaptive_gnn_sd_noclip_lr0p1_test_seed{s}"),
            loso=LOSO("adaptive_gnn_lr0p1"),
            # masked bone, lr 0.1 (unmasked lr0.1 bone 24.00 and lr0.01/clip5 bone 31.43 were missing-joint artifacts)
            bone=PKL(_BONEMASK.format(key="agb1"), BDSL_TEST_LABELS))},
        recipe="lr0p1",   # user-approved switch (val-selected base LR 0.1; docs/adaptive_attn_lr0p1_switch/)
        # the 6 "data newer than paper" pins of the lr-0.01 suite (2-seed) were removed with the lr0p1 switch
        pins={})),
    ("gnn_transformer", dict(
        names=dict(arch="GNN + Transformer$^\\ddagger$", smg8="GNN+Transformer$^\\ddagger$",
                   eff="GNN + Transformer$^\\ddagger$", suite="GNN+Transformer$^\\ddagger$"),
        # grad-norm clip 5.0 on both splits (docs/improvement_runs_2026-10/item1_gnntf_mixed_clip5.md)
        src=dict(test=PKL("work_dir/bdsl_gnn_transformer_si_clip5_seed{s}_test/eval_results/best_acc.pkl",
                          BDSL_TEST_LABELS),
                 suite=SUITE("bdsl_gnn_transformer_si_clip5"),
                 mixed=PKL("docs/audit_reruns_2026-10/raw/r5_gnn_transformer_sdtest_seed{s}.pkl",
                           BDSL_SD_TEST_LABELS),
                 loso=LOSO("gnn_transformer_clip5")),
        pins={})),
    ("st_gcn_vanilla", dict(
        names=dict(arch="ST-GCN~\\cite{stgcn}", smg8="ST-GCN", eff="ST-GCN", suite="ST-GCN"),
        src=dict(test=CSV("bdsl_st_gcn_vanilla_si_seed{s}_test"),
                 suite=SUITE("bdsl_st_gcn_vanilla_si"),
                 mixed=CSV("bdsl_st_gcn_vanilla_sd_noclip_test_seed{s}"),
                 loso=LOSO("st_gcn_vanilla")),
        recipes={"bestck": _bestck("st_gcn_vanilla")},
        recipe="bestck",   # true best-on-val checkpoint (user-approved)
        pins={})),
    ("gnn", dict(
        names=dict(arch="Attention GNN", smg8="Attention GNN", eff="Attention GNN", suite="Attention GNN"),
        src=dict(test=CSV("bdsl_gnn_si_seed{s}_test"),
                 suite=SUITE("bdsl_gnn_si"),
                 mixed=CSV("bdsl_gnn_sd_noclip_test_seed{s}"),
                 loso=LOSO("gnn")),
        recipes={"lr0p1": dict(
            test=CSV("bdsl_gnn_si_lr0p1_seed{s}_test"),
            suite=SUITE("bdsl_gnn_si_lr0p1"),
            mixed=CSV("bdsl_gnn_sd_noclip_lr0p1_test_seed{s}"),
            loso=LOSO("gnn_lr0p1"))},
        recipe="lr0p1")),   # user-approved switch (val-selected base LR 0.1)
    ("slgtformer", dict(
        names=dict(arch="SLGTFormer~\\cite{slgtformer}", smg8="SLGTFormer", eff="SLGTFormer", suite="SLGTFormer"),
        src=dict(test=CSV("bdsl_slgtformer_si_seed{s}_test"),
                 suite=SUITE("bdsl_slgtformer_si"),
                 # still the grad-clip-5.0 sd run (R5 re-score, job)
                 mixed=PKL("docs/audit_reruns_2026-10/raw/r5_slgtformer_sdtest_seed{s}.pkl", BDSL_SD_TEST_LABELS),
                 loso=LOSO("slgtformer")),
        recipes={"mixed_noclip": dict(  # job sd_noclip_slgt (training)
            mixed=CSV("bdsl_slgtformer_sd_noclip_test_seed{s}")),
                 "bestck": _bestck("slgtformer"),
                 # final: true best-on-val TEST checkpoint + unclipped signer-mixed run (all 3 seeds complete)
                 "bestck_mixed_noclip": dict(_bestck("slgtformer"),
                                             mixed=CSV("bdsl_slgtformer_sd_noclip_test_seed{s}"))},
        recipe="bestck_mixed_noclip")),
    ("gnn_lstm", dict(
        names=dict(arch="GNN + Bi-LSTM", smg8="GNN+Bi-LSTM", eff="GNN + Bi-LSTM", suite="GNN+Bi-LSTM"),
        src=dict(test=CSV("bdsl_gnn_lstm_si_seed{s}_test"),
                 suite=SUITE("bdsl_gnn_lstm_si"),
                 mixed=CSV("bdsl_gnn_lstm_sd_noclip_test_seed{s}"),
                 loso=LOSO("gnn_lstm")),
        recipes={"bestck": _bestck("gnn_lstm")},
        recipe="bestck")),
    ("block_gcn", dict(
        names=dict(arch="BlockGCN~\\cite{blockgcn}", smg8="BlockGCN", eff="BlockGCN", suite="BlockGCN",
                   bone="BlockGCN"),
        # base LR 0.01 (docs/blockgcn_lr01_switch/)
        src=dict(test=JSONL("docs/blockgcn_lr01_switch/raw/protoA.jsonl", "test_top1"),
                 suite=SUITE("bdsl_block_gcn_si_lr01"),
                 mixed=JSONL("docs/blockgcn_lr01_switch/raw/sd.jsonl", "sd_test"),
                 loso=LOSO("block_gcn_lr01"),
                 # masked bone (docs/p2_missing_joint_fixes_2026-10/; unmasked was 65.54 +- 1.63)
                 bone=PKL(_BONEMASK.format(key="bgb"), BDSL_TEST_LABELS)))),
    ("pose_lstm", dict(
        names=dict(arch="Pose-LSTM", smg8="Pose-LSTM", eff="Pose-LSTM", suite="Pose-LSTM"),
        src=dict(test=CSV("bdsl_pose_lstm_si_seed{s}_test"),
                 suite=SUITE("bdsl_pose_lstm_si"),
                 mixed=CSV("bdsl_pose_lstm_sd_noclip_test_seed{s}"),
                 loso=LOSO("pose_lstm")),
        recipes={"bestck": _bestck("pose_lstm")},
        recipe="bestck")),
])
POSE_INPUT = "27-kp pose (none)"

_SR = "docs/baselines_rerun/"
# Non-pose / prior-method rows of tab:arch panel (a). test/f1/mixed sources; SMG = Mixed - TEST.
ARCH_A_OTHER = collections.OrderedDict([
    ("rgb", [
        dict(key="videomae", name="VideoMAE ViT-B/16~\\cite{videomae}", input="RGB video (Kinetics)",
             test=JSON(_SR + "item4to6_rgb_foundation.json", ["VideoMAE", "seeds"], "top1"), test_suffix="$^\\S$",
             f1=JSON(_SR + "item4to6_rgb_foundation.json", ["VideoMAE", "seeds"], "f1"),
             mixed=JSON(_SR + "item4to6_rgb_foundation.json", ["VideoMAE", "seeds"], "mixed")),
        dict(key="s3d",  # cluster seeds 0r/1r/2r (Windows seeds 0/1 unverifiable), tools/p2_s3d_cluster_seeds.py
             name="S3D~\\cite{s3d}", input="RGB video (Kinetics)",
             test=JSON("docs/p2_rerun_fixes_2026-10/s3d_cluster_seeds.json", ["S3D", "seeds"], "top1"),
             f1=JSON("docs/p2_rerun_fixes_2026-10/s3d_cluster_seeds.json", ["S3D", "seeds"], "f1"),
             mixed=JSON("docs/p2_rerun_fixes_2026-10/s3d_cluster_seeds.json", ["S3D", "seeds"], "mixed")),
    ]),
    ("foundation", [
        dict(key="unisign", name="Uni-Sign, fine-tuned~\\cite{unisign}", input="RTMPose-133 (CSL-News)",
             test=JSON(_SR + "item4to6_rgb_foundation.json", ["Uni-Sign", "seeds"], "top1"),
             f1=JSON(_SR + "item4to6_rgb_foundation.json", ["Uni-Sign", "seeds"], "f1"),
             mixed=JSON(_SR + "item4to6_rgb_foundation.json", ["Uni-Sign", "seeds"], "mixed")),
        dict(key="shubert", name="SHuBERT, frozen probe~\\cite{shubert}", input="RGB multi-stream features",
             test=JSON(_SR + "item4to6_rgb_foundation.json", ["SHuBERT", "seeds"], "top1"),
             f1=JSON(_SR + "item4to6_rgb_foundation.json", ["SHuBERT", "seeds"], "f1"),
             mixed=JSON(_SR + "item4to6_rgb_foundation.json", ["SHuBERT", "seeds"], "mixed")),
    ]),
    ("prior_rerun", [
        dict(key="attn_bilstm_author", name="Attention Bi-LSTM $+$ RQ~\\cite{bdslw60}$^d$", input=POSE_INPUT,
             test=JSON(_SR + "item1_bdslw60_baselines.json", ["bdsl_attn_bilstm_author"], "test_top1"),
             f1=JSON(_SR + "item1_bdslw60_baselines.json", ["bdsl_attn_bilstm_author"], "macro_f1"),
             mixed=JSON(_SR + "item1_bdslw60_baselines.json", ["bdsl_attn_bilstm_author"], "mixed"),
             pins={}),
        dict(key="svm_original", name="Linear SVM $+$ RQ~\\cite{bdslw60}$^e$", input=POSE_INPUT,
             test=JSON(_SR + "item1_bdslw60_baselines.json", ["svm_original", "test_top1"]),
             f1=JSON(_SR + "item1_bdslw60_baselines.json", ["svm_original", "test_macro_f1"]),
             mixed=JSON(_SR + "item1_bdslw60_baselines.json", ["svm_original", "mixed_sd_test"]),
             pins={}),
        dict(key="svm_flipped", name="Linear SVM $+$ RQ, hand-flipped~\\cite{bdslw60}$^{e,f}$", input=POSE_INPUT,
             test=JSON(_SR + "item1_bdslw60_baselines.json", ["svm_flipped", "test_top1"]),
             f1=JSON(_SR + "item1_bdslw60_baselines.json", ["svm_flipped", "test_macro_f1"]),
             mixed=JSON(_SR + "item1_bdslw60_baselines.json", ["svm_flipped", "mixed_sd_test"])),
        dict(key="spoter_orig", name="SPOTER~\\cite{spoter}", input=POSE_INPUT,
             cost=dict(params=JSON(_SR + "item2_spoter.json", ["bdsl_spoter_orig", "params_M"]),
                       gflops=JSON(_SR + "item2_spoter.json", ["bdsl_spoter_orig", "gflops_thop"])),
             test=JSON(_SR + "item2_spoter.json", ["bdsl_spoter_orig", "test_top1"]),
             f1=JSON(_SR + "item2_spoter.json", ["bdsl_spoter_orig", "macro_f1"]),
             mixed=JSON(_SR + "item2_spoter.json", ["bdsl_spoter_orig", "mixed_sdtest"])),
        dict(key="spoter_bdsl", name="BdSL-SPOTER~\\cite{bdslspoter}", input=POSE_INPUT,
             cost=dict(params=JSON(_SR + "item2_spoter.json", ["bdsl_spoter", "params_M"]),
                       gflops=JSON(_SR + "item2_spoter.json", ["bdsl_spoter", "gflops_thop"])),
             test=JSON(_SR + "item2_spoter.json", ["bdsl_spoter", "test_top1"]),
             f1=JSON(_SR + "item2_spoter.json", ["bdsl_spoter", "macro_f1"]),
             mixed=JSON(_SR + "item2_spoter.json", ["bdsl_spoter", "mixed_sdtest"])),
        dict(key="spoter_native", name="BdSL-SPOTER, released recipe~\\cite{bdslspoter}", input=POSE_INPUT,
             cost=dict(params=JSON(_SR + "item2_spoter.json", ["bdsl_spoter_native", "params_M"]),
                       gflops=JSON(_SR + "item2_spoter.json", ["bdsl_spoter_native", "gflops_thop"])),
             test=JSON(_SR + "item2_spoter.json", ["bdsl_spoter_native", "test_top1"]),
             f1=JSON(_SR + "item2_spoter.json", ["bdsl_spoter_native", "macro_f1"]),
             mixed=JSON(_SR + "item2_spoter.json", ["bdsl_spoter_native", "mixed_sdtest"])),
    ]),
])

# tab:arch panel (b): BdSLW401, S15-selected checkpoints scored on S04/S08 (docs/w401_sdval/README.md);
# Mixed = best-on-mixed-val checkpoint scored on the mixed TEST split.
_W401_PKL = "work_dir/{exp}_seed{{s}}/eval_results/best_acc.pkl"
ARCH_B = [
    dict(key="w401_st_gcn", name="ST-GCN~\\cite{stgcn}",
         test=CSV("bdsl_st_gcn_w401_sdval_test_seed{s}"),
         f1=PKL(_W401_PKL.format(exp="bdsl_st_gcn_w401_sdval_test"), W401_TEST_LABELS),
         mixed=TXT("work_dir/bdsl_st_gcn_w401_sd_seed{s}/mixed_test_top1.txt")),
    dict(key="w401_ctr_gcn", name="CTR-GCN~\\cite{ctrgcn}",
         test=CSV("bdsl_ctr_gcn_w401_sdval_test_seed{s}"),
         f1=PKL(_W401_PKL.format(exp="bdsl_ctr_gcn_w401_sdval_test"), W401_TEST_LABELS),
         mixed=TXT("work_dir/bdsl_ctr_gcn_w401_sd_seed{s}/mixed_test_top1.txt")),
    dict(key="w401_block_gcn", name="BlockGCN~\\cite{blockgcn}",   # base LR 0.01 (S15-selected)
         test=CSV("bdsl_block_gcn_w401_lr01_sdval_test_seed{s}"),
         f1=PKL(_W401_PKL.format(exp="bdsl_block_gcn_w401_lr01_sdval_test"), W401_TEST_LABELS),
         mixed=CSV("bdsl_block_gcn_w401_sd_lr01_test_seed{s}")),
]

# Fragment registry: name -> (paper anchor, line preceding the rows in the hand-typed tex).
# The anchor is a unique string inside the table (label or caption); --check parses
# the rows that follow `after` up to the next \multicolumn / \bottomrule.
FRAGMENTS = collections.OrderedDict([
    ("tab_arch_a_pose", ("\\label{tab:arch}", "\\emph{Pose models, re-run under the shared recipe}")),
    ("tab_arch_a_rgb", ("\\label{tab:arch}", "\\emph{RGB models, re-run on the same split}")),
    ("tab_arch_a_foundation", ("\\label{tab:arch}", "\\emph{Foundation models, same split, different input footing}")),
    ("tab_arch_a_prior_rerun", ("\\label{tab:arch}", "\\emph{Prior Bangla and sign-pose methods, re-run on the same split}")),
    ("tab_arch_a_prior_reported", ("\\label{tab:arch}", "\\emph{Prior Bangla results, as reported under their own protocols}")),
    ("tab_arch_b_w401", ("\\label{tab:arch}", "\\textbf{(b) BdSLW401, held-out TEST signers S04 and S08}")),
    ("tab_smg8_rows", ("\\label{tab:smg8}", "\\midrule")),
    ("tab_eff_rows", ("\\caption{Model efficiency vs held-out-TEST accuracy.", "\\midrule")),
    ("tab_suite_rows", ("\\label{tab:suite}", "\\midrule")),
    ("tab_bone_rows", ("\\label{tab:bone}", "\\midrule")),
    ("tab_losoboot_rows", ("\\caption{LOSO signer-cluster bootstrap (BlockGCN", "\\midrule")),
    ("tab_losovar_rows", ("\\label{tab:losovar}", "\\midrule")),
    ("tab_losofolds_rows", ("\\label{tab:losofolds}", "\\toprule")),
    ("tab_sig_a_rows", ("\\label{tab:sig}", "\\emph{(A) per-instance clip bootstrap}")),
    ("tab_sig_b_rows", ("\\label{tab:sig}", "\\emph{(B) signer-clustered LOSO bootstrap}")),
])

SIG_NAMES = {"Adaptive GNN": "Adaptive GNN", "SLGTFormer": "SLGTFormer", "Attention GNN": "Attention GNN",
             "GNN+Transformer": "GNN+Transformer$^\\ddagger$", "ST-GCN": "ST-GCN", "GNN+LSTM": "GNN+LSTM",
             "BlockGCN": "BlockGCN", "Pose LSTM": "Pose-LSTM"}


# =============================================================================
# Loaders
# =============================================================================
class SourceError(Exception):
    pass


class Val:
    """A resolved number: per-seed values (mean / population SD) or one scalar."""

    def __init__(self, values=None, scalar=None, src="", ddof=0):
        self.values = None if values is None else [float(v) for v in values]
        self.scalar = None if scalar is None else float(scalar)
        self.src = src
        self.ddof = ddof

    @property
    def mean(self):
        return self.scalar if self.values is None else float(np.mean(self.values))

    @property
    def sd(self):
        return None if self.values is None or len(self.values) < 2 else float(np.std(self.values, ddof=self.ddof))

    @property
    def multi(self):
        return self.values is not None and len(self.values) > 1


_CACHE = {}


def _csv_latest():
    if "csv" not in _CACHE:
        latest = {}
        with open(RESULTS_CSV, newline="") as f:
            for i, r in enumerate(csv.DictReader(f)):
                key = (r["Timestamp"], i)
                if r["Experiment"] not in latest or key >= latest[r["Experiment"]][0]:
                    latest[r["Experiment"]] = (key, float(r["Top1_Acc"]))
        _CACHE["csv"] = {k: v[1] for k, v in latest.items()}
    return _CACHE["csv"]


def _labels(path):
    if ("lab", path) not in _CACHE:
        names, labs = pickle.load(open(path, "rb"))
        _CACHE[("lab", path)] = dict(zip(names, [int(x) for x in labs]))
    return _CACHE[("lab", path)]


def _pkl_metrics(path, labels):
    """tools/p2_metric_suite._metrics_one on one score pkl (clips common to pkl and labels)."""
    if ("pkl", path, labels) not in _CACHE:
        lab = _labels(labels)
        d = {k: suite.to_prob(v) for k, v in pickle.load(open(path, "rb")).items()}
        common = sorted(set(d) & set(lab))
        if not common:
            raise SourceError(f"{path}: no clips in common with {labels}")
        m = suite._metrics_one(np.stack([d[k] for k in common]), np.array([lab[k] for k in common]))
        m.pop("correct", None)
        _CACHE[("pkl", path, labels)] = m
    return _CACHE[("pkl", path, labels)]


def _suite(stem):
    if ("suite", stem) not in _CACHE:
        m = suite.metrics_for(stem, _labels(BDSL_TEST_LABELS))
        if not m:
            raise SourceError(f"no TEST score dumps for stem {stem}")
        if m["seeds"] != len(SEEDS):
            raise SourceError(f"suite {stem}: {m['seeds']}/{len(SEEDS)} seeds have score dumps")
        _CACHE[("suite", stem)] = m
    return _CACHE[("suite", stem)]


def _loso_folds(name):
    """{heldout_signer: [per-seed held-out Top-1]} in file order. Same reading rule as
    tools/arch_significance.loso_perfold and tools/make_paper_figures.fig_loso."""
    path = f"docs/loso_heldout/{name}.csv"
    if not os.path.isfile(path):
        raise SourceError(f"missing {path}")
    per = collections.OrderedDict()
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["test_acc"]:
                per.setdefault(r["heldout_signer"], []).append(float(r["test_acc"]))
    return per


def r2(x):
    """Per-seed accuracy at results_final.csv precision (4-dp fraction == 2-dp percent)."""
    return round(float(x), 2)


def _seedvals(fmt_path, seeds, read):
    vals, missing = [], []
    for s in seeds:
        p = fmt_path.format(s=s)
        try:
            vals.append(read(p))
        except (FileNotFoundError, KeyError, SourceError):
            missing.append(s)
    if missing:
        raise SourceError(f"{fmt_path}: missing seed(s) {missing} ({len(vals)}/{len(seeds)} present)")
    return vals


_CACHE_STATS = collections.Counter()


def _cache_key(src, metric):
    return json.dumps({"src": src, "metric": metric}, sort_keys=True)


def _cached(src, metric):
    if "resolved" not in _CACHE:
        _CACHE["resolved"] = json.load(open(CACHE_FILE))["entries"] if os.path.isfile(CACHE_FILE) else {}
    e = _CACHE["resolved"].get(_cache_key(src, metric))
    if e is None:
        return None
    return Val(e["values"], e["scalar"], src="[cached] " + e["src"], ddof=e["ddof"])


def resolve(src, metric=None):
    """Resolve a source dict to a Val: raw source first, then (CACHE_MODE) the resolved-value cache."""
    err = None
    if CACHE_MODE != "only":
        try:
            v = _resolve_raw(src, metric)
            _CACHE_STATS["raw"] += 1
            return v
        except (SourceError, OSError, KeyError) as e:
            if CACHE_MODE == "off":
                raise SourceError(f"{e}") from e
            err = e
    v = _cached(src, metric)
    if v is None:
        raise SourceError(f"source unreadable and not in {CACHE_FILE}: {src} ({err})")
    _CACHE_STATS["cache"] += 1
    return v


def _resolve_raw(src, metric=None):
    """Resolve a source dict to a Val. `metric` picks a metric for suite/pkl sources
    ('top1', 'macrof1', 'mrr', 'r10', 'nll', 'brier', 'ens_top1', ...)."""
    k = src["kind"]
    if k == "csv":
        tab = _csv_latest()

        def read(exp):
            if exp not in tab:
                raise KeyError(exp)
            return r2(tab[exp] * 100)
        vals = _seedvals(src["exp"], src["seeds"], read)
        return Val(vals, src=f"{RESULTS_CSV}: {src['exp'].replace('{s}', '{' + ','.join(map(str, src['seeds'])) + '}')} "
                             f"(latest row each)")
    if k == "suite":
        m = _suite(src["stem"])
        metric = metric or "top1"
        desc = f"p2_metric_suite.metrics_for('{src['stem']}')"
        if metric.startswith("ens_"):
            return Val(scalar=m["ens"][metric[4:]], src=desc + f" ens.{metric[4:]}")
        return Val(scalar=m["single"][metric][0], src=desc + f" single.{metric}")
    if k == "pkl":
        metric = metric or "top1"
        vals = _seedvals(src["files"], src["seeds"], lambda p: _pkl_metrics(p, src["labels"])[metric])
        return Val(vals, src=f"{src['files']} vs {src['labels']} ({metric})")
    if k == "jsonl":
        rows = {}
        with open(src["path"]) as f:
            for line in f:
                if line.strip():
                    o = json.loads(line)
                    rows[int(o["seed"])] = o
        missing = [s for s in src["seeds"] if s not in rows]
        if missing:
            raise SourceError(f"{src['path']}: missing seed(s) {missing}")
        return Val([rows[s][src["key"]] for s in src["seeds"]], src=f"{src['path']} [{src['key']}]")
    if k == "json":
        node = json.load(open(src["path"]))
        for key in src["keys"]:
            node = node[key]
        desc = f"{src['path']} {'/'.join(src['keys'])}" + (f" [{src['field']}]" if src["field"] else "")
        if isinstance(node, dict):
            node = list(node.values())
        if isinstance(node, list):
            vals = [o[src["field"]] if src["field"] else o for o in node]
            return Val(vals, src=desc)
        return Val(scalar=float(node), src=desc)
    if k == "txt":
        def read(p):
            m = re.search(r"top1\s*=\s*([\d.]+)", open(p).read())
            if not m:
                raise SourceError(f"{p}: no top1=")
            return r2(float(m.group(1)))
        return Val(_seedvals(src["files"], src["seeds"], read), src=src["files"])
    if k == "loso":
        per = _loso_folds(src["file"])
        folds = [float(np.mean(v)) for v in per.values()]
        return Val(folds, src=f"docs/loso_heldout/{src['file']}.csv (fold = mean over seeds; sample SD over folds)",
                   ddof=1)
    raise ValueError(k)


def load_manual():
    import yaml
    return yaml.safe_load(open(MANUAL_FILE))


# =============================================================================
# Formatting
# =============================================================================
def f2(x, dp=2):
    return f"{x:.{dp}f}"


def pm(v, dp=2):
    return f"{v.mean:.{dp}f}$\\pm${v.sd:.{dp}f}" if v.multi else f"{v.mean:.{dp}f}"


def signed(x, dp=2):
    return f"${x:+.{dp}f}$"


def lead(s):
    """Leading decimal number of a rendered cell, as an exact Decimal."""
    return Decimal(re.search(r"-?\d+(?:\.\d+)?", s).group(0))


def smg(mixed_cell, test_cell):
    """SMG as printed: displayed Mixed mean minus displayed TEST mean (reader-verifiable)."""
    return signed(float(lead(mixed_cell) - lead(test_cell)))


def bold(s):
    return "\\textbf{" + s + "}"


def num(s):
    """Leading float of a rendered cell (for sorting / bold)."""
    m = re.search(r"-?\d+(?:\.\d+)?", s.replace("$+", "").replace("{", " "))
    return float(m.group(0)) if m else float("nan")


class Ctx:
    """Rendering context: pins on/off, recipe overrides, collected provenance and drift."""

    def __init__(self, live=False, recipe_override=None):
        self.live = live
        self.recipe_override = recipe_override or {}
        self.newer = []      # (fragment, row, column, paper value, live value, source, reason)
        self.prov = []       # (fragment, row, column, value, source)
        self.manual = load_manual()

    def pose_src(self, arch):
        spec = POSE[arch]
        src = dict(spec["src"])
        rec = self.recipe_override.get(arch, spec.get("recipe"))
        if rec:
            src.update(spec["recipes"][rec])
        return src

    def cell(self, frag, row, col, live_str, source, pins=None, metric=None):
        """Apply a pin (if any) and record provenance / drift."""
        out = live_str
        if pins and metric in pins and not self.live:
            pinned, reason, kind = pins[metric]
            if pinned != live_str:
                self.newer.append((frag, row, col, pinned, live_str, source, reason, kind))
            out = pinned
        self.prov.append((frag, row, col, out, source))
        return out


# =============================================================================
# Table builders. Each returns a list of rows; a row is (cells, trailing_comment)
# or a raw LaTeX line (str, e.g. "\\midrule").
# =============================================================================
SUITE_METRICS = [("macrof1", 2), ("ens_top1", 2), ("mrr", 3), ("r10", 2), ("nll", 3), ("brier", 3)]


def _pose_metrics(ctx, frag, arch, rowname, want):
    """Resolve and render the per-arch cells a table shows (`want`); shared by several tables."""
    spec = POSE[arch]
    src = ctx.pose_src(arch)
    # pins describe the BASE sources; they never apply once a row runs on another recipe
    pins = {} if ctx.recipe_override.get(arch, spec.get("recipe")) else spec.get("pins", {})
    test = resolve(src["test"])
    out = dict(test_val=test)
    out["test"] = ctx.cell(frag, rowname, "TEST", pm(test), test.src, pins, "test")
    if "mixed" in want or "smg" in want:
        mixed = resolve(src["mixed"])
        out["mixed"] = ctx.cell(frag, rowname, "Mixed", pm(mixed), mixed.src, pins, "mixed")
        out["smg"] = ctx.cell(frag, rowname, "SMG", smg(out["mixed"], out["test"]),
                              "displayed Mixed - displayed TEST", pins, "smg")
    for met, dp in SUITE_METRICS:
        if met in want:
            v = resolve(src["suite"], met)
            out[met] = ctx.cell(frag, rowname, met, f2(v.mean, dp), v.src, pins, met)
    if "loso" in want:
        lv = resolve(src["loso"])
        out["loso"] = ctx.cell(frag, rowname, "LOSO", pm(lv, 1), lv.src, pins, "loso")
    cost = ctx.manual["cost"][arch]
    out["params"], out["gflops"] = cost["params"], cost["gflops"]
    return out


def _bold_max(rows, col_idx):
    """Bold the max of each listed column (ties: all)."""
    for c in col_idx:
        best = max(num(r[0][c]) for r in rows)
        for r in rows:
            if num(r[0][c]) == best:
                r[0][c] = bold(r[0][c])


def build_arch_a_pose(ctx, frag):
    rows = []
    for arch, spec in POSE.items():
        m = _pose_metrics(ctx, frag, arch, spec["names"]["arch"], ["macrof1", "mixed", "loso"])
        cells = [spec["names"]["arch"], POSE_INPUT, m["params"], m["gflops"], m["test"], m["macrof1"],
                 m["mixed"], m["smg"], m["loso"]]
        rows.append((cells, spec.get("comments", {}).get("arch"), m["test_val"].mean))
    rows.sort(key=lambda r: -num(r[0][4]))
    _bold_max(rows, [4, 5, 8])
    return [(c, cm) for c, cm, _ in rows]


def _other_row(ctx, frag, spec):
    name = spec["name"]
    test, f1, mixed = resolve(spec["test"]), resolve(spec["f1"]), resolve(spec["mixed"])
    if "cost" in spec:
        p, g = resolve(spec["cost"]["params"]), resolve(spec["cost"]["gflops"])
        params, gflops = f"{p.mean:.1f}", f"{g.mean:.2f}"
        ctx.prov.append((frag, name, "Params", params, p.src))
        ctx.prov.append((frag, name, "GFLOPs", gflops, g.src))
    else:
        cost = ctx.manual["cost"][spec["key"]]
        params, gflops = cost["params"], cost["gflops"]
    pins = spec.get("pins", {})
    t = ctx.cell(frag, name, "TEST", pm(test), test.src, pins, "test")
    mx = ctx.cell(frag, name, "Mixed", pm(mixed), mixed.src, pins, "mixed")
    cells = [name, spec["input"], params, gflops, t + spec.get("test_suffix", ""),
             ctx.cell(frag, name, "Macro-F1", f2(f1.mean), f1.src, pins, "macrof1"), mx,
             ctx.cell(frag, name, "SMG", smg(mx, t), "displayed Mixed - displayed TEST", pins, "smg"),
             "--"]
    return (cells, None)


def build_arch_a_group(group):
    def b(ctx, frag):
        return [_other_row(ctx, frag, s) for s in ARCH_A_OTHER[group]]
    return b


def build_manual_rows(key):
    def b(ctx, frag):
        out = []
        for cells in ctx.manual[key]:
            ctx.prov.append((frag, cells[0], "*", "(row)", f"{MANUAL_FILE}: {key}"))
            out.append((list(cells), None))
        return out
    return b


def build_arch_b(ctx, frag):
    rows = []
    for spec in ARCH_B:
        name = spec["name"]
        test, f1, mixed = resolve(spec["test"]), resolve(spec["f1"], "macrof1"), resolve(spec["mixed"])
        cost = ctx.manual["cost"][spec["key"]]
        t = ctx.cell(frag, name, "TEST", pm(test), test.src)
        mx = ctx.cell(frag, name, "Mixed", pm(mixed), mixed.src)
        cells = [name, POSE_INPUT, cost["params"], cost["gflops"], t,
                 ctx.cell(frag, name, "Macro-F1", f2(f1.mean), f1.src), mx,
                 ctx.cell(frag, name, "SMG", smg(mx, t), "displayed Mixed - displayed TEST"),
                 "--"]
        rows.append((cells, None, test.mean))
    rows.sort(key=lambda r: -num(r[0][4]))
    _bold_max(rows, [4, 5])
    out = [(c, cm) for c, cm, _ in rows]
    for cells in ctx.manual["w401_reported"]:
        ctx.prov.append((frag, cells[0], "*", "(row)", f"{MANUAL_FILE}: w401_reported"))
        out.append((list(cells), None))
    return out


def build_smg8(ctx, frag):
    rows = []
    for arch, spec in POSE.items():
        m = _pose_metrics(ctx, frag, arch, spec["names"]["smg8"], ["mixed"])
        rows.append(([spec["names"]["smg8"], m["mixed"], m["test"], m["smg"]],
                     spec.get("comments", {}).get("smg8")))
    rows.sort(key=lambda r: -num(r[0][3]))
    return rows


def build_eff(ctx, frag):
    rows = []
    for arch, spec in POSE.items():
        m = _pose_metrics(ctx, frag, arch, spec["names"]["eff"], [])
        rows.append(([spec["names"]["eff"], m["params"], m["gflops"], f2(num(m["test"]))], None, 0))
    rows.sort(key=lambda r: -num(r[0][3]))
    _bold_max(rows, [1, 2, 3])
    return [(c, cm) for c, cm, _ in rows]


SUITE_ORDER = ["adaptive_gnn", "st_gcn_vanilla", "gnn", "slgtformer", "gnn_lstm", "block_gcn", "pose_lstm",
               "gnn_transformer"]


def build_suite(ctx, frag):
    rows = []
    for arch in SUITE_ORDER:
        spec = POSE[arch]
        m = _pose_metrics(ctx, frag, arch, spec["names"]["suite"], [k for k, _ in SUITE_METRICS])
        rows.append(([spec["names"]["suite"], m["test"], m["ens_top1"], m["macrof1"], m["mrr"], m["r10"],
                      m["nll"], m["brier"]], None))
    rows.sort(key=lambda r: -num(r[0][1]))   # sorted by TEST like tab:arch / eff
    return rows


def build_bone(ctx, frag):
    rows = []
    for arch in ["block_gcn", "adaptive_gnn"]:
        spec = POSE[arch]
        m = _pose_metrics(ctx, frag, arch, spec["names"]["bone"], [])
        bv = resolve(ctx.pose_src(arch)["bone"])
        rows.append(([spec["names"]["bone"], m["test"],
                      ctx.cell(frag, spec["names"]["bone"], "Bone", pm(bv), bv.src)], None))
    return rows


def _blockgcn_loso(ctx):
    return ctx.pose_src("block_gcn")["loso"]


def build_losoboot(ctx, frag):
    src = _blockgcn_loso(ctx)
    per = _loso_folds(src["file"])
    x = np.array([np.mean(v) for v in per.values()])
    rng = np.random.default_rng(1234)    # as tools/make_paper_figures.fig_loso / lr01_switch_report.loso
    bs = [rng.choice(x, len(x)).mean() for _ in range(10000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    s = f"docs/loso_heldout/{src['file']}.csv (signer-cluster bootstrap, 10k, seed 1234)"
    cells = [["mean", bold(f2(x.mean()))],
             ["bootstrap 95\\% CI", f"$[{lo:.2f},\\ {hi:.2f}]$"],
             ["range (min--max signer)", f"${x.min():.1f}$ -- ${x.max():.1f}$"],
             ["SD across signers", f2(x.std(ddof=1))],
             ["\\#signers", str(len(x))]]
    for c in cells:
        ctx.prov.append((frag, c[0], "value", c[1], s))
    return [(c, None) for c in cells]


def build_losofolds(ctx, frag):
    src = _blockgcn_loso(ctx)
    per = _loso_folds(src["file"])
    folds = sorted(((k, float(np.mean(v))) for k, v in per.items()), key=lambda kv: kv[1])
    ctx.prov.append((frag, "Top-1", "*", "(row)", f"docs/loso_heldout/{src['file']}.csv (fold mean over seeds)"))
    return [(["Signer"] + [k for k, _ in folds], None), "\\midrule",
            (["Top-1"] + [f"{v:.1f}" for _, v in folds], None)]


def build_losovar(ctx, frag):
    txt = open(VARDECOMP_FILE).read().split("######## archs:")[1]   # first block = all 8 archs
    vals = {}
    for key in ["sigma_signer", "sigma_model", "sigma_interact", "sigma_seed"]:
        m = re.search(key + r"\s*\([^)]*\)\s*=\s*([\d.]+)\s*pp", txt)
        if not m:
            raise SourceError(f"{VARDECOMP_FILE}: {key} not found")
        vals[key] = m.group(1)
    names = [("Held-out signer", "sigma_signer"), ("Architecture", "sigma_model"),
             ("Signer$\\times$architecture interaction", "sigma_interact"), ("Seed (initialization)", "sigma_seed")]
    rows = []
    for n, k in names:
        ctx.prov.append((frag, n, "SD", vals[k], f"{VARDECOMP_FILE} (block 'all 8') {k}"))
        rows.append(([n, vals[k]], None))
    return rows


_SIG_RE = re.compile(r"^\s*(.+?)\s+>\s+(.+?)\s+D=([+-]?[\d.]+)\s+CI\[[^\]]*\]\s+p=([\d.]+)\s+p_holm=([\d.]+)"
                     r"(?:.*?two-sided p2=([\d.]+) p2_holm=([\d.]+))?")


def parse_sig(text):
    """{'A': [(a, b, D, p_holm)], 'B': [...]} from tools/arch_significance.py output."""
    out, panel = {"A": [], "B": []}, None
    for line in text.splitlines():
        if line.startswith("## (A)"):
            panel = "A"
        elif line.startswith("## (B)"):
            panel = "B"
        m = _SIG_RE.match(line)
        if m and panel:
            a, b, d, _p, p_holm, _p2, p2_holm = m.groups()
            # tab:sig reports the TWO-SIDED Holm-adjusted p when the file has it (--two-sided)
            out[panel].append((a, b, d, p2_holm if p2_holm is not None else p_holm))
    return out


def _sig_rows(entries):
    rows = []
    for a, b, d, p in entries:
        pd = Decimal(p)
        p_s = "$<$0.001" if pd < Decimal("0.001") else str(pd.quantize(Decimal("0.001"), ROUND_HALF_UP))
        star = ("$***$" if pd < Decimal("0.001") else "$**$" if pd < Decimal("0.01")
                else "$*$" if pd < Decimal("0.05") else "n.s.")
        d_s = f"${'+' if not d.startswith(('-', '+')) else ''}{d}$"
        rows.append(([f"{SIG_NAMES[a]} $>$ {SIG_NAMES[b]}", d_s, p_s, star], None))
    return rows


def build_sig(panel):
    def b(ctx, frag):
        rows = _sig_rows(parse_sig(open(SIG_FILE).read())[panel])
        for r in rows:
            ctx.prov.append((frag, r[0][0], "*", "(row)", f"{SIG_FILE} panel ({panel})"))
        return rows
    return b


BUILDERS = {
    "tab_arch_a_pose": build_arch_a_pose,
    "tab_arch_a_rgb": build_arch_a_group("rgb"),
    "tab_arch_a_foundation": build_arch_a_group("foundation"),
    "tab_arch_a_prior_rerun": build_arch_a_group("prior_rerun"),
    "tab_arch_a_prior_reported": build_manual_rows("prior_reported"),
    "tab_arch_b_w401": build_arch_b,
    "tab_smg8_rows": build_smg8,
    "tab_eff_rows": build_eff,
    "tab_suite_rows": build_suite,
    "tab_bone_rows": build_bone,
    "tab_losoboot_rows": build_losoboot,
    "tab_losovar_rows": build_losovar,
    "tab_losofolds_rows": build_losofolds,
    "tab_sig_a_rows": build_sig("A"),
    "tab_sig_b_rows": build_sig("B"),
}
assert list(BUILDERS) == list(FRAGMENTS)


def render(rows, name):
    lines = [f"% GENERATED by tools/p2_make_tables.py ({name}) -- do not edit by hand; "
             f"sources: SPEC in the script + {MANUAL_FILE}"]
    for r in rows:
        if isinstance(r, str):
            lines.append(r)
            continue
        cells, comment = r
        line = " & ".join(cells) + " \\\\"
        if comment:
            line += " % " + comment
        lines.append(line)
    return "\n".join(lines) + "\n"


def generate(ctx):
    out = collections.OrderedDict()
    for name, builder in BUILDERS.items():
        out[name] = render(builder(ctx, name), name)
    return out


# =============================================================================
# --check: parse the paper's rows and diff cell by cell
# =============================================================================
def strip_comment(line):
    m = re.search(r"(?<!\\)%", line)
    return line[:m.start()] if m else line


def split_cells(line):
    s = strip_comment(line).strip()
    if s.endswith("\\\\"):
        s = s[:-2]
    return [c.strip() for c in re.split(r"(?<!\\)&", s)]


def is_data_row(line):
    s = strip_comment(line).strip()
    return "&" in s and s.endswith("\\\\") and not s.startswith("\\multicolumn")


_INPUT_RE = re.compile(r"^\s*\\input\{(generated/[^}]+)\}")


def paper_rows(tex_path, anchor, after):
    """Rows of one fragment as the paper currently renders them (inlining \\input{generated/...})."""
    text = open(tex_path).read()
    i = text.find(anchor)
    if i < 0:
        raise SourceError(f"{tex_path}: anchor not found: {anchor}")
    # the enclosing table: back to \begin{table}, forward to \end{tabular}
    start = text.rfind("\\begin{table", 0, i)   # table or table*
    end = text.find("\\end{tabular}", i)
    block = text[start:end].splitlines()
    try:
        j = next(k for k, l in enumerate(block) if after in strip_comment(l))
    except StopIteration:
        raise SourceError(f"{tex_path}: line '{after}' not found in table '{anchor}'")
    lines = []
    for l in block[j + 1:]:
        m = _INPUT_RE.match(l)
        if m:
            p = os.path.join(os.path.dirname(tex_path), m.group(1))
            p = p if p.endswith(".tex") else p + ".tex"
            if not os.path.isfile(p):
                raise SourceError(f"{tex_path}: \\input file missing: {p}")
            lines.extend(open(p).read().splitlines())
            continue
        lines.append(l)
    rows, used = [], False
    for l in lines:
        s = strip_comment(l).strip()
        if not s:
            continue
        if s.startswith("\\midrule") and not used:
            continue
        if is_data_row(l):
            rows.append(split_cells(l))
            used = True
            continue
        if s.startswith("\\midrule"):
            continue
        break
    return rows


def frag_rows(text):
    return [split_cells(l) for l in text.splitlines() if is_data_row(l)]


def norm(c):
    return re.sub(r"\s+", " ", c).strip()


def diff_rows(paper, gen):
    """List of human-readable mismatches between two row lists."""
    out = []
    pk = [norm(r[0]) for r in paper]
    gk = [norm(r[0]) for r in gen]
    if pk != gk:
        out.append(f"row order/set differs:\n        paper: {pk}\n        gen  : {gk}")
    pmap = {norm(r[0]): r for r in paper}
    for g in gen:
        p = pmap.get(norm(g[0]))
        if p is None:
            out.append(f"row missing in paper: {g[0]}")
            continue
        if len(p) != len(g):
            out.append(f"{g[0]}: {len(p)} cells in paper vs {len(g)} generated")
        for ci, (pc, gc) in enumerate(zip(p, g)):
            if norm(pc) != norm(gc):
                out.append(f"{g[0]} | col {ci}: paper '{pc}' != generated '{gc}'")
    return out


def recipe_previews():
    """Evaluate every non-active recipe; report what it would give (or what is missing)."""
    out = []
    for arch, spec in POSE.items():
        for rec, srcs in spec.get("recipes", {}).items():
            if rec == spec.get("recipe"):
                continue
            for key, src in srcs.items():
                try:
                    metric = "macrof1" if key == "suite" else None
                    v = resolve(src, metric)
                    dp = 1 if key == "loso" else 2
                    out.append(f"{arch}.{rec}.{key}: {pm(v, dp) if v.values else f2(v.mean)}  [{v.src}]")
                except (SourceError, OSError) as e:
                    out.append(f"{arch}.{rec}.{key}: not ready -- {e}")
    return out


def rerun_sig():
    """Run tools/arch_significance.py now and diff its tab:sig rows against SIG_FILE."""
    res = subprocess.run([sys.executable, "tools/arch_significance.py", *SIG_ARGS], cwd=ROOT, capture_output=True,
                         text=True, timeout=3600)
    if res.returncode != 0:
        return [f"arch_significance.py failed: {res.stderr[-500:]}"]
    fresh, saved = parse_sig(res.stdout), parse_sig(open(SIG_FILE).read())
    out = []
    for panel in "AB":
        a = [r[0] for r in _sig_rows(saved[panel])]
        b = [r[0] for r in _sig_rows(fresh[panel])]
        if a != b:
            out.append(f"tab:sig ({panel}): paper {[' | '.join(x) for x in a]}\n"
                       f"                  fresh  {[' | '.join(x) for x in b]}")
    head = [l for l in res.stdout.splitlines() if "seeds)" in l]
    return out, head


def main():
    global RESULTS_CSV, CACHE_MODE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="diff the paper against the generator, write nothing")
    ap.add_argument("--dry-run", action="store_true", help="print fragments to stdout, write nothing")
    ap.add_argument("--live", action="store_true", help="ignore pins (show the latest data everywhere)")
    ap.add_argument("--use-recipe", action="append", default=[], metavar="ARCH=RECIPE",
                    help="override a pose row's recipe for this run (preview), e.g. adaptive_gnn=lr0p1")
    ap.add_argument("--rerun-sig", action="store_true", help="with --check: re-run tools/arch_significance.py")
    ap.add_argument("--provenance", action="store_true", help="print the source of every generated cell")
    ap.add_argument("--out-dir", default=OUT_DIR, help=f"where fragments are written (default {OUT_DIR})")
    ap.add_argument("--ref-dir", default=REF_DIR, help=f"reference fragments for --check (default {REF_DIR})")
    ap.add_argument("--paper-tex", action="append", default=[], metavar="MAIN_TEX",
                    help="with --check: diff the rows of this LaTeX source (\\input{generated/...} inlined) instead")
    ap.add_argument("--results-csv", default=RESULTS_CSV, help=f"run log to read (default {RESULTS_CSV})")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--no-cache", action="store_true", help="raw sources only (needs work_dir/ score dumps and data/)")
    g.add_argument("--cache-only", action="store_true", help=f"read every source value from {CACHE_FILE}")
    args = ap.parse_args()
    RESULTS_CSV = args.results_csv
    CACHE_MODE = "off" if args.no_cache else "only" if args.cache_only else "fallback"

    override = dict(x.split("=", 1) for x in args.use_recipe)
    for a, r in override.items():
        if a not in POSE or r not in POSE[a].get("recipes", {}):
            ap.error(f"unknown recipe {a}={r}")
    ctx = Ctx(live=args.live, recipe_override=override)
    try:
        frags = generate(ctx)
    except SourceError as e:
        print(f"cannot generate: {e}\n(an active source is missing/incomplete; nothing was written)")
        return 2

    if args.provenance:
        for f, row, col, val, src in ctx.prov:
            print(f"{f:26s} {norm(row)[:40]:40s} {col:9s} {val:28s} <- {src}")

    if args.dry_run:
        for name, text in frags.items():
            print(f"===== {name}.tex\n{text}")
        return 0

    _report_sources()
    if not args.check:
        os.makedirs(args.out_dir, exist_ok=True)
        for name, text in frags.items():
            with open(os.path.join(args.out_dir, name + ".tex"), "w") as f:
                f.write(text)
        print(f"wrote {len(frags)} fragments to {args.out_dir}/")
        _report_newer(ctx)
        return 0

    n_bad = 0
    if not args.paper_tex:
        print(f"\n=== reference fragments in {args.ref_dir}/")
        for name in FRAGMENTS:
            p = os.path.join(args.ref_dir, name + ".tex")
            if not os.path.isfile(p):
                print(f"  [ERROR] {name}: missing {p}")
                n_bad += 1
                continue
            prow = frag_rows(open(p).read())
            diffs = diff_rows(prow, frag_rows(frags[name]))
            if diffs:
                n_bad += len(diffs)
                print(f"  [MISMATCH] {name} ({len(prow)} rows)")
                for x in diffs:
                    print(f"      - {x}")
            else:
                print(f"  [ok] {name} ({len(prow)} rows)")
    for tex in args.paper_tex:
        d = os.path.dirname(tex)
        print(f"\n=== {tex}")
        for name, (anchor, after) in FRAGMENTS.items():
            try:
                prow = paper_rows(tex, anchor, after)
            except SourceError as e:
                print(f"  [ERROR] {name}: {e}")
                n_bad += 1
                continue
            diffs = diff_rows(prow, frag_rows(frags[name]))
            uses_input = any(f"generated/{name}" in l for l in open(tex).read().splitlines()
                             if not l.lstrip().startswith("%"))
            tag = "input" if uses_input else "inline"
            if diffs:
                n_bad += len(diffs)
                print(f"  [MISMATCH] {name} ({tag}, {len(prow)} rows)")
                for x in diffs:
                    print(f"      - {x}")
            else:
                print(f"  [ok] {name} ({tag}, {len(prow)} rows)")
            if uses_input:
                p = os.path.join(d, "generated", name + ".tex")
                if open(p).read() != frags[name]:
                    print(f"      note: {p} differs from a fresh generation in comments/layout only"
                          if not diffs else f"      note: regenerate {p}")
    _report_newer(ctx)
    prev = recipe_previews()
    if prev:
        print("\n=== Pending recipes (not active; select via POSE[...]['recipe'] or --use-recipe)")
        for x in prev:
            print("  " + x)
    if args.rerun_sig:
        print("\n=== tab:sig re-run (tools/arch_significance.py now vs " + SIG_FILE + ")")
        r = rerun_sig()
        if isinstance(r, tuple):
            diffs, head = r
            for h in head:
                print("  " + h.strip())
            print("  identical" if not diffs else "\n".join("  DATA NEWER THAN PAPER: " + x for x in diffs))
        else:
            print("\n".join(r))
    kinds = collections.Counter(x[7] for x in ctx.newer)
    print(f"\n{'PASS' if n_bad == 0 else 'FAIL'}: {n_bad} unexplained mismatch(es); pinned cells that differ "
          f"from the data: {kinds.get(NEWER, 0)} data-newer, {kinds.get(ROUNDING, 0)} rounding")
    return 0 if n_bad == 0 else 1


def _report_sources():
    n_raw, n_cache = _CACHE_STATS["raw"], _CACHE_STATS["cache"]
    if n_cache:
        print(f"[sources] {n_raw} resolved from raw files, {n_cache} from {CACHE_FILE} "
              f"(raw score dumps / data not present; use --no-cache to require raw sources)")
    else:
        print(f"[sources] all {n_raw} resolved from raw files")


def _report_newer(ctx):
    seen, rows = set(), []
    for f, row, col, old, new, src, reason, kind in ctx.newer:
        k = (f, norm(row), col)
        if k not in seen:
            seen.add(k)
            rows.append((f, row, col, old, new, src, reason, kind))
    for kind, title in [(NEWER, "Data newer than paper"),
                        (ROUNDING, "Paper rounding inconsistencies (value as printed vs the convention)")]:
        sel = [r for r in rows if r[7] == kind]
        if not sel:
            continue
        print(f"\n=== {title} -- pinned to the paper's value; remove the pin in the SPEC to update")
        for f, row, col, old, new, src, reason, _ in sel:
            print(f"  {f} / {norm(row)} / {col}: paper {old} -> generator {new}\n      source: {src}\n      why: {reason}")


if __name__ == "__main__":
    sys.exit(main())
