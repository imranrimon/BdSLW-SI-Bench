#!/bin/bash -l
set -e
cd "$(dirname "$0")/../../.."  # repository root
source ~/miniconda3/etc/profile.d/conda.sh && conda activate bdsl_graph
python - <<'PY'
import sys, pickle, csv, statistics as S, collections
sys.path.insert(0, "tools")
import arch_significance  # numpy alias
import p2_metric_suite as suite
names, labs = pickle.load(open("data/bdsl_si/test_label.pkl", "rb")); lab = dict(zip(names, [int(x) for x in labs]))
for stem in ["bdsl_slgtformer_si", "bdsl_slgtformer_si_bestck", "bdsl_gnn_lstm_si", "bdsl_gnn_lstm_si_bestck",
             "bdsl_pose_lstm_si", "bdsl_pose_lstm_si_bestck", "bdsl_st_gcn_vanilla_si", "bdsl_st_gcn_vanilla_si_bestck"]:
    m = suite.metrics_for(stem, lab)
    print(stem, "seeds", m["seeds"], "macroF1 %.2f" % m["single"]["macrof1"][0], "ens top1 %.2f" % m["ens"]["top1"])
for n in ["slgtformer", "slgtformer.pre_fix_2026-10-22", "gnn_lstm", "gnn_lstm.pre_fix_2026-10-22"]:
    per = collections.OrderedDict()
    for r in csv.DictReader(open(f"docs/loso_heldout/{n}.csv")):
        per.setdefault(r["heldout_signer"], []).append(float(r["test_acc"]))
    f = [S.mean(v) for v in per.values()]
    print("LOSO", n, "%.2f +- %.2f (sample SD over 11 folds), min %.1f max %.1f" % (S.mean(f), S.stdev(f), min(f), max(f)))
PY
echo DONE
