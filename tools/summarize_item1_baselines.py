#!/usr/bin/env python
"""Summarise Baseline request item 1 (BdSLW60 dataset-paper baselines) into
docs/baselines_rerun/item1_bdslw60_baselines.md (+ .json).

Bi-LSTM rows: TEST Top-1 + macro-F1 from each seed's TEST-eval score file
(work_dir/<stem>_si_test_seed<N>/eval_results/best_acc.pkl, metric code shared with
tools/p2_metric_suite.py). Mixed (P2 symmetric protocol, docs/audit_reruns_2026-10/
r5_mixed_selection.md) = the best-on-sd-val checkpoint scored on data/bdsl_sd(_rq)/test
via tools/eval_ckpt_on_npy.py (score dump cached next to the run); old Mixed (best-epoch
sd-val Top-1) kept as a secondary column.
SD = population SD (np.std), the paper's convention. SVM rows from results.json.
"""
import json, os, pickle, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
from p2_metric_suite import _metrics_one, to_prob  # noqa: E402
import subprocess  # noqa: E402
import torch  # noqa: E402

TEST_LABEL = os.path.join(ROOT, "data/bdsl_si/test_label.pkl")
OUT = os.path.join(ROOT, "docs/baselines_rerun/item1_bdslw60_baselines")


def bilstm(stem):
    names, labels = pickle.load(open(TEST_LABEL, "rb")); lab = dict(zip(names, labels))
    rows = []
    for s in (0, 1, 2):
        r = {"seed": s}
        pk = os.path.join(ROOT, f"work_dir/{stem}_si_test_seed{s}/eval_results/best_acc.pkl")
        if os.path.exists(pk):
            d = {k: to_prob(v) for k, v in pickle.load(open(pk, "rb")).items()}
            ks = [k for k in d if k in lab]
            m = _metrics_one(np.stack([d[k] for k in ks]), np.array([lab[k] for k in ks]))
            r.update(test_top1=float(m["top1"]), macro_f1=float(m["macrof1"]), n_test=len(ks))
        wd = os.path.join(ROOT, f"work_dir/{stem}_sd_seed{s}")
        bp = os.path.join(wd, f"{stem}_sd_seed{s}_model-best.pt")
        final = os.path.join(wd, f"{stem}_sd_seed{s}_model-{int(_nep(stem)) - 1}.pt")
        if os.path.exists(bp) and os.path.exists(final):
            r["mixed_old"] = float(torch.load(bp, map_location="cpu")["acc"]) * 100
            dump = os.path.join(wd, "sd_test_scores_best.pkl")
            if not os.path.exists(dump) or os.path.getmtime(dump) < os.path.getmtime(bp):
                subprocess.check_call([sys.executable, "tools/eval_ckpt_on_npy.py", "--config", os.path.join(wd, "config.yaml"),
                                       "--weights", bp, "--data", "data/bdsl_sd_rq/test_data.npy",
                                       "--label", "data/bdsl_sd_rq/test_label.pkl", "--dump", dump], cwd=ROOT)
            sn, sl = pickle.load(open(os.path.join(ROOT, "data/bdsl_sd_rq/test_label.pkl"), "rb"))
            d = pickle.load(open(dump, "rb"))
            m = _metrics_one(np.stack([to_prob(d[k]) for k in sn]), np.asarray(sl))
            r["mixed"] = float(m["top1"]); r["mixed_macro_f1"] = float(m["macrof1"])
        vp = os.path.join(ROOT, f"work_dir/{stem}_si_seed{s}/{stem}_si_seed{s}_model-best.pt")
        if os.path.exists(vp):
            ck = torch.load(vp, map_location="cpu"); r["val_top1"] = float(ck["acc"]) * 100; r["best_epoch"] = int(ck["epoch"])
        rows.append(r)
    return rows


def _nep(stem):
    import yaml
    return yaml.safe_load(open(os.path.join(ROOT, f"config/{stem}_sd.yaml")))["num_epoch"]


def agg(rows, k):
    v = [r[k] for r in rows if k in r]
    return (float(np.mean(v)), float(np.std(v)), len(v)) if v else None


def fmt(a):
    return "–" if a is None else f"{a[0]:.2f} ± {a[1]:.2f}" + ("" if a[2] == 3 else f" (n={a[2]})")


def main():
    res = {}
    lines = ["# Item 1 — BdSLW60 dataset-paper baselines under Protocol A", "",
             "Protocol A: train {1,4,5,6,8,9,11,12} / val 15 / TEST {2,13}; 27-keypoint pose, T=120. "
             "Mixed = signer-dependent split `data/bdsl_sd`: checkpoint (or SVM C) selected on sd val (655), "
             "scored on sd TEST (1,367) — P2's symmetric protocol; 'old mixed' = best-epoch sd-val Top-1 (previous rule). "
             "SMG = Mixed − TEST. ± = population SD over seeds. grad_clip_norm unset (0.0).", ""]
    lr_file = os.path.join(ROOT, "work_dir/bdsl_attn_bilstm_si_lrsel.txt")
    lr = open(lr_file).read().strip() if os.path.exists(lr_file) else "pending"
    for stem, label in (("bdsl_attn_bilstm", f"Attention Bi-LSTM + RQ — shared recipe (Track A; {lr})"),
                        ("bdsl_attn_bilstm_author", "Attention Bi-LSTM + RQ — authors' recipe (Adam 3e-5, bs64, 200 ep)")):
        rows = bilstm(stem); res[stem] = rows
        t, f, mx = agg(rows, "test_top1"), agg(rows, "macro_f1"), agg(rows, "mixed")
        smg = None if (t is None or mx is None) else (mx[0] - t[0], 0.0, 3)
        mo = agg(rows, "mixed_old")
        lines += [f"## {label}", "", "| seed | val Top-1 (U15) | TEST Top-1 | macro-F1 | Mixed (sd TEST) | SMG | old mixed (sd val best) | old SMG |",
                  "|---|--:|--:|--:|--:|--:|--:|--:|"]
        for r in rows:
            g = lambda k: f"{r[k]:.2f}" if k in r else "–"
            sm = f"{r['mixed'] - r['test_top1']:+.2f}" if "mixed" in r and "test_top1" in r else "–"
            so = f"{r['mixed_old'] - r['test_top1']:+.2f}" if "mixed_old" in r and "test_top1" in r else "–"
            lines.append(f"| {r['seed']} | {g('val_top1')} | {g('test_top1')} | {g('macro_f1')} | {g('mixed')} | {sm} | {g('mixed_old')} | {so} |")
        lines += [f"| **mean ± SD** | {fmt(agg(rows, 'val_top1'))} | **{fmt(t)}** | {fmt(f)} | {fmt(mx)} | "
                  f"{'–' if smg is None else f'{smg[0]:+.2f}'} | {fmt(mo)} | "
                  f"{'–' if (mo is None or t is None) else f'{mo[0] - t[0]:+.2f}'} |", ""]
    for tag in ("original", "flipped"):
        p = os.path.join(ROOT, f"work_dir/bdsl_svm_baseline_{tag}/results.json")
        if not os.path.exists(p):
            lines += [f"## Linear SVM ({tag}) — pending", ""]; continue
        r = json.load(open(p)); res[f"svm_{tag}"] = r
        lines += [f"## Linear SVM — {tag}" + (" (authors' hand-dominance correction; uses _L/_R clip metadata at test time)" if tag == "flipped" else ""), "",
                  "Deterministic (sklearn SVC linear): one value, no seeds.", "",
                  "| C | SI val Top-1 | SI TEST Top-1 | sd val | sd TEST |", "|--:|--:|--:|--:|--:|"]
        for c in r["C_grid"]:
            lines.append(f"| {c} | {r['si'][str(c)]['val']:.2f} | {r['si'][str(c)]['test']:.2f} | {r['sd'][str(c)]['val']:.2f} | {r['sd'][str(c)]['test']:.2f} |")
        lines += ["", f"C on SI val = {r['C_selected']}: **TEST Top-1 {r['test_top1']:.2f}**, macro-F1 {r['test_macro_f1']:.2f}. "
                  f"C on sd val = {r['C_selected_sd']}: **Mixed (sd TEST) {r['mixed_sd_test']:.2f}** (macro-F1 {r['mixed_sd_test_macro_f1']:.2f}), "
                  f"**SMG {r['smg']:+.2f}**. Old rule (max sd-val over C) {r['mixed_old_max_sd_val']:.2f}, old SMG {r['smg_old']:+.2f}.", ""]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT + ".md", "w").write("\n".join(lines) + "\n")
    json.dump(res, open(OUT + ".json", "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
