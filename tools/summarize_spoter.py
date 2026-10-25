#!/usr/bin/env python
"""Summarise the SPOTER-family baselines (baseline item 2) into
docs/baselines_rerun/item2_spoter.md (+ .json), in the requested format: per-seed
values and mean +- population SD for held-out TEST Top-1, macro-F1 (single-model
3-seed mean, same estimator as tools/p2_metric_suite.py), signer-mixed accuracy
(best-epoch Top-1 on data/bdsl_sd val, as in tab:smg8) and SMG = mixed - TEST,
plus params / thop GFLOPs at T=120, V=27.
"""
import json, os, pickle, sys
import numpy as np
import torch, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from tools.p2_metric_suite import _metrics_one, to_prob  # noqa: E402

SEEDS = (0, 1, 2)


def labels(path):
    n, l = pickle.load(open(path, "rb")); return dict(zip(n, l))


def test_metrics(pkl, lab):
    d = {k: to_prob(v) for k, v in pickle.load(open(pkl, "rb")).items()}
    ks = sorted(set(d) & set(lab))
    m = _metrics_one(np.stack([d[k] for k in ks]), np.array([lab[k] for k in ks]))
    return m["top1"], m["macrof1"], len(ks)


def ms(v):
    v = [x for x in v if x is not None]
    return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (None, None, 0)


def cost(cfg):
    from thop import profile
    from model.spoter import Model
    c = yaml.safe_load(open(cfg))
    m = Model(**c["model_args"]).eval()
    params = sum(p.numel() for p in m.parameters()) / 1e6
    f, _ = profile(Model(**c["model_args"]).eval(), inputs=(torch.randn(1, 3, 120, 27, 1),), verbose=False)
    return params, f / 1e9


def main():
    lab = labels("data/bdsl_si/test_label.pkl")
    sdlab = labels("data/bdsl_sd/test_label.pkl")
    rows, out = [], {}
    tracks = [("BdSL-SPOTER (shared recipe)", "bdsl_spoter", "main"),
              ("SPOTER, original (shared recipe)", "bdsl_spoter_orig", "main"),
              ("BdSL-SPOTER (native released recipe)", "bdsl_spoter_native", "native")]
    for name, stem, kind in tracks:
        top1, f1, mixed, oldmixed, notes = [], [], [], [], []
        for s in SEEDS:
            sdw = f"work_dir/{stem}_sd_seed{s}"
            if kind == "main":
                tp = f"work_dir/{stem}_si_test_seed{s}/eval_results/best_acc.pkl"
                sdb = f"{sdw}/{stem}_sd_seed{s}_model-best.pt"
                om = torch.load(sdb, map_location="cpu")["acc"] * 100 if os.path.exists(f"{sdw}/.done") and os.path.exists(sdb) else None
            else:
                tp = f"work_dir/{stem}_si_seed{s}/eval_results/test_scores.pkl"
                rj = f"{sdw}/result.json"
                om = json.load(open(rj))["best_val"] * 100 if os.path.exists(rj) else None
            oldmixed.append(om)
            # primary Mixed: best-on-sd-val checkpoint scored on data/bdsl_sd/test (tools/eval_ckpt_on_npy.py dump)
            sp = f"{sdw}/eval_results/sdtest_scores.pkl"
            mx = test_metrics(sp, sdlab)[0] if os.path.exists(sp) else None
            if os.path.exists(tp):
                t, f, n = test_metrics(tp, lab); top1.append(t); f1.append(f)
                if n != len(lab): notes.append(f"seed{s}: {n}/{len(lab)} TEST clips scored")
            else:
                top1.append(None); f1.append(None); notes.append(f"seed{s}: TEST missing")
            mixed.append(mx)
            if mx is None: notes.append(f"seed{s}: mixed missing")
        smg = [m - t if (m is not None and t is not None) else None for m, t in zip(mixed, top1)]
        cfg = f"config/{stem}_si.yaml" if kind == "main" else None
        lr = yaml.safe_load(open(cfg)).get("base_lr") if cfg else "AdamW 3e-4 / OneCycle max 1e-3"
        p, g = cost(cfg) if cfg else (None, None)
        if kind == "native":
            from model.spoter import Model
            from thop import profile
            p = sum(x.numel() for x in Model(num_class=60, variant="bdsl_spoter", num_layers=8).parameters()) / 1e6
            g = profile(Model(num_class=60, variant="bdsl_spoter", num_layers=8).eval(),
                        inputs=(torch.randn(1, 3, 120, 27, 1),), verbose=False)[0] / 1e9
        oldsmg = [m - t if (m is not None and t is not None) else None for m, t in zip(oldmixed, top1)]
        out[stem] = dict(name=name, base_lr=lr, test_top1=top1, macro_f1=f1, mixed_sdtest=mixed, smg=smg,
                         mixed_sdval_old=oldmixed, smg_old=oldsmg,
                         params_M=p, gflops_thop=g, notes=notes)
        rows.append((name, lr, top1, f1, mixed, smg, p, g, notes, oldmixed, oldsmg))

    fmt = lambda v: "/".join("—" if x is None else f"{x:.2f}" for x in v)
    pm = lambda v: ("—" if ms(v)[0] is None else f"{ms(v)[0]:.2f} ± {ms(v)[1]:.2f}" + ("" if ms(v)[2] == 3 else f" (n={ms(v)[2]})"))
    L = ["# Item 2 — SPOTER-family baselines under Protocol A + signer-mixed split", "",
         "Protocol A (train {1,4,5,6,8,9,11,12}, val U15, TEST {2,13}); 27-kpt pose, T=120, mirror OFF.",
         "Shared-recipe rows: SGD+nesterov, wd 1e-4, warm-up 5, step [60,80], 100 ep, batch 16, **no grad-clip** "
         "(grad_clip_norm unset = 0.0, matching the headline Protocol-A/LOSO runs); base LR selected on val U15 at seed 0 "
         "from {0.1, 0.05, 0.01, 0.001}. Native row: BdSL-SPOTER's released recipe (tools/train_spoter_native.py; "
         "grad-clip 1.0 is part of that recipe). TEST = best-val-epoch checkpoint on held-out signers 2/13. "
         "Mixed = best-on-sd-val checkpoint scored on data/bdsl_sd/test (1,367 clips; P2 symmetric-selection protocol, "
         "docs/audit_reruns_2026-10/r5_mixed_selection.md). SMG = Mixed − TEST, per seed. Secondary: old Mixed = "
         "best-epoch Top-1 on sd val (selected on the reporting set) and its SMG. "
         "± = population SD over 3 seeds. Macro-F1 = single-model mean (tools/p2_metric_suite.py estimator).", "",
         "| Method | base LR | Params (M) | GFLOPs (thop) | TEST Top-1 | Macro-F1 | Mixed (sd test) | SMG (pp) | old Mixed (sd val) | old SMG |",
         "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for name, lr, t, f, m, s, p, g, _, om, os_ in rows:
        L.append(f"| {name} | {lr} | {p:.2f} | {g:.3f} | {pm(t)} | {pm(f)} | {pm(m)} | {pm(s)} | {pm(om)} | {pm(os_)} |")
    L += ["", "## Per-seed values (seeds 0/1/2)", ""]
    for name, lr, t, f, m, s, p, g, notes, om, os_ in rows:
        L.append(f"- **{name}** — TEST Top-1 {fmt(t)}; Macro-F1 {fmt(f)}; Mixed (sd test) {fmt(m)}; SMG {fmt(s)}; "
                 f"old Mixed (sd val) {fmt(om)}; old SMG {fmt(os_)}"
                 + (f"  _(notes: {'; '.join(notes)})_" if notes else ""))
    os.makedirs("docs/baselines_rerun", exist_ok=True)
    open("docs/baselines_rerun/item2_spoter.md", "w").write("\n".join(L) + "\n")
    json.dump(out, open("docs/baselines_rerun/item2_spoter.json", "w"), indent=1, default=float)
    print("\n".join(L))


if __name__ == "__main__":
    main()
