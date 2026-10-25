#!/usr/bin/env python
"""Aggregate tools/tta_eval.py outputs into the TTA tables (docs/tta_grl_2026-10/).

Protocol A (TEST signers 2 and 13, each adapted separately): per seed, pooled Top-1 over
both signers' 1,365 clips; mean +/- population SD over seeds (the paper's convention for
Protocol-A rows); Delta vs none = mean of per-seed Deltas. Paired bootstrap: per-clip
correctness averaged over the seeds, clips resampled within signer (10k, seed 1234),
two-sided p = 2 * min(P(D* <= 0), P(D* >= 0)).

LOSO (11 held-out signers): fold value = mean over that fold's complete seeds; mean +/-
sample SD over folds (Table 7 convention); signer-clustered bootstrap over the 11 folds
(10k, seed 1234) of the fold-level Delta -> 95% CI and two-sided p. Only (fold, seed)
pairs that have both the method and `none` enter (truncated runs are skipped upstream).
"""
import argparse, glob, json, os
from collections import defaultdict
import numpy as np

NB, BSEED = 10000, 1234
MODELS = ["block_gcn", "adaptive_gnn", "slgtformer"]
MNAME = {"block_gcn": "BlockGCN (lr 0.01)", "adaptive_gnn": "Adaptive GNN (lr 0.01)", "slgtformer": "SLGTFormer"}


def load(raw):
    recs = []
    for p in sorted(glob.glob(os.path.join(raw, "tta_*.jsonl"))):
        for ln in open(p):
            try:
                recs.append(json.loads(ln))
            except Exception:
                pass
    # last record wins for a key
    d = {}
    for r in recs:
        d[(r["model"], r["protocol"], r["fold"], r["seed"], r["signer"], r["method"])] = r
    return d


def pval(boot):
    return float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())))


SUPERSEDED = {"cmvn", "cmvn_c", "cmvn_adabn"}  # first CMVN run also standardized the visibility channel (invalid)


def method_order(methods):
    methods = set(methods) - SUPERSEDED
    base = ["none", "adabn", "cmvnxy", "cmvnxy_c", "cmvnxy_adabn"]
    tents = sorted([m for m in methods if m.startswith("tent_lr")], key=lambda m: float(m[7:]))
    return [m for m in base if m in methods] + tents + sorted(set(methods) - set(base) - set(tents) - {"skipped_truncated"})


def protoA(D, model, rng):
    keys = [k for k in D if k[0] == model and k[1] == "A" and k[5] != "skipped_truncated"]
    methods = method_order({k[5] for k in keys})
    seeds = sorted({k[3] for k in keys})
    rows = {}
    for m in methods:
        accs, deltas, per_sig, cm, cn, sigs = [], [], defaultdict(list), [], [], None
        used = []
        for s in seeds:
            ks = [(model, "A", "A", s, g, m) for g in (2, 13)]
            kn = [(model, "A", "A", s, g, "none") for g in (2, 13)]
            if not all(k in D for k in ks + kn):
                continue
            c = np.concatenate([D[k]["correct"] for k in ks]); c0 = np.concatenate([D[k]["correct"] for k in kn])
            sg = np.concatenate([[g] * D[k]["n"] for g, k in zip((2, 13), ks)])
            accs.append(c.mean() * 100); deltas.append((c.mean() - c0.mean()) * 100)
            for g, k in zip((2, 13), ks):
                per_sig[g].append(D[k]["acc"])
            cm.append(c); cn.append(c0); sigs = sg; used.append(s)
        if not accs:
            continue
        cm, cn = np.mean(cm, 0), np.mean(cn, 0)
        diff = cm - cn
        boot = np.empty(NB)
        idx = {g: np.where(sigs == g)[0] for g in (2, 13)}
        for b in range(NB):
            sel = np.concatenate([rng.choice(ix, len(ix)) for ix in idx.values()])
            boot[b] = diff[sel].mean() * 100
        extra = {}
        if m.startswith("tent_lr"):
            off = []
            for s in used:
                ks = [(model, "A", "A", s, g, m) for g in (2, 13)]
                off.append(np.concatenate([D[k]["offline_correct"] for k in ks]).mean() * 100)
            extra["offline"] = (np.mean(off), np.std(off))
        rows[m] = dict(seeds=used, acc=(np.mean(accs), np.std(accs)), per_seed=accs, delta=np.mean(deltas),
                       ci=(np.percentile(boot, 2.5), np.percentile(boot, 97.5)), p=pval(boot),
                       u02=np.mean(per_sig[2]), u13=np.mean(per_sig[13]), **extra)
    return rows


def loso(D, model, rng):
    keys = [k for k in D if k[0] == model and k[1] == "loso" and k[5] != "skipped_truncated"]
    methods = method_order({k[5] for k in keys})
    folds = sorted({k[2] for k in keys})
    skipped = sorted({(k[2], k[3]) for k in D if k[0] == model and k[1] == "loso" and k[5] == "skipped_truncated"})
    rows = {}
    for m in methods:
        fv, f0, nseeds, off = [], [], [], []
        for f in folds:
            a, a0, ao = [], [], []
            for s in (0, 1, 2):
                km = [k for k in D if k[:4] == (model, "loso", f, s) and k[5] == m]
                kn = [k for k in D if k[:4] == (model, "loso", f, s) and k[5] == "none"]
                if km and kn:
                    a.append(D[km[0]]["acc"]); a0.append(D[kn[0]]["acc"])
                    if "acc_offline" in D[km[0]]:
                        ao.append(D[km[0]]["acc_offline"])
            if a:
                fv.append(np.mean(a)); f0.append(np.mean(a0)); nseeds.append(len(a))
                if ao:
                    off.append(np.mean(ao))
        if not fv:
            continue
        fv, f0 = np.array(fv), np.array(f0)
        d = fv - f0
        boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(NB)])
        rows[m] = dict(n_folds=len(fv), seed_runs=int(sum(nseeds)), acc=(fv.mean(), fv.std(ddof=1)),
                       delta=d.mean(), ci=(np.percentile(boot, 2.5), np.percentile(boot, 97.5)), p=pval(boot),
                       n_up=int((d > 0).sum()), fold_delta=dict(zip(folds, d.round(2).tolist())),
                       offline=(np.mean(off), np.std(off, ddof=1)) if off else None)
    return rows, skipped


def tent_collapse(D, model):
    out = {}
    for k, r in D.items():
        if k[0] != model or not k[5].startswith("tent_lr"):
            continue
        tr = r.get("traj") or []
        if not tr:
            continue
        q = max(1, len(tr) // 4)
        e0 = np.mean([t["entropy"] for t in tr[:q]]); e1 = np.mean([t["entropy"] for t in tr[-q:]])
        a0 = np.mean([t["batch_acc"] for t in tr[:q]]) * 100; a1 = np.mean([t["batch_acc"] for t in tr[-q:]]) * 100
        o = out.setdefault(k[5], dict(n=0, e0=[], e1=[], a0=[], a1=[], min_cls=[], max_share=[], off_cls=[], off_share=[]))
        o["n"] += 1; o["e0"].append(e0); o["e1"].append(e1); o["a0"].append(a0); o["a1"].append(a1)
        o["min_cls"].append(r["n_pred_classes"]); o["max_share"].append(r["max_class_share"])
        o["off_cls"].append(r.get("offline_n_pred_classes", np.nan)); o["off_share"].append(r.get("offline_max_class_share", np.nan))
    return out


def fmt_p(p):
    return "<0.001" if p < 1e-3 else f"{p:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="docs/tta_grl_2026-10/raw")
    ap.add_argument("--md", default="docs/tta_grl_2026-10/tta_tables.md")
    ap.add_argument("--json", default="docs/tta_grl_2026-10/raw/tta_summary.json")
    a = ap.parse_args()
    D = load(a.raw)
    rng = np.random.default_rng(BSEED)
    L, J = [], {}
    for model in MODELS:
        A = protoA(D, model, np.random.RandomState(BSEED))
        Lr, skipped = loso(D, model, rng)
        if not A and not Lr:
            continue
        J[model] = dict(A={m: {k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()} for m, r in A.items()},
                        loso={m: {k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()} for m, r in Lr.items()},
                        skipped=skipped)
        L.append(f"\n## {MNAME[model]}\n")
        if skipped:
            L.append(f"Truncated LOSO runs excluded (fold, seed): {skipped}\n")
        L.append("| method | TEST Top-1 (mean ± SD, seeds) | Δ vs none | 95% CI (paired, clip bootstrap) | p | U02 | U13 | LOSO mean ± SD (folds) | Δ vs none | 95% CI (signer-clustered) | p | folds improved |")
        L.append("|---|---|--:|---|--:|--:|--:|---|--:|---|--:|--:|")
        for m in method_order(set(A) | set(Lr)):
            ra, rl = A.get(m), Lr.get(m)
            ca = (f"{ra['acc'][0]:.2f} ± {ra['acc'][1]:.2f} ({len(ra['seeds'])})", f"{ra['delta']:+.2f}",
                  f"[{ra['ci'][0]:+.2f}, {ra['ci'][1]:+.2f}]", fmt_p(ra["p"]), f"{ra['u02']:.2f}", f"{ra['u13']:.2f}") if ra else ("--",) * 6
            cl = (f"{rl['acc'][0]:.2f} ± {rl['acc'][1]:.2f} ({rl['n_folds']}; {rl['seed_runs']} runs)", f"{rl['delta']:+.2f}",
                  f"[{rl['ci'][0]:+.2f}, {rl['ci'][1]:+.2f}]", fmt_p(rl["p"]), f"{rl['n_up']}/{rl['n_folds']}") if rl else ("--",) * 5
            if m == "none":
                ca = ca[:1] + ("--", "--", "--") + ca[4:]
                cl = cl[:1] + ("--", "--", "--", "--")
            L.append(f"| {m} | " + " | ".join(ca + cl) + " |")
        tents = [m for m in method_order(set(A) | set(Lr)) if m.startswith("tent_lr")]
        if tents:
            L.append("\nTent: ONLINE accuracy above (reference protocol); OFFLINE second pass after the epoch:")
            for m in tents:
                oa = A.get(m, {}).get("offline"); ol = Lr.get(m, {}).get("offline") if m in Lr else None
                L.append(f"- {m}: TEST offline {oa[0]:.2f} ± {oa[1]:.2f}" if oa else f"- {m}: TEST --")
                if ol:
                    L[-1] += f"; LOSO offline {ol[0]:.2f} ± {ol[1]:.2f}"
            C = tent_collapse(D, model)
            L.append("\nTent collapse diagnostics (over all adapted (run, signer) units; first vs last quarter of the 1-epoch trajectory):\n")
            L.append("| lr | units | entropy first→last q | online batch acc first→last q | min #pred classes (online / offline) | max single-class share (online / offline) |")
            L.append("|---|--:|---|---|---|---|")
            for m in tents:
                c = C.get(m)
                if not c:
                    continue
                L.append(f"| {m[7:]} | {c['n']} | {np.mean(c['e0']):.3f} → {np.mean(c['e1']):.3f} | "
                         f"{np.mean(c['a0']):.1f} → {np.mean(c['a1']):.1f} | {min(c['min_cls'])} / {np.nanmin(c['off_cls']):.0f} | "
                         f"{max(c['max_share']):.3f} / {np.nanmax(c['off_share']):.3f} |")
        L.append("\nPer-fold LOSO Δ (fold value = seed mean):\n")
        for m, rl in Lr.items():
            if m != "none":
                L.append(f"- {m}: " + ", ".join(f"U{f} {v:+.1f}" for f, v in rl["fold_delta"].items()))
    os.makedirs(os.path.dirname(a.md), exist_ok=True)
    open(a.md, "w").write("# TTA results (auto-generated by tools/tta_report.py)\n" + "\n".join(L) + "\n")
    json.dump(J, open(a.json, "w"), indent=1, default=float)
    print("\n".join(L))


if __name__ == "__main__":
    main()
