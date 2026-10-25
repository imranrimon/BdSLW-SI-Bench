#!/usr/bin/env python
"""P2 missing-joint fixes: aggregate tasks A-E into docs/p2_missing_joint_fixes_2026-10/results.md.

Inputs: raw/scores/*.pkl + index.jsonl (tools/p2_mjfix_score.py), raw/probe_*.jsonl
(tools/decodability_probe_v2.py), results_final.csv (cue final epochs), and the paired-SMG
*_test score dumps in work_dir. Means are over seeds with population SD (as P2). Paired
bootstraps resample TEST clips (10k, per-instance; the TEST set has only 2 signers).
  python tools/p2_mjfix_report.py
"""
import csv, collections, json, os, pickle, re, statistics as st, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
D = "docs/p2_missing_joint_fixes_2026-10"
SC = f"{D}/raw/scores"
SEEDS = (0, 1, 2)
NB = 10000
if not hasattr(np, "_core"):
    import numpy.core as _c
    sys.modules.setdefault("numpy._core", _c)
    for _s in ("multiarray", "numeric", "umath", "_multiarray_umath"):
        if f"numpy.core.{_s}" in sys.modules: sys.modules.setdefault(f"numpy._core.{_s}", sys.modules[f"numpy.core.{_s}"])

LAB = {sp: dict(zip(*pickle.load(open(f"data/bdsl_si/{sp}_label.pkl", "rb")))) for sp in ("val", "test")}


def ms(xs, nd=2):
    xs = [x for x in xs if x is not None]
    if not xs: return "--"
    return f"{st.mean(xs):.{nd}f} ± {st.pstdev(xs):.{nd}f}" if len(xs) > 1 else f"{xs[0]:.{nd}f} (n=1)"


def per(xs): return " / ".join("--" if x is None else f"{x:.2f}" for x in xs)


def pk(key):
    p = f"{SC}/{key}.pkl"
    return pickle.load(open(p, "rb")) if os.path.exists(p) else None


def acc(sc, lab):
    return None if sc is None else 100.0 * np.mean([np.argmax(v) == lab[k] for k, v in sc.items()])


def corr(sc, lab, clips):
    return np.array([int(np.argmax(sc[c]) == lab[c]) for c in clips])


def boot(a, b, seed=20260826):
    """paired per-clip bootstrap of mean(a) - mean(b); a, b 0/1 (or fractional) arrays."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(NB, len(a)))
    d = (a[idx].mean(1) - b[idx].mean(1)) * 100
    lo, hi = np.percentile(d, [2.5, 97.5])
    return (a.mean() - b.mean()) * 100, lo, hi, float(min(1.0, min((d <= 0).mean(), (d >= 0).mean()) * 2))


def fmt_boot(r): return f"{r[0]:+.2f} pp [{r[1]:+.2f}, {r[2]:+.2f}], p={r[3]:.4f}" if r else "--"


def ens(keys, lab):
    s = [pk(k) for k in keys]
    if any(x is None for x in s): return None
    clips = sorted(s[0])
    return {c: sum(np.asarray(x[c]) for x in s) for c in clips}


# ------------------------------------------------------------------ A
def task_a(L):
    L.append("## A. Masked bone stream and joint+bone fusion\n")
    L.append("Bone = child − parent along graph.sign_27 inward edges; masked version zeroes a bone whose child OR parent is "
             "undetected (only the two hand-root bones change: body wrist → hand wrist, 10.0% of missing-joint entries in valid "
             "frames; median |old bone| there 1.17–1.31 vs 0.11 for real bones). Best-on-val checkpoint scored on held-out TEST "
             "(signers 2/13, 1,365 clips). Fusion s = s_joint + α·s_bone on logits; α=1 is P2's rule (fixed, not tuned); "
             "'α val' picks α∈{0,0.05,…,3} maximizing the 3-seed-ensemble VAL (U15) accuracy and applies it to TEST.\n")
    archs = [("BlockGCN (lr 0.01, P2 recipe)", "bgj", "bgb_old", "bgb_new", "65.54 ± 1.63", "+7.03 [+5.13, +8.94]"),
             ("Adaptive GNN (P2 recipe: lr 0.01, clip 5)", "agj", "agb_old", "agb_new", "31.43 ± 0.60", "--"),
             ("Adaptive GNN (lr 0.1, candidate recipe)", "agj1", "agb1_old", "agb1_new", "(in flight)", "--")]
    L.append("| arch | joint TEST | bone TEST old (unmasked) | bone TEST new (masked) | P2 bone |")
    L.append("|---|--:|--:|--:|--:|")
    out = {}
    for name, j, bo, bn, p2, _ in archs:
        row = []
        for k in (j, bo, bn):
            v = [acc(pk(f"{k}_s{s}_test"), LAB["test"]) for s in SEEDS]
            out[(name, k)] = v
            row.append(f"{per(v)} → **{ms(v)}**")
        L.append(f"| {name} | " + " | ".join(row) + f" | {p2} |")
    L.append("")
    L.append("Fusion (3-seed ensembles, paired clip bootstrap over 1,365 TEST clips):\n")
    L.append("| arch | bone | joint ens | bone ens | fused α=1 | fused−joint α=1 | per-seed fused α=1 (mean ± SD) | α (val) | fused α_val TEST | fused−joint α_val |")
    L.append("|---|---|--:|--:|--:|---|--:|--:|--:|---|")
    for name, j, bo, bn, p2, p2f in archs:
        for tag, b in (("old (unmasked)", bo), ("new (masked)", bn)):
            J = ens([f"{j}_s{s}_test" for s in SEEDS], LAB["test"]); B = ens([f"{b}_s{s}_test" for s in SEEDS], LAB["test"])
            Jv = ens([f"{j}_s{s}_val" for s in SEEDS], LAB["val"]); Bv = ens([f"{b}_s{s}_val" for s in SEEDS], LAB["val"])
            if J is None or B is None:
                L.append(f"| {name} | {tag} | -- | -- | -- | -- | -- | -- | -- | -- |"); continue
            clips = sorted(J); y = LAB["test"]
            jc = corr(J, y, clips); bc = corr(B, y, clips)
            fc = np.array([int(np.argmax(J[c] + B[c]) == y[c]) for c in clips])
            pf = []
            for s in SEEDS:
                js, bs = pk(f"{j}_s{s}_test"), pk(f"{b}_s{s}_test")
                pf.append(100 * np.mean([np.argmax(js[c] + bs[c]) == y[c] for c in clips]))
            av, fva, dva = "--", "--", "--"
            if Jv is not None and Bv is not None:
                vc = sorted(Jv); grid = np.round(np.arange(0, 3.0001, 0.05), 2)
                va = [np.mean([np.argmax(Jv[c] + a * Bv[c]) == LAB["val"][c] for c in vc]) for a in grid]
                best = max(va); cand = [a for a, v in zip(grid, va) if v == best]
                a_star = min(cand, key=lambda a: abs(a - 1.0))
                fa = np.array([int(np.argmax(J[c] + a_star * B[c]) == y[c]) for c in clips])
                av, fva, dva = f"{a_star:.2f} (val {100 * best:.2f})", f"{100 * fa.mean():.2f}", fmt_boot(boot(fa, jc))
            L.append(f"| {name} | {tag} | {100 * jc.mean():.2f} | {100 * bc.mean():.2f} | {100 * fc.mean():.2f} | "
                     f"{fmt_boot(boot(fc, jc))} | {ms(pf)} | {av} | {fva} | {dva} |")
        # masked vs unmasked bone, same seeds
        Bo = ens([f"{bo}_s{s}_test" for s in SEEDS], LAB["test"]); Bn = ens([f"{bn}_s{s}_test" for s in SEEDS], LAB["test"])
        if Bo is not None and Bn is not None:
            clips = sorted(Bo)
            L.append(f"| {name} | masked − unmasked bone (ensembles) | | | | {fmt_boot(boot(corr(Bn, LAB['test'], clips), corr(Bo, LAB['test'], clips)))} | | | | |")
    L.append("\nP2 (BlockGCN): bone 65.54 ± 1.63; joint 71.72 → fused 78.75 ensemble, fused − joint +7.03 [+5.13, +8.94].\n")


# ------------------------------------------------------------------ B
def csv_final(exp, ep=99):
    v = None
    for r in csv.reader(open("results_final.csv")):
        if len(r) > 3 and r[1] == exp and r[2] == str(ep): v = float(r[3]) * 100
    return v


def task_b(L):
    L.append("## B. Masked cue perturbations (BlockGCN lr 0.01, final-epoch held-out TEST, P2 estimator)\n")
    tl = LAB["test"]
    ctl = [acc(pk(f"cuectl_s{s}_test"), tl) for s in SEEDS]
    L.append("| cue | per seed | mean ± SD | Δ vs control | P2 |")
    L.append("|---|---|--:|--:|--:|")
    cm = st.mean([x for x in ctl if x is not None]) if any(ctl) else 68.86
    L.append(f"| none (control, final epoch) | {per(ctl)} | {ms(ctl)} | -- | 68.9 ± 0.8 |")
    rows = [("speed-normalized (P2, unmasked)", "bdsl_cue_speednorm_lr01_seed{s}", "cue_speed_old", "56.0 ± 1.3"),
            ("speed-normalized, MASKED", "bdsl_cue_speednorm_masked_lr01_seed{s}", "cue_speed_new", ""),
            ("root-centred (P2, unmasked)", "bdsl_cue_rootcenter_lr01_seed{s}", "cue_root_old", "70.3 ± 1.1"),
            ("root-centred, MASKED", "bdsl_cue_rootcenter_masked_lr01_seed{s}", "cue_root_new", "")]
    for name, exp, key, p2 in rows:
        v = [csv_final(exp.format(s=s)) for s in SEEDS]
        vs = [acc(pk(f"{key}_s{s}_test"), tl) for s in SEEDS]
        if all(x is None for x in v): v = vs
        d = f"{st.mean([x for x in v if x is not None]) - cm:+.2f}" if any(x is not None for x in v) else "--"
        chk = "" if all(a is None or b is None or abs(a - b) < 0.01 for a, b in zip(v, vs)) else " (CSV≠rescore!)"
        L.append(f"| {name} | {per(v)} | **{ms(v)}**{chk} | {d} | {p2} |")
    L.append("\nPaired clip bootstrap on the 3-seed final-epoch ensembles (TEST):\n")
    C = ens([f"cuectl_s{s}_test" for s in SEEDS], tl)
    for nm, a, b in (("speed masked − speed unmasked", "cue_speed_new", "cue_speed_old"),
                     ("speed masked − control", "cue_speed_new", None), ("speed unmasked − control", "cue_speed_old", None),
                     ("root masked − root unmasked", "cue_root_new", "cue_root_old"),
                     ("root masked − control", "cue_root_new", None), ("root unmasked − control", "cue_root_old", None)):
        A = ens([f"{a}_s{s}_test" for s in SEEDS], tl); B = C if b is None else ens([f"{b}_s{s}_test" for s in SEEDS], tl)
        if A is None or B is None: L.append(f"- {nm}: --"); continue
        clips = sorted(A)
        L.append(f"- {nm}: ensemble {100 * corr(A, tl, clips).mean():.2f} vs {100 * corr(B, tl, clips).mean():.2f}: "
                 f"{fmt_boot(boot(corr(A, tl, clips), corr(B, tl, clips)))}")
    L.append("")


# ------------------------------------------------------------------ C
P2_LEAK = {"sup_blockgcn_lr01": "0.853 ± 0.021 / 0.975 ± 0.002", "sup_adaptive_lr0p01_s0_m75": "0.911 / 0.986",
           "ssl_bdsl_only": "0.913 / 0.979", "ssl_xlingual": "0.871 / 0.967", "ssl_dgxl": "0.824 / 0.957",
           "grl_lr01_l01": "0.804 (cc)", "grl_lr01_l05": "0.832 (cc)", "grl_lr01_l10": "0.832 (cc)"}


def task_c(L):
    L.append("## C. Signer-decodability probe v2 (tools/decodability_probe_v2.py)\n")
    def load(p):
        return {json.loads(l)["tag"]: json.loads(l) for l in open(p)} if os.path.exists(p) else {}
    v2, v1, vv = load(f"{D}/raw/probe_v2.jsonl"), load(f"{D}/raw/probe_v1.jsonl"), load(f"{D}/raw/probe_v2_withval.jsonl")
    if not v2: L.append("(pending)\n"); return
    base = v2.get("rawmean_unmasked", {}); basem = v2.get("rawmean_masked", {})
    n, S = next(iter(v2.values()))["n"], next(iter(v2.values()))["S"]
    L.append(f"v2 = feeder-processed test-time input (120-frame centre crop, root-normalised), held-out clips (SI TEST + "
             f"labelled pretrain pool: n={n}, S={S} signers, none seen in any supervised training), recording-grouped "
             f"StratifiedGroupKFold, mean of 3 CV seeds. v1 = the P2 probe protocol (raw 300-frame arrays of the first 4000 "
             f"SI-TRAIN clips, ungrouped CV) re-run here for the same rows. wc is ungrouped by construction (one recording per "
             f"signer×class). Δ = v2 cc minus the raw-input baseline (unmasked / masked joint means).\n")
    L.append("| encoder | flag | v2 cc | v2 marginal | v2 wc* | v2 cc ungrouped | Δcc vs raw unmasked / masked | v2+val cc | v1 cc / mrg / wc | P2 (cc / wc) |")
    L.append("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    order = ["rawmean_unmasked", "rawmean_masked", "missrate_len", "randinit_blockgcn", "randinit_adaptive",
             "sup_blockgcn_lr01_s0", "sup_blockgcn_lr01_s1", "sup_blockgcn_lr01_s2",
             "sup_adaptive_lr0p01_s0_m75", "sup_adaptive_lr0p01_s0_m59", "sup_adaptive_lr0p01_s1_m59", "sup_adaptive_lr0p01_s2_m89",
             "sup_adaptive_lr0p1_s0", "sup_adaptive_lr0p1_s1", "sup_adaptive_lr0p1_s2", "ssl_bdsl_only",
             "grl_lr01_l01", "grl_lr01_l05", "grl_lr01_l10", "ssl_xlingual", "ssl_dgxl"]
    f = lambda d, k: f"{d[k]:.3f}" if d and k in d else "--"
    for t in order:
        r = v2.get(t)
        if not r: continue
        dl = (f"{r['cc'] - base['cc']:+.3f} / {r['cc'] - basem['cc']:+.3f}" if base and basem else "--")
        o = v1.get(t)
        old = f"{f(o, 'cc')} / {f(o, 'mrg')} / {f(o, 'wc')}" if o else "--"
        p2 = P2_LEAK.get(t, P2_LEAK.get(re.sub(r"_s\d$", "", t), "")) if not t.startswith("sup_blockgcn") else P2_LEAK["sup_blockgcn_lr01"] + " (3-seed)"
        L.append(f"| {t} | {r.get('flag', '')} | {f(r, 'cc')} | {f(r, 'mrg')} | {f(r, 'wc')} | {f(r, 'cc_ungrouped')} | {dl} | "
                 f"{f(vv.get(t), 'cc')} | {old} | {p2} |")
    def grp(prefix, k):
        v = [v2[t][k] for t in v2 if t.startswith(prefix)]
        return ms(v, 3) if v else "--"
    L.append("")
    L.append(f"Seed summaries (v2): supervised BlockGCN lr0.01 cc {grp('sup_blockgcn_lr01_s', 'cc')}, wc {grp('sup_blockgcn_lr01_s', 'wc')}; "
             f"Adaptive lr0.01 best ckpts (m59/m59/m89) cc {ms([v2[t]['cc'] for t in ('sup_adaptive_lr0p01_s0_m59', 'sup_adaptive_lr0p01_s1_m59', 'sup_adaptive_lr0p01_s2_m89') if t in v2], 3)}; "
             f"Adaptive lr0.1 cc {grp('sup_adaptive_lr0p1_s', 'cc')}.")
    L.append("\n*wc: within-class signer probe; ungrouped by necessity (each signer has one recording per class), so it measures "
             "same-recording trial similarity, not identity transfer to new recordings.\n")


# ------------------------------------------------------------------ D
def task_d(L):
    L.append("## D. Validation-selected matched-split SMG (Adaptive GNN, lr 0.01 = P2 recipe, seed 0)\n")
    old = {42: 99.19, 43: 99.34, 44: 98.61, 45: 98.83, 46: 98.90}
    idx = {}
    if os.path.exists(f"{SC}/index.jsonl"):
        for l in open(f"{SC}/index.jsonl"):
            r = json.loads(l); idx[r["key"]] = r["top1"]
    L.append("| split | P2: best epoch on own TEST | new run: val-selected → TEST | new run: best on own TEST (all ckpts) | new run: final epoch TEST | new run: best val (epoch) |")
    L.append("|---|--:|--:|--:|--:|--:|")
    vs, bt, fe = [], [], []
    for r in range(42, 47):
        v = idx.get(f"rand{r}_valsel_test")
        sp = f"{SC}/rand{r}_scan.json"
        b = fin = bv = None
        if os.path.exists(sp):
            sc = {int(k): x for k, x in json.load(open(sp)).items()}
            b = max(x["test"] for x in sc.values()); fin = sc[max(sc)]["test"]
            e = max(sc, key=lambda k: sc[k]["val"]); bv = f"{sc[e]['val']:.2f} (ep {e})"
        vs.append(v); bt.append(b); fe.append(fin)
        g = lambda z: "--" if z is None else f"{z:.2f}"
        L.append(f"| rand{r} | {old[r]:.2f} | {g(v)} | {g(b)} | {g(fin)} | {bv or '--'} |")
    def mm(x):
        x = [a for a in x if a is not None]
        return f"{np.mean(x):.2f} [{min(x):.2f}, {max(x):.2f}] (n={len(x)})" if x else "--"
    L.append(f"| **mean [min, max]** | 98.97 [98.61, 99.34] | **{mm(vs)}** | {mm(bt)} | {mm(fe)} | |")
    x = [a for a in vs if a is not None]
    if x:
        L.append(f"\nSMG vs held-out TEST 79.07 (Adaptive GNN lr 0.01, 3-seed mean): P2 (test-selected) 98.97 − 79.07 = +19.90; "
                 f"**val-selected {np.mean(x):.2f} − 79.07 = {np.mean(x) - 79.07:+.2f} pp**.")
    L.append("")


# ------------------------------------------------------------------ E
def task_e(L):
    L.append("## E. Paired-instance SMG decomposition: signer familiarity vs same-recording familiarity\n")
    L.append("BlockGCN lr 0.01 (P2 paired protocol: select on U15, score best ckpt once on the frozen focus-signer clips; seeds 0/1/2). "
             "`seenother` = the focus signer's clips of 30 classes in training (matched swap, |train| = 5,748), tested on the "
             "frozen clips of the OTHER 30 classes; halves A and B together cover exactly the frozen set. Accuracies are on the "
             "identical frozen clips for all three conditions.\n")
    W = "work_dir"; stem = "bdsl_block_gcn_si_lr01_pairsmg"
    L.append("| focus | seed | unseen | seen via other classes | seen_matched (same class/recording) | signer fam. (other−unseen) | same-recording fam. (matched−other) | paired gap (matched−unseen) |")
    L.append("|---|---|--:|--:|--:|--:|--:|--:|")
    allc = collections.defaultdict(list)
    summ = {}
    for f in (2, 13):
        lab = None; per_seed = []
        for s in SEEDS:
            def ld(cond):
                p = f"{W}/{stem}_U{f:02d}_{cond}_seed{s}_test/eval_results/best_acc.pkl"
                return pickle.load(open(p, "rb")) if os.path.exists(p) else None
            u, m, oa, ob = ld("unseen"), ld("seen_matched"), ld("seenother_A"), ld("seenother_B")
            if lab is None:
                n, l = pickle.load(open(f"data/bdsl_si_pairsmg/U{f:02d}_unseen/test_label.pkl", "rb")); lab = dict(zip(n, l))
            if None in (u, m, oa, ob):
                L.append(f"| U{f:02d} | {s} | -- | -- | -- | -- | -- | -- |"); continue
            o = {**oa, **ob}
            assert set(o) == set(u) == set(m), "frozen sets differ"
            clips = sorted(u)
            cu, cm_, co = corr(u, lab, clips), corr(m, lab, clips), corr(o, lab, clips)
            per_seed.append((cu, co, cm_))
            a = [100 * x.mean() for x in (cu, co, cm_)]
            L.append(f"| U{f:02d} | {s} | {a[0]:.2f} | {a[1]:.2f} | {a[2]:.2f} | {a[1] - a[0]:+.2f} | {a[2] - a[1]:+.2f} | {a[2] - a[0]:+.2f} |")
        if len(per_seed) == 3:
            cu, co, cm_ = (np.mean([p[i] for p in per_seed], 0) for i in range(3))
            summ[f] = (cu, co, cm_)
            for i, x in enumerate((cu, co, cm_)): allc[i].append(x)
            mu = lambda i: ms([100 * p[i].mean() for p in per_seed])
            L.append(f"| **U{f:02d} mean ± SD** | | {mu(0)} | {mu(1)} | {mu(2)} | "
                     f"{ms([100 * (p[1].mean() - p[0].mean()) for p in per_seed])} | {ms([100 * (p[2].mean() - p[1].mean()) for p in per_seed])} | "
                     f"{ms([100 * (p[2].mean() - p[0].mean()) for p in per_seed])} |")
    L.append("\nPer-clip paired bootstrap on the 3-seed-mean correctness of each frozen clip:\n")
    for f, (cu, co, cm_) in summ.items():
        L.append(f"- U{f:02d}: signer familiarity (other − unseen) {fmt_boot(boot(co, cu))}; same-recording (matched − other) "
                 f"{fmt_boot(boot(cm_, co))}; total paired gap {fmt_boot(boot(cm_, cu))}")
    if len(summ) == 2:
        cu, co, cm_ = (np.concatenate(allc[i]) for i in range(3))
        tot = (cm_.mean() - cu.mean()) * 100
        L.append(f"- pooled U02+U13 ({len(cu)} clips): signer familiarity {fmt_boot(boot(co, cu))}; same-recording "
                 f"{fmt_boot(boot(cm_, co))}; total {fmt_boot(boot(cm_, cu))}; share of the paired gap explained by "
                 f"signer familiarity alone: {100 * (co.mean() - cu.mean()) * 100 / tot:.0f}% (clips pooled; per-signer "
                 f"means are the independent unit, n=2)")
    L.append("\nP2: paired gap +30.36 ± 6.06 (U02 +25.41 ± 3.17, U13 +35.30 ± 3.81).\n")


if __name__ == "__main__":
    L = ["# P2 missing-joint fixes — generated results (tools/p2_mjfix_report.py)\n"]
    for t in (task_a, task_b, task_c, task_d, task_e):
        try:
            t(L)
        except Exception as e:  # keep the other sections
            import traceback; L.append(f"\n**{t.__name__} failed: {e!r}**\n```\n{traceback.format_exc()}\n```\n")
    open(f"{D}/results.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
