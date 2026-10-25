#!/usr/bin/env python
"""Aggregate the P2 Adaptive GNN / Attention GNN lr-0.1 switch: old (lr 0.01) vs new (lr 0.1).

Reads results_final.csv, the evals in docs/adaptive_attn_lr0p1_switch/raw/ (tools/lr0p1_switch_eval.py,
scripts/hpc/lr0p1_switch/probes.sh), docs/loso_heldout/, and the TEST score dumps; recomputes the
metric suite (tools/p2_metric_suite.metrics_for), significance (tools/arch_significance.py --lr0p1)
and the LOSO variance decomposition (tools/variance_decompose.py). The same code computes the old
column, which therefore checks the aggregation against the numbers printed in P2.
  python tools/lr0p1_switch_report.py
"""
import csv, collections, json, os, re, statistics as st, subprocess, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import tools.p2_metric_suite as M  # noqa: E402

OUT = ROOT / "docs" / "adaptive_attn_lr0p1_switch"
RAW = OUT / "raw"
DER = ROOT / "docs" / "loso_heldout" / "derived"
SEEDS = (0, 1, 2)
PRETTY = {"adaptive_gnn": "Adaptive GNN", "gnn": "Attention GNN"}
ARCHS = ("adaptive_gnn", "gnn")
# P2 numbers this switch can move (paper as of 2026-10-11)
P2 = {"blockgcn_test": 68.82, "videomae": (80.93, 2.11), "unisign": (83.61, 1.08),
      "trackB": {"adaptive_gnn": (77.49, 1.72)}, "slgt_trackB": 83.91,
      "leaderboard_rest": {"GNN+Transformer": 75.55, "ST-GCN": 74.51, "SLGTFormer": 73.26,
                           "GNN+Bi-LSTM": 71.57, "BlockGCN": 68.82, "Pose-LSTM": 61.64}}

rows = collections.defaultdict(list)          # exp -> [(epoch, top1%)]
for r in csv.reader(open(ROOT / "results_final.csv")):
    if len(r) < 4: continue
    try: rows[r[1]].append((int(r[2]), float(r[3]) * 100))
    except ValueError: pass


def best(exp): return max(v for _, v in rows[exp]) if rows.get(exp) else None
def lastrow(exp): return rows[exp][-1][1] if rows.get(exp) else None
def ms(xs, nd=2):
    xs = [x for x in xs if x is not None]
    if not xs: return "--"
    return f"{st.mean(xs):.{nd}f} ± {st.pstdev(xs):.{nd}f}" if len(xs) > 1 else f"{xs[0]:.{nd}f}"
def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None
def per(xs): return " / ".join("--" if x is None else f"{x:.2f}" for x in xs)
def raw(section):
    p = RAW / f"{section}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []
def lastby(section, *keys):
    d = {}
    for r in raw(section): d[tuple(r[k] for k in keys)] = r     # later lines win (idempotent re-runs)
    return d
def write(name, title, lines):
    (OUT / name).write_text(f"# {title}\n\n" + "\n".join(lines) + "\n"); print("wrote", name)


SUMMARY = {}   # key -> (old, new) strings for the README


# ---------------------------------------------------------------- 1. Protocol A + metric suite
def test_top1(a, new):
    return [lastrow(f"bdsl_{a}_si{'_lr0p1' if new else ''}_seed{s}_test") for s in SEEDS]


def suite(pkls):
    lab = M.load_labels()
    orig = M.find_seed_pkls
    M.find_seed_pkls = lambda stem: pkls
    try: return M.metrics_for("x", lab)
    finally: M.find_seed_pkls = orig


def pk(a, new, s):
    p = ROOT / f"work_dir/bdsl_{a}_si{'_lr0p1' if new else ''}_seed{s}_test/eval_results/best_acc.pkl"
    if not p.exists() and a == "adaptive_gnn" and not new and s == 2:
        p = RAW / "old_lr0p01_adaptive_gnn_seed2_test_best_acc.pkl"
    return str(p) if p.exists() else None


def protoA():
    L = ["Protocol A, BdSLW60 held-out TEST signers 2/13 (1,365 clips). New = the 3 existing lr-0.1 runs "
         "(`config/lrsweep/bdsl_<arch>_si_lr0p1.yaml`, best checkpoint on val U15) scored with "
         "`main.py --phase test --config config/bdsl_<arch>_si_test.yaml` -> "
         "`work_dir/bdsl_<arch>_si_lr0p1_seed{S}_test/eval_results/best_acc.pkl`. Old = the P2 runs "
         "(`work_dir/bdsl_<arch>_si_seed{S}_test`). ± = population SD over 3 seeds.", "",
         "| arch | lr | per seed (TEST Top-1) | mean ± SD | best-val epoch / val Top-1 |", "|---|---|---|--:|---|"]
    pa = lastby("protoA", "arch", "seed")
    for a in ARCHS:
        o, n = test_top1(a, False), test_top1(a, True)
        ep = " / ".join(f"{pa[(a, s)]['best_val_epoch']}" if (a, s) in pa else "--" for s in SEEDS)
        vb = [best(f"bdsl_{a}_si_lr0p1_seed{s}") for s in SEEDS]
        L.append(f"| {PRETTY[a]} | 0.01 (old) | {per(o)} | {ms(o)} | |")
        L.append(f"| {PRETTY[a]} | **0.1 (new)** | {per(n)} | **{ms(n)}** | ep {ep}; val {per(vb)} (mean {mean(vb) or 0:.2f}) |")
        rc = [pa[(a, s)]["test_top1_recheck"] if (a, s) in pa else None for s in SEEDS]
        L.append(f"| {PRETTY[a]} | 0.1 recheck (side-effect-free scorer) | {per(rc)} | {ms(rc)} | |")
        SUMMARY[f"test_{a}"] = (ms(o), ms(n))
    L += ["", "## Metric suite (tab:suite, §metrics calibration/per-signer paragraphs)", "",
          "`tools/p2_metric_suite.metrics_for` on the TEST score dumps: single-model 3-seed mean (± pop. SD for Top-1); "
          "ens = 3-seed softmax ensemble. Old Adaptive GNN in P2 used only seeds 0/1 (its seed-2 `_test/eval_results/` "
          "is empty); `old (3 seeds)` adds a side-effect-free seed-2 dump of the same checkpoint (model-89) "
          "the seed-2 `_test` run loaded.", "",
          "| arch | variant | seeds | Top-1 ± SD | Top-1 ens | Top-5 | Macro-F1 | MRR | R@10 | ECE % | ECE ens % | NLL | Brier | per-signer ens U13 / U2 |",
          "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|"]
    out = {}
    for a in ARCHS:
        variants = [("old (P2)", [pk(a, False, s) for s in SEEDS if pk(a, False, s) and not (a == "adaptive_gnn" and s == 2)])]
        if a == "adaptive_gnn":
            variants.append(("old (3 seeds)", [pk(a, False, s) for s in SEEDS if pk(a, False, s)]))
        variants.append(("**new lr 0.1**", [pk(a, True, s) for s in SEEDS if pk(a, True, s)]))
        for tag, pl in variants:
            m = suite(pl) if pl else None
            if not m: L.append(f"| {PRETTY[a]} | {tag} | 0 | -- |"); continue
            s, e = m["single"], m["ens"]; ps = m["per_signer"]
            L.append(f"| {PRETTY[a]} | {tag} | {m['seeds']} | {s['top1'][0]:.2f} ± {s['top1'][1]:.2f} | {e['top1']:.2f} | "
                     f"{s['top5'][0]:.2f} | {s['macrof1'][0]:.2f} | {s['mrr'][0]:.3f} | {s['r10'][0]:.2f} | {s['ece'][0]:.2f} | "
                     f"{e['ece']:.2f} | {s['nll'][0]:.3f} | {s['brier'][0]:.3f} | "
                     f"{ps.get('13', (float('nan'),))[0]:.1f} / {ps.get('2', ps.get('02', (float('nan'),)))[0]:.1f} |")
            out[(a, tag)] = m
    for a in ARCHS:
        o = out.get((a, "old (P2)")); n = out.get((a, "**new lr 0.1**"))
        if o and n:
            SUMMARY[f"f1_{a}"] = (f"{o['single']['macrof1'][0]:.2f}", f"{n['single']['macrof1'][0]:.2f}")
            SUMMARY[f"suite_{a}"] = (
                f"ens {o['ens']['top1']:.2f}, MRR {o['single']['mrr'][0]:.3f}, R@10 {o['single']['r10'][0]:.2f}, NLL {o['single']['nll'][0]:.3f}, Brier {o['single']['brier'][0]:.3f}, ECE {o['single']['ece'][0]:.2f} (ens {o['ens']['ece']:.2f})",
                f"ens {n['ens']['top1']:.2f}, MRR {n['single']['mrr'][0]:.3f}, R@10 {n['single']['r10'][0]:.2f}, NLL {n['single']['nll'][0]:.3f}, Brier {n['single']['brier'][0]:.3f}, ECE {n['single']['ece'][0]:.2f} (ens {n['ens']['ece']:.2f})")
    json.dump({f"{a}|{t}": v for (a, t), v in out.items()}, open(RAW / "metric_suite.json", "w"), indent=1, default=float)
    write("01_protocolA_metric_suite.md", "1 — Protocol A held-out TEST + metric suite (Table 7, tab:suite, efficiency, tracks)", L)


# ---------------------------------------------------------------- 2. LOSO
def loso_load(f):
    d = collections.defaultdict(dict)
    p = ROOT / "docs" / "loso_heldout" / f
    if not p.exists(): return d
    for r in csv.DictReader(open(p)):
        if r["test_acc"]: d[r["heldout_signer"]][int(r["seed"])] = float(r["test_acc"])
    return d


def loso_stats(d):
    fm = {s: np.mean(list(v.values())) for s, v in d.items()}; x = np.array(list(fm.values()))
    rng = np.random.default_rng(1234); bs = [rng.choice(x, len(x)).mean() for _ in range(10000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(mean=x.mean(), lo=lo, hi=hi, mn=x.min(), mx=x.max(), sd=x.std(ddof=1), folds=len(x),
                runs=sum(len(v) for v in d.values()), fm=fm)


def loso():
    L = ["Held-out-signer Top-1 per fold (each fold's best-on-fold-val checkpoint scored on the fold's held-out "
         "TEST signer, as `scripts/hpc/slurm_loso_heldout_allarch.sbatch`); fold value = mean over seeds; stats over "
         "the 11 folds; 95% CI = signer-clustered percentile bootstrap over folds (10k, seed 1234; the estimator of "
         "docs/baselines_rerun/item3_loso_all8.md and the BlockGCN \"74.4±16.1 [64.7, 82.9]\"); SD = sample SD "
         "across folds (Table 7 LOSO column = mean ± this SD). New per-fold CSVs: "
         "`docs/loso_heldout/{adaptive_gnn,gnn}_lr0p1.csv` (old: `{adaptive_gnn,gnn}.csv`, untouched).", "",
         "| arch | lr | mean | 95% CI | min–max | SD (sample) | folds | seed-runs | Table 7 cell |",
         "|---|---|--:|--:|--:|--:|--:|--:|--:|"]
    per_fold = {}
    for a in ARCHS:
        for tag, f in [("0.01 (old)", f"{a}.csv"), ("**0.1 (new)**", f"{a}_lr0p1.csv")]:
            d = loso_load(f)
            if not d: L.append(f"| {PRETTY[a]} | {tag} | -- | | | | | | |"); continue
            S = loso_stats(d)
            L.append(f"| {PRETTY[a]} | {tag} | {S['mean']:.2f} | [{S['lo']:.2f}, {S['hi']:.2f}] | {S['mn']:.1f}–{S['mx']:.1f} | "
                     f"{S['sd']:.2f} | {S['folds']} | {S['runs']} | {S['mean']:.1f}±{S['sd']:.1f} |")
            per_fold[(a, tag)] = d
            SUMMARY.setdefault(f"loso_{a}", ["", ""])[0 if "old" in tag else 1] = \
                f"{S['mean']:.2f} [{S['lo']:.2f}, {S['hi']:.2f}], {S['mn']:.1f}–{S['mx']:.1f}, SD {S['sd']:.2f} ({S['runs']} runs)"
    L += ["", "Per fold (seed 0/1/2), folds sorted by signer:", ""]
    for (a, tag), d in per_fold.items():
        L.append(f"- {PRETTY[a]} {tag}: " + "; ".join(f"{s}: " + "/".join(f"{d[s][k]:.1f}" for k in sorted(d[s]))
                                                     for s in sorted(d, key=lambda z: int(z[1:]))))
    write("02_loso.md", "2 — LOSO, 11 held-out signers (Table 7 LOSO column, sec:archtable signer-clustered ranking)", L)


# ---------------------------------------------------------------- 3. Mixed / SMG
def sd_mixed():
    sdr = lastby("sd", "arch", "seed")
    L = ["Signer-mixed `data/bdsl_sd`, unclipped, symmetric rule: checkpoint selected on sd val, scored once on sd "
         "test (1,367 clips) via `tools/run_resumable_seed.py --test-config` "
         "(`config/lr0p1_switch/bdsl_<arch>_sd_noclip_lr0p1{,_test}.yaml`). Old = `bdsl_<arch>_sd_noclip_test_seed*` "
         "(lr 0.01). SMG = Mixed − Protocol-A TEST mean (same lr on both sides).", "",
         "| arch | lr | Mixed per seed | Mixed mean ± SD | recheck (side-effect-free) | sd-val best | TEST mean | **SMG** |",
         "|---|---|---|--:|---|---|--:|--:|"]
    for a in ARCHS:
        for new in (False, True):
            stem = f"bdsl_{a}_sd_noclip{'_lr0p1' if new else ''}"
            mx = [lastrow(f"{stem}_test_seed{s}") for s in SEEDS]
            vb = [best(f"{stem}_seed{s}") for s in SEEDS]
            rc = [sdr[(a, s)]["sd_test"] if new and (a, s) in sdr else None for s in SEEDS]
            t = mean(test_top1(a, new))
            smg = (mean(mx) - t) if (mean(mx) is not None and t is not None and None not in mx) else None
            L.append(f"| {PRETTY[a]} | {'**0.1 (new)**' if new else '0.01 (old)'} | {per(mx)} | {ms(mx)} | {per(rc) if new else ''} | "
                     f"{per(vb)} | {t if t is None else f'{t:.2f}'} | {'--' if smg is None else f'{smg:+.2f}'} |")
            SUMMARY.setdefault(f"mixed_{a}", ["", ""])[int(new)] = f"Mixed {ms(mx)}, SMG {'--' if smg is None else f'{smg:+.2f}'}"
    write("03_sd_mixed.md", "3 — Signer-mixed split, Mixed/SMG columns (Table 7, tab:smg8)", L)


# ---------------------------------------------------------------- 4. matched random splits
def rand_splits():
    new = lastby("rand", "split")
    L = ["Five matched random signer-mixed splits `data/bdsl_rand42..46`, Adaptive GNN, one model per split (seed 0). "
         "Old (P2) = lr 0.01, test feeder = the split's own test, best of the every-5-epoch evals (epochs 4, 9, …, 99). "
         "New = lr 0.1 (`config/lr0p1_switch/bdsl_adaptive_gnn_rand4X_lr0p1.yaml`), selected on the split's own val as "
         "in the BlockGCN lr-0.01 switch, with checkpoints saved at exactly the old eval epochs (save_interval 5), so the "
         "paper's estimator (best epoch on own test) is recomputed on the same epoch set.", "",
         "| split | old lr 0.01 best-on-own-test | new lr 0.1 best-on-own-test (same 20 epochs) | new val-selected → test | new val best |",
         "|---|--:|--:|--:|--:|"]
    o, a, v = [], [], []
    for r in range(42, 47):
        ob = best(f"bdsl_adaptive_gnn_rand{r}_seed0"); o.append(ob)
        nr = new.get((r,), {}); a.append(nr.get("old_estimator_max_test")); v.append(nr.get("val_selected_test"))
        f = lambda x: "--" if x is None else f"{x:.2f}"
        L.append(f"| rand{r} | {f(ob)} | {f(a[-1])} | {f(v[-1])} | {f(nr.get('val_best'))} |")
    def rng(x):
        x = [y for y in x if y is not None]
        return f"{st.mean(x):.2f} [{min(x):.1f}, {max(x):.1f}]" if x else "--"
    L += [f"| **mean [min, max]** | {rng(o)} | **{rng(a)}** | {rng(v)} | |", ""]
    t_old, t_new = mean(test_top1("adaptive_gnn", False)), mean(test_top1("adaptive_gnn", True))
    L.append(f"Headline Adaptive GNN SMG (tab:smg, Fig. 1, abstract): old {mean(o):.2f} − {t_old:.2f} = {mean(o) - t_old:+.2f} (P2: +19.90).")
    if all(x is not None for x in a) and t_new is not None:
        L.append(f"New, same estimator: {mean(a):.2f} − {t_new:.2f} = **{mean(a) - t_new:+.2f}**; "
                 f"val-selected: {mean(v):.2f} − {t_new:.2f} = {mean(v) - t_new:+.2f}.")
        SUMMARY["matched_smg"] = (f"{mean(o):.2f} [{min(o):.1f}, {max(o):.1f}] − 79.07 = +19.90",
                                  f"{mean(a):.2f} [{min(a):.1f}, {max(a):.1f}] − {t_new:.2f} = {mean(a) - t_new:+.2f} (val-selected {mean(v):.2f} → {mean(v) - t_new:+.2f})")
    write("04_matched_splits.md", "4 — Matched random splits, Adaptive GNN (tab:smg, Fig. 1, abstract, conclusion)", L)


# ---------------------------------------------------------------- 5. leak probe
def probes():
    p = RAW / "probes.log"
    vals = collections.defaultdict(dict)
    if p.exists():
        for line in open(p):
            m = re.search(r"\[([^|\]]+)\|(cc|mrg|wc)\]", line)
            if not m: continue
            tag, kind = m.group(1), m.group(2)
            v = re.search(r"L_norm\^cond = ([0-9.]+)", line) if kind == "wc" else re.search(r"L_norm\s*=\s*([0-9.]+)", line)
            if v: vals[tag][kind] = float(v.group(1))
    L = ["Same probe settings as docs/audit_reruns_2026-10/r4_lnorm.md and the BlockGCN lr-0.01 switch "
         "(cc = `--class-conditioned` n=4000, the P2 leak-table \\Lnorm; marginal n=6000; within-class n=6000, "
         "the P2 \\Lnorm^wc column). `config/probe_adaptive_gnn_si.yaml`; new = best-on-val checkpoints of the "
         "lr-0.1 runs. P2's row (0.911 / 0.986) is the single old seed-0 periodic checkpoint m75.", "",
         "| encoder | cc (P2 \\Lnorm) | marginal | within-class (P2 \\Lnorm^wc) |", "|---|--:|--:|--:|",
         "| old P2 row: lr 0.01 seed 0 m75 (r4) | 0.911 | 0.843 | 0.986 |"]
    for tag in sorted(vals):
        d = vals[tag]
        L.append(f"| {tag} | {d.get('cc', float('nan')):.3f} | {d.get('mrg', float('nan')):.3f} | {d.get('wc', float('nan')):.3f} |")
    for lab, pref in [("old lr 0.01, 3 seeds (best ckpts m59/m59/m89)", "sup_adaptive_lr0p01_"), ("**new lr 0.1, 3 seeds**", "sup_adaptive_lr0p1_")]:
        ks = [k for k in vals if k.startswith(pref)]
        if ks:
            cell = lambda kind: ms([vals[k].get(kind) for k in ks], 3)
            L.append(f"| {lab} mean ± SD | {cell('cc')} | {cell('mrg')} | {cell('wc')} |")
            if "lr0p1" in pref: SUMMARY["leak"] = ("0.911 / 0.986 (1 encoder)", f"{cell('cc')} / {cell('wc')} (3 seeds)")
    L += ["", "Raw log: docs/adaptive_attn_lr0p1_switch/raw/probes.log"]
    write("05_leak_probe.md", "5 — Signer decodability of the supervised Adaptive GNN (leak table, sec:leak decoupling)", L)


# ---------------------------------------------------------------- 6. bone
def bone():
    old = [lastrow(f"bdsl_adaptive_gnn_bone_clip5_seed{s}_test") for s in SEEDS]
    new = [lastrow(f"bdsl_adaptive_gnn_bone_lr0p1_test_seed{s}") for s in SEEDS]
    vb = [best(f"bdsl_adaptive_gnn_bone_lr0p1_seed{s}") for s in SEEDS]
    L = ["Adaptive GNN bone stream, held-out TEST (tab:bone). Old = `bdsl_adaptive_gnn_bone_clip5` (lr 0.01, grad clip 5.0). "
         "New = `config/lr0p1_switch/bdsl_adaptive_gnn_bone_lr0p1.yaml` (lr 0.1, no clipping = the new joint recipe; "
         "is_vector False as before), best-on-val checkpoint, `--test-config ..._bone_lr0p1_test.yaml`.", "",
         "| | per seed | mean ± SD |", "|---|---|--:|",
         f"| old bone (lr 0.01, clip 5) | {per(old)} | {ms(old)} |",
         f"| **new bone (lr 0.1, no clip)** | {per(new)} | **{ms(new)}** |",
         f"| new bone val best | {per(vb)} | {ms(vb)} |",
         f"| joint, new (Protocol A lr 0.1) | {per(test_top1('adaptive_gnn', True))} | {ms(test_top1('adaptive_gnn', True))} |", ""]
    f = RAW / "fusion_adaptive_lr0p1.md"
    if f.exists(): L += ["Joint (lr 0.1) + bone (lr 0.1) score fusion, paired clip bootstrap (`tools/fusion_bootstrap.py` procedure):", "", "```", f.read_text().strip(), "```"]
    SUMMARY["bone"] = (ms(old), ms(new))
    write("06_bone.md", "6 — Two-stream: Adaptive GNN bone (tab:bone, app:bone)", L)


# ---------------------------------------------------------------- 7. significance
def significance():
    txt = {}
    for tag, args in [("old", []), ("new", ["--lr0p1"])]:
        r = subprocess.run([sys.executable, "tools/arch_significance.py"] + args, capture_output=True, text=True, cwd=ROOT)
        txt[tag] = r.stdout + (r.stderr[-2000:] if r.returncode else "")
    (DER / "arch_significance_adaptattn_lr0p1.txt").write_text(txt["new"])
    L = ["`tools/arch_significance.py` (per-instance clip bootstrap on the 3-seed TEST ensembles + signer-clustered "
         "LOSO bootstrap, adjacent ranks, Holm within family). Old = current P2 inputs (BlockGCN lr 0.01, "
         "GNN+Transformer clip 5, Adaptive/Attention lr 0.01; note the old Adaptive ensemble has only seeds 0/1). "
         "New = `--lr0p1` (Adaptive + Attention lr-0.1 TEST pkls and LOSO CSVs). Output also in "
         "`docs/loso_heldout/derived/arch_significance_adaptattn_lr0p1.txt`.", "",
         "## New (lr 0.1)", "", "```", txt["new"].strip(), "```", "", "## Old (reproduces P2 tab:sig)", "", "```", txt["old"].strip(), "```"]
    write("07_significance.md", "7 — Pairwise architecture significance (sec:archtable, tab:sig)", L)


# ---------------------------------------------------------------- 8. variance decomposition
def vardecomp():
    src = DER / "heldout_as_results_blockgcn_lr01.csv"
    new = {a: loso_load(f"{a}_lr0p1.csv") for a in ARCHS}
    if not all(sum(len(v) for v in new[a].values()) == 33 for a in ARCHS):
        write("08_variance_decomp.md", "8 — LOSO variance decomposition (sec:loso, tab:losovar)", ["LOSO lr-0.1 incomplete; not run."]); return
    out_csv = DER / "heldout_as_results_adaptattn_lr0p1.csv"
    keep = [l for l in open(src) if not re.match(r"heldout,bdsl_(adaptive_gnn|gnn)_si_loso", l)]
    for a in ARCHS:
        for sig, sd in sorted(new[a].items()):
            for seed, acc in sorted(sd.items()):
                keep.append(f"heldout,bdsl_{a}_si_loso_test{sig[1:]}_seed{seed},0,{acc / 100.0},,,\n")
    out_csv.write_text("".join(keep))
    txt = ""
    for tag, args in [("all 8", []), ("4 arch", ["--archs", "bdsl_block_gcn_si", "bdsl_adaptive_gnn_si", "bdsl_st_gcn_vanilla_si", "bdsl_slgtformer_si"]),
                      ("2 arch", ["--archs", "bdsl_block_gcn_si", "bdsl_adaptive_gnn_si"])]:
        for lab, c in [("NEW", out_csv), ("OLD", src)]:
            r = subprocess.run([sys.executable, "tools/variance_decompose.py", "--csv", str(c)] + args, capture_output=True, text=True, cwd=ROOT)
            txt += f"######## {lab} archs: {tag}\n" + r.stdout + "\n"
    (DER / "variance_decomp_adaptattn_lr0p1.txt").write_text(txt)
    L = ["`tools/variance_decompose.py` on held-out-signer LOSO accuracy (8 archs × 11 signers × 3 seeds; "
         "GNN+Transformer clip 5, BlockGCN lr 0.01). New input "
         "`docs/loso_heldout/derived/heldout_as_results_adaptattn_lr0p1.csv` = the current P2 input "
         "(`heldout_as_results_blockgcn_lr01.csv`) with the Adaptive GNN and Attention GNN cells swapped to lr 0.1. "
         "Full output: `docs/loso_heldout/derived/variance_decomp_adaptattn_lr0p1.txt`.", "",
         "| design | input | σ signer | σ arch | σ interaction | σ seed | signer/arch | signer/seed |", "|---|---|--:|--:|--:|--:|--:|--:|"]
    for block in txt.split("######## ")[1:]:
        head = block.splitlines()[0]
        g = lambda k: (re.search(rf"{k}\s*\([^)]*\)\s*=\s*([0-9.]+)", block) or [None, "nan"])[1]
        r1 = (re.search(r"signer / sigma_model =\s*([0-9.]+)x", block) or [None, "nan"])[1]
        r2 = (re.search(r"signer / sigma_seed\s*=\s*([0-9.]+)x", block) or [None, "nan"])[1]
        lab, _, tag = head.partition(" archs: ")
        L.append(f"| {tag} | {lab} | {g('sigma_signer')} | {g('sigma_model')} | {g('sigma_interact')} | {g('sigma_seed')} | {r1}× | {r2}× |")
        if tag == "all 8":
            SUMMARY.setdefault("vardecomp", ["", ""])[0 if lab == "OLD" else 1] = \
                f"signer {g('sigma_signer')}, arch {g('sigma_model')} ({r1}×), interaction {g('sigma_interact')}, seed {g('sigma_seed')} ({r2}×)"
    write("08_variance_decomp.md", "8 — LOSO variance decomposition (sec:loso, tab:losovar)", L)


# ---------------------------------------------------------------- 9. leaderboard-level claims
def claims():
    n = {a: test_top1(a, True) for a in ARCHS}
    if any(None in v for v in n.values()):
        write("09_claims.md", "9 — Leaderboard-level claims", ["Protocol-A TEST incomplete."]); return
    ad, at = mean(n["adaptive_gnn"]), mean(n["gnn"])
    adsd = st.pstdev(n["adaptive_gnn"])
    board = dict(P2["leaderboard_rest"]); board["Adaptive GNN"] = ad; board["Attention GNN"] = at
    order = sorted(board.items(), key=lambda kv: -kv[1])
    vm, vsd = P2["videomae"]; us, usd = P2["unisign"]; tb, tbsd = P2["trackB"]["adaptive_gnn"]
    L = ["Track-A leaderboard (Table 7 (a) pose rows, efficiency table) with the two archs at lr 0.1 "
         "(other rows as in P2):", "", "| rank | arch | TEST Top-1 |", "|--:|---|--:|"]
    L += [f"| {i + 1} | {k} | {v:.2f} |" for i, (k, v) in enumerate(order)]
    L += ["",
          f"- Leader: **{order[0][0]}** at {order[0][1]:.2f} (P2: Adaptive GNN 79.07 ± 0.89). Top-to-bottom spread "
          f"{order[0][1]:.2f} → {order[-1][1]:.2f} = {order[0][1] - order[-1][1]:.1f} pp (P2: 79.07 → 61.64, \"~17 pp\").",
          f"- Adaptive GNN − BlockGCN = {ad - P2['blockgcn_test']:+.2f} pp (P2 §eff: +10.3; compared there with the RTMPose extractor gain +9.3).",
          f"- VideoMAE {vm} ± {vsd} vs Adaptive GNN {ad:.2f} ± {adsd:.2f}: diff {vm - ad:+.2f} pp "
          f"(P2: \"within seed noise of Adaptive GNN\", diff +1.86 vs 79.07 ± 0.89).",
          f"- Uni-Sign {us} ± {usd} vs best Track-A pose model {order[0][1]:.2f}: Uni-Sign ahead by {us - order[0][1]:+.2f} pp "
          f"(P2: \"above every shared-recipe (Track A) pose model\").",
          f"- Track B (tab:tracks): Adaptive GNN best-practice {tb} ± {tbsd} (`bdsl_adaptive_gnn_si_bp`, also base LR 0.1, batch 64, "
          f"65 epochs); Track A now {ad:.2f} ± {adsd:.2f}, Δ = {tb - ad:+.2f} (P2: −1.58).",
          f"- Pareto (§eff): Adaptive GNN (5.2 M / 5.3 GFLOPs) is Pareto-optimal iff no cheaper model is more accurate; "
          f"{'still the most accurate pose model, so still Pareto-optimal' if order[0][0] == 'Adaptive GNN' else 'NOT the most accurate pose model any more'}. "
          f"Attention GNN (4.3 M / 4.1 GFLOPs) at {at:.2f}: "
          f"{'also on the Pareto front (cheaper than Adaptive GNN, more accurate than every cheaper model)' if all(v < at for k, v in board.items() if k in ('GNN+Transformer', 'ST-GCN', 'GNN+Bi-LSTM', 'BlockGCN', 'Pose-LSTM')) else 'dominated by a cheaper, more accurate model'}.",
          f"- Limitations para: \"SLGTFormer ±4.77 vs Adaptive GNN ±0.89\" → ±{adsd:.2f}."]
    SUMMARY["leader"] = ("Adaptive GNN 79.07", f"{order[0][0]} {order[0][1]:.2f}")
    write("09_claims.md", "9 — Leaderboard-level claims (sec:archtable, sec:eff, app:tracks, limitations)", L)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
    for f in (protoA, loso, sd_mixed, rand_splits, probes, bone, significance, vardecomp, claims):
        try: f()
        except FileNotFoundError as e: print(f"[{f.__name__}] skipped: {type(e).__name__}: {e}")
        except OSError: raise  # unreadable storage must not leave a stale section behind
        except Exception as e: print(f"[{f.__name__}] skipped: {type(e).__name__}: {e}")
    json.dump(SUMMARY, open(RAW / "summary.json", "w"), indent=1)
    print(json.dumps(SUMMARY, indent=1))
