#!/usr/bin/env python
"""Stage-2 tables for the signer-normalized recipe (docs/signernorm_2026-10/README.md).

Baseline values come from tools/p2_make_tables.POSE (the paper's current sources, recipe=None), so the
"P2" columns match the paper generator. Signer-norm (SN) values come from this project's runs:
  TEST   work_dir/<sn_stem>_seed{S}_test/eval_results/best_acc.pkl (best-on-val ckpt, --phase test)
  Mixed  work_dir/bdsl_<arch>_sd_sn_<r>_seed{S}_test/... (best-on-sd-val ckpt scored on sd TEST)
  LOSO   work_dir/bdsl_block_gcn_si_lr01_sn_<r>_loso_test{NN}_seed{S}_test/... -> docs/loso_heldout/
         block_gcn_lr01_sn_<r>.csv, gated by check_run_complete.train_status (complete training run)
Every run's training is gated for completeness (final epoch reached) before its numbers are used.

  python tools/signernorm_stage2_report.py [--recipe abc] [--write-loso]
"""
import argparse, csv, glob, json, os, pickle, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools")); sys.path.insert(0, ROOT)
os.chdir(ROOT)
import p2_make_tables as T  # noqa: E402
from check_run_complete import train_status  # noqa: E402
from signernorm_report import summarize, hier_ci  # noqa: E402
from p2_metric_suite import load_labels  # noqa: E402

SD_LAB = "data/bdsl_sd/test_label.pkl"
RNG = np.random.default_rng(1234)


def gate(runs):
    bad = [r for r in runs if not train_status(os.path.join(ROOT, "work_dir", r))["complete"]]
    return bad


def active_src(spec, recipe="__active__"):
    """The paper's active sources for an arch (src updated by its selected recipe, as p2_make_tables.Ctx)."""
    src = dict(spec["src"])
    rec = spec.get("recipe") if recipe == "__active__" else recipe
    if rec:
        src.update(spec["recipes"][rec])
    return src, rec


def rows_for(r):
    """(arch, label, P2 baseline sources, SN Protocol-A stem, SN Mixed stem)."""
    out = []
    for arch, spec in T.POSE.items():
        if arch == "block_gcn":
            out.append((arch, "BlockGCN (lr 0.01)", active_src(spec)[0], f"bdsl_block_gcn_si_lr01_sn_{r}",
                        f"bdsl_block_gcn_sd_sn_{r}"))
        elif arch in ("adaptive_gnn", "gnn"):
            out.append((arch, f"{arch} (lr 0.1 = current P2)", active_src(spec)[0], f"bdsl_{arch}_si_lr0p1_sn_{r}",
                        f"bdsl_{arch}_sd_lr0p1_sn_{r}"))
            out.append((arch, f"{arch} (lr 0.01, earlier P2)", active_src(spec, None)[0], f"bdsl_{arch}_si_sn_{r}",
                        f"bdsl_{arch}_sd_sn_{r}"))
        else:
            src, rec = active_src(spec)
            out.append((arch, f"{arch}" + (f" [{rec}]" if rec else ""), src, f"bdsl_{arch}_si_sn_{r}",
                        f"bdsl_{arch}_sd_sn_{r}"))
    return out


def disp(x):
    """The 2-dp displayed value (as rendered in the tables)."""
    return float(f"{x:.2f}")


def try_resolve(src, metric=None):
    try:
        return T.resolve(src, metric)
    except Exception as e:  # noqa: BLE001
        return None


def fmt(v):
    if v is None: return "--"
    return f"{v.mean:.2f} ± {v.sd:.2f}" if v.multi else f"{v.mean:.2f}"


def mixed_sn(sd_stem):
    vals = []
    for s in (0, 1, 2):
        p = f"work_dir/{sd_stem}_seed{s}_test/eval_results/best_acc.pkl"
        if not os.path.isfile(p):
            return None, f"Mixed pending ({sd_stem})"
        vals.append(T.r2(T._pkl_metrics(p, SD_LAB)["top1"]))
    bad = gate([f"{sd_stem}_seed{s}" for s in (0, 1, 2)])
    return (T.Val(vals), "INCOMPLETE " + ",".join(bad)) if bad else (T.Val(vals), "")


def arch_table(r):
    lab = load_labels()
    print(f"\n## Table-7-style rows: P2 recipe vs signer-normalized '{r}' (Protocol A TEST, Mixed = sd TEST)\n")
    print("Δ = difference of the 2-dp displayed 3-seed means; CI = hierarchical bootstrap (seeds resampled per arm, "
          "1,365 clips resampled paired across arms; 10k, percentile). SMG from displayed means.\n")
    print("| arch | P2 TEST | SN TEST | Δ TEST | hierarchical 95% CI, p | P2 Macro-F1 | SN Macro-F1 | "
          "P2 Mixed | SN Mixed | P2 SMG | SN SMG | notes |")
    print("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|")
    out = {}
    for arch, label, src, st, sdst in rows_for(r):
        bt = try_resolve(src["test"]); bm = try_resolve(src["mixed"])
        bsuite = src["suite"]["stem"]
        B = summarize(bsuite, bsuite, lab, [0, 1, 2])
        S = summarize(st, st, lab, [0, 1, 2])
        notes = []
        if S is None or len(S["seeds"]) < 3:
            print(f"| {label} | {fmt(bt)} | (pending: {0 if S is None else len(S['seeds'])}/3) |||||||||"); continue
        bad = gate([f"{st}_seed{s}" for s in (0, 1, 2)])
        if bad: notes.append("INCOMPLETE " + ",".join(bad))
        st_v = T.Val([T.r2(S["top1"][s]) for s in (0, 1, 2)])
        sm, note = mixed_sn(sdst)
        if note: notes.append(note)
        d = disp(st_v.mean) - disp(bt.mean)
        lo, hi, p = hier_ci(S["C"], B["C"]) if B else (np.nan,) * 3
        f1b = np.mean(list(B["f1"].values())) if B else float("nan")
        f1s = f"{np.mean(list(S['f1'].values())):.2f} ± {np.std(list(S['f1'].values())):.2f}"
        smg_b = f"{disp(bm.mean) - disp(bt.mean):+.2f}" if (bm and bt) else "--"
        smg_s = f"{disp(sm.mean) - disp(st_v.mean):+.2f}" if sm else "--"
        print(f"| {label} | {fmt(bt)} | {fmt(st_v)} | {d:+.2f} | [{lo:+.2f}, {hi:+.2f}], p={p:.4f} | {f1b:.2f} | {f1s} | "
              f"{fmt(bm)} | {fmt(sm)} | {smg_b} | {smg_s} | {'; '.join(notes)} |")
        out[label] = dict(p2_test=bt.mean if bt else None, sn_test=st_v.mean, sn_sd=st_v.sd, d=d,
                          p2_mixed=bm.mean if bm else None, sn_mixed=sm.mean if sm else None)
    return out


def write_loso(r):
    rows, missing, bad = [], [], []
    for fold in sorted(glob.glob("data/bdsl_si_loso/test_U*_val_U*")):
        fd = os.path.basename(fold); nn = fd.split("_")[1][1:]
        names, labs = pickle.load(open(f"{fold}/test_label.pkl", "rb"))
        lab = dict(zip(names, labs))
        for s in (0, 1, 2):
            run = f"bdsl_block_gcn_si_lr01_sn_{r}_loso_test{nn}_seed{s}"
            p = f"work_dir/{run}_test/eval_results/best_acc.pkl"
            if not os.path.isfile(p):
                missing.append(run); continue
            if not train_status(os.path.join(ROOT, "work_dir", run))["complete"]:
                bad.append(run); continue
            sc = pickle.load(open(p, "rb"))
            assert set(sc) == set(lab), f"{p}: clip set != fold test set"
            acc = 100 * np.mean([int(np.argmax(sc[k]) == lab[k]) for k in sc])
            rows.append((nn, s, f"U{nn}", f"{acc:.2f}"))
    out = f"docs/loso_heldout/block_gcn_lr01_sn_{r}.csv"
    if not missing and not bad:
        with open(out, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["fold", "seed", "heldout_signer", "test_acc"]); w.writerows(rows)
        print(f"wrote {out} ({len(rows)} rows)")
    return rows, missing, bad


def loso_table(r, rows):
    base = T._loso_folds("block_gcn_lr01")
    new = {}
    for nn, s, sig, acc in rows:
        new.setdefault(sig, []).append(float(acc))
    folds = [f for f in base if f in new and len(new[f]) == 3]
    B = np.array([np.mean(base[f]) for f in folds]); N = np.array([np.mean(new[f]) for f in folds])
    d = N - B
    idx = RNG.integers(0, len(folds), size=(10000, len(folds)))
    ci = lambda v: np.percentile(v[idx].mean(1), [2.5, 97.5])
    bs = d[idx].mean(1); p = (bs <= 0).mean() if d.mean() >= 0 else (bs >= 0).mean()
    print(f"\n## BlockGCN held-out LOSO ({len(folds)}/11 folds x 3 seeds), fold = mean over seeds\n")
    print("| | mean | 95% CI (signer-clustered bootstrap) | SD (sample, folds) | min–max |")
    print("|---|--:|--:|--:|--:|")
    for nm, v in (("P2 BlockGCN lr 0.01", B), (f"signer-norm {r}", N)):
        lo, hi = ci(v)
        print(f"| {nm} | {v.mean():.2f} | [{lo:.2f}, {hi:.2f}] | {v.std(ddof=1):.2f} | {v.min():.1f}–{v.max():.1f} |")
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"| Δ (paired by signer) | {d.mean():+.2f} | [{lo:+.2f}, {hi:+.2f}], p={p:.4f} | {d.std(ddof=1):.2f} | "
          f"{d.min():+.1f}–{d.max():+.1f} |")
    print("\nPer fold (P2 → SN, seeds):")
    for f, b, n in zip(folds, B, N):
        print(f"- {f}: {b:.1f} → {n:.1f} ({'/'.join(f'{x:.1f}' for x in new[f])}) Δ {n - b:+.1f}")


def probe_table(path="docs/signernorm_2026-10/raw/probe_v2_signernorm.jsonl"):
    if not os.path.isfile(path):
        print("\n(probe pending)"); return
    rows = [json.loads(l) for l in open(path) if l.strip()]
    grp = {}
    for o in rows:
        grp.setdefault(o["tag"].split("@")[0].rsplit("_seed", 1)[0], []).append(o)
    print("\n## Signer decodability (probe v2 protocol, own-feeder inputs; L_norm, mean ± SD over encoder seeds)\n")
    print("| encoder | n seeds | cc (P2 L_norm) | marginal | within-class | cc ungrouped |")
    print("|---|--:|--:|--:|--:|--:|")
    for k, v in grp.items():
        f = lambda key: f"{np.mean([o[key] for o in v]):.3f} ± {np.std([o[key] for o in v]):.3f}"
        print(f"| {k} | {len(v)} | {f('cc')} | {f('mrg')} | {f('wc')} | {f('cc_ungrouped')} |")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--recipe", default="abc")
    a = ap.parse_args()
    arch_table(a.recipe)
    rows, missing, bad = write_loso(a.recipe)
    if missing or bad:
        print(f"\nLOSO: {len(rows)}/33 complete; missing TEST eval {len(missing)}, incomplete training {len(bad)}: "
              f"{(missing + bad)[:6]}")
    if rows:
        loso_table(a.recipe, rows)
    probe_table()


if __name__ == "__main__":
    main()
