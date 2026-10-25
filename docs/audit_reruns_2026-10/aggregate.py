#!/usr/bin/env python
"""Aggregate the 2026-10-07 audit reruns (job log of scripts/hpc/slurm_audit_reruns_2026_10.sbatch)
into r3_blockgcn_lr.md, r4_lnorm.md, r5_mixed_selection.md. CPU-only; reads the raw/ score dumps.

  python docs/audit_reruns_2026-10/aggregate.py slurm_logs/audit_reruns_<JOB>.out [slurm_logs/audit_reruns_clip5_<JOB2>.out]
"""
import os, re, sys, pickle, statistics as st
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(OUT, "raw")
LOG = sys.argv[1]
txt = "\n".join(open(f, encoding="utf-8", errors="replace").read() for f in sys.argv[1:])


def labels(path):
    n, l = pickle.load(open(path, "rb"))
    return dict(zip(n, l))


LAB = {"si_test": labels("data/bdsl_si/test_label.pkl"), "si_val": labels("data/bdsl_si/val_label.pkl"),
       "sd_test": labels("data/bdsl_sd/test_label.pkl"), "sd_val": labels("data/bdsl_sd/val_label.pkl")}


def score(tag, split):
    """Top-1 and macro-F1 (same per-class F1 averaging as tools/p2_metric_suite._metrics_one)."""
    p = os.path.join(RAW, tag + ".pkl")
    if not os.path.exists(p):
        return None
    sc = pickle.load(open(p, "rb")); lab = LAB[split]
    names = [k for k in sc if k in lab]
    y = np.array([lab[k] for k in names]); pred = np.array([int(np.argmax(sc[k])) for k in names])
    f1s = []
    for c in range(int(max(y.max(), pred.max())) + 1):
        tp = ((pred == c) & (y == c)).sum(); fp = ((pred == c) & (y != c)).sum(); fn = ((pred != c) & (y == c)).sum()
        if tp + fp + fn == 0:
            continue
        pr = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn) if tp + fn else 0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0)
    return dict(top1=(pred == y).mean() * 100, f1=float(np.mean(f1s)) * 100, n=len(y))


def ms(xs):
    return f"{st.mean(xs):.2f} ± {st.pstdev(xs):.2f}"


# ---------------- R3 ----------------
jid = re.search(r"audit_reruns_(\d+)", LOG); jid = jid.group(1) if jid else "?"
r = ["# R3 — BlockGCN base-LR selection (P2 audit item 4)", "",
     f"Source: SLURM job {jid} (`scripts/hpc/slurm_audit_reruns_2026_10.sbatch`), side-effect-free "
     "`tools/eval_ckpt_on_npy.py` on each run's `*_model-best.pt` (best epoch on SI val U15). "
     "TEST = data/bdsl_si/test (signers 2/13, 1365 clips). Macro-F1 = mean per-class F1 over classes present "
     "(same rule as tools/p2_metric_suite.py). SD = population SD (paper convention).", "",
     "## (a) SI TEST of the lr 0.01 runs vs the shared-recipe lr 0.1 headline", "",
     "| run | lr | seed | val (ckpt sel.) | TEST Top-1 | TEST macro-F1 |", "|---|--:|--:|--:|--:|--:|"]
for lab, lr, pre in [("bdsl_block_gcn_si_lr01_scratch", 0.01, "r3_lr001"), ("bdsl_block_gcn_si (headline)", 0.1, "r3_lr01_headline")]:
    t1, f1, vv = [], [], []
    for s in range(3):
        te = score(f"{pre}_test_seed{s}", "si_test")
        va = score(f"{pre}_val_seed{s}", "si_val") if pre == "r3_lr001" else None
        if te:
            t1.append(te["top1"]); f1.append(te["f1"])
        if va:
            vv.append(va["top1"])
        r.append(f"| {lab} | {lr} | {s} | {va['top1']:.2f} |" if va else f"| {lab} | {lr} | {s} | — |")
        r[-1] += f" {te['top1']:.2f} | {te['f1']:.2f} |" if te else " MISSING | MISSING |"
    if t1:
        r.append(f"| **{lab} mean** | {lr} | — | {ms(vv) if vv else '—'} | **{ms(t1)}** | **{ms(f1)}** |")
open(os.path.join(OUT, "r3_blockgcn_lr.md"), "w").write("\n".join(r) + "\n")

# ---------------- R5 ----------------
SI = {"block_gcn": 65.23, "adaptive_gnn": 79.07, "st_gcn_vanilla": 74.51, "gnn": 73.97, "slgtformer": 73.26,
      "gnn_lstm": 71.57, "pose_lstm": 61.64, "gnn_transformer": 56.92}
OLD = {"block_gcn": 98.47, "adaptive_gnn": 98.93, "st_gcn_vanilla": 98.93, "gnn": 98.12, "slgtformer": 99.03,
       "gnn_lstm": 98.88, "pose_lstm": 95.83, "gnn_transformer": 99.03}
NAME = {"block_gcn": "BlockGCN", "adaptive_gnn": "Adaptive GNN", "st_gcn_vanilla": "ST-GCN", "gnn": "Attention GNN",
        "slgtformer": "SLGTFormer", "gnn_lstm": "GNN+Bi-LSTM", "pose_lstm": "Pose-LSTM", "gnn_transformer": "GNN+Transformer"}
r = ["# R5 — Symmetric selection for the signer-mixed (\"Mixed\") accuracy (P2 audit item 7)", "",
     f"Source: SLURM job {jid}. For each `work_dir/bdsl_<arch>_sd_seed<N>`, the `*_model-best.pt` checkpoint "
     "(best epoch on data/bdsl_sd/val, 655 clips) is scored on the held-out data/bdsl_sd/test (1367 clips, "
     "signers shared with train) — i.e. selected on val, reported on test, the same protocol as the SI TEST numbers. "
     "Old Mixed = best-epoch sd-val Top-1 (selected on the reporting set). SI TEST = paper tab:arch. SD = population.", "",
     "| Arch | sd-val (re-eval of best ckpt) per seed | **Mixed (sd TEST) per seed** | Mixed (sd TEST) mean ± SD | macro-F1 (sd TEST) | old Mixed (sd val best) | SI TEST | old SMG | **new SMG** |",
     "|---|---|---|--:|--:|--:|--:|--:|--:|"]
for a in SI:
    te = [score(f"r5_{a}_sdtest_seed{s}", "sd_test") for s in range(3)]
    va = [score(f"r5_{a}_sdval_seed{s}", "sd_val") for s in range(3)]
    tt = [x["top1"] for x in te if x]; ff = [x["f1"] for x in te if x]; vv = [x["top1"] for x in va if x]
    miss = [s for s in range(3) if te[s] is None]
    new = st.mean(tt) if tt else float("nan")
    r.append(f"| {NAME[a]} | {'/'.join(f'{x:.2f}' for x in vv)} | {'/'.join(f'{x:.2f}' for x in tt)}"
             f"{' (missing seeds ' + str(miss) + ')' if miss else ''} | {ms(tt) if tt else '—'} | {ms(ff) if ff else '—'} | "
             f"{OLD[a]:.2f} | {SI[a]:.2f} | {OLD[a]-SI[a]:+.2f} | **{new-SI[a]:+.2f}** |")
open(os.path.join(OUT, "r5_mixed_selection.md"), "w").write("\n".join(r) + "\n")

# ---------------- R4 ----------------
pat_g = re.compile(r"\[([^\]|]+)\|(cc8k|cc|mrg)\](?: \[class-conditioned\])? SIGNER decodability: balanced acc = ([\d.]+) \+/- ([\d.]+).*?L_norm = ([\-\d.]+)\s+\(n=(\d+), S=(\d+)\)")
pat_w = re.compile(r"\[([^\]|]+)\|(wcall|wc)\] WITHIN-CLASS.*?L_norm\^cond = ([\-\d.na]+).*?classes used (\d+) \(skipped (\d+).*?n=(\d+)")
res = {}
for m in pat_g.finditer(txt):
    res.setdefault(m.group(1), {})[m.group(2)] = (float(m.group(5)), int(m.group(6)), int(m.group(7)))
for m in pat_w.finditer(txt):
    res.setdefault(m.group(1), {})[m.group(2)] = (float(m.group(3)), int(m.group(6)), f"{m.group(4)} used/{m.group(5)} skipped")
P2 = {"sup_blockgcn_m69": "0.763", "ssl_bdsl_only": "0.913", "ssl_xlingual": "0.871", "ssl_dgxl": "0.824",
      "sup_adaptive_m75": "0.911", "sup_adaptive_m59": "0.911 (P2 used latest periodic ckpt = m75)",
      "grl_l01": "0.760", "grl_l05": "0.722", "grl_l10": "0.732"}
P1 = {"sup_blockgcn_m69": ("0.625", "0.968"), "ssl_bdsl_only": ("0.847", "0.979"), "ssl_xlingual": ("0.788", "0.967"),
      "ssl_dgxl": ("0.727", "0.957")}
r = ["# R4 — Signer-decodability L_norm under three definitions (cross-paper item 5)", "",
     f"Source: SLURM job {jid}; current committed `tools/decodability_probe.py`, frozen-backbone features "
     "(mean-pooled over T,V,M), 5-fold LogReg, chance-normalised balanced signer accuracy.",
     "- **cc** = `--class-conditioned` (subtract per-class mean feature, then ONE global signer probe) — P2's definition; n=4000 first clips (BdSLW60), as in P2's scripts.",
     "- **mrg** = marginal (no conditioning) — P1 tab:probe; n=6000 (all 5748 BdSLW60 train clips).",
     "- **wc** = `--within-class` L_norm^cond (signer probe fit WITHIN each sign class, averaged) — P1's primary measure; n=6000.",
     "- BdSLW401: signer regex `S(\\d+)` (16 train signers). cc and marginal on a fixed-seed random 8000-clip sample (P2's setting); within-class on ALL 38,876 train clips (an 8000-clip sample leaves <3 clips/signer in every class -> all 401 classes skipped).", "",
     "| Encoder (ckpt) | cc (P2 def.) | marginal | within-class | P2 reports | P1 reports (marg / wc) |",
     "|---|--:|--:|--:|--:|--:|"]
def g(t, k):
    v = res.get(t, {}).get(k)
    return f"{v[0]:.3f}" if v else "—"
rows = ["sup_blockgcn_m69", "ssl_bdsl_only", "ssl_xlingual", "ssl_dgxl", "sup_adaptive_m75", "sup_adaptive_m59",
        "grl_l01", "grl_l05", "grl_l10"]
for t in rows:
    p1 = P1.get(t, ("—", "—"))
    r.append(f"| {t} | {g(t,'cc')} | {g(t,'mrg')} | {g(t,'wc')} | {P2.get(t,'—')} | {p1[0]} / {p1[1]} |")
# clipped BlockGCN (grad_clip 5.0) = recipe-matched lambda=0 control for the GRL table (TEST 68.47)
cl = []
for k in ("cc", "mrg", "wc"):
    xs = [res.get(f"sup_blockgcn_clip5_s{s}", {}).get(k) for s in range(3)]
    xs = [x[0] for x in xs if x]
    cl.append(f"{ms(xs).replace(' ± ', '±')} ({'/'.join(f'{x:.3f}' for x in xs)})" if xs else "— (follow-up job pending)")
r.append(f"| sup_blockgcn_clip5 (λ=0 GRL control, 3 seeds, best ckpt) | {cl[0]} | {cl[1]} | {cl[2]} | — (P2 uses unclipped 0.763) | — |")
for arch, p2 in [("ctrgcn", "0.69 ± 0.00"), ("blockgcn", "0.43 ± 0.13")]:
    out = []
    for k, kk in (("cc", "cc8k"), ("mrg", "mrg"), ("wc", "wcall")):
        # prefer the P2-matched cc (n=8000) and the full-set within-class from the follow-up job
        xs = [res.get(f"w401_{arch}_s{s}", {}).get(kk) or res.get(f"w401_{arch}_s{s}", {}).get(k) for s in range(3)]
        xs = [x if (x and x[0] == x[0]) else None for x in xs]
        xs = [x[0] for x in xs if x]
        out.append(f"{ms(xs).replace(' ± ', '±')} ({'/'.join(f'{x:.3f}' for x in xs)})" if xs else "—")
    r.append(f"| W401 {arch} (3 seeds) | {out[0]} | {out[1]} | {out[2]} | {p2} | — |")
r += ["", "Within-class usable classes (used/skipped): " + "; ".join(
    f"{t}: {(v.get('wcall') or v['wc'])[2]}" for t, v in res.items() if ("wc" in v or "wcall" in v))]
grl = os.path.join(RAW, "r4_grl_sweep_rerun.md")
if os.path.exists(grl):
    r += ["", "## eval_grl_sweep.py rerun (P2's GRL-table tool, verbatim)", "", open(grl).read()]
open(os.path.join(OUT, "r4_lnorm.md"), "w").write("\n".join(r) + "\n")
print("wrote r3_blockgcn_lr.md r4_lnorm.md r5_mixed_selection.md in", OUT)
