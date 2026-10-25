#!/usr/bin/env python
"""GRL signer-adversarial sweep at 3 seeds: held-out-TEST Top-1 + class-conditioned L_norm.

Exactly the paper's tab:grl procedure (tools/eval_grl_sweep.py, as run by
scripts/hpc/lr01_switch/tail_J4.sh): the val-selected `<exp>_model-best.pt` of each
`bdsl_block_gcn_si_signeradv_lr01_gl{01,05,10}_seed{0,1,2}` run; Top-1 = signer_adv class
head on the canonical TEST signers {2,13}; L_norm = class-conditioned (per-class-mean
residualised) logistic signer probe on the first 4000 BdSLW60-SI TRAIN clips
(eval_grl_sweep.cc_probe). Optionally (--wc) also the within-class L_norm^cond via
tools/decodability_probe.py --within-class (n=6000), as in docs/blockgcn_lr01_switch/06.

Writes per-run JSON lines (--jsonl, idempotent: finished runs skipped) and a markdown
table (--md) with mean +/- population SD (the paper's convention for tab:grl's control)
and the sample SD.
"""
import argparse, json, os, re, subprocess, sys
import numpy as np
import torch, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from tools.eval_grl_sweep import build_model, features_and_logits, load_split, cc_probe  # noqa: E402

CFG = "config/lr01_switch/bdsl_block_gcn_si_signeradv_lr01.yaml"
LAMS = {"01": 0.1, "05": 0.5, "10": 1.0}
# lambda = 0 control: the headline lr-0.01 BlockGCN (docs/blockgcn_lr01_switch/01 and 06)
CONTROL = dict(top1=[68.06, 69.01, 69.38], cc=[0.824, 0.871, 0.863], wc=[0.972, 0.978, 0.976])
PAPER = {0.1: (50.70, 0.804), 0.5: (43.81, 0.832), 1.0: (45.27, 0.832)}  # current single-seed tab:grl


def ckpt(tag, s):
    exp = f"bdsl_block_gcn_si_signeradv_lr01_gl{tag}_seed{s}"
    return exp, f"work_dir/{exp}/{exp}_model-best.pt"


def wc_probe(path):
    out = subprocess.run([sys.executable, "-u", "tools/decodability_probe.py", "--mode", "signer", "--within-class",
                          "--config", "config/probe_blockgcn_si.yaml", "--ckpt", path, "--max-clips", "6000"],
                         capture_output=True, text=True, check=True).stdout
    return float(re.search(r"L_norm\^cond = ([0-9.]+)", out).group(1))


def ms(v):
    v = np.asarray(v, float)
    return v.mean(), v.std(), (v.std(ddof=1) if len(v) > 1 else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="docs/tta_grl_2026-10/raw/grl_3seed.jsonl")
    ap.add_argument("--md", default="docs/tta_grl_2026-10/grl_3seed.md")
    ap.add_argument("--wc", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.jsonl), exist_ok=True)
    rows = {}
    if os.path.exists(a.jsonl):
        for ln in open(a.jsonl):
            r = json.loads(ln); rows[(r["lam"], r["seed"])] = r
    cfg = yaml.safe_load(open(CFG))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xte, _, y_te = load_split(cfg, "test")
    Xtr, names_tr, y_tr = load_split(cfg, "train")
    n = min(4000, len(Xtr))
    Xtr, names_tr, cls_tr = Xtr[:n], names_tr[:n], y_tr[:n]
    sig_tr = np.array([int(re.search(r"U(\d+)W", s).group(1)) for s in names_tr])
    for tag, lam in LAMS.items():
        for s in (0, 1, 2):
            if (lam, s) in rows and (not a.wc or "wc" in rows[(lam, s)]):
                continue
            exp, path = ckpt(tag, s)
            if not os.path.exists(path):
                print(f"[missing] {path}"); continue
            final = [f for f in os.listdir(f"work_dir/{exp}") if re.match(rf"{exp}_model-99\.pt$", f)]
            if s > 0 and not final:
                print(f"[incomplete] {exp}: no final-epoch checkpoint yet"); continue
            r = rows.get((lam, s)) or {}
            if "top1" not in r:
                model = build_model(cfg, path, dev)
                _, logits = features_and_logits(model, Xte, dev)
                top1 = float((logits.argmax(1) == y_te).mean()) * 100
                F, _ = features_and_logits(model, Xtr, dev)
                lk, lksd, chance, lnorm, S = cc_probe(F, sig_tr, cls_tr)
                r = dict(lam=lam, seed=s, exp=exp, top1=top1, leak_acc=lk * 100, cc=lnorm)
            if a.wc and "wc" not in r:
                r["wc"] = wc_probe(path)
            rows[(lam, s)] = r
            print(json.dumps(r), flush=True)
            with open(a.jsonl, "a") as f:
                f.write(json.dumps(r) + "\n")

    L = ["| $\\lambda$ | seeds | TEST Top-1 mean ± SD (per seed) | $L_{norm}$ (cc) mean ± SD (per seed) | $L^{wc}_{norm}$ | paper (seed 0 only) |",
         "|---|--:|---|---|---|---|"]
    t, c, w = ms(CONTROL["top1"]), ms(CONTROL["cc"]), ms(CONTROL["wc"])
    ts = " / ".join("%.2f" % v for v in CONTROL["top1"]); cs = " / ".join("%.3f" % v for v in CONTROL["cc"])
    L.append(f"| 0 (control) | 3 | {t[0]:.2f} ± {t[1]:.2f} ({ts}) | "
             f"{c[0]:.3f} ± {c[1]:.3f} ({cs}) | {w[0]:.3f} ± {w[1]:.3f} | 68.82 / 0.853 |")
    for lam in sorted(set(LAMS.values())):
        rr = [rows[(lam, s)] for s in (0, 1, 2) if (lam, s) in rows]
        if not rr:
            continue
        t = ms([r["top1"] for r in rr]); c = ms([r["cc"] for r in rr])
        wv = [r["wc"] for r in rr if "wc" in r]
        wcs = f"{ms(wv)[0]:.3f} ± {ms(wv)[1]:.3f}" if wv else "--"
        ts = " / ".join("%.2f" % r["top1"] for r in rr)
        cs = " / ".join("%.3f" % r["cc"] for r in rr)
        L.append(f"| {lam:g} | {len(rr)} | {t[0]:.2f} ± {t[1]:.2f} (sample SD {t[2]:.2f}; {ts}) | "
                 f"{c[0]:.3f} ± {c[1]:.3f} (sample SD {c[2]:.3f}; {cs}) | {wcs} | {PAPER[lam][0]:.2f} / {PAPER[lam][1]:.3f} |")
    L.append("\nSD = population SD over seeds (the convention of the paper's control row, 0.853 ± 0.021); sample SD in brackets.")
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
