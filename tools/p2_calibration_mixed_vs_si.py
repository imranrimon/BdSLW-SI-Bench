#!/usr/bin/env python3
"""E: does signer-mixing inflate CALIBRATION (not just accuracy)? Compares ECE/NLL/
Brier on ASL Citizen held-out TEST between the participant-DISJOINT (SI) and
participant-MIXED score files (3-seed ensemble). Re-scores saved softmax; no retrain."""
import glob, pickle, numpy as np
def load_ens(base, labelpkl):
    names, labs = pickle.load(open(labelpkl, "rb")); lab = dict(zip(names, labs))
    pkls = sorted(glob.glob(f"work_dir/{base}_seed*/eval_results/best_acc.pkl"))
    per = [{k: np.asarray(v, np.float64) for k, v in pickle.load(open(p, "rb")).items()} for p in pkls]
    common = sorted(set(lab) & set.intersection(*[set(d) for d in per]))
    P = np.stack([np.mean([_sm(d[k]) for d in per], 0) for k in common]); y = np.array([lab[k] for k in common])
    return P, y, len(pkls)
def _sm(v):
    v = np.asarray(v, np.float64).ravel(); 
    if v.min() >= 0 and abs(v.sum()-1) < 1e-3: return v
    e = np.exp(v - v.max()); return e/e.sum()
def cal(P, y):
    N = len(y); pred = P.argmax(1); conf = P.max(1); corr = (pred==y).astype(float)
    o = np.argsort(conf); conf, corr2 = conf[o], corr[o]; ece = 0.0
    for b in np.array_split(np.arange(N), 15):
        if len(b): ece += len(b)/N*abs(corr2[b].mean()-conf[b].mean())
    nll = -np.mean([np.log(max(P[i, y[i]], 1e-12)) for i in range(N)])
    oh = np.zeros_like(P); oh[np.arange(N), y] = 1
    return corr.mean()*100, ece*100, nll, float(np.mean(((P-oh)**2).sum(1)))
for tag, base, lp in [("SI (disjoint)","aslcitizen_si_test","data/aslcitizen_si/test_label.pkl"),
                      ("MIXED","aslcitizen_mixed_test","data/aslcitizen_mixed/test_label.pkl")]:
    try:
        P, y, ns = load_ens(base, lp); acc, ece, nll, br = cal(P, y)
        print(f"{tag:14} n={len(y)} seeds={ns}  Top1(R@1) {acc:.2f}  ECE {ece:.2f}  NLL {nll:.3f}  Brier {br:.3f}")
    except FileNotFoundError as e: print(f"{tag}: ERR {e}")
    except OSError: raise  # unreadable storage must not silently drop a row
    except Exception as e: print(f"{tag}: ERR {e}")
