#!/usr/bin/env python
"""Linear-SVM baseline of the BdSLW60 dataset paper (Rubaiyeat et al., MTAP 2025;
arXiv 2402.08635, Sec. 4.2.4; authors' code pipeline_SVM_POSE_HAND.ipynb), re-run on
our 27-keypoint footing.

Faithful to the authors: per clip the valid frames are uniformly stretched by frame
duplication to a fixed length (their PROLONGED set), flattened, StandardScaler'd and
fed to sklearn SVC(kernel='linear'); optionally left-dominant clips are re-mapped to
right dominance (their FLIPPED set). Our footing: 27 keypoints (x, y, confidence)
from data/bdsl_si / data/bdsl_sd, deterministic window-120 centre crop (the feeder's
eval transform), so the fixed length is 120 (their 164 was their max clip length).
Each clip is first calibrated as in the authors' pipeline: every present point's
(x, y) minus the shoulder midpoint of the first valid frame (they used the video's
first frame; confidence channel untouched).
Fixes vs the authors' notebook: the scaler is fit on train only (they refit it on
the test set), and C is selected on the validation signer (they fixed C=1).

  SI   : train -> pick C on val (U15) -> TEST (U02, U13).
  mixed: train(bdsl_sd) -> pick C on sd val -> score sd TEST (1,367 clips; P2's
         symmetric Mixed protocol). Secondary: max sd-val accuracy over C (old rule).
SVC(linear) is deterministic, so a single run is reported (no seeds).
"""
import argparse, json, os, pickle, sys, time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from feeders.tools import center_choose, valid_frame_length  # noqa: E402

LEN = 120
CS = [0.001, 0.01, 0.1, 1.0]
RH, LH = list(range(7, 17)), list(range(17, 27))


def calibrate(clip):
    """Authors' calibration: subtract the first valid frame's shoulder midpoint from
    every present point's (x, y). clip (3, T, 27)."""
    out = clip.copy()
    present = (clip > 0).any(axis=0)                        # (T, 27)
    both = present[:, 1] & present[:, 2]
    if both.any():
        t0 = int(np.argmax(both))
        ref = 0.5 * (clip[:2, t0, 1] + clip[:2, t0, 2])
        out[:2][:, present] = clip[:2][:, present] - ref[:, None]
    return out


def flip_to_right(clip):
    """Authors' hand-dominance correction: mirror the hands horizontally about the
    shoulder midpoint and swap them (face/body untouched). clip (3, T, 27)."""
    out = clip.copy()
    mid = 0.5 * (clip[0, :, 1] + clip[0, :, 2])            # (T,)
    for src, dst in ((RH, LH), (LH, RH)):
        h = clip[:, :, src].copy()
        present = h[0] != 0
        h[0] = np.where(present, 2 * mid[:, None] - h[0], 0.0)
        out[:, :, dst] = h
    return out


def features(split_dir, split, flip):
    data = np.load(os.path.join(split_dir, f"{split}_data.npy"), mmap_mode="r")
    names, labels = pickle.load(open(os.path.join(split_dir, f"{split}_label.pkl"), "rb"))
    X = np.zeros((len(labels), 3 * LEN * 27), np.float32)
    for i in range(len(labels)):
        c = center_choose(np.asarray(data[i], np.float32), LEN)[:, :, :, 0]   # (3, <=LEN, 27)
        L = max(1, min(valid_frame_length(c[..., None]), c.shape[1]))
        c = calibrate(c[:, :L])
        if flip and os.path.splitext(str(names[i]))[0].endswith("_L"):
            c = flip_to_right(c)
        idx = np.floor(np.arange(LEN) * L / LEN).astype(int)                   # PROLONGED
        X[i] = c[:, idx].reshape(-1)
    return X, np.asarray(labels), list(names)


def fit_eval(Xtr, ytr, evals, C):
    sc = StandardScaler().fit(Xtr)
    clf = SVC(kernel="linear", C=C).fit(sc.transform(Xtr), ytr)
    out = {}
    for k, (X, y, names) in evals.items():
        s = clf.decision_function(sc.transform(X))
        pred = clf.classes_[s.argmax(1)]
        out[k] = dict(acc=float((pred == y).mean() * 100), scores=dict(zip(names, s)), classes=clf.classes_)
    return out


def macro_f1(scores, classes, names, y):
    pred = np.array([classes[np.argmax(scores[n])] for n in names])
    f = []
    for c in np.unique(y):
        tp = ((pred == c) & (y == c)).sum(); fp = ((pred == c) & (y != c)).sum(); fn = ((pred != c) & (y == c)).sum()
        f.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return float(np.mean(f) * 100)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flip", action="store_true", help="authors' FLIPPED hand-dominance correction (uses _L/_R clip metadata)")
    ap.add_argument("--out", default="work_dir/bdsl_svm_baseline")
    a = ap.parse_args()
    tag = "flipped" if a.flip else "original"
    out = os.path.join(ROOT, f"{a.out}_{tag}"); os.makedirs(out, exist_ok=True)
    t0 = time.time()
    si = {s: features(os.path.join(ROOT, "data/bdsl_si"), s, a.flip) for s in ("train", "val", "test")}
    sd = {s: features(os.path.join(ROOT, "data/bdsl_sd"), s, a.flip) for s in ("train", "val", "test")}
    print(f"[svm/{tag}] features {si['train'][0].shape} in {time.time()-t0:.0f}s", flush=True)
    res = {"variant": tag, "len": LEN, "C_grid": CS, "si": {}, "sd": {}}
    si_runs, sd_runs = {}, {}
    for C in CS:
        r = fit_eval(si["train"][0], si["train"][1], {"val": si["val"], "test": si["test"]}, C)
        si_runs[C] = r
        res["si"][str(C)] = {"val": r["val"]["acc"], "test": r["test"]["acc"]}
        r2 = fit_eval(sd["train"][0], sd["train"][1], {"val": sd["val"], "test": sd["test"]}, C)
        sd_runs[C] = r2
        res["sd"][str(C)] = {"val": r2["val"]["acc"], "test": r2["test"]["acc"]}
        print(f"[svm/{tag}] C={C}: SI val {r['val']['acc']:.2f} TEST {r['test']['acc']:.2f} | "
              f"mixed val {r2['val']['acc']:.2f} test {r2['test']['acc']:.2f}  ({time.time()-t0:.0f}s)", flush=True)
    Csel = max(CS, key=lambda c: (si_runs[c]["val"]["acc"], -c))
    te = si_runs[Csel]["test"]
    _, yte, nte = si["test"]
    res.update(C_selected=Csel, test_top1=te["acc"],
               test_macro_f1=macro_f1(te["scores"], te["classes"], nte, yte),
               C_selected_sd=(Csd := max(CS, key=lambda c: (sd_runs[c]["val"]["acc"], -c))),
               mixed_sd_test=sd_runs[Csd]["test"]["acc"],
               mixed_sd_test_macro_f1=macro_f1(sd_runs[Csd]["test"]["scores"], sd_runs[Csd]["test"]["classes"],
                                               sd["test"][2], sd["test"][1]),
               mixed_old_max_sd_val=max(sd_runs[c]["val"]["acc"] for c in CS))
    res["smg"] = res["mixed_sd_test"] - res["test_top1"]
    res["smg_old"] = res["mixed_old_max_sd_val"] - res["test_top1"]
    pickle.dump(te["scores"], open(os.path.join(out, "test_scores_Csel.pkl"), "wb"))
    json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("si", "sd")}, indent=1))


if __name__ == "__main__":
    main()
