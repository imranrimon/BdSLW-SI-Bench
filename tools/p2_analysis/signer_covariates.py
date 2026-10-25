#!/usr/bin/env python
"""P2 analysis 1 (stage 1): per-signer pose covariates for the 11 LOSO signers.

Reads the per-clip MediaPipe cache (data/bdsl_cache/<class>/<clip>.npz, array
'data' of shape (3, T, 27, 1): channels x, y, visibility; joints 0-6 body
[nose, R-sh, L-sh, R-elb, L-elb, R-wri, L-wri], 7-16 right hand, 17-26 left
hand; hand landmarks carry no confidence, a missing hand is all-zero) and the
raw video headers (data/bdslw60_clips/<class>/<clip>.mp4, for fps/resolution).

Only clips that appear in the LOSO bundles' test_label.pkl (the exact clips each
signer is evaluated on as held-out signer) are used, so covariates describe the
same clip set as the held-out accuracy.

Outputs (JSON): docs/p2_analysis_2026-10/signer_covariates.json
  per_signer: covariate means, per_clip_csv path
  mmd: per fold, MMD(held-out signer, fold TRAIN signers) using the
       tools/p2_analysis/pose_mmd.py clip descriptor + RBF MMD (median
       heuristic), averaged over `--mmd-draws` draws of n=`--mmd-n` per side.

Usage: nice -n 10 python tools/p2_analysis/signer_covariates.py [--workers 3]
"""
import argparse, glob, json, os, pickle, re, sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from tools.p2_analysis.pose_mmd import clip_descriptor, mmd_rbf  # noqa: E402

LOSO_DIR = os.path.join(ROOT, "data/bdsl_si_loso")
CACHE = os.path.join(ROOT, "data/bdsl_cache")
CLIPS = os.path.join(ROOT, "data/bdslw60_clips")
OUT_DIR = os.path.join(ROOT, "docs/p2_analysis_2026-10")
SIGNERS = [1, 2, 4, 5, 6, 8, 9, 11, 12, 13, 15]
NAME_RE = re.compile(r"^U(\d+)W(\d+)([A-Z]*)(?:_(\d+))?_trial_(\d+)_([LR])\.mp4$", re.I)

RSH, LSH, REL, LEL, RWR, LWR = 1, 2, 3, 4, 5, 6
RH = list(range(7, 17))
LH = list(range(17, 27))
# hand-local index 0 = wrist, 4 = middle-finger MCP (HAND_INDICES = [0,4,5,8,9,...])
H_WRIST, H_MIDMCP = 0, 4


def fold_dirs():
    out = {}
    for d in sorted(glob.glob(os.path.join(LOSO_DIR, "test_U*_val_U*"))):
        m = re.search(r"test_U(\d+)_val_U(\d+)", d)
        out[int(m.group(1))] = (d, int(m.group(2)))
    return out


def video_meta(path):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    r = (cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_WIDTH),
         cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return r


def clip_stats(task):
    name, cls = task
    base = os.path.splitext(name)[0]
    p = os.path.join(CACHE, cls, base + ".npz")
    with np.load(p) as h:
        a = np.asarray(h["data"], dtype=np.float64)[..., 0]  # (3,T,27)
    m = NAME_RE.match(name)
    signer, suffix = int(m.group(1)), m.group(6).upper()
    meta = video_meta(os.path.join(CLIPS, cls, name))
    fps, W, H = (meta[0], meta[1], meta[2]) if meta else (np.nan, np.nan, np.nan)
    # aspect correction: MediaPipe x,y are normalised by width/height separately;
    # rescale y by H/W so distances are isotropic in units of frame width.
    asp = (H / W) if (meta and W > 0) else 1.0
    xy = a[:2].copy()
    xy[1] *= asp
    vis = a[2]
    T = a.shape[1]
    present = np.abs(a[:2]).sum(0) > 0               # (T,27)
    rh_pres = present[:, RH].any(1)
    lh_pres = present[:, LH].any(1)
    body_pres = present[:, :7].all(1)

    def dist(i, j, mask=None):
        d = np.linalg.norm(xy[:, :, i] - xy[:, :, j], axis=0)
        ok = present[:, i] & present[:, j]
        if mask is not None:
            ok &= mask
        return d[ok]

    sw = dist(RSH, LSH)
    sw_med = float(np.median(sw)) if len(sw) else np.nan

    def path_len(j):
        p = xy[:, :, j]
        ok = present[:-1, j] & present[1:, j]
        st = np.linalg.norm(np.diff(p, axis=1), axis=0)
        return float(st[ok].sum())

    # motion-based dominant hand from the BODY wrists (always tracked, unlike hand landmarks)
    pr, pl = path_len(RWR), path_len(LWR)
    left_dom_index = pl / (pr + pl) if (pr + pl) > 0 else np.nan
    # hand-landmark presence-based: which hand MediaPipe tracks more
    # joint speed (shoulder-widths per frame) over valid consecutive pairs
    st = np.linalg.norm(np.diff(xy, axis=1), axis=0)          # (T-1,27)
    ok = present[:-1] & present[1:]
    speed_all = float(st[ok].mean() / sw_med) if ok.any() and sw_med > 0 else np.nan
    dom_wr = LWR if (left_dom_index == left_dom_index and left_dom_index > 0.5) else RWR
    okw = ok[:, dom_wr]
    speed_domwrist = float(st[okw, dom_wr].mean() / sw_med) if okw.any() and sw_med > 0 else np.nan
    speed_domwrist_ps = speed_domwrist * fps if fps == fps else np.nan  # per second

    arm_r = np.median(dist(RSH, REL)) + np.median(dist(REL, RWR)) if present[:, [RSH, REL, RWR]].all(1).any() else np.nan
    arm_l = np.median(dist(LSH, LEL)) + np.median(dist(LEL, LWR)) if present[:, [LSH, LEL, LWR]].all(1).any() else np.nan
    hs = np.concatenate([dist(RH[H_WRIST], RH[H_MIDMCP]), dist(LH[H_WRIST], LH[H_MIDMCP])])
    hand_size = float(np.median(hs)) if len(hs) else np.nan
    sig_hand_pres = lh_pres if suffix == "L" else rh_pres
    dom_hand_pres = lh_pres if dom_wr == LWR else rh_pres
    return dict(
        clip=name, cls=cls, signer=signer, suffix=suffix, T=T, fps=fps, W=W, H=H,
        rh_missing=float(1 - rh_pres.mean()), lh_missing=float(1 - lh_pres.mean()),
        both_missing=float((~rh_pres & ~lh_pres).mean()),
        sighand_missing=float(1 - sig_hand_pres.mean()),       # hand named by filename suffix
        domhand_missing=float(1 - dom_hand_pres.mean()),       # motion-dominant hand
        body_missing=float(1 - body_pres.mean()),
        wrist_lowvis=float(((vis[:, [RWR, LWR]] < 0.5) & present[:, [RWR, LWR]]).mean()),
        left_dom_index=left_dom_index, motion_left_dom=float(left_dom_index > 0.5),
        speed_all=speed_all, speed_domwrist=speed_domwrist, speed_domwrist_ps=speed_domwrist_ps,
        shoulder_width=sw_med,
        arm_len_norm=float(np.nanmean([arm_r, arm_l]) / sw_med) if sw_med > 0 else np.nan,
        hand_size_norm=float(hand_size / sw_med) if sw_med > 0 else np.nan,
        duration_s=T / fps if fps and fps == fps and fps > 0 else np.nan,
    )


def load_raw(name, cls):
    with np.load(os.path.join(CACHE, cls, os.path.splitext(name)[0] + ".npz")) as h:
        return h["data"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--mmd-n", type=int, default=400)
    ap.add_argument("--mmd-draws", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    folds = fold_dirs()
    assert sorted(folds) == SIGNERS, sorted(folds)
    classes = json.load(open(os.path.join(LOSO_DIR, "test_U01_val_U02", "classes.json")))
    if isinstance(classes, dict) and "classes" in classes:
        classes = classes["classes"]
    idx2cls = dict(enumerate(classes))

    clips = {}   # signer -> list of (name, cls)
    for s, (d, _v) in folds.items():
        names, labels = pickle.load(open(os.path.join(d, "test_label.pkl"), "rb"))
        clips[s] = [(n, idx2cls[int(l)]) for n, l in zip(names, labels)]
        assert all(NAME_RE.match(n) and int(NAME_RE.match(n).group(1)) == s for n in names)

    tasks = [t for s in SIGNERS for t in clips[s]]
    print(f"{len(tasks)} clips over {len(SIGNERS)} signers")
    with Pool(a.workers) as pool:
        rows = pool.map(clip_stats, tasks, chunksize=64)
    import csv
    per_clip_csv = os.path.join(OUT_DIR, "signer_covariates_per_clip.csv")
    with open(per_clip_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    keys = [k for k in rows[0] if k not in ("clip", "cls", "signer", "suffix")]
    per_signer = {}
    for s in SIGNERS:
        rs = [r for r in rows if r["signer"] == s]
        d = {k: float(np.nanmean([r[k] for r in rs])) for k in keys}
        d["n_clips"] = len(rs)
        d["suffix_L_frac"] = float(np.mean([r["suffix"] == "L" for r in rs]))
        d["T_median"] = float(np.median([r["T"] for r in rs]))
        # agreement of motion-based dominance with filename suffix
        d["suffix_motion_agree"] = float(np.mean([(r["suffix"] == "L") == (r["motion_left_dom"] > 0.5) for r in rs]))
        d["fps_values"] = sorted({round(r["fps"], 2) for r in rs if r["fps"] == r["fps"]})
        d["resolutions"] = sorted({f"{int(r['W'])}x{int(r['H'])}" for r in rs if r["W"] == r["W"]})
        per_signer[f"U{s:02d}"] = d
    agree_all = float(np.mean([(r["suffix"] == "L") == (r["motion_left_dom"] > 0.5) for r in rows]))

    # ---- MMD: held-out signer vs that fold's TRAIN signers (9 signers) and vs all other 10
    descs = {}
    for s in SIGNERS:
        dd = [clip_descriptor(load_raw(n, c)) for n, c in clips[s]]
        descs[s] = np.array([x for x in dd if x is not None])
    mmd = {}
    for s in SIGNERS:
        val = folds[s][1]
        train_s = [t for t in SIGNERS if t not in (s, val)]
        res = {}
        for tag, pool_s in (("fold_train", train_s), ("all_other10", [t for t in SIGNERS if t != s])):
            src_all = np.vstack([descs[t] for t in pool_s])
            vals = []
            for k in range(a.mmd_draws):
                rng = np.random.default_rng(1000 * s + k)
                X = src_all[rng.choice(len(src_all), min(a.mmd_n, len(src_all)), replace=False)]
                Y = descs[s][rng.choice(len(descs[s]), min(a.mmd_n, len(descs[s])), replace=False)]
                vals.append(mmd_rbf(X, Y))
            res[tag] = dict(mean=float(np.mean(vals)), sd=float(np.std(vals, ddof=1)), draws=[float(v) for v in vals],
                            pool_signers=pool_s)
        res["val_signer"] = val
        mmd[f"U{s:02d}"] = res
        per_signer[f"U{s:02d}"]["mmd_fold_train"] = res["fold_train"]["mean"]
        per_signer[f"U{s:02d}"]["mmd_all_other10"] = res["all_other10"]["mean"]
        print(f"U{s:02d} mmd fold_train={res['fold_train']['mean']:.4f}±{res['fold_train']['sd']:.4f} "
              f"other10={res['all_other10']['mean']:.4f}")

    out = dict(
        source=dict(cache=CACHE, clips=CLIPS, loso_bundles=LOSO_DIR, per_clip_csv=per_clip_csv,
                    descriptor="tools/p2_analysis/pose_mmd.py:clip_descriptor + mmd_rbf",
                    mmd_n=a.mmd_n, mmd_draws=a.mmd_draws),
        n_clips_total=len(rows), suffix_vs_motion_dominance_agreement=agree_all,
        per_signer=per_signer, mmd=mmd)
    jp = os.path.join(OUT_DIR, "signer_covariates.json")
    json.dump(out, open(jp, "w"), indent=1)
    print("wrote", jp, per_clip_csv)


if __name__ == "__main__":
    main()
