#!/usr/bin/env python3
"""P2 analysis 2 -- RGB + pose late fusion on Protocol-A TEST (signers U02/U13).

Fuses per-clip class posteriors of RGB / foundation models with pose models:
    p_fused = w * p_rgb + (1 - w) * p_pose          (softmax-probability averaging)
w is chosen on the VAL signer (U15) ONLY, per seed pair, from a fixed grid; TEST is
never used for selection. Seed i of the RGB model is paired with seed i of the pose model.

Sources (all per-clip, keyed by clip name):
  VideoMAE  work_dir/rgb_videomae_bdsl_si_rerun_seed{0,1,2}/{test_scores,val_best_scores}.pkl  (probabilities)
  S3D       work_dir/rgb_s3d_bdsl_si_seed{0r,1r,2r}/eval_results/{testset_scores,best_acc}.pkl  (logits; best_acc = VAL)
  Uni-Sign  external/Uni-Sign/out/bdslw60_si_ft_lowdata[_seed{1,2}]_eval/test_tmp_pres.txt   (hard text labels, TEST only)
  SHuBERT   work_dir/shubert_probe_seg/si_test_pred_seed{0,1,2}.pkl                          (probabilities, TEST only, 1364 clips)
  pose      TEST: work_dir/bdsl_<stem>_seed{S}_test/eval_results/best_acc.pkl (logits)
            VAL : work_dir/<train run>/eval_results/best_acc.pkl (logits at the best-val epoch = the TEST checkpoint)
            SLGTFormer Track-B = slgtformer_si_bp; Adaptive GNN = adaptive_gnn_si (shared recipe);
            BlockGCN = block_gcn_si_lr01 (train run bdsl_block_gcn_si_lr01_scratch_seed{S}).

Writes docs/p2_analysis_2026-10/rgb_pose_fusion.{md,json}.
Run (repo root): nice -n 10 python tools/p2_analysis/rgb_pose_fusion.py
"""
import json, os, pickle, re, sys
import numpy as np

if not hasattr(np, "_core"):  # pkls written under numpy>=2
    import numpy.core as _c
    sys.modules.setdefault("numpy._core", _c)
    for _s in ("multiarray", "numeric", "umath", "_multiarray_umath", "_exceptions", "numerictypes", "_dtype", "_methods"):
        _m = sys.modules.get("numpy.core." + _s)
        if _m is not None:
            sys.modules.setdefault("numpy._core." + _s, _m)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = "docs/p2_analysis_2026-10"
SEEDS = [0, 1, 2]
WGRID = np.round(np.arange(0, 1.0001, 0.05), 2)
NBOOT = 10000
RNG = np.random.default_rng(20261022)


def key(n):
    return os.path.splitext(os.path.basename(n))[0]


def labels(split):
    n, l = pickle.load(open(f"data/bdsl_si/{split}_label.pkl", "rb"))
    return {key(a): int(b) for a, b in zip(n, l)}


CLASSES = json.load(open("data/bdsl_si/classes.json"))["classes"]
YT, YV = labels("test"), labels("val")
TEST_KEYS = sorted(YT)
VAL_KEYS = sorted(YV)


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


def load_scores(path, keys):
    d = pickle.load(open(path, "rb"))
    d = {key(k): np.asarray(v, dtype=np.float64) for k, v in d.items()}
    missing = [k for k in keys if k not in d]
    extra = [k for k in d if k not in set(keys)]
    m = np.stack([d[k] for k in keys]) if not missing else None
    is_prob = m is not None and m.min() >= 0 and np.allclose(m.sum(1), 1, atol=1e-3)
    return (m if is_prob else (softmax(m) if m is not None else None)), dict(
        path=path, n=len(d), missing=len(missing), extra=len(extra), raw_is_prob=bool(is_prob))


POSE = {
    "SLGTFormer (Track-B)": dict(test="work_dir/bdsl_slgtformer_si_bp_seed{s}_test/eval_results/best_acc.pkl",
                                 val="work_dir/bdsl_slgtformer_si_bp_seed{s}/eval_results/best_acc.pkl"),
    "Adaptive GNN": dict(test="work_dir/bdsl_adaptive_gnn_si_seed{s}_test/eval_results/best_acc.pkl",
                         val="work_dir/bdsl_adaptive_gnn_si_seed{s}/eval_results/best_acc.pkl"),
    "BlockGCN (lr 0.01)": dict(test="work_dir/bdsl_block_gcn_si_lr01_seed{s}_test/eval_results/best_acc.pkl",
                               val="work_dir/bdsl_block_gcn_si_lr01_scratch_seed{s}/eval_results/best_acc.pkl"),
}
S3D_DIR = {0: "rgb_s3d_bdsl_si_seed0r", 1: "rgb_s3d_bdsl_si_seed1r", 2: "rgb_s3d_bdsl_si_seed2r"}  # cluster re-trains (2026-10-22)
RGB = {
    "VideoMAE": dict(test="work_dir/rgb_videomae_bdsl_si_rerun_seed{s}/test_scores.pkl",
                     val="work_dir/rgb_videomae_bdsl_si_rerun_seed{s}/val_best_scores.pkl"),
    "S3D": dict(test="work_dir/{d}/eval_results/testset_scores.pkl", val="work_dir/{d}/eval_results/best_acc.pkl"),
}
UNISIGN = {0: "external/Uni-Sign/out/bdslw60_si_ft_lowdata_eval",
           1: "external/Uni-Sign/out/bdslw60_si_ft_lowdata_seed1_eval",
           2: "external/Uni-Sign/out/bdslw60_si_ft_lowdata_seed2_eval"}  # seed index 0 = Uni-Sign seed 42
SHUBERT = "work_dir/shubert_probe_seg/si_test_pred_seed{s}.pkl"


def fmt(path, s):
    return path.format(s=s, d=S3D_DIR[s])


def acc(P, y):
    return float((P.argmax(1) == y).mean() * 100)


def boot(ca, cb):
    """paired bootstrap of mean(ca)-mean(cb) over clips; ca, cb per-clip correctness (may be seed-averaged)."""
    n = len(ca)
    idx = RNG.integers(0, n, size=(NBOOT, n))
    d = (ca[idx].mean(1) - cb[idx].mean(1)) * 100
    obs = float((ca.mean() - cb.mean()) * 100)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = float((d <= 0).mean())  # one-sided H1: a > b
    return dict(delta=obs, ci=[float(lo), float(hi)], p_one_sided=p)


def holm(ps):
    order = np.argsort(ps)
    m, prev, out = len(ps), 0.0, [None] * len(ps)
    for r, i in enumerate(order):
        prev = max(prev, min(1.0, (m - r) * ps[i]))
        out[i] = prev
    return out


def pick_w(Pa_v, Pb_v, yv):
    """argmax val Top-1 over WGRID; ties -> w closest to 0.5 (pre-declared)."""
    accs = np.array([acc(w * Pa_v + (1 - w) * Pb_v, yv) for w in WGRID])
    best = np.flatnonzero(accs == accs.max())
    j = best[np.argmin(np.abs(WGRID[best] - 0.5))]
    return float(WGRID[j]), float(accs[j]), accs.tolist()


def main():
    R = {"protocol": "Protocol A; TEST = signers U02+U13 (data/bdsl_si/test_label.pkl), VAL = U15 (data/bdsl_si/val_label.pkl)",
         "w_grid": WGRID.tolist(), "n_boot": NBOOT, "alignment": {}, "single": {}, "fusion": {}, "notes": []}
    yt = np.array([YT[k] for k in TEST_KEYS]); yv = np.array([YV[k] for k in VAL_KEYS])
    R["n_test"], R["n_val"] = len(yt), len(yv)

    # ---------- load + alignment ----------
    M = {}  # model -> seed -> {"test": P, "val": P}
    for name, spec in {**POSE, **RGB}.items():
        M[name] = {}
        for s in SEEDS:
            Pt, it = load_scores(fmt(spec["test"], s), TEST_KEYS)
            Pv, iv = load_scores(fmt(spec["val"], s), VAL_KEYS)
            R["alignment"][f"{name}/seed{s}"] = dict(test=it, val=iv)
            if Pt is None or Pv is None:
                print(f"ALIGNMENT FAILURE {name} seed {s}: {it} {iv}"); sys.exit(1)
            M[name][s] = dict(test=Pt, val=Pv)

    # Uni-Sign hard predictions (TEST only) -- check names + refs against our labels
    US = {}
    for s, d in UNISIGN.items():
        def rd(f):
            o = {}
            for line in open(f, encoding="utf-8"):
                m = re.match(r"sample: ([^,]+), [\w-]+: ?(.*)$", line.rstrip("\n"))
                if m: o[key(m[1])] = m[2].strip()
            return o
        pr, rf = rd(f"{d}/test_tmp_pres.txt"), rd(f"{d}/test_tmp_refs.txt")
        ref_ok = all(CLASSES.index(rf[k]) == YT[k] for k in TEST_KEYS if k in rf)
        unk = sum(1 for k in TEST_KEYS if pr.get(k) not in CLASSES)
        pred = np.array([CLASSES.index(pr[k]) if pr.get(k) in CLASSES else -1 for k in TEST_KEYS])
        R["alignment"][f"Uni-Sign/seed{s}"] = dict(path=d, n=len(pr), missing=sum(k not in pr for k in TEST_KEYS),
                                                   refs_match_labels=bool(ref_ok), preds_not_a_class=int(unk))
        US[s] = pred
    # SHuBERT probs (TEST only, 1364 clips)
    SH = {}
    for s in SEEDS:
        d = pickle.load(open(SHUBERT.format(s=s), "rb"))
        names = [key(n) for n in d["names"]]
        lab_ok = all(YT[n] == int(l) for n, l in zip(names, d["label"]))
        # SHuBERT uses its own class indexing: derive the column permutation from (its label, our label)
        # pairs over the clip names and require it to be a consistent bijection before remapping.
        pairs = {(int(l), YT[n]) for n, l in zip(names, d["label"])}
        fwd = dict(pairs)
        bij = len(pairs) == len(fwd) == len(set(fwd.values())) == 60
        if not bij:
            print("SHuBERT label mapping is not a bijection"); sys.exit(1)
        P = np.asarray(d["prob"], dtype=np.float64)
        Pm = np.zeros_like(P)
        for a, b in fwd.items():
            Pm[:, b] = P[:, a]
        missing = sorted(set(TEST_KEYS) - set(names))
        SH[s] = dict(names=names, P=Pm)
        R["alignment"][f"SHuBERT/seed{s}"] = dict(path=SHUBERT.format(s=s), n=len(names), missing=missing,
                                                  labels_identical_indexing=bool(lab_ok), label_permutation_bijective=bool(bij),
                                                  n_identity_columns=int(sum(a == b for a, b in fwd.items())))

    # ---------- single models ----------
    corr = {}  # model -> seed -> per-clip correctness on TEST
    for name in M:
        a = [acc(M[name][s]["test"], yt) for s in SEEDS]
        av = [acc(M[name][s]["val"], yv) for s in SEEDS]
        corr[name] = {s: (M[name][s]["test"].argmax(1) == yt).astype(float) for s in SEEDS}
        R["single"][name] = dict(test_per_seed=a, test_mean=float(np.mean(a)), test_sd=float(np.std(a)),
                                 val_per_seed=av, val_mean=float(np.mean(av)))
    a = [float((US[s] == yt).mean() * 100) for s in SEEDS]
    corr["Uni-Sign"] = {s: (US[s] == yt).astype(float) for s in SEEDS}
    R["single"]["Uni-Sign"] = dict(test_per_seed=a, test_mean=float(np.mean(a)), test_sd=float(np.std(a)), val_per_seed=None,
                                   seed_ids=[42, 1, 2])
    a = [float((SH[s]["P"].argmax(1) == np.array([YT[n] for n in SH[s]["names"]])).mean() * 100) for s in SEEDS]
    R["single"]["SHuBERT"] = dict(test_per_seed=a, test_mean=float(np.mean(a)), test_sd=float(np.std(a)), n=len(SH[0]["names"]))
    best_single = max((n for n in R["single"]), key=lambda n: R["single"][n]["test_mean"])
    R["best_single_overall"] = best_single

    # ---------- val-tuned probability fusion ----------
    pairs = [(r, p, "RGB+pose") for r in RGB for p in POSE]
    pairs += [("SLGTFormer (Track-B)", "Adaptive GNN", "control: pose+pose"),
              ("VideoMAE", "S3D", "control: RGB+RGB")]
    fused_corr = {}
    for A, B, kind in pairs:
        rows = []
        for s in SEEDS:
            w, vacc, curve = pick_w(M[A][s]["val"], M[B][s]["val"], yv)
            Pf = w * M[A][s]["test"] + (1 - w) * M[B][s]["test"]
            rows.append(dict(seed=s, w=w, val_acc=vacc, test_acc=acc(Pf, yt),
                             test_acc_equal_w=acc(0.5 * M[A][s]["test"] + 0.5 * M[B][s]["test"], yt)))
            fused_corr.setdefault((A, B), {})[s] = (Pf.argmax(1) == yt).astype(float)
        t = [r["test_acc"] for r in rows]
        stronger = A if R["single"][A]["test_mean"] >= R["single"][B]["test_mean"] else B
        fc = np.mean([fused_corr[(A, B)][s] for s in SEEDS], 0)
        e = dict(kind=kind, a=A, b=B, per_seed=rows, test_mean=float(np.mean(t)), test_sd=float(np.std(t)),
                 equal_w_mean=float(np.mean([r["test_acc_equal_w"] for r in rows])),
                 stronger_component=stronger,
                 gain_vs_stronger_component_per_seed=[r["test_acc"] - R["single"][stronger]["test_per_seed"][r["seed"]] for r in rows],
                 boot_vs_stronger_component=boot(fc, np.mean([corr[stronger][s] for s in SEEDS], 0)),
                 boot_vs_best_single=boot(fc, np.mean([corr[best_single][s] for s in SEEDS], 0)),
                 oracle_either_correct=float(np.mean([np.maximum(corr[A][s], corr[B][s]).mean() * 100 for s in SEEDS])),
                 both_wrong=float(np.mean([((1 - corr[A][s]) * (1 - corr[B][s])).mean() * 100 for s in SEEDS])),
                 disagree_rate=float(np.mean([(M[A][s]["test"].argmax(1) != M[B][s]["test"].argmax(1)).mean() * 100 for s in SEEDS])))
        R["fusion"][f"{A} + {B}"] = e
    rgbpose = [k for k, v in R["fusion"].items() if v["kind"] == "RGB+pose"]
    for tag in ("boot_vs_stronger_component", "boot_vs_best_single"):
        ph = holm([R["fusion"][k][tag]["p_one_sided"] for k in rgbpose])
        for k, p in zip(rgbpose, ph):
            R["fusion"][k][tag]["p_holm_6"] = p

    # ---------- Uni-Sign: no logits, no VAL predictions -> parameter-free plurality vote only ----------
    A, B = "VideoMAE", "SLGTFormer (Track-B)"
    vote_rows, vote_corr = [], {}
    for s in SEEDS:
        w, _, _ = pick_w(M[A][s]["val"], M[B][s]["val"], yv)
        fb = (w * M[A][s]["test"] + (1 - w) * M[B][s]["test"]).argmax(1)  # pre-declared tie fallback
        pa, pb, pu = M[A][s]["test"].argmax(1), M[B][s]["test"].argmax(1), US[s]
        out = np.where(pa == pb, pa, np.where(pu == pa, pa, np.where(pu == pb, pb, fb)))
        vote_corr[s] = (out == yt).astype(float)
        vote_rows.append(dict(seed=s, unisign_seed=[42, 1, 2][s], test_acc=float(vote_corr[s].mean() * 100),
                              all_three_disagree_pct=float(((pa != pb) & (pu != pa) & (pu != pb)).mean() * 100)))
    vc = np.mean([vote_corr[s] for s in SEEDS], 0)
    t = [r["test_acc"] for r in vote_rows]
    R["unisign_vote"] = dict(members=["Uni-Sign", A, B], rule="plurality; all-disagree -> val-tuned VideoMAE+SLGTFormer fusion",
                             per_seed=vote_rows, test_mean=float(np.mean(t)), test_sd=float(np.std(t)),
                             boot_vs_best_single=boot(vc, np.mean([corr[best_single][s] for s in SEEDS], 0)),
                             boot_vs_unisign=boot(vc, np.mean([corr["Uni-Sign"][s] for s in SEEDS], 0)),
                             oracle_unisign_slgt=float(np.mean([np.maximum(corr["Uni-Sign"][s], corr[B][s]).mean() * 100 for s in SEEDS])),
                             oracle_unisign_videomae=float(np.mean([np.maximum(corr["Uni-Sign"][s], corr[A][s]).mean() * 100 for s in SEEDS])))

    # ---------- SHuBERT: no VAL predictions -> equal-weight (NOT val-tuned), 1364 clips ----------
    rows = []
    for s in SEEDS:
        idx = [TEST_KEYS.index(n) for n in SH[s]["names"]]
        Pp = M["SLGTFormer (Track-B)"][s]["test"][idx]
        y = yt[idx]
        rows.append(dict(seed=s, n=len(idx), shubert=acc(SH[s]["P"], y), slgt=acc(Pp, y),
                         fused_equal_w=acc(0.5 * SH[s]["P"] + 0.5 * Pp, y)))
    R["shubert_equal_w"] = dict(per_seed=rows, fused_mean=float(np.mean([r["fused_equal_w"] for r in rows])),
                                fused_sd=float(np.std([r["fused_equal_w"] for r in rows])),
                                slgt_mean_on_1364=float(np.mean([r["slgt"] for r in rows])))

    os.makedirs(OUT, exist_ok=True)
    json.dump(R, open(f"{OUT}/rgb_pose_fusion.json", "w"), indent=1)
    write_md(R)
    print(open(f"{OUT}/rgb_pose_fusion.md").read())


def write_md(R):
    L = ["# P2 analysis 2 -- RGB + pose late fusion (Protocol A TEST)", "",
         "Script: `tools/p2_analysis/rgb_pose_fusion.py`; numbers: `docs/p2_analysis_2026-10/rgb_pose_fusion.json`.", "",
         "**Protocol.** TEST = held-out signers U02+U13 (n = %d clips, `data/bdsl_si/test_label.pkl`); VAL = U15 (n = %d, "
         "`data/bdsl_si/val_label.pkl`). Fusion = softmax-probability averaging `w*p_A + (1-w)*p_B`; w picked on VAL only "
         "from {0, 0.05, ..., 1} (ties -> w nearest 0.5), separately per seed pair; seed i of A paired with seed i of B "
         "(S3D seeds 0r/1r/2r = cluster re-trains; Uni-Sign seed index 0 = its seed 42). mean +- SD = population SD over the 3 seed pairs. "
         "Bootstrap: paired clip-level, %d resamples, on per-clip correctness averaged over the 3 seed pairs; one-sided p "
         "(H1: fusion better); Holm over the 6 RGB x pose pairs. Clip-level CIs ignore within-signer correlation "
         "(only 2 TEST signers, so a signer-clustered CI is impossible) -- they are per-instance, not signer-generalisation, CIs."
         % (R["n_test"], R["n_val"], R["n_boot"]), "",
         "## Alignment check (done first)", "",
         "All pose (3 models x 3 seeds), VideoMAE and S3D TEST dumps contain exactly the 1,365 TEST clip names and all VAL dumps "
         "exactly the 655 U15 names (0 missing / 0 extra; rows aligned by clip name, not by position). Uni-Sign TEST predictions "
         "cover all 1,365 names and its reference labels match `test_label.pkl` for every clip. SHuBERT covers 1,364/1,365 "
         "(missing: %s); its class indices differ from `classes.json` (%d/60 identical) but map by a consistent bijection derived from the clip names, applied before scoring." % (", ".join(R["alignment"]["SHuBERT/seed0"]["missing"]), R["alignment"]["SHuBERT/seed0"]["n_identity_columns"]), "",
         "Score types: VideoMAE dumps are already probabilities; pose and S3D dumps are logits (softmaxed here). "
         "Uni-Sign saves only decoded text labels (no logits) and no VAL predictions; SHuBERT saves TEST probabilities only "
         "(no VAL). Hence Uni-Sign and SHuBERT cannot enter val-tuned probability fusion (see the last two sections).", "",
         "## Single models (TEST Top-1, per seed)", "",
         "| Model | seed 0 | seed 1 | seed 2 | mean +- SD | VAL (U15) mean |", "|---|--:|--:|--:|--:|--:|"]
    for n, v in R["single"].items():
        ps = " | ".join("%.2f" % x for x in v["test_per_seed"])
        L.append("| %s | %s | %.2f +- %.2f | %s |" % (n, ps, v["test_mean"], v["test_sd"],
                                                     "%.2f" % v["val_mean"] if v.get("val_mean") is not None else "n/a"))
    L += ["", "Best single model by TEST mean: **%s**." % R["best_single_overall"], "",
          "## Val-tuned fusion (TEST Top-1)", "",
          "| Pair | kind | w (A) per seed | fused per seed | fused mean +- SD | equal-w mean | Delta vs stronger component [95%% CI], p_Holm | Delta vs best single (%s) [95%% CI], p_Holm | oracle (either correct) | both wrong |"
          % R["best_single_overall"], "|---|---|---|---|--:|--:|---|---|--:|--:|"]
    for k, e in R["fusion"].items():
        bs, bb = e["boot_vs_stronger_component"], e["boot_vs_best_single"]
        L.append("| %s | %s | %s | %s | %.2f +- %.2f | %.2f | %+.2f [%+.2f, %+.2f], p=%.4f%s (vs %s) | %+.2f [%+.2f, %+.2f], p=%.4f%s | %.2f | %.2f |" % (
            k, e["kind"], "/".join("%.2f" % r["w"] for r in e["per_seed"]),
            "/".join("%.2f" % r["test_acc"] for r in e["per_seed"]), e["test_mean"], e["test_sd"], e["equal_w_mean"],
            bs["delta"], bs["ci"][0], bs["ci"][1], bs.get("p_holm_6", bs["p_one_sided"]), "" if "p_holm_6" in bs else " (raw)",
            e["stronger_component"], bb["delta"], bb["ci"][0], bb["ci"][1], bb.get("p_holm_6", bb["p_one_sided"]),
            "" if "p_holm_6" in bb else " (raw)", e["oracle_either_correct"], e["both_wrong"]))
    v = R["unisign_vote"]
    L += ["", "Controls (pose+pose, RGB+RGB) use the identical val-tuned rule; their p-values are raw (not in the Holm family).",
          "Oracle / both-wrong are TEST-only diagnostics of complementarity (an upper bound, never used for selection).", "",
          "## Uni-Sign (hard labels only): parameter-free plurality vote", "",
          "Members: Uni-Sign + VideoMAE + SLGTFormer (Track-B); when all three disagree, the val-tuned VideoMAE+SLGTFormer "
          "fusion decides (pre-declared). No parameter is fit on TEST.", "",
          "| seed pair | Uni-Sign seed | vote TEST Top-1 | all-3-disagree % |", "|---|--:|--:|--:|"]
    for r in v["per_seed"]:
        L.append("| %d | %d | %.2f | %.2f |" % (r["seed"], r["unisign_seed"], r["test_acc"], r["all_three_disagree_pct"]))
    b1, b2 = v["boot_vs_best_single"], v["boot_vs_unisign"]
    L += ["", "Vote: %.2f +- %.2f; vs best single %+.2f [%+.2f, %+.2f] (p=%.4f, raw); vs Uni-Sign %+.2f [%+.2f, %+.2f] (p=%.4f, raw). "
          "Oracle Uni-Sign|SLGTFormer = %.2f, Uni-Sign|VideoMAE = %.2f." % (
              v["test_mean"], v["test_sd"], b1["delta"], b1["ci"][0], b1["ci"][1], b1["p_one_sided"],
              b2["delta"], b2["ci"][0], b2["ci"][1], b2["p_one_sided"], v["oracle_unisign_slgt"], v["oracle_unisign_videomae"]), "",
          "## SHuBERT: equal-weight fusion with SLGTFormer (NOT val-tuned; no SHuBERT VAL dumps), 1,364 clips", ""]
    s = R["shubert_equal_w"]
    L.append("Per seed (SHuBERT / SLGTFormer / fused): " + "; ".join(
        "s%d %.2f / %.2f / %.2f" % (r["seed"], r["shubert"], r["slgt"], r["fused_equal_w"]) for r in s["per_seed"]) +
        ". Fused mean %.2f +- %.2f vs SLGTFormer %.2f on the same 1,364 clips." % (s["fused_mean"], s["fused_sd"], s["slgt_mean_on_1364"]))
    L += ["", "## Source files", ""]
    for k, a in R["alignment"].items():
        if "test" in a:
            L.append("- %s: TEST `%s`; VAL `%s`" % (k, a["test"]["path"], a["val"]["path"]))
        else:
            L.append("- %s: `%s`" % (k, a["path"]))
    L += ["- labels: `data/bdsl_si/test_label.pkl`, `data/bdsl_si/val_label.pkl`, classes `data/bdsl_si/classes.json`"]
    open(f"{OUT}/rgb_pose_fusion.md", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
