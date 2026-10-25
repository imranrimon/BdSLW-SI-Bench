#!/usr/bin/env python
"""P2 analysis 1 (stage 2): per-signer difficulty vs pose covariates (exploratory, n=11).

Inputs
  docs/loso_heldout/<file>.csv              held-out-signer Top-1 (paper file choice:
                                            BlockGCN lr0.01, GNN+Transformer clip5)
  docs/p2_analysis_2026-10/signer_covariates.json / _per_clip.csv
                                            (tools/p2_analysis/signer_covariates.py)
  optional clip-level held-out score dumps  $P2_ANALYSIS_TMP/loso_dumps/ (default work_dir/p2_analysis_tmp)
                                            <arch>_test<XX>_seed<S>.pkl ({clip: logits}),
                                            produced by tools/eval_ckpt_on_npy.py --dump on each
                                            fold's model-best.pt and the fold's test_data.npy.
Analyses
  (a) Spearman(rho) of each signer-level covariate with signer mean accuracy (mean over
      8 archs x 3 seeds); exact permutation p (11! is too many -> 200k random perms);
      BH-FDR and Holm across the covariate family; leave-one-signer-out rho range.
  (b) statsmodels MixedLM on the 88 signer x arch cell means:
      Ma (requested): acc ~ z(cov) + (1|arch)
      Mb (honest):    acc ~ z(cov) + (1|arch) + (1|signer)  (crossed, vc_formula)
      Ma treats the 8 arch rows of a signer as independent -> anti-conservative for a
      signer-level covariate; Mb's SE reflects n=11.
  (c) clip-level, within-signer (if dumps exist): logistic GLM
      correct ~ left_hand_clip + hand_missing + speed + log(T) + C(signer) + C(class)
      with signer-clustered SEs, and per-signer accuracy split by the filename hand tag.
Output: docs/p2_analysis_2026-10/signer_difficulty.json
"""
import csv, glob, json, os, pickle, re, warnings

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "docs/p2_analysis_2026-10")
DUMPS = os.path.join(os.environ.get("P2_ANALYSIS_TMP", "work_dir/p2_analysis_tmp"), "loso_dumps")
ARCH_FILES = [("Adaptive GNN", "adaptive_gnn"), ("ST-GCN", "st_gcn_vanilla"), ("SLGTFormer", "slgtformer"),
              ("GNN+Transformer", "gnn_transformer_clip5"), ("Attention GNN", "gnn"),
              ("BlockGCN", "block_gcn_lr01"), ("GNN+LSTM", "gnn_lstm"), ("Pose LSTM", "pose_lstm")]
# pre-specified covariate family (signer-level means of per-clip values)
COVS = [
    ("left_hand_frac_tag", "fraction of clips tagged _L (filename hand tag)"),
    ("left_hand_frac_tracked", "fraction of clips where MediaPipe tracks the LEFT hand in more frames than the right"),
    ("left_dom_index", "body-wrist motion share of the left wrist (0.5 = balanced)"),
    ("both_missing", "fraction of frames with NO hand detected"),
    ("sighand_missing", "fraction of frames where the tagged signing hand is undetected"),
    ("wrist_lowvis", "fraction of wrist detections with visibility < 0.5"),
    ("speed_all", "mean joint speed (shoulder widths / frame, 30 fps)"),
    ("T", "clip length (frames, 30 fps)"),
    ("shoulder_width", "shoulder width (fraction of frame width; camera framing)"),
    ("arm_len_norm", "upper+fore-arm length / shoulder width"),
    ("hand_size_norm", "wrist-to-middle-MCP / shoulder width"),
    ("aspect_4x3_frac", "fraction of clips recorded at 4:3 (960x720/640x480) vs 16:9"),
    ("mmd_fold_train", "pose-motion MMD(held-out signer, that fold's 9 train signers)"),
]
NPERM = 100000
RNG = np.random.default_rng(7)


def loso_acc():
    rows = []
    for name, f in ARCH_FILES:
        with open(os.path.join(ROOT, "docs/loso_heldout", f + ".csv")) as fh:
            for r in csv.DictReader(fh):
                rows.append((name, r["heldout_signer"], int(r["seed"]), float(r["test_acc"])))
    return pd.DataFrame(rows, columns=["arch", "signer", "seed", "acc"])


def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)
    q = np.empty(n); prev = 1.0
    for i in range(n - 1, -1, -1):
        prev = min(prev, p[o[i]] * n / (i + 1)); q[o[i]] = prev
    return q


def holm(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p); q = np.empty(n); run = 0
    for k, i in enumerate(o):
        run = max(run, (n - k) * p[i]); q[i] = min(1, run)
    return q


def perm_spearman(x, y):
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    obs = stats.spearmanr(x, y)[0]
    perms = np.array([np.corrcoef(rx, RNG.permutation(ry))[0, 1] for _ in range(NPERM)])
    return obs, float((np.abs(perms) >= abs(obs) - 1e-12).mean())


def clip_level(pc):
    files = sorted(glob.glob(os.path.join(DUMPS, "*_test*_seed*.pkl")))
    if not files:
        return None
    names = {}
    for sdir in glob.glob(os.path.join(ROOT, "data/bdsl_si_loso/test_U*_val_U*")):
        n, l = pickle.load(open(os.path.join(sdir, "test_label.pkl"), "rb"))
        names.update(dict(zip(n, [int(x) for x in l])))
    recs = []
    used = []
    for f in files:
        m = re.search(r"([a-z_0-9]+)_test(\d+)_seed(\d+)\.pkl$", os.path.basename(f))
        arch, fold, seed = m.group(1), int(m.group(2)), int(m.group(3))
        sc = pickle.load(open(f, "rb"))
        used.append(f)
        for clip, logit in sc.items():
            recs.append((arch, fold, seed, clip, int(np.argmax(logit) == names[clip])))
    cl = pd.DataFrame(recs, columns=["arch", "fold", "seed", "clip", "correct"])
    cl = cl.merge(pc, on="clip", how="left", validate="many_to_one")
    assert cl["signer"].notna().all()
    assert (cl["fold"] == cl["signer"]).all()
    out = dict(dump_files=used, n_clip_preds=len(cl), archs=sorted(cl.arch.unique()),
               folds=sorted(int(x) for x in cl.fold.unique()))
    out["by_arch"] = {}
    for arch_tag, sub in [("pooled", cl)] + [(a, cl[cl.arch == a]) for a in sorted(cl.arch.unique())]:
        out["by_arch"][arch_tag] = _clip_stats(sub.copy())
    return out


def _clip_stats(cl):
    out = dict(n_clip_preds=len(cl), folds=sorted(int(x) for x in cl.fold.unique()),
               archs=sorted(cl.arch.unique()), overall_acc=float(cl.correct.mean() * 100))
    # per-signer accuracy split by filename hand tag
    tab = cl.groupby(["signer", "suffix"])["correct"].agg(["mean", "size"]).unstack()
    out["acc_by_tag"] = {f"U{int(s):02d}": {t: dict(acc=float(tab.loc[s, ("mean", t)]) * 100 if not np.isnan(tab.loc[s, ("mean", t)]) else None,
                                                  n=int(tab.loc[s, ("size", t)]) if not np.isnan(tab.loc[s, ("size", t)]) else 0)
                                              for t in ("L", "R")} for s in tab.index}
    cl["track_left"] = (cl["lh_missing"] < cl["rh_missing"]).astype(int)
    tab2 = cl.groupby(["signer", "track_left"])["correct"].agg(["mean", "size"]).unstack()
    out["acc_by_tracked_hand"] = {f"U{int(s):02d}": {("L" if t else "R"): dict(acc=float(tab2.loc[s, ("mean", t)]) * 100,
                                                                             n=int(tab2.loc[s, ("size", t)]))
                                                    for t in (0, 1) if not np.isnan(tab2.loc[s, ("size", t)])}
                                  for s in tab2.index}
    # pooled within-signer contrast
    cl["left_tag"] = (cl["suffix"] == "L").astype(int)
    import statsmodels.formula.api as smf
    cl["logT"] = np.log(cl["T"])
    for c in ("both_missing", "speed_all", "logT"):
        cl[c + "_z"] = (cl[c] - cl[c].mean()) / cl[c].std()
    res = {}
    arch_fe = " + C(arch)" if cl.arch.nunique() > 1 else ""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tag, form in (
                ("tag_only", "correct ~ left_tag + C(signer) + C(cls)" + arch_fe),
                ("tracked_only", "correct ~ track_left + C(signer) + C(cls)" + arch_fe),
                ("full", "correct ~ left_tag + both_missing_z + speed_all_z + logT_z + C(signer) + C(cls)" + arch_fe)):
            m = smf.glm(form, cl, family=__import__("statsmodels.api").api.families.Binomial()).fit(
                cov_type="cluster", cov_kwds={"groups": cl["signer"].astype(int)})
            keep = [k for k in m.params.index if not k.startswith("C(") and k != "Intercept"]
            res[tag] = {k: dict(log_odds=float(m.params[k]), se_cluster=float(m.bse[k]),
                                odds_ratio=float(np.exp(m.params[k])),
                                ci95_or=[float(np.exp(m.params[k] - 1.96 * m.bse[k])), float(np.exp(m.params[k] + 1.96 * m.bse[k]))],
                                p=float(m.pvalues[k])) for k in keep}
    out["glm_within_signer"] = res
    out["glm_note"] = ("Binomial GLM with signer and class fixed effects; SEs clustered by signer (11 clusters "
                       "-> cluster-robust SEs are themselves noisy). Clip covariates z-scored over all clips.")
    # signers with enough left-tag clips: paired within-signer L-vs-R difference
    diffs = []
    for s, d in out["acc_by_tag"].items():
        if d["L"]["n"] >= 30 and d["R"]["n"] >= 30:
            diffs.append((s, d["L"]["acc"] - d["R"]["acc"], d["L"]["n"], d["R"]["n"]))
    out["L_minus_R_signers_n30"] = diffs
    return out


def main():
    acc = loso_acc()
    cov = json.load(open(os.path.join(D, "signer_covariates.json")))
    pc = pd.read_csv(os.path.join(D, "signer_covariates_per_clip.csv"))
    pc["aspect_4x3"] = (pc["W"] / pc["H"] < 1.5).astype(float)
    pc["left_tracked"] = (pc["lh_missing"] < pc["rh_missing"]).astype(float)
    sig = pd.DataFrame(cov["per_signer"]).T
    sig.index.name = "signer"
    sig["left_hand_frac_tag"] = sig["suffix_L_frac"].astype(float)
    sig["left_hand_frac_tracked"] = pc.groupby("signer")["left_tracked"].mean().values
    sig["aspect_4x3_frac"] = pc.groupby("signer")["aspect_4x3"].mean().values
    assert list(pc.groupby("signer").size().index) == [int(s[1:]) for s in sig.index]
    cell = acc.groupby(["signer", "arch"])["acc"].mean().reset_index()
    sacc = cell.groupby("signer")["acc"].mean()
    table = cell.pivot(index="signer", columns="arch", values="acc")[[n for n, _ in ARCH_FILES]]
    table["mean_8arch"] = sacc
    table["sd_across_arch"] = table[[n for n, _ in ARCH_FILES]].std(1)
    table = table.sort_values("mean_8arch")
    # arch-rank stability across signers: Kendall W of arch ranking within each signer
    R = table[[n for n, _ in ARCH_FILES]].rank(axis=1, ascending=False).values
    m, n = R.shape
    S = ((R.sum(0) - R.sum(0).mean()) ** 2).sum()
    kendall_w = 12 * S / (m ** 2 * (n ** 3 - n))

    # (a) Spearman family
    sp = []
    for c, desc in COVS:
        x = sig.loc[sacc.index, c].astype(float).values
        y = sacc.values
        rho, p = perm_spearman(x, y)
        loo = [stats.spearmanr(np.delete(x, i), np.delete(y, i))[0] for i in range(len(x))]
        sp.append(dict(covariate=c, desc=desc, rho=float(rho), p_perm=p,
                       loo_min=float(np.min(loo)), loo_max=float(np.max(loo)),
                       loo_drop_for_min=sacc.index[int(np.argmin(loo))], loo_drop_for_max=sacc.index[int(np.argmax(loo))]))
    q_bh, q_h = bh([r["p_perm"] for r in sp]), holm([r["p_perm"] for r in sp])
    for r, a, b in zip(sp, q_bh, q_h):
        r["q_bh"], r["p_holm"] = float(a), float(b)

    # (b) mixed models
    import statsmodels.formula.api as smf
    mm = []
    cell2 = cell.merge(sig[[c for c, _ in COVS]].astype(float), left_on="signer", right_index=True)
    cell2["one"] = 1
    for c, _d in COVS:
        z = (cell2[c] - sig[c].astype(float).mean()) / sig[c].astype(float).std()
        cell2["z"] = z
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ma = smf.mixedlm("acc ~ z", cell2, groups="arch").fit(reml=True)
            mb = smf.mixedlm("acc ~ z", cell2, groups="one",
                             vc_formula={"arch": "0 + C(arch)", "signer": "0 + C(signer)"}).fit(reml=True)
        mm.append(dict(covariate=c,
                       Ma_beta_per_sd=float(ma.params["z"]), Ma_se=float(ma.bse["z"]), Ma_p=float(ma.pvalues["z"]),
                       Mb_beta_per_sd=float(mb.params["z"]), Mb_se=float(mb.bse["z"]), Mb_p=float(mb.pvalues["z"]),
                       Mb_signer_var=float(dict(zip(mb.model.exog_vc.names, mb.vcomp))["signer"])))
    for key in ("Ma_p", "Mb_p"):
        qq = holm([r[key] for r in mm])
        for r, q in zip(mm, qq):
            r[key.replace("_p", "_p_holm")] = float(q)

    # two-predictor exploratory model: handedness + hand-missing (the two outlier signers)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for c in ("left_hand_frac_tag", "both_missing", "mmd_fold_train"):
            cell2[c + "_z"] = (cell2[c] - sig[c].astype(float).mean()) / sig[c].astype(float).std()
        m2 = smf.mixedlm("acc ~ left_hand_frac_tag_z + both_missing_z", cell2, groups="one",
                         vc_formula={"arch": "0 + C(arch)", "signer": "0 + C(signer)"}).fit(reml=True)
        m0 = smf.mixedlm("acc ~ 1", cell2, groups="one",
                         vc_formula={"arch": "0 + C(arch)", "signer": "0 + C(signer)"}).fit(reml=True)
    v0 = dict(zip(m0.model.exog_vc.names, m0.vcomp))["signer"]
    v2 = dict(zip(m2.model.exog_vc.names, m2.vcomp))["signer"]
    two = dict(formula="acc ~ z(left_hand_frac_tag) + z(both_missing) + (1|arch) + (1|signer)",
               params={k: dict(beta=float(m2.params[k]), se=float(m2.bse[k]), p=float(m2.pvalues[k]))
                       for k in ("left_hand_frac_tag_z", "both_missing_z")},
               signer_var_null=float(v0), signer_var_model=float(v2),
               signer_var_explained_frac=float(1 - v2 / v0))

    # honest small-n check of the two-predictor model: OLS on the 11 signer means (t, df = 8)
    import statsmodels.api as sm
    Xs = sig.loc[sacc.index, ["left_hand_frac_tag", "both_missing"]].astype(float)
    Xs = (Xs - Xs.mean()) / Xs.std()
    ols = sm.OLS(sacc.values, sm.add_constant(Xs.values)).fit()
    two["ols_signer_means"] = dict(n=11, df_resid=float(ols.df_resid), r2=float(ols.rsquared), r2_adj=float(ols.rsquared_adj),
                                   beta=dict(zip(["const", "left_hand_frac_tag_z", "both_missing_z"], map(float, ols.params))),
                                   p=dict(zip(["const", "left_hand_frac_tag_z", "both_missing_z"], map(float, ols.pvalues))))
    loo_r2 = []
    for i in range(11):
        keep = np.arange(11) != i
        loo_r2.append(float(sm.OLS(sacc.values[keep], sm.add_constant(Xs.values[keep])).fit().rsquared))
    two["ols_signer_means"]["loo_r2_range"] = [min(loo_r2), max(loo_r2)]
    two["note"] = ("POST HOC: both predictors were chosen after inspecting the covariate table (U01 = mostly "
                   "left-tagged, U12 = hand-dropout outlier); MixedLM Wald p-values treat 11 signers as large-sample.")
    print("two-predictor OLS on signer means:", two["ols_signer_means"])

    cl = clip_level(pc)
    out = dict(
        sources=dict(acc={n: f"docs/loso_heldout/{f}.csv" for n, f in ARCH_FILES},
                     covariates="docs/p2_analysis_2026-10/signer_covariates.json",
                     per_clip="docs/p2_analysis_2026-10/signer_covariates_per_clip.csv"),
        per_signer_accuracy=table.round(3).to_dict(orient="index"),
        kendall_w_arch_rank_across_signers=float(kendall_w),
        signer_covariates={s: {c: float(sig.loc[s, c]) for c, _ in COVS} for s in sig.index},
        spearman=sp, mixed_models=mm, two_predictor_model=two, clip_level=cl,
        caveats=["n=11 signers: every signer-level association is exploratory and underpowered "
                 "(|rho| must exceed ~0.62 for p<0.05 even before multiplicity correction).",
                 "Ma treats the 8 architecture means of a signer as independent observations of a signer-level "
                 "covariate (pseudo-replication) -> its SE is anti-conservative; Mb adds a signer random intercept."])
    json.dump(out, open(os.path.join(D, "signer_difficulty.json"), "w"), indent=1, default=float)

    pd.set_option("display.width", 250)
    print(table.round(1))
    print("Kendall W (arch ranking agreement across signers) =", round(kendall_w, 3))
    print(pd.DataFrame(sp)[["covariate", "rho", "p_perm", "q_bh", "p_holm", "loo_min", "loo_max", "loo_drop_for_min"]].round(3))
    print(pd.DataFrame(mm).round(3))
    print(json.dumps(two, indent=1))
    if cl:
        for a, v in cl["by_arch"].items():
            print("==", a, "n", v["n_clip_preds"], "folds", v["folds"], "acc", round(v["overall_acc"], 2))
            print(json.dumps({k: v[k] for k in ("glm_within_signer", "L_minus_R_signers_n30")}, indent=1, default=float))


if __name__ == "__main__":
    main()
