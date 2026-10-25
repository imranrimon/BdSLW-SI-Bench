"""P1 ★#2 / P2 ★#2 — Representational mechanism probes on frozen pretrained features.

Three read-outs from a frozen pretrained BlockGCN backbone (backbone-only ckpt):
  --mode signer   : linear decodability of SIGNER identity from BdSL features
                    (the P2 "signer-leakage" metric; chance = 1/8 = 12.5%).
  --mode language : linear decodability of source LANGUAGE from pooled features
                    (P1 mechanism: a working GRL should LOWER this in xling vs pool).
  --mode cka      : linear CKA between THIS encoder and a second (--ckpt-b) on the
                    same BdSL data (how much cross-lingual pretraining reshapes reps).

Features are mean-pooled over (V, M) and over T -> one vector per clip. A
LogisticRegression probe (sklearn) is fit train->held-out; we report balanced
accuracy vs chance. Runs on the existing checkpoints; GPU for extraction.
"""
import argparse, os, re, pickle, sys
import numpy as np
import torch, yaml

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
import importlib
def _imp(dotted):
    m,_,c = dotted.rpartition("."); return getattr(importlib.import_module(m), c)


def _alias_numpy_core():
    """Checkpoints pickled under numpy>=2 reference the private module
    'numpy._core'; alias it to 'numpy.core' so torch.load can unpickle them in
    this numpy<2 environment (bdsl_graph ships numpy 1.x)."""
    if hasattr(np, "_core"):
        return
    import numpy.core as _core
    sys.modules.setdefault("numpy._core", _core)
    for sub in ("multiarray", "numeric", "umath", "_multiarray_umath",
                "_exceptions", "numerictypes", "_dtype", "_methods"):
        m = sys.modules.get(f"numpy.core.{sub}")
        if m is not None:
            sys.modules.setdefault(f"numpy._core.{sub}", m)


def _extract_backbone_sd(sd, strip_prefix=None):
    """Unwrap common checkpoint containers and isolate the backbone params.

    Handles: (a) {'weights'|'state_dict'|'model_state_dict': OrderedDict}
    wrappers (e.g. main_signer_adv.py saves under 'weights'), and (b) a
    'backbone.' key prefix from composite models (signer_adv wraps the BlockGCN
    as self.backbone). Auto-detects the prefix when strip_prefix is None.
    """
    for wrap in ("weights", "state_dict", "model_state_dict", "model", "net"):
        if isinstance(sd, dict) and wrap in sd and isinstance(sd[wrap], dict):
            sd = sd[wrap]
            break
    if strip_prefix is None:  # auto: strip 'backbone.' if it dominates
        n_bb = sum(k.startswith("backbone.") for k in sd)
        strip_prefix = "backbone." if n_bb > 0.5 * len(sd) else ""
    if strip_prefix:
        sd = {k[len(strip_prefix):]: v for k, v in sd.items()
              if k.startswith(strip_prefix)}
    return sd, strip_prefix


def build_backbone(cfg_path, ckpt, device, strip_prefix=None):
    cfg = yaml.safe_load(open(cfg_path))
    Backbone = _imp(cfg["backbone"])
    kw = dict(cfg.get("backbone_args", {})); kw["return_features"] = True
    try:
        bb = Backbone(**kw).to(device).eval()
    except TypeError:
        # some backbones don't accept return_features; feats_from_array bypasses
        # forward() anyway (it walks data_bn + layers directly), so drop it.
        kw.pop("return_features", None)
        bb = Backbone(**kw).to(device).eval()
    _alias_numpy_core()
    raw = torch.load(ckpt, map_location=device)
    sd, used_prefix = _extract_backbone_sd(raw, strip_prefix)
    missing, unexpected = bb.load_state_dict(sd, strict=False)
    n_model = len(bb.state_dict())
    loaded = n_model - len(missing)
    print(f"  loaded {os.path.basename(ckpt)}  prefix='{used_prefix}'  "
          f"backbone params {loaded}/{n_model}  (missing={len(missing)} unexpected={len(unexpected)})")
    if loaded < 0.9 * n_model:
        raise RuntimeError(
            f"only {loaded}/{n_model} backbone params loaded from {ckpt} -- probe "
            f"would read near-random features. Check --strip-prefix / config match.")
    return bb, kw


@torch.no_grad()
def feats_from_array(bb, X, device, bs=64):
    """X: (N,C,T,V,M) -> (N, Cf) mean-pooled over T,V,M via the pretrain adapter path."""
    out = []
    for i in range(0, len(X), bs):
        x = torch.tensor(np.asarray(X[i:i+bs]), dtype=torch.float32, device=device)
        N, C, T, V, M = x.shape
        y = x.permute(0,4,3,1,2).contiguous().view(N, M*V*C, T)
        y = bb.data_bn(y)
        y = y.view(N,M,V,C,T).permute(0,1,3,4,2).contiguous().view(N*M, C, T, V)
        for layer in bb.layers: y = layer(y)
        y = y.mean(dim=-1)                                # (N*M, Cf, T')
        y = y.view(N, M, -1, y.size(-1)).mean(dim=1)      # (N, Cf, T')
        out.append(y.mean(dim=-1).cpu().numpy())          # (N, Cf) pool T'
    return np.concatenate(out, 0)


def probe(F, y, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score
    F = StandardScaler().fit_transform(F)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(F, y):
        clf = LogisticRegression(max_iter=2000, C=1.0, multi_class="auto")
        clf.fit(F[tr], y[tr])
        accs.append(balanced_accuracy_score(y[te], clf.predict(F[te])))
    return float(np.mean(accs)), float(np.std(accs))


def linear_cka(A, B):
    A = A - A.mean(0); B = B - B.mean(0)
    hsic = np.linalg.norm(A.T @ B, "fro")**2
    return hsic / (np.linalg.norm(A.T @ A, "fro") * np.linalg.norm(B.T @ B, "fro"))


def probe_within_class(F, signer, cls, min_signers=2, min_per_signer=3, seed=0):
    r"""The PROPER class-conditioned signer-leakage measure (paper $\Lnorm^{cond}$).

        L_norm^cond = (1/C) * sum_c (BA^(c)_signer - 1/S_c) / (1 - 1/S_c)

    For EACH sign class c we fit a signer classifier using ONLY the clips of that
    class (so sign-label structure cannot help the probe), take the chance-
    normalised balanced signer accuracy within that class, and average over the
    usable classes. This is genuinely "signer decoded WITHIN each sign class" --
    NOT the weaker per-class-mean feature centering (--class-conditioned), which
    only removes the class-mean component before one GLOBAL probe.

    A class is usable only if it has >=min_signers signers each with
    >=min_per_signer clips (else a within-class signer split is degenerate).
    Returns (L_norm^cond, per-class chance-normalised accs, n_used, n_skipped).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score
    per_class = []
    used = skipped = 0
    for c in np.unique(cls):
        m = cls == c
        uy, yv = np.unique(signer[m], return_inverse=True)
        Sc = len(uy)
        counts = np.bincount(yv)
        if Sc < min_signers or counts.min() < min_per_signer:
            skipped += 1
            continue
        Fc = StandardScaler().fit_transform(F[m])
        n_splits = int(min(5, counts.min()))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        accs = [balanced_accuracy_score(
                    yv[te],
                    LogisticRegression(max_iter=2000, C=1.0).fit(Fc[tr], yv[tr]).predict(Fc[te]))
                for tr, te in skf.split(Fc, yv)]
        ba = float(np.mean(accs))
        per_class.append((ba - 1.0 / Sc) / (1.0 - 1.0 / Sc))
        used += 1
    lcond = float(np.mean(per_class)) if per_class else float("nan")
    return lcond, per_class, used, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt-b", default=None, help="second encoder (cka mode)")
    ap.add_argument("--mode", choices=["signer", "language", "cka"], required=True)
    ap.add_argument("--data", default="data/bdsl_si/train_data.npy")
    ap.add_argument("--label", default="data/bdsl_si/train_label.pkl")
    ap.add_argument("--max-clips", type=int, default=4000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--class-conditioned", action="store_true",
                    help="subtract per-CLASS mean feature before the signer probe (removes sign-label confounds)")
    ap.add_argument("--within-class", action="store_true",
                    help="compute the PROPER class-conditioned L_norm^cond: fit a signer probe "
                         "WITHIN each sign class and average chance-normalised accuracy over classes "
                         "(the paper formula). Supersedes --class-conditioned when set.")
    ap.add_argument("--min-signers", type=int, default=2,
                    help="[--within-class] skip a class with fewer signers than this")
    ap.add_argument("--min-per-signer", type=int, default=3,
                    help="[--within-class] skip a class where any signer has fewer clips than this")
    ap.add_argument("--signer-regex", default=r"U(\d+)W",
                    help="regex with one capture group giving the signer id from a clip name "
                         r"(BdSLW60: 'U(\d+)W'; BdSLW401: 'S(\d+)')")
    ap.add_argument("--sample-random", action="store_true",
                    help="randomly subsample max-clips (fixed seed) instead of taking the first "
                         "max-clips; use for large class-ordered sets like BdSLW401")
    ap.add_argument("--class-label", default="data/bdsl_si/train_label.pkl",
                    help="label pkl giving the sign class per clip (for --class-conditioned)")
    ap.add_argument("--strip-prefix", default=None,
                    help="submodule prefix to strip from checkpoint keys (auto-detects "
                         "'backbone.' by default; pass '' to disable)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bb, _ = build_backbone(a.config, a.ckpt, dev, a.strip_prefix)

    if a.mode in ("signer", "cka"):
        X = np.load(a.data, mmap_mode="r")
        names, clslab = pickle.load(open(a.label, "rb"))
        clslab = np.asarray(clslab)
        n = min(a.max_clips, len(X))
        if a.sample_random and n < len(X):
            # representative sample for large sets (e.g. BdSLW401's ~80k clips are
            # class-ordered, so X[:n] would see only the first few classes).
            idx = np.sort(np.random.default_rng(0).choice(len(X), size=n, replace=False))
            X = X[idx]; names = [names[i] for i in idx]; clslab = clslab[idx]
        else:
            X = X[:n]; names = names[:n]; clslab = clslab[:n]
        F = feats_from_array(bb, X, dev)
        if a.mode == "signer":
            sig = np.array([int(re.search(a.signer_regex, nm).group(1)) for nm in names])
            _, yv = np.unique(sig, return_inverse=True)
            if a.within_class:
                # PROPER class-conditioned measure: signer decoded WITHIN each sign class.
                lcond, per_c, used, skipped = probe_within_class(
                    F, yv, np.asarray(clslab),
                    min_signers=a.min_signers, min_per_signer=a.min_per_signer)
                pc = np.array(per_c)
                print(f"\n[{a.tag}] WITHIN-CLASS signer decodability (paper L_norm^cond):"
                      f" L_norm^cond = {lcond:.3f}"
                      f"  | per-class mean {pc.mean():.3f} +/- {pc.std():.3f}"
                      f"  | classes used {used} (skipped {skipped}, too few signers/clips)"
                      f"  | n={n}")
                print("  (formula: (1/C) sum_c (BA^(c)_signer - 1/S_c)/(1 - 1/S_c); "
                      "within-class probe, sign label cannot leak)")
                return
            cc = ""
            if a.class_conditioned:
                # remove the per-class-mean component so the probe cannot use sign-label structure
                Fc = F.copy()
                for c in np.unique(clslab):
                    m = clslab == c
                    Fc[m] = Fc[m] - Fc[m].mean(0, keepdims=True)
                F = Fc; cc = " [class-conditioned]"
            acc, sd = probe(F, yv)
            S = len(np.unique(yv)); chance = 1.0/S
            lnorm = (acc - chance) / (1 - chance)
            print(f"\n[{a.tag}]{cc} SIGNER decodability: balanced acc = {acc*100:.2f} +/- {sd*100:.2f}"
                  f"  | chance {chance*100:.2f}  | leakage = {(acc-chance)*100:+.2f} pp"
                  f"  | L_norm = {lnorm:.3f}  (n={n}, S={S})")
        else:
            bb2, _ = build_backbone(a.config, a.ckpt_b, dev, a.strip_prefix)
            F2 = feats_from_array(bb2, X, dev)
            print(f"\n[{a.tag}] linear CKA(encoderA, encoderB) = {linear_cka(F, F2):.4f}  (n={n})")
    else:  # language
        from feeders.pretrain_feeder import PretrainFeeder
        cfg = yaml.safe_load(open(a.config)); p = cfg["pretrain"]
        feeder = PretrainFeeder(manifest_path=p["manifest"], targets_path=p["targets"],
                                window_size=int(p.get("window_size",120)), num_channels=3,
                                num_joints=27, num_person=1, random_crop=False,
                                max_clips=a.max_clips, return_language=True)
        Xs, langs = [], []
        for i in range(len(feeder)):
            x, _c, lg = feeder[i]; Xs.append(np.asarray(x)); langs.append(int(lg))
        X = np.stack(Xs); yv = np.array(langs)
        F = feats_from_array(bb, X, dev)
        acc, sd = probe(F, yv)
        chance = 1.0/len(np.unique(yv))
        import collections
        print(f"\n[{a.tag}] LANGUAGE decodability: balanced acc = {acc*100:.2f} +/- {sd*100:.2f}"
              f"  | chance {chance*100:.2f}  | langs {dict(collections.Counter(yv))}  (n={len(yv)})")


if __name__ == "__main__":
    main()
