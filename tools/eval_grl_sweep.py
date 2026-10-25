"""E3 -- GRL signer-adversarial lambda sweep: SI test accuracy + signer leakage.

For each signer-adversarial checkpoint (trained by main_signer_adv.py, saved
under key 'weights' with a backbone./class_head./signer_head. structure) this
computes, on ONE GPU:

  * SI test Top-1  -- the real signer_adv forward (class head on backbone
                      features), on the canonical held-out TEST signers {2,13}.
  * signer leakage -- class-conditioned linear decodability of signer identity
                      from the SAME backbone features used by the classifier
                      (residualize per-class mean z' = z - mu_y, then a
                      LogisticRegression probe; L_norm = (acc-chance)/(1-chance)).

Both read-outs use `model.backbone(x)` directly, so accuracy reproduces training
and leakage is measured on exactly the features the classifier sees.

Usage (one GPU):
  python tools/eval_grl_sweep.py \\
    --config config/bdsl_block_gcn_si_signeradv.yaml \\
    --ckpts \\
      0.1=work_dir/bdsl_block_gcn_si_signeradv_gl01_seed0/bdsl_block_gcn_si_signeradv_gl01_seed0_model-best.pt \\
      0.5=work_dir/bdsl_block_gcn_si_signeradv_gl05_seed0/bdsl_block_gcn_si_signeradv_gl05_seed0_model-best.pt \\
      1.0=work_dir/bdsl_block_gcn_si_signeradv_gl10_seed0/bdsl_block_gcn_si_signeradv_gl10_seed0_model-best.pt
"""
import argparse, os, re, pickle, sys
import numpy as np
import torch, yaml

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from tools.decodability_probe import _imp, probe


def build_model(cfg, ckpt, device):
    ma = dict(cfg["model_args"])
    sd = torch.load(ckpt, map_location=device)
    w = sd["weights"] if "weights" in sd else sd
    # infer num_signer / feat_dim from the checkpoint so we can load strict.
    num_signer = w["signer_head.2.weight"].shape[0]
    feat_dim = w["class_head.weight"].shape[1]
    Model = _imp(cfg["model"])
    model = Model(backbone=ma["backbone"], backbone_args=ma["backbone_args"],
                  num_class=ma.get("num_class", 60), num_signer=num_signer,
                  feat_dim=feat_dim, grl_lambda=ma.get("grl_lambda", 1.0)).to(device).eval()
    missing, unexpected = model.load_state_dict(w, strict=False)
    assert not missing, f"missing keys: {missing[:5]}"
    return model


@torch.no_grad()
def features_and_logits(model, X, device, bs=64):
    feats, logits = [], []
    for i in range(0, len(X), bs):
        x = torch.tensor(np.asarray(X[i:i+bs]), dtype=torch.float32, device=device)
        f = model.backbone(x, 1.0)
        if isinstance(f, tuple):
            f = f[0]
        feats.append(f.cpu().numpy())
        logits.append(model.class_head(f).cpu().numpy())
    return np.concatenate(feats), np.concatenate(logits)


def load_split(cfg, split):
    fa = cfg["test_feeder_args"]
    base = os.path.dirname(fa["data_path"])
    X = np.load(os.path.join(base, f"{split}_data.npy"), mmap_mode="r")
    names, labels = pickle.load(open(os.path.join(base, f"{split}_label.pkl"), "rb"))
    return X, names, np.asarray(labels)


def cc_probe(F, signer, cls):
    Fc = F.copy()
    for c in np.unique(cls):
        m = cls == c
        Fc[m] = Fc[m] - Fc[m].mean(0, keepdims=True)
    _, y = np.unique(signer, return_inverse=True)
    acc, sd = probe(Fc, y)
    S = len(np.unique(y)); chance = 1.0 / S
    return acc, sd, chance, (acc - chance) / (1 - chance), S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True, help="lambda=path pairs")
    ap.add_argument("--max-clips", type=int, default=4000)
    ap.add_argument("--out", default=None, help="optional markdown output path")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # test signers (accuracy) and train signers (leakage) once.
    Xte, names_te, y_te = load_split(cfg, "test")
    Xtr, names_tr, y_tr = load_split(cfg, "train")
    n = min(a.max_clips, len(Xtr))
    Xtr, names_tr, cls_tr = Xtr[:n], names_tr[:n], y_tr[:n]
    sig_tr = np.array([int(re.search(r"U(\d+)W", s).group(1)) for s in names_tr])

    rows = []
    for pair in a.ckpts:
        lam, path = pair.split("=", 1)
        print(f"\n=== lambda={lam}  {os.path.basename(path)} ===")
        model = build_model(cfg, path, dev)
        # accuracy on held-out TEST signers
        _, logits = features_and_logits(model, Xte, dev)
        acc_test = float((logits.argmax(1) == y_te).mean()) * 100
        # leakage on TRAIN features (class-conditioned)
        F, _ = features_and_logits(model, Xtr, dev)
        lk, lksd, chance, lnorm, S = cc_probe(F, sig_tr, cls_tr)
        print(f"  SI test Top-1 = {acc_test:.2f}%   "
              f"signer leakage acc = {lk*100:.2f}+/-{lksd*100:.2f} "
              f"(chance {chance*100:.1f}, S={S}) -> L_norm={lnorm:.3f}")
        rows.append((float(lam), acc_test, lk * 100, lnorm))

    rows.sort()
    print("\n" + "=" * 56 + "\nlambda   SI-Top1    leak-acc   L_norm")
    for lam, acc, lk, ln in rows:
        print(f"{lam:5.2f}   {acc:7.2f}   {lk:7.2f}   {ln:6.3f}")
    if a.out:
        with open(a.out, "w") as f:
            f.write("| $\\lambda$ | SI Top-1 (%) | signer leak acc (%) | $L_{norm}$ |\n")
            f.write("|---|---|---|---|\n")
            for lam, acc, lk, ln in rows:
                f.write(f"| {lam:g} | {acc:.2f} | {lk:.2f} | {ln:.3f} |\n")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
