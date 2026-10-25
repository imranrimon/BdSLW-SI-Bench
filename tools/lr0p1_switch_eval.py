#!/usr/bin/env python
"""Evaluations for the P2 Adaptive GNN / Attention GNN lr-0.1 switch.

Sections (each idempotent; skipped when its inputs are missing):
  protoA   Protocol-A TEST (signers 2/13) of the 6 lr-0.1 best checkpoints via
           `main.py --phase test` with config/bdsl_<arch>_si_test.yaml (as the original
           work_dir/bdsl_<arch>_si_seed*_test runs did) -> work_dir/bdsl_<arch>_si_lr0p1_seed{S}_test/
           eval_results/best_acc.pkl (+ one results_final.csv row each, like the originals).
  oldpkl   side-effect-free TEST score dump of the OLD lr-0.01 Adaptive GNN seed 2 (its
           work_dir/bdsl_adaptive_gnn_si_seed2_test/eval_results/ is empty, so P2's metric suite and
           significance used only 2 Adaptive seeds) -> docs/adaptive_attn_lr0p1_switch/raw/
  rand     matched random splits (Adaptive GNN): val-selected checkpoint scored on own test, and the
           old estimator = max over the saved every-5-epoch checkpoints (= the old eval epochs) of
           own-test Top-1
  sd       signer-mixed: best-on-sd-val checkpoint scored on data/bdsl_sd/test (cross-check of the
           run_resumable_seed --test-config row) + sd-val best
  loso     held-out-signer Top-1 for every lr-0.1 LOSO fold-seed -> docs/loso_heldout/<arch>_lr0p1.csv
  bone     bone-stream TEST (from the --test-config dumps) + joint(lr0.1)+bone fusion bootstrap
Side-effect-free sections append JSON lines to docs/adaptive_attn_lr0p1_switch/raw/<section>.jsonl.

  python tools/lr0p1_switch_eval.py --sections protoA oldpkl rand sd loso bone
"""
import argparse, glob, json, os, pickle, re, subprocess, sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _s in ("multiarray", "numeric", "_multiarray_umath"):   # numpy-2 pickles on numpy-1
    try:
        import numpy.core as _nc
        sys.modules.setdefault("numpy._core", _nc)
        sys.modules.setdefault("numpy._core." + _s, getattr(_nc, _s))
    except Exception:
        pass
from main import import_class  # noqa: E402

RAW = ROOT / "docs" / "adaptive_attn_lr0p1_switch" / "raw"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ARCHS = ("adaptive_gnn", "gnn")
SEEDS = (0, 1, 2)
SI_TEST = ("data/bdsl_si/test_data.npy", "data/bdsl_si/test_label.pkl")
_DATA = {}


def _dataset(cfg, data, label):
    key = (cfg["feeder"], json.dumps(cfg["test_feeder_args"], sort_keys=True), data, label)
    if key not in _DATA:
        Feeder = import_class(cfg["feeder"])
        fa = dict(cfg["test_feeder_args"]); fa["data_path"] = data; fa["label_path"] = label
        _DATA[key] = Feeder(**fa)
    return _DATA[key]


def score(cfg_path, ckpt, data, label, dump=None):
    """Top-1 (%) of a checkpoint on an npy bundle (same procedure as tools/eval_ckpt_on_npy.py)."""
    cfg = yaml.safe_load(open(cfg_path))
    model = import_class(cfg["model"])(**cfg["model_args"]).to(DEV).eval()
    ck = torch.load(ckpt, map_location="cpu")
    w = ck["weights"] if isinstance(ck, dict) and "weights" in ck else ck
    model.load_state_dict({k.replace("module.", ""): v for k, v in w.items()}, strict=False)
    ds = _dataset(cfg, data, label)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.get("test_batch_size", 64), shuffle=False)
    names, out, cor, tot, off = list(ds.sample_name), {}, 0, 0, 0
    with torch.no_grad():
        for x, y, *_ in dl:
            o = model(x.float().to(DEV))
            o = (o[0] if isinstance(o, (tuple, list)) else o).cpu().numpy()
            cor += int((o.argmax(1) == y.numpy()).sum()); tot += len(y)
            for i in range(len(y)):
                out[names[off + i]] = o[i]
            off += len(y)
    if dump:
        os.makedirs(os.path.dirname(dump), exist_ok=True)
        pickle.dump(out, open(dump, "wb"))
    return 100.0 * cor / max(tot, 1)


def emit(section, rec):
    RAW.mkdir(parents=True, exist_ok=True)
    with open(RAW / f"{section}.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(section, rec, flush=True)


def done(section, **key):
    """True if a JSON line with these key values is already recorded (chunked reruns skip it)."""
    p = RAW / f"{section}.jsonl"
    if not p.exists(): return False
    for l in open(p):
        r = json.loads(l)
        if all(r.get(k) == v for k, v in key.items()): return True
    return False


def best(exp):
    p = ROOT / "work_dir" / exp / f"{exp}_model-best.pt"
    return str(p) if p.exists() else None


def last(exp, ep=99):
    p = ROOT / "work_dir" / exp / f"{exp}_model-{ep}.pt"
    return str(p) if p.exists() else None


def protoA():
    for a in ARCHS:
        for s in SEEDS:
            exp = f"bdsl_{a}_si_lr0p1_seed{s}"
            ck = best(exp)
            if not ck or not last(exp): continue
            texp = f"{exp}_test"
            pk = ROOT / "work_dir" / texp / "eval_results" / "best_acc.pkl"
            fresh = pk.exists() and pk.stat().st_mtime > os.path.getmtime(ck)
            if fresh and done("protoA", arch=a, seed=s): continue          # idempotent on chunk reruns
            if not fresh:
                subprocess.check_call([sys.executable, "-u", "main.py", "--config", f"config/bdsl_{a}_si_test.yaml",
                                       "--phase", "test", "--weights", os.path.relpath(ck, ROOT),
                                       "-Experiment_name", texp, "--seed", str(s)], cwd=ROOT)
            # cross-check with the side-effect-free scorer (same checkpoint, same TEST bundle)
            emit("protoA", {"arch": a, "seed": s, "ckpt": os.path.relpath(ck, ROOT),
                            "best_val_epoch": int(torch.load(ck, map_location="cpu").get("epoch", -1)),
                            "test_top1_recheck": score(ROOT / f"config/bdsl_{a}_si_test.yaml", ck, *SI_TEST)})


def oldpkl():
    exp = "bdsl_adaptive_gnn_si_seed2"
    ck = ROOT / "work_dir" / exp / f"{exp}_model-89.pt"      # the checkpoint its _test run loaded
    dump = str(RAW / "old_lr0p01_adaptive_gnn_seed2_test_best_acc.pkl")
    if ck.exists() and not (os.path.exists(dump) and done("oldpkl", exp=exp)):
        emit("oldpkl", {"exp": exp, "ckpt": os.path.relpath(ck, ROOT),
                        "test_top1": score(ROOT / "config/bdsl_adaptive_gnn_si_test.yaml", str(ck), *SI_TEST, dump=dump)})


def rand():
    for r in range(42, 47):
        exp = f"bdsl_adaptive_gnn_rand{r}_lr0p1_seed0"
        ck = best(exp)
        if not ck or not last(exp) or done("rand", split=r): continue
        cfg = ROOT / "work_dir" / exp / "config.yaml"
        te = (f"data/bdsl_rand{r}/test_data.npy", f"data/bdsl_rand{r}/test_label.pkl")
        all_ck = sorted(glob.glob(str(ROOT / "work_dir" / exp / f"{exp}_model-[0-9]*.pt")),
                        key=lambda p: int(re.search(r"model-(\d+)\.pt$", p).group(1)))
        per = {int(re.search(r"model-(\d+)\.pt$", p).group(1)): score(cfg, p, *te) for p in all_ck}
        emit("rand", {"split": r, "val_selected_test": score(cfg, ck, *te),
                      "val_best": float(torch.load(ck, map_location="cpu").get("acc", -1)) * 100,
                      "old_estimator_max_test": max(per.values()), "n_ckpts": len(per),
                      "ckpt_epochs": sorted(per), "per_epoch_test": per})


def sd():
    for a in ARCHS:
        for s in SEEDS:
            exp = f"bdsl_{a}_sd_noclip_lr0p1_seed{s}"
            ck = best(exp)
            if not ck or not last(exp) or done("sd", arch=a, seed=s): continue
            cfg = ROOT / "work_dir" / exp / "config.yaml"
            emit("sd", {"arch": a, "seed": s,
                        "sd_test": score(cfg, ck, "data/bdsl_sd/test_data.npy", "data/bdsl_sd/test_label.pkl"),
                        "sd_val_best": float(torch.load(ck, map_location="cpu").get("acc", -1)) * 100})


def loso():
    for a in ARCHS:
        out = ROOT / "docs" / "loso_heldout" / f"{a}_lr0p1.csv"
        rows = ["fold,seed,heldout_signer,test_acc"]
        for d in sorted(glob.glob(str(ROOT / "work_dir" / f"bdsl_{a}_si_lr0p1_loso_test*_seed*"))):
            exp = os.path.basename(d)
            m = re.match(r".*_loso_test(\d+)_seed(\d+)$", exp)
            if not m or not best(exp) or not last(exp): continue
            fold, seed = m.group(1), m.group(2)
            fd = glob.glob(str(ROOT / f"data/bdsl_si_loso/test_U{fold}_val_U*"))
            if not fd: continue
            ck = best(exp)
            cache = [json.loads(l) for l in open(RAW / "loso.jsonl")] if (RAW / "loso.jsonl").exists() else []
            hit = [r for r in cache if r["exp"] == exp and r["ckpt_mtime"] == os.path.getmtime(ck)]
            if hit:
                acc = hit[-1]["heldout_top1"]                      # cached (chunked reruns stay cheap)
            else:
                acc = score(ROOT / "work_dir" / exp / "config.yaml", ck,
                            f"{fd[0]}/test_data.npy", f"{fd[0]}/test_label.pkl")
                emit("loso", {"exp": exp, "ckpt_mtime": os.path.getmtime(ck), "heldout_top1": acc})
            rows.append(f"{fold},{seed},U{fold},{acc:.2f}")
        out.write_text("\n".join(rows) + "\n")
        print(f"loso {a}: {len(rows) - 1} fold-seed rows -> {out}")


def bone():
    for s in SEEDS:
        exp = f"bdsl_adaptive_gnn_bone_lr0p1_seed{s}"
        ck = best(exp)
        if not ck or not last(exp) or done("bone", seed=s): continue
        emit("bone", {"seed": s, "bone_test": score(ROOT / "config/lr0p1_switch/bdsl_adaptive_gnn_bone_lr0p1_test.yaml",
                                                    ck, "data/bdsl_si/test_data_bone.npy", "data/bdsl_si/test_label.pkl"),
                      "bone_val_best": float(torch.load(ck, map_location="cpu").get("acc", -1)) * 100})
    jp = [ROOT / f"work_dir/bdsl_adaptive_gnn_si_lr0p1_seed{s}_test/eval_results/best_acc.pkl" for s in SEEDS]
    bp = [ROOT / f"work_dir/bdsl_adaptive_gnn_bone_lr0p1_test_seed{s}/eval_results/best_acc.pkl" for s in SEEDS]
    if all(p.exists() for p in jp + bp) and not (RAW / "fusion_adaptive_lr0p1.md").exists():
        import io, contextlib
        import tools.fusion_bootstrap as F
        F.load = lambda seed: (pickle.load(open(jp[seed], "rb")), pickle.load(open(bp[seed], "rb")))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            F.main()
        (RAW / "fusion_adaptive_lr0p1.md").write_text(buf.getvalue())
        print(buf.getvalue())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", nargs="+", required=True,
                    choices=["protoA", "oldpkl", "rand", "sd", "loso", "bone"])
    for sec in ap.parse_args().sections:
        globals()[sec]()
