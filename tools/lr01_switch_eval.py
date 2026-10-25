#!/usr/bin/env python
"""Side-effect-free evaluations for the P2 BlockGCN lr-0.01 switch (no training, no CSV writes).

Sections (each idempotent; skipped when its inputs are missing):
  protoA  TEST score dumps for the lr-0.01 Protocol-A runs
          -> work_dir/bdsl_block_gcn_si_lr01_seed{S}_test/eval_results/best_acc.pkl
  sd      signer-mixed: best-on-sd-val checkpoint scored on data/bdsl_sd/test
  rand    matched random splits: best-on-own-val checkpoint scored on own test (val-selected),
          plus the old estimator = max over saved checkpoints of own-test Top-1
  cuectl  cue control: final-epoch (model-99) lr-0.01 checkpoints scored on TEST
  sslold  the OLD lr-0.1 native pose-SSL fine-tunes (J4) scored on TEST (they were only val)
  loso    held-out-signer Top-1 for every lr-0.01 LOSO fold -> docs/loso_heldout/block_gcn_lr01.csv
Results are appended as JSON lines to docs/blockgcn_lr01_switch/raw/<section>.jsonl.

  python tools/lr01_switch_eval.py --sections protoA sd rand cuectl sslold loso
"""
import argparse, glob, json, os, pickle, re, sys
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

RAW = ROOT / "docs" / "blockgcn_lr01_switch" / "raw"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
_DATA = {}


def _dataset(cfg, data, label):
    key = (data, label)
    if key not in _DATA:
        Feeder = import_class(cfg["feeder"])
        fa = dict(cfg["test_feeder_args"]); fa["data_path"] = data; fa["label_path"] = label
        _DATA[key] = Feeder(**fa)
    return _DATA[key]


def score(cfg_path, ckpt, data, label, dump=None):
    """Top-1 (%) of a checkpoint on an npy bundle; optional {clip: logits} dump."""
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


def best(exp):
    p = ROOT / "work_dir" / exp / f"{exp}_model-best.pt"
    return str(p) if p.exists() else None


def last(exp, ep=99):
    p = ROOT / "work_dir" / exp / f"{exp}_model-{ep}.pt"
    return str(p) if p.exists() else None


SI_TEST = ("data/bdsl_si/test_data.npy", "data/bdsl_si/test_label.pkl")


def protoA():
    for s in (0, 1, 2):
        exp = f"bdsl_block_gcn_si_lr01_scratch_seed{s}"
        ck = best(exp)
        if not ck: continue
        dump = str(ROOT / f"work_dir/bdsl_block_gcn_si_lr01_seed{s}_test/eval_results/best_acc.pkl")
        emit("protoA", {"seed": s, "test_top1": score(ROOT / "work_dir" / exp / "config.yaml", ck, *SI_TEST, dump=dump), "ckpt": ck})


def sd():
    for s in (0, 1, 2):
        exp = f"bdsl_block_gcn_sd_lr01_seed{s}"
        ck = best(exp)
        if not ck or not last(exp): continue
        cfg = ROOT / "work_dir" / exp / "config.yaml"
        emit("sd", {"seed": s, "sd_test": score(cfg, ck, "data/bdsl_sd/test_data.npy", "data/bdsl_sd/test_label.pkl"),
                    "sd_val_best": float(torch.load(ck, map_location="cpu").get("acc", -1)) * 100})


def rand():
    for r in range(42, 47):
        exp = f"bdsl_block_gcn_rand{r}_lr01_seed0"
        ck = best(exp)
        if not ck or not last(exp): continue
        cfg = ROOT / "work_dir" / exp / "config.yaml"
        te = (f"data/bdsl_rand{r}/test_data.npy", f"data/bdsl_rand{r}/test_label.pkl")
        all_ck = sorted(glob.glob(str(ROOT / "work_dir" / exp / f"{exp}_model-[0-9]*.pt")),
                        key=lambda p: int(re.search(r"model-(\d+)\.pt$", p).group(1)))
        per = {int(re.search(r"model-(\d+)\.pt$", p).group(1)): score(cfg, p, *te) for p in all_ck}
        emit("rand", {"split": r, "val_selected_test": score(cfg, ck, *te),
                      "old_estimator_max_test": max(per.values()), "n_ckpts": len(per),
                      "max_test_at_eval_epochs": max([v for e, v in per.items() if e % 5 == 4] or [float("nan")])})


def cuectl():
    for s in (0, 1, 2):
        exp = f"bdsl_block_gcn_si_lr01_scratch_seed{s}"
        ck = last(exp)
        if not ck: continue
        emit("cuectl", {"seed": s, "final_epoch_test": score(ROOT / "work_dir" / exp / "config.yaml", ck, *SI_TEST)})


def sslold():
    for s in (0, 1, 2):
        exp = f"bdsl_ssl_bdslonly_si_ft_seed{s}"
        ck = best(exp)
        if not ck: continue
        emit("sslold", {"seed": s, "test_top1": score(ROOT / "work_dir" / exp / "config.yaml", ck, *SI_TEST)})


def loso():
    out = ROOT / "docs" / "loso_heldout" / "block_gcn_lr01.csv"
    rows = ["fold,seed,heldout_signer,test_acc"]
    for d in sorted(glob.glob(str(ROOT / "work_dir" / "bdsl_block_gcn_si_lr01_loso_test*_seed*"))):
        exp = os.path.basename(d)
        m = re.match(r".*_loso_test(\d+)_seed(\d+)$", exp)
        if not m or not best(exp) or not last(exp): continue
        fold, seed = m.group(1), m.group(2)
        fd = glob.glob(str(ROOT / f"data/bdsl_si_loso/test_U{fold}_val_U*"))
        if not fd: continue
        acc = score(ROOT / "work_dir" / exp / "config.yaml", best(exp), f"{fd[0]}/test_data.npy", f"{fd[0]}/test_label.pkl")
        rows.append(f"{fold},{seed},U{fold},{acc:.2f}")
    out.write_text("\n".join(rows) + "\n")
    print(f"loso: {len(rows) - 1} fold-seed rows -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", nargs="+", required=True,
                    choices=["protoA", "sd", "rand", "cuectl", "sslold", "loso"])
    for sec in ap.parse_args().sections:
        globals()[sec]()
