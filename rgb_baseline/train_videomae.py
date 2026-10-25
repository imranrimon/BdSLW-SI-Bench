#!/usr/bin/env python
"""P2 Tier-1 #3 — VideoMAE RGB baseline on BdSLW60-SI (unseen-signer).

Reuses the BdSLW60RGBDataset of rgb_baseline (SI signer split + RGB decode). Loads the
Kinetics-pretrained VideoMAE-base from the local HF cache (offline compute nodes),
replaces the head with 60 classes, fine-tunes, and reports held-out TEST-signer
Top-1. Runs in the dedicated `videomae` conda env (transformers installed there).

Usage (in env `videomae`):
  HF_HUB_OFFLINE=1 python -m rgb_baseline.train_videomae \
      --root data/bdslw60_clips --classes-json <classes.json> --seed 0 \
      --exp-name rgb_videomae_bdsl_si_seed0 --epochs 20
"""
import argparse, csv, datetime, json, os, pickle, sys
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from rgb_baseline.rgb_dataset import BdSLW60RGBDataset  # noqa: E402

MODEL_ID = "MCG-NJU/videomae-base-finetuned-kinetics"


def build_loader(root, split, cj, train, bs, nw, split_dir=None):
    names_pkl = os.path.join(split_dir, f"{split}_label.pkl") if split_dir else None
    ds = BdSLW60RGBDataset(root=root, split=split, classes_json=cj, num_frames=16,
                           size=224, resize_short=256, train=train, normalize="kinetics",
                           names_pkl=names_pkl)
    return ds, DataLoader(ds, batch_size=bs, shuffle=train, num_workers=nw, drop_last=train, pin_memory=True)


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval(); corr = n = 0
    for clip, label, _ in loader:
        pv = clip.permute(0, 2, 1, 3, 4).to(dev)        # (B,C,T,H,W)->(B,T,C,H,W)
        logits = model(pixel_values=pv).logits
        corr += int((logits.argmax(1).cpu() == label).sum()); n += len(label)
    return corr / max(1, n)


@torch.no_grad()
def predict(model, loader, dev):
    """Per-clip softmax keyed by clip file name, plus Top-1."""
    model.eval(); out = {}; corr = n = 0
    names = loader.dataset.sample_name; i = 0
    for clip, label, _ in loader:
        pv = clip.permute(0, 2, 1, 3, 4).to(dev)
        prob = torch.softmax(model(pixel_values=pv).logits.float(), 1).cpu()
        for p in prob:
            out[names[i]] = p.numpy(); i += 1
        corr += int((prob.argmax(1) == label).sum()); n += len(label)
    return out, corr / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/bdslw60_clips")
    ap.add_argument("--classes-json", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exp-name", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--num-worker", type=int, default=8)
    ap.add_argument("--split-dir", default=None,
                    help="optional pose-bundle dir (e.g. data/bdsl_sd): split = its {train,val,test}_label.pkl clip names "
                         "instead of the canonical SI signer split")
    ap.add_argument("--out-dir", default=None, help="default work_dir/<exp-name>; holds last.pt (resume), best.pt, scores")
    a = ap.parse_args()
    out_dir = a.out_dir or os.path.join("work_dir", a.exp_name)
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import VideoMAEForVideoClassification

    _, trl = build_loader(a.root, "train", a.classes_json, True, a.bs, a.num_worker, a.split_dir)
    _, val = build_loader(a.root, "val", a.classes_json, False, a.bs, a.num_worker, a.split_dir)
    teds, tel = build_loader(a.root, "test", a.classes_json, False, a.bs, a.num_worker, a.split_dir)
    ncls = len(teds.classes)
    print(f"VideoMAE-SI: train batches {len(trl)} | val {len(val.dataset)} | test {len(teds)} | classes {ncls}")

    model = VideoMAEForVideoClassification.from_pretrained(
        MODEL_ID, num_labels=ncls, ignore_mismatched_sizes=True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    best_val = 0.0; best_state = None; start_ep = 0; hist = []
    last_p, best_p = os.path.join(out_dir, "last.pt"), os.path.join(out_dir, "best.pt")
    if os.path.exists(last_p):                       # resume after requeue
        ck = torch.load(last_p, map_location="cpu")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_ep, best_val, hist = ck["epoch"] + 1, ck["best_val"], ck.get("hist", [])
        if os.path.exists(best_p):
            best_state = torch.load(best_p, map_location="cpu")
        print(f"[resume] from {last_p}: start epoch {start_ep}, best-val {best_val*100:.2f}")
    for ep in range(start_ep, a.epochs):
        model.train(); tot = 0.0
        for clip, label, _ in trl:
            pv = clip.permute(0, 2, 1, 3, 4).to(dev); label = label.to(dev)
            loss = ce(model(pixel_values=pv).logits, label)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss.item())
        vscores, va = predict(model, val, dev)
        hist.append({"epoch": ep, "val_top1": va, "loss": tot / max(1, len(trl))})
        print(f"[ep {ep}] loss {tot/max(1,len(trl)):.4f} val {va*100:.2f}")
        if va > best_val:
            best_val = va; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, best_p)
            with open(os.path.join(out_dir, "val_best_scores.pkl"), "wb") as f:
                pickle.dump(vscores, f)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep,
                    "best_val": best_val, "hist": hist}, last_p + ".tmp")
        os.replace(last_p + ".tmp", last_p)
    if best_state:
        model.load_state_dict(best_state)
    tscores, te = predict(model, tel, dev)
    with open(os.path.join(out_dir, "test_scores.pkl"), "wb") as f:
        pickle.dump(tscores, f)
    json.dump({"exp": a.exp_name, "split_dir": a.split_dir, "best_val": best_val, "test": te, "hist": hist},
              open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    print(f"DONE {a.exp_name}: best-val {best_val*100:.2f} | TEST {te*100:.2f}")
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("results_final.csv", "a", newline="") as f:
        csv.writer(f).writerow([ts, a.exp_name, a.epochs, f"{te:.4f}", f"{best_val:.4f}", "videomae_test", "path4"])


if __name__ == "__main__":
    main()
