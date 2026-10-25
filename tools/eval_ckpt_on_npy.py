#!/usr/bin/env python
"""Side-effect-free checkpoint eval: load a trained model + a data/label npy and
report Top-1 (optionally dump a score_dict). Does NOT touch results_final.csv or
create work_dirs -- used to verify whether the LOSO CSV numbers are held-out-TEST
or VAL signer (P1 caveat check).

Usage:
  python tools/eval_ckpt_on_npy.py --config work_dir/<run>/config.yaml \
      --weights work_dir/<run>/<run>_model-best.pt \
      --data data/bdsl_si_loso/test_U08_val_U01/test_data.npy \
      --label data/bdsl_si_loso/test_U08_val_U01/test_label.pkl
"""
import argparse, importlib, os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, yaml


def import_class(name):
    mod, cls = name.rsplit(".", 1)
    return getattr(importlib.import_module(mod), cls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--dump", default=None, help="optional pkl path for {clip:score}")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    Model = import_class(cfg["model"])
    model = Model(**cfg["model_args"]).to(dev).eval()
    ck = torch.load(a.weights, map_location="cpu")
    w = ck["weights"] if isinstance(ck, dict) and "weights" in ck else ck
    w = {k.replace("module.", ""): v for k, v in w.items()}
    missing, unexpected = model.load_state_dict(w, strict=False)
    if missing or unexpected:
        print(f"[warn] missing={len(missing)} unexpected={len(unexpected)}", file=sys.stderr)

    # build the test feeder exactly as main.py would, but on the given npy
    Feeder = import_class(cfg["feeder"])
    fa = dict(cfg["test_feeder_args"])
    fa["data_path"] = a.data
    fa["label_path"] = a.label
    ds = Feeder(**fa)
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg.get("test_batch_size", 64),
                                         shuffle=False, num_workers=0)
    names = list(ds.sample_name)

    scores, offset = {}, 0
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0], batch[1]
            out = model(x.float().to(dev))
            if isinstance(out, (tuple, list)):
                out = out[0]
            out = out.cpu().numpy()
            pred = out.argmax(1)
            yy = y.numpy()
            correct += int((pred == yy).sum()); total += len(yy)
            for i in range(len(yy)):
                scores[names[offset + i]] = out[i]
            offset += len(yy)

    acc = correct / max(total, 1)
    print(f"top1={acc*100:.2f}  n={total}  data={a.data}")
    if a.dump:
        pickle.dump(scores, open(a.dump, "wb"))
    return acc


if __name__ == "__main__":
    main()
