#!/usr/bin/env python3
"""Aggregate Baseline items 4-6 (P2 Table 7): TEST Top-1 + macro-F1 for VideoMAE,
S3D, Uni-Sign, SHuBERT on Protocol A, and signer-mixed ("Mixed") accuracy + SMG for
VideoMAE and Uni-Sign. Side-effect-free except writing the report.

Mixed = the sd-VAL-selected model scored on data/bdsl_sd TEST (1,367 clips; P2 protocol R5,
docs/audit_reruns_2026-10/r5_mixed_selection.md). The old estimator (best-epoch sd-val Top-1) is
kept as the secondary column `mixed_val`.
SD = population SD (matches the paper tables). Missing runs are reported as such.
Writes docs/baselines_rerun/item4to6_rgb_foundation.{md,json}.
"""
import json, os, pickle, re
import numpy as np

UNI = "external/Uni-Sign/out"


def macro_f1(y, yh):
    f = []
    for c in sorted(set(y)):
        tp = sum(a == c and b == c for a, b in zip(y, yh))
        fp = sum(a != c and b == c for a, b in zip(y, yh))
        fn = sum(a == c and b != c for a, b in zip(y, yh))
        f.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return float(np.mean(f) * 100)


def from_scores(scores, labels):
    """scores: {clip name: prob/logit vector}; labels: {clip name: int}."""
    k = sorted(set(scores) & set(labels))
    if len(k) != len(labels):
        print(f"  WARNING: {len(k)}/{len(labels)} test clips scored")
    y = [labels[x] for x in k]; yh = [int(np.argmax(scores[x])) for x in k]
    return dict(top1=float(np.mean([a == b for a, b in zip(y, yh)]) * 100), f1=macro_f1(y, yh), n=len(k))


def si_test_labels():
    n, l = pickle.load(open("data/bdsl_si/test_label.pkl", "rb"))
    return dict(zip(n, l))


def sd_labels(split):
    n, l = pickle.load(open(f"data/bdsl_sd/{split}_label.pkl", "rb"))
    return {os.path.splitext(x)[0] + ".mp4": y for x, y in zip(n, l)}


def unisign_eval(d):
    def rd(f):
        out = {}
        for line in open(f):
            m = re.match(r"sample: ([^,]+), [\w-]+: ?(.*)$", line.rstrip("\n"))
            if m: out[m[1]] = m[2].strip()
        return out
    p, r = rd(f"{d}/test_tmp_pres.txt"), rd(f"{d}/test_tmp_refs.txt")
    k = sorted(r); y = [r[x] for x in k]; yh = [p.get(x, "") for x in k]
    return dict(top1=float(np.mean([a == b for a, b in zip(y, yh)]) * 100), f1=macro_f1(y, yh), n=len(k))


def unisign_best_dev(d):
    L = [json.loads(x) for x in open(f"{d}/log.txt") if x.startswith("{")]
    return dict(mixed=max(x["test_top1_acc_pi"] for x in L), epochs=len(L))


def agg(vals):
    v = [x for x in vals if x is not None]
    return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (None, None, 0)


def main():
    lab = si_test_labels()
    R = {}

    # VideoMAE: SI re-run (per-clip dumps) + SD
    vm = {"seeds": {}}
    for s in (0, 1, 2):
        e = {}
        p = f"work_dir/rgb_videomae_bdsl_si_rerun_seed{s}/test_scores.pkl"
        if os.path.exists(p): e.update(from_scores(pickle.load(open(p, "rb")), lab))
        q = f"work_dir/rgb_videomae_bdsl_sd_seed{s}"
        if os.path.exists(f"{q}/summary.json"):
            e["mixed_val"] = json.load(open(f"{q}/summary.json"))["best_val"] * 100
            e["mixed"] = from_scores(pickle.load(open(f"{q}/test_scores.pkl", "rb")), sd_labels("test"))["top1"]
        vm["seeds"][s] = e
    R["VideoMAE"] = vm

    # S3D: seeds 0,1 (Windows runs) + seed2r (cluster re-train)
    s3 = {"seeds": {}}
    for s, d in [(0, "rgb_s3d_bdsl_si_seed0"), (1, "rgb_s3d_bdsl_si_seed1"), (2, "rgb_s3d_bdsl_si_seed2r")]:
        p = f"work_dir/{d}/eval_results/testset_scores.pkl"
        s3["seeds"][s] = from_scores(pickle.load(open(p, "rb")), lab) if os.path.exists(p) else {}
        q = f"work_dir/rgb_s3d_bdsl_sd_seed{s}/eval_results/testset_scores.pkl"   # written only after all 30 epochs
        if os.path.exists(q):
            v = pickle.load(open(f"work_dir/rgb_s3d_bdsl_sd_seed{s}/eval_results/best_acc.pkl", "rb"))
            s3["seeds"][s]["mixed_val"] = from_scores(v, sd_labels("val"))["top1"]
            s3["seeds"][s]["mixed"] = from_scores(pickle.load(open(q, "rb")), sd_labels("test"))["top1"]
    R["S3D"] = s3

    # Uni-Sign: SI seed 42 (original run) + 1, 2; SD seeds 42, 1, 2
    us = {"seeds": {}}
    for s in (42, 1, 2):
        e = {}
        d = f"{UNI}/bdslw60_si_ft_lowdata_eval" if s == 42 else f"{UNI}/bdslw60_si_ft_lowdata_seed{s}_eval"
        if os.path.exists(f"{d}/test_tmp_pres.txt"): e.update(unisign_eval(d))
        q = f"{UNI}/bdslw60_sd_ft_lowdata_seed{s}"
        if os.path.exists(f"{q}/log.txt"):
            b = unisign_best_dev(q)
            if b["epochs"] >= 100: e["mixed_val"] = b["mixed"]
            else: e["mixed_partial_epochs"] = b["epochs"]
        if os.path.exists(f"{q}_eval/test_tmp_pres.txt"):
            e["mixed"] = unisign_eval(f"{q}_eval")["top1"]
        us["seeds"][s] = e
    R["Uni-Sign"] = us

    # SHuBERT frozen probe (features fixed; seeds vary probe init)
    sh = {"seeds": {}}
    for s in (0, 1, 2):
        p = f"work_dir/shubert_probe_seg/si_test_pred_seed{s}.pkl"
        if os.path.exists(p):
            d = pickle.load(open(p, "rb")); y = list(d["label"]); yh = list(d["prob"].argmax(1))
            sh["seeds"][s] = dict(top1=float(np.mean([a == b for a, b in zip(y, yh)]) * 100), f1=macro_f1(y, yh), n=len(y))
        else:
            sh["seeds"][s] = {}
        q = f"work_dir/shubert_probe_seg/sd_test_pred_seed{s}.pkl"
        if os.path.exists(q):
            sh["seeds"][s]["mixed_val"] = json.load(open(q + ".json"))["best_val"]
            d = pickle.load(open(q, "rb"))
            sh["seeds"][s]["mixed"] = float(np.mean(d["prob"].argmax(1) == d["label"]) * 100)
    R["SHuBERT"] = sh

    lines = ["# Baseline items 4-6 — RGB + foundation baselines (P2 Table 7)", "",
             "Protocol A held-out TEST signers {2,13}. **Mixed = the model selected on `data/bdsl_sd/val` scored on "
             "`data/bdsl_sd/test` (1,367 clips, signers shared with train)** — the P2 R5 protocol "
             "(docs/audit_reruns_2026-10/r5_mixed_selection.md). `Mixed (old)` = best-epoch sd-val Top-1, the superseded "
             "estimator, kept for reference. mean ± SD = population SD over seeds. SMG = Mixed − TEST (pp). Each model "
             "keeps its original recipe.", "",
             "| Model | TEST Top-1 | Macro-F1 | Mixed (sd test) | SMG | Mixed (old, sd-val best) | seeds (TEST/Mixed) |",
             "|---|--:|--:|--:|--:|--:|--:|"]
    for name, r in R.items():
        sd = r["seeds"]
        t = agg([e.get("top1") for e in sd.values()]); f = agg([e.get("f1") for e in sd.values()])
        m = agg([e.get("mixed") for e in sd.values()]); mo = agg([e.get("mixed_val") for e in sd.values()])
        r["summary"] = dict(top1=t, f1=f, mixed=m, mixed_val=mo)
        fmt = lambda a: "–" if a[0] is None else f"{a[0]:.2f} ± {a[1]:.2f}"
        smg = "–" if (t[0] is None or m[0] is None) else f"{m[0] - t[0]:+.2f}"
        lines.append(f"| {name} | {fmt(t)} | {fmt(f)} | {fmt(m)} | {smg} | {fmt(mo)} | {t[2]}/{m[2]} |")
    lines += ["", "## Params / GFLOPs (fills Table 7 cost cells)", "",
              "GFLOPs follow the paper's convention: raw `thop` count (multiply-accumulates of Linear/Conv layers; "
              "attention matmuls not counted), single clip.", "",
              "| Model | Params (M) | GFLOPs | What is counted |", "|---|--:|--:|---|",
              "| S3D | 7.97 | 36.3 | full network, thop, 1×3×32×224² training clip |",
              "| Uni-Sign (fine-tuned) | 587.7 | 9.9 | pose encoder 5.3 M + mT5-base 582.4 M (incl. 250k-vocab embeddings); thop, one "
              "teacher-forced forward on the median SI-test clip (39 frames); excludes RTMPose extraction and autoregressive decode steps |",
              "| SHuBERT (frozen probe) | 87.4 | 3.3 | SHuBERT encoder at inference (87.33 M; pretraining heads excluded) + linear "
              "probe (0.05 M); analytic thop-equivalent at the median 38 feature frames; EXCLUDES the two frozen DINOv2 ViT-S/14 "
              "crop encoders (+43.2 M params, ≈+629 GFLOPs for 3 crops × 38 frames) and MediaPipe/YOLO |", ""]
    lines += ["## Provenance and caveats", "",
              "- **VideoMAE SI is a full 3-seed re-run** (original recipe: 20 ep, bs 6, lr 5e-5; job) because the "
              "original runs saved no predictions. New per-seed TEST 78.10/83.15/81.54 = 80.93 ± 2.11 vs the paper's "
              "78.76 ± 3.78 (82.64/73.63/80.00). Same seeds do NOT reproduce per-seed (non-deterministic decode/augment "
              "workers + cuDNN); the means agree within one seed-SD. Macro-F1 is from the re-run.",
              "- **S3D: the paper's 77.11 ± 1.42 is a 2-seed mean** (seeds 0,1; Windows runs). Seed 2 never reached its "
              "TEST eval (training stopped near epoch 19 of 30), so it was re-trained in full on the cluster "
              "(`rgb_s3d_bdsl_si_seed2r`, same recipe, clip root `data/bdslw60_clips`) → 75.16. 3-seed = 76.46 ± 1.48.",
              "- **Uni-Sign: the paper's 82.34 is one seed** (`--seed` default 42). Seeds 1, 2 added (identical low-data "
              "recipe, dev-selected on U15, job) → 3-seed 83.61 ± 1.08. NB: this now sits ~0.3 pp below Track-B "
              "SLGTFormer (83.91).",
              "- **SHuBERT**: frozen features are fixed; seeds vary only the probe init. Seed 0 reproduces 67.60 exactly. "
              "Features exist for 9,264/9,307 clips → SI TEST n = 1,364/1,365; sd test n = 1,359/1,367 (Mixed computed on "
              "the clips that have features).",
              "- **Mixed footing** = the pose `data/bdsl_sd` clip names (train 5,746 / val 655 / test 1,367; 11 signers "
              "shared across splits), mapped to the per-trial RGB clips / Uni-Sign poses by name. VideoMAE and S3D sd-test "
              "errors are 10 and 3 clips per seed respectively (different clips per seed, same count at ceiling).",
              "- Mixed is near-ceiling for every model (98.9–99.8), so SMG differences are driven by the TEST column.", ""]
    lines += ["", "## Per-seed values", ""]
    for name, r in R.items():
        lines.append(f"**{name}**")
        for s, e in r["seeds"].items():
            parts = [f"{k} {v:.2f}" for k, v in e.items() if k in ("top1", "f1", "mixed", "mixed_val")]
            parts += [f"{k} {v}" for k, v in e.items() if k in ("n", "mixed_partial_epochs")]
            lines.append(f"- seed {s}: " + (", ".join(parts) if parts else "not available yet"))
        lines.append("")
    os.makedirs("docs/baselines_rerun", exist_ok=True)
    open("docs/baselines_rerun/item4to6_rgb_foundation.md", "w").write("\n".join(lines) + "\n")
    json.dump(R, open("docs/baselines_rerun/item4to6_rgb_foundation.json", "w"), indent=1, default=float)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
