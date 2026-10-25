"""S3D Protocol-A scores from the three cluster runs (2026-10-22).

The original S3D seeds 0/1 were Windows runs with no training log, so their completeness could not
be verified; they were re-trained on the cluster as rgb_s3d_bdsl_si_seed{0,1}r (job, same
recipe as seed2r). Scoring reuses tools/rgb_foundation_scores.py; the Mixed (bdsl_sd) runs are unchanged.
    python tools/p2_s3d_cluster_seeds.py
"""
import json, os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rgb_foundation_scores import from_scores, si_test_labels, sd_labels

lab = si_test_labels()
out = {"seeds": {}, "source": "work_dir/rgb_s3d_bdsl_si_seed{0,1,2}r (cluster); Mixed from rgb_s3d_bdsl_sd_seed{s}"}
for s in (0, 1, 2):
    d = f"rgb_s3d_bdsl_si_seed{s}r"
    out["seeds"][s] = from_scores(pickle.load(open(f"work_dir/{d}/eval_results/testset_scores.pkl", "rb")), lab)
    q = f"work_dir/rgb_s3d_bdsl_sd_seed{s}/eval_results/testset_scores.pkl"
    v = pickle.load(open(f"work_dir/rgb_s3d_bdsl_sd_seed{s}/eval_results/best_acc.pkl", "rb"))
    out["seeds"][s]["mixed_val"] = from_scores(v, sd_labels("val"))["top1"]
    out["seeds"][s]["mixed"] = from_scores(pickle.load(open(q, "rb")), sd_labels("test"))["top1"]
p = "docs/p2_rerun_fixes_2026-10/s3d_cluster_seeds.json"
json.dump({"S3D": out}, open(p, "w"), indent=1)
for s, r in out["seeds"].items():
    print(s, {k: round(v, 2) for k, v in r.items()})
