#!/bin/bash
# P2 missing-joint fixes: build/verify the new data dirs ON A COMPUTE NODE (login-node Lustre writes stalled).
set -eo pipefail
cd "$(dirname "$0")/../../.."  # repository root
source ~/miniconda3/etc/profile.d/conda.sh; conda activate bdsl_graph
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONIOENCODING=utf-8
echo "== $(hostname) $(date)"
python -u preprocessing/generate_bone_data_masked.py --data-dir data/bdsl_si --output-dir data/bdsl_si_bonemask
for f in 2 13; do python -u preprocessing/generate_paired_smg_otherclass_bundle.py --focus-signer $f 2>&1 | grep -v -E "it/s|s/it"; done
python -u - <<'PY'
import numpy as np, pickle, sys
sys.path[:0] = [".", "preprocessing"]
import make_cue_bundle_masked as CM
from generate_bone_data import _joint_to_bone
from graph.sign_27 import Graph
for sp in ("train", "val", "test"):
    X = np.load(f"data/bdsl_si/{sp}_data.npy", mmap_mode="r")
    for cue in ("speednorm_masked", "rootcenter_masked"):
        Y = np.load(f"data/bdsl_cue_{cue}/{sp}_data.npy", mmap_mode="r")
        bad = sum(not np.array_equal(CM.transform(np.asarray(X[i]), cue)[0], np.asarray(Y[i])) for i in range(len(X)))
        print(f"verify cue {cue} {sp}: {len(X)} clips, mismatches {bad}", flush=True)
    J = np.asarray(X); O = np.load(f"data/bdsl_si/{sp}_data_bone.npy"); N = np.load(f"data/bdsl_si_bonemask/{sp}_data_bone.npy")
    print(f"verify bone {sp}: old file == generate_bone_data(joint): {np.array_equal(O, _joint_to_bone(J, Graph().inward))}; "
          f"joints changed by masking: {np.where(np.abs(O - N).sum(axis=(0, 1, 2, 4)) > 0)[0].tolist()}", flush=True)
PY
echo "== DATAPREP DONE $(date)"
