#!/bin/bash
# Joint(lr01) + bone(lr01, is_vector as the old clip5 bone recipe) fusion, paired clip bootstrap.
python - <<'PY' | tee docs/blockgcn_lr01_switch/raw/fusion_lr01_vec.md
import pickle, sys
sys.path.insert(0, ".")
import tools.fusion_bootstrap as F
def load(seed):
    j = pickle.load(open(f"work_dir/bdsl_block_gcn_si_lr01_seed{seed}_test/eval_results/best_acc.pkl", "rb"))
    b = pickle.load(open(f"work_dir/bdsl_block_gcn_si_bone_vec_lr01_test_seed{seed}/eval_results/best_acc.pkl", "rb"))
    return j, b
F.load = load
F.main()
PY
