#!/bin/bash
# GRL sweep (Top-1 + class-conditioned L_norm, eval_grl_sweep procedure) and joint+bone fusion at lr 0.01.
set -o pipefail
W=work_dir; mkdir -p docs/blockgcn_lr01_switch/raw
python -u tools/eval_grl_sweep.py --config config/lr01_switch/bdsl_block_gcn_si_signeradv_lr01.yaml \
  --out docs/blockgcn_lr01_switch/raw/grl_sweep_lr01.md --ckpts \
  0.1=$W/bdsl_block_gcn_si_signeradv_lr01_gl01_seed0/bdsl_block_gcn_si_signeradv_lr01_gl01_seed0_model-best.pt \
  0.5=$W/bdsl_block_gcn_si_signeradv_lr01_gl05_seed0/bdsl_block_gcn_si_signeradv_lr01_gl05_seed0_model-best.pt \
  1.0=$W/bdsl_block_gcn_si_signeradv_lr01_gl10_seed0/bdsl_block_gcn_si_signeradv_lr01_gl10_seed0_model-best.pt
for l in 01 05 10; do
  ck=$W/bdsl_block_gcn_si_signeradv_lr01_gl${l}_seed0/bdsl_block_gcn_si_signeradv_lr01_gl${l}_seed0_model-best.pt
  [ -f "$ck" ] || continue
  for m in "--class-conditioned|cc|4000" "|mrg|6000" "--within-class|wc|6000"; do
    IFS='|' read -r flag tag n <<< "$m"
    python -u tools/decodability_probe.py --mode signer $flag --config config/probe_blockgcn_si.yaml --ckpt "$ck" --max-clips $n --tag "grl_lr01_l$l|$tag" 2>&1 \
      | grep -E "decodability|L_norm\^cond =" | tee -a docs/blockgcn_lr01_switch/raw/probes.log
  done
done
python - <<'PY' | tee docs/blockgcn_lr01_switch/raw/fusion_lr01.md
import os, pickle, sys
sys.path.insert(0, ".")
import tools.fusion_bootstrap as F
def load(seed):
    j = pickle.load(open(f"work_dir/bdsl_block_gcn_si_lr01_seed{seed}_test/eval_results/best_acc.pkl", "rb"))
    b = pickle.load(open(f"work_dir/bdsl_block_gcn_si_bone_lr01_test_seed{seed}/eval_results/best_acc.pkl", "rb"))
    return j, b
F.load = load
F.main()
PY
