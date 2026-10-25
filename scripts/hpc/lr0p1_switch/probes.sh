#!/bin/bash
# Leak-table probes (sec:leak, "supervised Adaptive GNN") for the lr-0.1 Adaptive GNN, 3 seeds, best-on-val
# checkpoints; plus the OLD lr-0.01 seeds 1/2 best checkpoints for an old 3-seed comparison (old seed 0 =
# m59 0.910 / P2's m75 0.911, docs/audit_reruns_2026-10/r4_lnorm.md). Same settings as r4 and the BlockGCN
# lr-0.01 switch: cc n=4000, marginal n=6000, within-class n=6000.
set -o pipefail
W=work_dir; OUT=docs/adaptive_attn_lr0p1_switch/raw/probes.log; mkdir -p "$(dirname $OUT)"
pr () {  # $1 tag $2 cfg $3 ckpt $4 n
  local tag="$1" cfg="$2" ck="$3" n="$4"
  [ -f "$ck" ] || { echo "PROBE $tag MISSING $ck" | tee -a $OUT; return; }
  grep -q "\[${tag}|wc\]" "$OUT" 2>/dev/null && { echo "PROBE $tag already done"; return; }   # idempotent
  echo "PROBE $tag cfg=$cfg ckpt=$ck" | tee -a $OUT
  python -u tools/decodability_probe.py --mode signer --class-conditioned --config "$cfg" --ckpt "$ck" --max-clips 4000 --tag "${tag}|cc"  2>&1 | grep -E "decodability" | tee -a $OUT
  python -u tools/decodability_probe.py --mode signer                     --config "$cfg" --ckpt "$ck" --max-clips "$n"  --tag "${tag}|mrg" 2>&1 | grep -E "decodability" | tee -a $OUT
  python -u tools/decodability_probe.py --mode signer --within-class      --config "$cfg" --ckpt "$ck" --max-clips "$n"  --tag "${tag}|wc"  2>&1 | grep -E "L_norm\^cond =" | tee -a $OUT
}
for s in 0 1 2; do
  pr "sup_adaptive_lr0p1_s$s" config/probe_adaptive_gnn_si.yaml $W/bdsl_adaptive_gnn_si_lr0p1_seed$s/bdsl_adaptive_gnn_si_lr0p1_seed${s}_model-best.pt 6000
done
pr "sup_adaptive_lr0p01_s1_m59" config/probe_adaptive_gnn_si.yaml $W/bdsl_adaptive_gnn_si_seed1/bdsl_adaptive_gnn_si_seed1_model-59.pt 6000
pr "sup_adaptive_lr0p01_s2_m89" config/probe_adaptive_gnn_si.yaml $W/bdsl_adaptive_gnn_si_seed2/bdsl_adaptive_gnn_si_seed2_model-89.pt 6000
pr "sup_adaptive_lr0p01_s0_m59" config/probe_adaptive_gnn_si.yaml $W/bdsl_adaptive_gnn_si_seed0/bdsl_adaptive_gnn_si_seed0_model-59.pt 6000
