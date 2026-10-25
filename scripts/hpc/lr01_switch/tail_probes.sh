#!/bin/bash
# Leak-table + GRL-control probes for the lr-0.01 BlockGCN (same settings as docs/audit_reruns_2026-10/r4_lnorm.md).
set -o pipefail
W=work_dir; OUT=docs/blockgcn_lr01_switch/raw/probes.log; mkdir -p "$(dirname $OUT)"
pr () {  # $1 tag $2 cfg $3 ckpt $4 n
  local tag="$1" cfg="$2" ck="$3" n="$4"
  [ -f "$ck" ] || { echo "PROBE $tag MISSING $ck" | tee -a $OUT; return; }
  echo "PROBE $tag cfg=$cfg ckpt=$ck" | tee -a $OUT
  python -u tools/decodability_probe.py --mode signer --class-conditioned --config "$cfg" --ckpt "$ck" --max-clips 4000 --tag "${tag}|cc"  2>&1 | grep -E "decodability" | tee -a $OUT
  python -u tools/decodability_probe.py --mode signer                     --config "$cfg" --ckpt "$ck" --max-clips "$n"  --tag "${tag}|mrg" 2>&1 | grep -E "decodability" | tee -a $OUT
  python -u tools/decodability_probe.py --mode signer --within-class      --config "$cfg" --ckpt "$ck" --max-clips "$n"  --tag "${tag}|wc"  2>&1 | grep -E "L_norm\^cond =" | tee -a $OUT
}
for s in 0 1 2; do
  pr "sup_blockgcn_lr01_s$s" config/probe_block_gcn_si.yaml $W/bdsl_block_gcn_si_lr01_scratch_seed$s/bdsl_block_gcn_si_lr01_scratch_seed${s}_model-best.pt 6000
done
