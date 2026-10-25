#!/bin/bash -l
set -e
cd "$(dirname "$0")/../../.."  # repository root
source ~/miniconda3/etc/profile.d/conda.sh && conda activate bdsl_graph
hostname
python tools/arch_significance.py --bestck --lr0p1 --two-sided > docs/loso_heldout/derived/arch_significance_final_2026-10-22_twosided.txt
python tools/arch_significance.py --bestck --lr0p1 > /tmp/sig_final_check_$$.txt
diff /tmp/sig_final_check_$$.txt docs/loso_heldout/derived/arch_significance_final_2026-10-22.txt && echo FINAL_REPRODUCED
echo DONE
