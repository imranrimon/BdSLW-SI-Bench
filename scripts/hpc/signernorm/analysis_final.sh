#!/bin/bash
set -o pipefail
O=docs/signernorm_2026-10
python -u tools/signernorm_probe.py --out $O/raw/probe_v2_signernorm.jsonl --runs bdsl_adaptive_gnn_si_lr0p1_seed0 bdsl_adaptive_gnn_si_lr0p1_seed1 bdsl_adaptive_gnn_si_lr0p1_seed2 bdsl_adaptive_gnn_si_lr0p1_sn_abc_seed0 bdsl_adaptive_gnn_si_lr0p1_sn_abc_seed1 bdsl_adaptive_gnn_si_lr0p1_sn_abc_seed2
python -u tools/check_run_complete.py $(ls -d work_dir/*_sn_* | grep -v -E '\.partial_') > $O/completeness.txt 2>&1; echo "check rc=$?" >> $O/completeness.txt
for c in block_gcn_lr01 block_gcn_lr01_sn_abc; do python -u tools/check_run_complete.py --loso-csv docs/loso_heldout/$c.csv >> $O/completeness.txt 2>&1; done
python -u tools/signernorm_stage2_report.py > $O/stage2_tables.md 2> $O/stage2_tables.err
python -u tools/signernorm_make_tables.py > $O/generated_tables.log 2>&1; echo "make_tables rc=$?" >> $O/generated_tables.log
echo "=== final analysis done $(date) ==="
