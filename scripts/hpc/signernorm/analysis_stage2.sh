#!/bin/bash
# Stage-2 analysis on a compute node: completeness gate, tables, oracle row. Run from repo root.
set -o pipefail
O=docs/signernorm_2026-10
python -u tools/check_run_complete.py $(ls -d work_dir/*_sn_* | grep -v -E '\.partial_') > $O/completeness.txt 2>&1; echo "check rc=$?" >> $O/completeness.txt
python -u tools/check_run_complete.py --loso-csv docs/loso_heldout/block_gcn_lr01.csv >> $O/completeness.txt 2>&1
python -u tools/signernorm_stage2_report.py > $O/stage2_tables.md 2> $O/stage2_tables.err
python -u tools/check_run_complete.py --loso-csv docs/loso_heldout/block_gcn_lr01_sn_abc.csv >> $O/completeness.txt 2>&1
python -u tools/signernorm_report.py --stems bdsl_block_gcn_si_lr01_sn_ab bdsl_block_gcn_si_lr01_sn_abc bdsl_block_gcn_si_lr01_sn_ab_oraclehand \
   --labels +a+b +a+b+c "+a+b + ORACLE handedness (metadata)" > $O/oracle_table.md 2> $O/oracle_table.err
python -u tools/signernorm_stage2_report.py > $O/stage2_tables.md 2> $O/stage2_tables.err
echo "=== analysis done $(date) ==="
