#!/bin/bash
set -o pipefail
O=docs/signernorm_2026-10
python -u tools/signernorm_make_tables.py > $O/generated_tables.log 2>&1; echo "make_tables rc=$?" >> $O/generated_tables.log
python -u tools/signernorm_stage2_report.py > $O/stage2_tables.md 2> $O/stage2_tables.err; echo "report rc=$?" >> $O/generated_tables.log
echo "=== tables done $(date) ===" >> $O/generated_tables.log
