#!/bin/bash -l
set -e
cd "$(dirname "$0")/../../.."  # repository root
source ~/miniconda3/etc/profile.d/conda.sh && conda activate bdsl_graph
hostname
python tools/p2_rerun_fixes_derived.py
python tools/p2_make_tables.py --check > docs/p2_rerun_fixes_2026-10/raw/p2_make_tables_check_after_derived_beforegen.txt 2>&1 || true
python tools/p2_make_tables.py --check --use-recipe slgtformer=bestck --use-recipe gnn_lstm=bestck --use-recipe pose_lstm=bestck --use-recipe st_gcn_vanilla=bestck > docs/p2_rerun_fixes_2026-10/raw/p2_make_tables_check_bestck_preview.txt 2>&1 || true
python tools/p2_make_tables.py --dry-run > docs/p2_rerun_fixes_2026-10/raw/p2_make_tables_dryrun_after.txt 2>&1
python tools/p2_make_tables.py --dry-run --use-recipe slgtformer=bestck --use-recipe gnn_lstm=bestck --use-recipe pose_lstm=bestck --use-recipe st_gcn_vanilla=bestck > docs/p2_rerun_fixes_2026-10/raw/p2_make_tables_dryrun_bestck.txt 2>&1
python tools/check_run_complete.py --p2-sweep --md docs/p2_rerun_fixes_2026-10/incomplete_runs.md > /dev/null
echo DONE
