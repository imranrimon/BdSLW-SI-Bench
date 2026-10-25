#!/usr/bin/env bash
# Run a task list (one shell command per line, e.g. scripts/hpc/<program>/tasks_*.txt)
# sequentially on the current machine, from the repository root. Blank lines and lines
# starting with '#' are skipped. Training commands go through tools/run_resumable_seed.py,
# so re-running a list resumes unfinished runs and skips finished ones.
#
#   bash scripts/run_tasks_local.sh scripts/hpc/lr01_switch/tasks_J1.txt [more lists...]
set -euo pipefail
cd "$(dirname "$0")/.."
[ "$#" -ge 1 ] || { echo "usage: $0 TASKS.txt [TASKS.txt ...]" >&2; exit 2; }
mkdir -p slurm_logs/local
for list in "$@"; do
  n=0
  while IFS= read -r cmd || [ -n "$cmd" ]; do
    case "$cmd" in ''|'#'*) continue ;; esac
    n=$((n + 1))
    log="slurm_logs/local/$(basename "$list" .txt)_task$(printf %03d "$n").log"
    echo "[$(date '+%F %T')] $list #$n -> $log"
    bash -c "$cmd" >> "$log" 2>&1 || echo "  task $n failed (see $log)"
  done < "$list"
done
