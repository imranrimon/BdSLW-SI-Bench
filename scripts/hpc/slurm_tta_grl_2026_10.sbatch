#!/bin/bash -l
# =====================================================================
# P2 TTA (label-free unseen-signer adaptation) + GRL 3-seed fill (2026-10).
# Runs one packed task list on this job's GPUs (round-robin lanes, CUDA pinned per lane).
#   TASKS  = scripts/hpc/tta_grl_2026_10/<list>.txt (one shell command per line)
#   MAXPAR = concurrent tasks PER GPU (default and cap 3; SLGTFormer lists use 1)
#   TAIL   = optional command run after all tasks finished (evals), on GPU 0
# 8-h chunks: USR1 15 min before the limit -> scontrol requeue; every task is resume-safe
# (tools/tta_eval.py skips finished (run,signer,method) keys; GRL training goes through
# tools/main_signer_adv_resumable.py, which resumes from <exp>_resume.pt).
# Submit: sbatch --gpus 1 --export=ALL,TASKS=...,MAXPAR=3 -J tta_bg scripts/hpc/slurm_tta_grl_2026_10.sbatch
# =====================================================================
#SBATCH -J tta_grl
#SBATCH -c 12
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=96G
#SBATCH --gpus 1
#SBATCH -p <PARTITION>              # <-- set to your GPU partition (`sinfo -s`)
#SBATCH -t 8:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --open-mode=append
#SBATCH --requeue
# (optional) #SBATCH --exclude=<bad-nodes>
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs/tta_grl_2026_10
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdsl_graph
export PYTHONIOENCODING=utf-8
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4

trap 'echo "[chunk] $(date "+%F %T") time limit near -> requeue $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
ok=0; for i in $(seq 1 12); do touch work_dir/.rw_$$ 2>/dev/null && { rm -f work_dir/.rw_$$; ok=1; break; }; echo "scratch not ready ($i/12)"; sleep 10; done
[ "$ok" = 1 ] || { echo "scratch down on $(hostname) -> requeue"; scontrol requeue "$SLURM_JOB_ID"; exit 1; }
if ! python -c "import torch,sys; n=torch.cuda.device_count(); sys.exit(0 if n>0 and all(torch.zeros(1).cuda(i).add_(1).item()==1.0 for i in range(n)) else 1)" 2>/dev/null; then
  echo "GPU unusable on $(hostname) -> requeue"; scontrol requeue "$SLURM_JOB_ID"; exit 1
fi

MAXPAR=${MAXPAR:-3}; [ "$MAXPAR" -le 3 ] || MAXPAR=3
IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
NG=${#GPUS[@]}
TB=$(basename "$TASKS" .txt)
echo "host $(hostname)  GPUs ${GPUS[*]}  MAXPAR/GPU $MAXPAR  tasks $TASKS  start $(date '+%F %T')  restart ${SLURM_RESTART_COUNT:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

lane () {  # $1 = gpu index in GPUS, $2 = file with this GPU's commands
  local gi=$1 f=$2 k=0
  while IFS= read -r CMD; do
    [ -z "$CMD" ] && continue
    k=$((k+1)); local LOG="slurm_logs/tta_grl_2026_10/${TB}_g${gi}_task$(printf %03d $k).log"
    echo "[gpu ${GPUS[$gi]} launch $k $(date '+%F %T')] $CMD"
    CUDA_VISIBLE_DEVICES=${GPUS[$gi]} bash -c "$CMD" >> "$LOG" 2>&1 &
    while [ "$(jobs -rp | wc -l)" -ge "$MAXPAR" ]; do sleep 30; done
  done < "$f"
  wait || true
}

TMP=$(mktemp -d)
n=0
while IFS= read -r CMD; do
  [ -z "$CMD" ] && continue
  echo "$CMD" >> "$TMP/g$((n % NG))"; n=$((n+1))
done < "$TASKS"
for gi in $(seq 0 $((NG-1))); do [ -f "$TMP/g$gi" ] && lane "$gi" "$TMP/g$gi" & done
wait || true
echo "all $n tasks finished $(date '+%F %T')"
grep -l -E "Traceback|Error" slurm_logs/tta_grl_2026_10/${TB}_g*_task*.log 2>/dev/null | sed 's/^/  check: /' || true
if [ -n "${TAIL:-}" ]; then echo "[tail] $TAIL"; CUDA_VISIBLE_DEVICES=${GPUS[0]} bash -c "$TAIL" || true; fi
echo "=== $TB done $(date '+%F %T') ==="
