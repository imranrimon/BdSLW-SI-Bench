#!/bin/bash -l
# =====================================================================
# Leave-one-signer-out (LOSO) as a job array: the 11 folds run in
# parallel, one GPU each. Submit from the repository root:
#     sbatch scripts/hpc/slurm_loso_array.sbatch
#     CONFIG=config/lr01_switch/... sbatch --export=ALL scripts/hpc/slurm_loso_array.sbatch
#
# NOTE: pre-build the fold data ONCE before submitting so 11 tasks don't
# regenerate it at the same time (CPU-only, fast) - run fold U01 alone first:
#   python tools/run_loso.py --single config/bdsl_block_gcn_si.yaml \
#       --test-signers 1 --seeds 0 --skip-existing
# then submit this array for the rest.
# =====================================================================
#SBATCH -J slgt_loso
#SBATCH -a 0-10                      # 11 folds
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p <PARTITION>              # <-- set to your GPU partition (`sinfo -s`)
#SBATCH -t 24:00:00
#SBATCH --requeue
#SBATCH -o slurm_logs/%x_%A_%a.out
#SBATCH -e slurm_logs/%x_%A_%a.err

# NB: no set -u (conda MKL activation not nounset-safe)
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdsl_graph
export PYTHONIOENCODING=utf-8
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"
# scratch write-trigger guard + requeue off nodes where scratch never mounts
ok=0; for i in $(seq 1 12); do touch work_dir/.rw_$$ 2>/dev/null && { rm -f work_dir/.rw_$$; ok=1; break; }; echo "scratch not ready ($i/12)"; sleep 10; done
[ "$ok" = 1 ] || { echo "scratch down on $(hostname) -> requeue"; scontrol requeue "$SLURM_JOB_ID"; exit 1; }

# CONFIG env selects the arch (default BlockGCN). Fold bundles in data/bdsl_si_loso are
# arch-agnostic and reused; run_loso logs <config-stem>_loso_test<XX>_seed<N>.
CFG="${CONFIG:-config/bdsl_block_gcn_si.yaml}"
EXTRA="${EXTRA_ARGS:-}"
SIGNERS=(1 4 5 6 8 9 11 12 2 13 15)
TS=${SIGNERS[$SLURM_ARRAY_TASK_ID]}
echo "LOSO fold: test signer U$TS  (config $CFG)"
python -u tools/run_loso.py --single "$CFG" \
    --test-signers "$TS" --seeds 0 1 2 --skip-existing $EXTRA
