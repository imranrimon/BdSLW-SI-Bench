#!/bin/bash -l
# =====================================================================
# Masked pose SSL pre-training (SHuBERT-style, BlockGCN backbone) with
# main_pretrain.py. CONFIG selects the recipe; the default is the
# BdSL-only pool used for the SSL-initialisation row of the paper
# (work_dir/bdsl_shubert_pretrain_bdsl_only/pretrained_epoch30.pt).
# Prerequisites: preprocessing/build_ssl_pool_manifest.py and
# preprocessing/compute_pretrain_targets.py --feature-mode pose_motion.
# Submit from the repository root:
#     sbatch scripts/hpc/slurm_ssl_pretrain.sbatch
# =====================================================================
#SBATCH -J slgt_ssl_pretrain
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p <PARTITION>              # <-- fill from `sinfo -s`
##SBATCH -w <node>                # <-- optional: pin a node
#SBATCH -t 36:00:00
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdsl_graph
export PYTHONIOENCODING=utf-8

CFG="${CONFIG:-config/bdsl_shubert_pretrain_bdsl_only.yaml}"
python -u main_pretrain.py --config "$CFG" --seed 0
echo "SSL pretrain done ($CFG)"
