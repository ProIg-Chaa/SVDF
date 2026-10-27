#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --signal=USR1@1800
#SBATCH --no-requeue
#SBATCH --exclude=paraai-n32-h-01-agent-4
#SBATCH --job-name=pai_three_models_full
#SBATCH --output=/home/bingxing2/home/scx9ftv/gushuo/experiments/20261017_pai_three_models_full/%x_%A_%a.out
#SBATCH --error=/home/bingxing2/home/scx9ftv/gushuo/experiments/20261017_pai_three_models_full/%x_%A_%a.err
#SBATCH --array=0-5%4
set -euo pipefail
module load singularity/3.9.2-with-suid
TASK_ROOT=/home/bingxing2/home/scx9ftv/gushuo
TASK_EXP=/home/bingxing2/home/scx9ftv/gushuo/experiments/20261017_pai_three_models_full
TASK_TMP=$TASK_EXP/runtime_tmp/$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID
mkdir -p "$TASK_TMP/tmp" "$TASK_TMP/hf"
singularity exec --nv --bind "$TASK_ROOT:$TASK_ROOT" --pwd "$TASK_EXP" "$TASK_ROOT/containers/nvidia-pytorch-24.01-py3-arm64.sif" env PYTHONPATH="$TASK_EXP/audit_source:$TASK_EXP/source:$TASK_ROOT/container_python/py310" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HOME="$TASK_TMP/hf" TMPDIR="$TASK_TMP/tmp" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 python -u "$TASK_EXP/worker.py" --phase full --lane "$SLURM_ARRAY_TASK_ID"
