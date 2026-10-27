#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --array=0-1%2
#SBATCH --no-requeue
#SBATCH --exclude=paraai-n32-h-01-agent-4
#SBATCH --job-name=rethink_core4_h05
#SBATCH --output=/home/bingxing2/home/scx9ftv/gushuo/output/experiments/20261009_rethinker_core4_theta05/%x_%A_%a.out
#SBATCH --error=/home/bingxing2/home/scx9ftv/gushuo/output/experiments/20261009_rethinker_core4_theta05/%x_%A_%a.err
set -euo pipefail
module load singularity/3.9.2-with-suid
R=/home/bingxing2/home/scx9ftv/gushuo
E=/home/bingxing2/home/scx9ftv/gushuo/experiments/20261009_rethinker_core4_theta05
O=/home/bingxing2/home/scx9ftv/gushuo/output/experiments/20261009_rethinker_core4_theta05
T=$O/shell_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$T/tmp" "$T/hf"
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:?}"
test "${#GPUS[@]}" -eq 1
singularity exec --nv --bind "$R:$R" --pwd "$E/source" "$R/containers/nvidia-pytorch-24.01-py3-arm64.sif" env CUDA_VISIBLE_DEVICES="${GPUS[0]}" PYTHONPATH="$E/source:$R/container_python/py310" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HOME="$T/hf" TMPDIR="$T/tmp" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u "$E/worker.py" --model rethinker
