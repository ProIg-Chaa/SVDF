#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --no-requeue
#SBATCH --exclude=paraai-n32-h-01-agent-4
#SBATCH --job-name=r1_same_position_push
#SBATCH --array=0-1%2
#SBATCH --output=/home/bingxing2/home/scx9ftv/gushuo/experiments/20261013_same_position_fact_push/output/job_%A_%a.out
#SBATCH --error=/home/bingxing2/home/scx9ftv/gushuo/experiments/20261013_same_position_fact_push/output/job_%A_%a.err
set -euo pipefail
module load singularity/3.9.2-with-suid
R=/home/bingxing2/home/scx9ftv/gushuo
E=$R/experiments/20261013_same_position_fact_push
T=$E/output/runtime_$SLURM_JOB_ID
mkdir -p "$T/tmp" "$T/hf"
phase=$1
case "$phase" in smoke|full) ;; *) exit 3;; esac
case "$CUDA_VISIBLE_DEVICES" in *,*) exit 2;; esac
singularity exec --nv --bind "$R:$R" --pwd "$E/source" "$R/containers/nvidia-pytorch-24.01-py3-arm64.sif" env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" PYTHONPATH="$E/source:$R/container_python/py310" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$T/hf" TMPDIR="$T/tmp" OMP_NUM_THREADS=4 python "$E/run.py" --phase "$phase" --lane "$SLURM_ARRAY_TASK_ID"
if [ "$phase" = full ]; then
 singularity exec --bind "$R:$R" --pwd "$E/source" "$R/containers/nvidia-pytorch-24.01-py3-arm64.sif" env PYTHONPATH="$E/source:$R/container_python/py310" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 python "$E/closeout_cpu.py"
fi
