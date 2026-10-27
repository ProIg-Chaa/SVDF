#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --time-min=00:03:00
#SBATCH --no-requeue
#SBATCH --exclude=paraai-n32-h-01-agent-4
#SBATCH --job-name=r1_fixed_ID1_guard_check
#SBATCH --output=/home/bingxing2/home/scx9ftv/gushuo/experiments/20261013_same_position_fact_push/output/guard_check_%j.out
#SBATCH --error=/home/bingxing2/home/scx9ftv/gushuo/experiments/20261013_same_position_fact_push/output/guard_check_%j.err
set -euo pipefail
R=/home/bingxing2/home/scx9ftv/gushuo
E=$R/experiments/20261013_same_position_fact_push
python3 "$E/continuation_resource_gate.py"
T=$E/output/runtime_$SLURM_JOB_ID
mkdir -p "$T/tmp" "$T/hf"
module load singularity/3.9.2-with-suid
singularity exec --nv --bind "$R:$R" --pwd "$E/source" "$R/containers/nvidia-pytorch-24.01-py3-arm64.sif" env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" PYTHONPATH="$E/source:$R/container_python/py310" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$T/hf" TMPDIR="$T/tmp" OMP_NUM_THREADS=4 python "$E/engineering_guard_point_check.py"
