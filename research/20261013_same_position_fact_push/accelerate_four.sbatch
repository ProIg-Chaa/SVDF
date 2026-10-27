#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --exclude=paraai-n32-h-01-agent-4
#SBATCH --array=0-3%4
#SBATCH --job-name=r1_fact_push_four
#SBATCH --output=/home/bingxing2/home/scx9ftv/gushuo/experiments/20261013_same_position_fact_push/output/accel4_%A_%a.out
#SBATCH --error=/home/bingxing2/home/scx9ftv/gushuo/experiments/20261013_same_position_fact_push/output/accel4_%A_%a.err
set -euo pipefail
E=/home/bingxing2/home/scx9ftv/gushuo/experiments/20261013_same_position_fact_push
R=/home/bingxing2/home/scx9ftv/gushuo
python3 "$E/accelerate_four_check.py"
python3 "$E/continuation_resource_gate.py"
T=$E/output/runtime_$SLURM_JOB_ID
mkdir -p "$T/tmp" "$T/hf"
module load singularity/3.9.2-with-suid
case "$CUDA_VISIBLE_DEVICES" in *,*) exit 2;; esac
singularity exec --nv --bind "$R:$R" --pwd "$E/source" "$R/containers/nvidia-pytorch-24.01-py3-arm64.sif" env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" PYTHONPATH="$E/source:$R/container_python/py310" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$T/hf" TMPDIR="$T/tmp" OMP_NUM_THREADS=4 python "$E/run_fourcard.py" --phase full --lane "$SLURM_ARRAY_TASK_ID"
singularity exec --bind "$R:$R" --pwd "$E/source" "$R/containers/nvidia-pytorch-24.01-py3-arm64.sif" env PYTHONPATH="$E/source:$R/container_python/py310" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 python "$E/closeout_cpu_numeric.py"
