#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --exclude=paraai-n32-h-01-agent-4
#SBATCH --job-name=r1_bias78_close
#SBATCH --array=0-1%2
#SBATCH --output=/home/bingxing2/home/scx9ftv/gushuo/experiments/20261013_bias_amplification78/output/wrapped_%A_%a.out
#SBATCH --error=/home/bingxing2/home/scx9ftv/gushuo/experiments/20261013_bias_amplification78/output/wrapped_%A_%a.err
set -euo pipefail
R=/home/bingxing2/home/scx9ftv/gushuo
E=$R/experiments/20261013_bias_amplification78
bash "$E/worker.sbatch" probes full
module load singularity/3.9.2-with-suid
# Generation process has exited. CPU-only verification uses no --nv and CUDA is hidden.
singularity exec --bind "$R:$R" --pwd "$E/source" "$R/containers/nvidia-pytorch-24.01-py3-arm64.sif" env CUDA_VISIBLE_DEVICES="" PYTHONPATH="$E/source:$R/container_python/py310" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 python "$E/slurm_cpu_closeout.py"
