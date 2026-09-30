#!/bin/bash
# Batch job for examples/gpu/two_layer_cabbeling_gpu.jl.
# Edit the lines marked EDIT for your cluster, then submit with `sbatch run_gpu.slurm`.
# One-off setup (login node, which has internet):
#   julia --project=$REPO/examples/gpu -e 'using Pkg; Pkg.instantiate(); using CUDA; CUDA.set_runtime_version!(v"12.6"); Pkg.precompile()'

#SBATCH --job-name=cabbeling-gpu
#SBATCH --partition=gpu          # EDIT: your GPU partition
#SBATCH --account=my_account     # EDIT: your project/account (delete if not needed)
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

module purge
module load julia                # EDIT: or put juliaup's julia on PATH instead

export JULIA_DEPOT_PATH=$HOME/.julia   # EDIT: a large filesystem (work/scratch) if $HOME is small
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

REPO=$HOME/ThermoShenanigans.jl        # EDIT: where you copied the repository
RUN_DIR=$HOME/runs/cabbeling-$SLURM_JOB_ID   # EDIT: where output should go (scratch is best)

mkdir -p $RUN_DIR
cd $RUN_DIR

nvidia-smi
julia --project=$REPO/examples/gpu $REPO/examples/gpu/two_layer_cabbeling_gpu.jl
