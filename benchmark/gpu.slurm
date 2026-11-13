#!/bin/bash
# Batch job that runs benchmark/benchmark.jl on a GPU, then the full test suite (TEST_GROUP=all) with
# TEST_ARCHITECTURE=GPU, which compares every exported diagnostic between the GPU and the CPU (test/test_architectures.jl).
# Edit the lines marked EDIT for your cluster, then submit with `sbatch gpu.slurm`.
#
# Both run in the environment benchmark/Project.toml, which holds CUDA and the test dependencies, so that the job needs
# no internet. One-off setup (login node, which has internet; Julia 1.11 or later, for [sources]):
#   julia --project=$REPO/benchmark -e 'using Pkg; Pkg.instantiate(); using CUDA; CUDA.set_runtime_version!(v"12.6"); Pkg.precompile()'
#   julia --project=$REPO/benchmark -e 'using CUDA'   # in a new session, which fetches the CUDA runtime it now selects
#
# The tests run as `julia test/runtests.jl` in that environment rather than through `Pkg.test`, whose fresh test
# environment would have to download CUDA. Results to report back: the benchmark table and the test summary at the end
# of the output file.

#SBATCH --job-name=thermoshenanigans-gpu
#SBATCH --partition=gpu          # EDIT: your GPU partition
#SBATCH --account=my_account     # EDIT: your project/account (delete if not needed)
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00          # compiling every diagnostic for the GPU and the CPU dominates the test time
#SBATCH --output=%x-%j.out

module purge
module load julia                # EDIT: or put juliaup's julia on PATH instead

export JULIA_DEPOT_PATH=$HOME/.julia   # EDIT: a large filesystem (work/scratch) if $HOME is small
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JULIA_PKG_OFFLINE=true          # fail at once, rather than wait for a download, if the setup missed a package

REPO=$HOME/ThermoShenanigans.jl        # EDIT: where you copied the repository

cd $REPO
nvidia-smi
julia --project=benchmark -e 'using CUDA; CUDA.versioninfo()'

echo "=== Benchmark ==="
julia --project=benchmark benchmark/benchmark.jl GPU
benchmark_status=$?

echo "=== Tests (TEST_ARCHITECTURE=GPU, TEST_GROUP=all) ==="
TEST_ARCHITECTURE=GPU TEST_GROUP=all julia --project=benchmark test/runtests.jl
test_status=$?

echo "Benchmark exit status: $benchmark_status, test exit status: $test_status"
exit $(( benchmark_status != 0 || test_status != 0 ))
