#!/bin/bash
#SBATCH --job-name=cbl_v5_joint
#SBATCH --account=astro
#SBATCH --partition=m5,m6
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=logs/cbl_v5_joint_%j.out
#SBATCH --error=logs/cbl_v5_joint_%j.err

# ophfe1 cluster modules — all required for CBL Python bindings
module load python/3.14.3
module load astro/swig/4.1.1 astro/gsl/2.7.1 astro/fftw/3.3.10
module load astro/hdf5/1.12.2 astro/cfitsio/3.47 astro/blas/3.11.0 astro/boost/1.70.0

# OpenBLAS thread caps — required on login node, healthy on compute nodes
# to keep emcee's multiprocessing pool from oversubscribing the CPUs.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# CBL paths (edit for the cluster's CBL install location)
export CBL_HOME=${CBL_HOME:-$HOME/CosmoBolognaLib}
export PYTHONPATH=$CBL_HOME/Python:$PYTHONPATH
export LD_LIBRARY_PATH=$CBL_HOME:$LD_LIBRARY_PATH
# (DYLD_LIBRARY_PATH is macOS-only; clusters typically use LD_LIBRARY_PATH)

# Project location
PROJ=${PROJ:-$HOME/workspace/void_analysis}
cd $PROJ

# Knobs — change here, not in the python script
NWALKERS=64
NSTEPS=8000
NTHREADS=${SLURM_CPUS_PER_TASK:-16}
FIELD=NREC50           # or BIT
RMIN=7.0

echo "Launching joint MCMC: $NWALKERS walkers × $NSTEPS steps, $NTHREADS threads"
echo "  field = $FIELD, R_min = $RMIN"
echo "  CBL_HOME = $CBL_HOME"
echo "  starting at: $(date)"

python scripts/cbl_mcmc_v5_joint.py \
    --field $FIELD \
    --rmin $RMIN \
    --nwalkers $NWALKERS \
    --nsteps $NSTEPS \
    --nthreads $NTHREADS \
    --tag cluster

echo "done at: $(date)"
