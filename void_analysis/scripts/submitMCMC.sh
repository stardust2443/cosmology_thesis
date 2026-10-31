#!/bin/bash
#SBATCH --job-name=vsf_mcmc
#SBATCH --output=mcmc_%A_%a.out
#SBATCH --error=mcmc_%A_%a.err
#SBATCH --array=0-7
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16    # Scaled up for heavy parallelization
#SBATCH --mem=32G             # Increased memory to support 16 parallel threads
#SBATCH --time=05:00:00       # Extended walltime for rigorous convergence
#SBATCH --partition=m1,m7

# --- BitVF VSF MCMC | SLURM Array Job ---

# Configuration for Publication-Quality MCMC
NSTEPS=3000                  # Tripled to guarantee the autocorrelation time (tau) converges
NWALKERS=32                   # Doubled to better map out the 3D parameter space
PROJECT_ROOT="/home/STUDENTI/arun.mitra/workspace/mcmc_analysis"
SCRIPT_DIR="${PROJECT_ROOT}/scripts"

# Define the 8 run configurations
FIELDS=("BIT" "BIT" "BIT" "BIT" "NREC50" "NREC50" "NREC50" "NREC50")
SUBSAMPLES=("20p" "10p" "5p" "2_5p" "20p" "10p" "5p" "2_5p")

# Get this job's configuration
FIELD=${FIELDS[$SLURM_ARRAY_TASK_ID]}
SUBSAMPLE=${SUBSAMPLES[$SLURM_ARRAY_TASK_ID]}

echo "============================================"
echo "  Job ID:     $SLURM_JOB_ID"
echo "  Array ID:   $SLURM_ARRAY_TASK_ID"
echo "  Field:      $FIELD"
echo "  Subsample:  $SUBSAMPLE"
echo "  Steps:      $NSTEPS"
echo "  Walkers:    $NWALKERS"
echo "  Cores:      $SLURM_CPUS_PER_TASK"
echo "============================================"

# CRITICAL: Prevents numpy/OpenBLAS from spawning too many threads on cluster
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# cd into scripts dir so Python's os.path.dirname(__file__) + ../data/ resolves correctly
cd "${SCRIPT_DIR}" || exit 1

source ~/cosmo_env/bin/activate

# Run the MCMC
~/cosmo_env/bin/python3 "${SCRIPT_DIR}/run_mcmc_vsf.py" \
    --field "$FIELD" \
    --subsample "$SUBSAMPLE" \
    --fit-delta-v \
    --nsteps "$NSTEPS" \
    --nwalkers "$NWALKERS"

echo "Job $SLURM_ARRAY_TASK_ID completed at $(date)"