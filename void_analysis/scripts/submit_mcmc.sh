#!/bin/bash
#SBATCH --job-name=vsf_mcmc
#SBATCH --output=mcmc_%A_%a.out
#SBATCH --error=mcmc_%A_%a.err
#SBATCH --array=0-7
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --partition=standard

# ─── BitVF VSF MCMC — SLURM Array Job ─────────────────────────────
#
# Submits 8 independent MCMC runs:
#   Jobs 0-3: BitVF true displacement field (BIT), subsamples 20p/10p/5p/2_5p
#   Jobs 4-7: OT reconstructed field (NREC50), subsamples 20p/10p/5p/2_5p
#
# Each job runs the 3D sampler (Ω_m, σ_8, δ_v) with 3000 steps × 32 walkers.
#
# Usage:
#   sbatch submit_mcmc.sh
#
# ───────────────────────────────────────────────────────────────────

# Configuration
NSTEPS=3000
NWALKERS=32

# IMPORTANT: SLURM copies this script to /var/spool/slurmd/ before running,
# so $(dirname "$0") won't work. Use the absolute path instead.
MCMC_DIR="${HOME}/workspace/mcmc_analysis"
SCRIPT_DIR="${MCMC_DIR}/scripts"

# Activate the cosmo_env conda environment
source activate cosmo_env 2>/dev/null || conda activate cosmo_env 2>/dev/null || true

# cd into the scripts directory so relative paths (../data/) resolve correctly
cd "${SCRIPT_DIR}" || exit 1

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
echo "  Working dir: $(pwd)"
echo "  Python:      $(which python3)"
echo "============================================"

# Run the MCMC
python3 "${SCRIPT_DIR}/run_mcmc_vsf.py" \
    --field "$FIELD" \
    --subsample "$SUBSAMPLE" \
    --fit-delta-v \
    --nsteps "$NSTEPS" \
    --nwalkers "$NWALKERS"

echo "Job $SLURM_ARRAY_TASK_ID completed at $(date)"
