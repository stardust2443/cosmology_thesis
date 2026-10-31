#!/bin/bash
# run_validation_test1.sh
# Run joint MCMC fits at R_min = 7.0 Mpc/h using Jackknife errors instead of Poisson errors.

PYTHON="/Users/arun/miniforge3/bin/python3.12"
SCRIPT="scripts/cbl_mcmc_v5_joint.py"

WALKERS=32
STEPS=800
FIELD="NREC50"
DATA_DIR="data/vsf_jk_simple0"

echo "=== STARTING VALIDATION TEST 1 (JACKKNIFE COVARIANCE DIAGNOSTIC) ==="
echo "Settings: walkers=$WALKERS, steps=$STEPS, field=$FIELD, data_dir=$DATA_DIR"

# 1. Run TOUCH (OV00) with Jackknife errors
echo ""
echo "--- Running TOUCH (OV00) with Jackknife errors ---"
$PYTHON $SCRIPT --field $FIELD --overlap OV00 --rmin 7.0 --nwalkers $WALKERS --nsteps $STEPS --data-dir $DATA_DIR --tag jk

# 2. Run CENTER (OV05) with Jackknife errors
echo ""
echo "--- Running CENTER (OV05) with Jackknife errors ---"
$PYTHON $SCRIPT --field $FIELD --overlap OV05 --rmin 7.0 --nwalkers $WALKERS --nsteps $STEPS --data-dir $DATA_DIR --tag jk

echo ""
echo "=== VALIDATION TEST 1 MCMC RUNS COMPLETE ==="
