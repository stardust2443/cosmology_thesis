#!/bin/bash
# run_validation_test2.sh
# Run joint MCMC fits for the scale-cut sensitivity scan (Test 2)

PYTHON="/Users/arun/miniforge3/bin/python3.12"
SCRIPT="scripts/cbl_mcmc_v5_joint.py"

WALKERS=32
STEPS=800
FIELD="NREC50"

echo "=== STARTING VALIDATION TEST 2 (SCALE-CUT SCAN) ==="
echo "Settings: walkers=$WALKERS, steps=$STEPS, field=$FIELD"

# 1. OV00 (TOUCH) runs
for rmin in 7.0 8.0 9.0; do
    echo ""
    echo "--- Running TOUCH (OV00) at R_min = $rmin ---"
    $PYTHON $SCRIPT --field $FIELD --overlap OV00 --rmin $rmin --nwalkers $WALKERS --nsteps $STEPS
done

# 2. OV05 (CENTER) runs
for rmin in 7.0 8.0 9.0; do
    echo ""
    echo "--- Running CENTER (OV05) at R_min = $rmin ---"
    $PYTHON $SCRIPT --field $FIELD --overlap OV05 --rmin $rmin --nwalkers $WALKERS --nsteps $STEPS
done

echo ""
echo "=== VALIDATION TEST 2 MCMC RUNS COMPLETE ==="
