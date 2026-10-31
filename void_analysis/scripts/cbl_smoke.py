#!/usr/bin/env python3
"""
cbl_smoke.py

Phase 0 smoke test:
  - Confirm CBL Python bindings load
  - Construct a Cosmology object at the Aletheia truth
  - Call size_function for a few radii under both "SvdW" and "Vdn"
  - Print the values

If this script runs to completion and prints sensible (monotonically falling)
n(R) values, CBL is ready for use. Run as:

    /Users/arun/miniforge3/bin/python3.12 scripts/cbl_smoke.py
"""
import os, sys
import numpy as np

# Sanity-check the env even if .zshrc isn't sourced for this shell
if 'CosmoBolognaLib' not in sys.modules:
    cbl_home = os.environ.get('CBL_HOME', '/Users/arun/projects/CosmoBolognaLib')
    sys.path.insert(0, os.path.join(cbl_home, 'Python'))
    os.environ.setdefault('DYLD_LIBRARY_PATH', cbl_home)

import CosmoBolognaLib as cbl
print(f"[OK] CBL imported from {cbl.__file__}")

# ─── Aletheia cosmology ──────────────────────────────────────────────
OMEGA_M = 0.315
OMEGA_B = 0.0493
H0      = 0.6736   # little h
N_S     = 0.9649
SIGMA8  = 0.811
SCALAR_AMP   = 2.194e-9
SCALAR_PIVOT = 0.05
TAU     = 0.0925

print("\n[Step 1] Constructing Cosmology(LCDM, Aletheia) ...")
cosmo = cbl.cosmology.LCDM(
    OMEGA_M, OMEGA_B, 0.0,             # Omega_M, Omega_b, Omega_radiation
    H0, SCALAR_AMP, SCALAR_PIVOT,
    N_S, TAU, True
)
cosmo.set_parameter("sigma8", SIGMA8)
print(f"        Omega_M={OMEGA_M}, sigma8={SIGMA8}, h={H0}")
print(f"        deltac(z=0) = {cosmo.deltac(0.0):.4f}")

# ─── Construct SizeFunction wrapper ──────────────────────────────────
# CBL's size_function is a method on cbl.cosmology.SizeFunction, not LCDM.
# The class takes a shared_ptr<Cosmology> in its constructor.
print("\n[Step 2] Constructing SizeFunction(cosmo) ...")
sf = cbl.cosmology.SizeFunction(cosmo)

# Signature (from Headers/SizeFunction.h line 280):
#   size_function(RV, redshift, model, b_eff,
#                 slope=0.854, offset=0.420,
#                 deltav_NL=-0.795, del_c=1.69,
#                 method_Pk="EisensteinHu", ...)
# Models: "SvdW", "Vdn", "linear"

R_grid = np.array([3., 5., 8., 10., 15., 20., 30.])
z = 0.0
b_eff = 1.0   # treat DM tracers as unbiased for the smoke test

print("\n[Step 3] Evaluating size_function(R, z=0)")
print(f"\n{'R [Mpc/h]':>12} | {'SvdW n(R)':>16} | {'Vdn n(R) (b=1)':>20}")
print("-" * 56)
for R in R_grid:
    n_svdw = sf.size_function(float(R), z, "SvdW", b_eff)
    n_vdn  = sf.size_function(float(R), z, "Vdn",  b_eff)
    print(f"{R:>12.2f} | {n_svdw:>16.6e} | {n_vdn:>20.6e}")

# ─── Quick monotonicity check ─────────────────────────────────────────
n_vdn = np.array([sf.size_function(float(R), z, "Vdn", b_eff) for R in R_grid])
diffs = np.diff(n_vdn)
print(f"\n[Step 4] Monotonicity check: dn/dR is "
      f"{'monotonically decreasing ✓' if (diffs < 0).all() else 'NOT monotone ✗'}")

print("\n[OK] cbl_smoke complete. CBL is ready to use.")
