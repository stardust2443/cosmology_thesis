#!/usr/bin/env python3
"""
cbl_mcmc_v2_fixed_bias.py

Follow-up to cbl_mcmc_v1.py. The v1 MCMC found χ²/ν=1.4 (good!) but
Ω_m and σ_8 were biased low because nuisance parameters (B_slope, B_offset)
absorbed amplitude signal. This script fixes the nuisance parameters at
their v1-MAP values and re-runs with only (Ω_m, σ_8) free.

If the cosmology posteriors now enclose Aletheia truth, we have a
defensible cosmological result on DM tracers, modulo the bias-calibration
assumption.

Run:
    /Users/arun/miniforge3/bin/python3.12 scripts/cbl_mcmc_v2_fixed_bias.py
"""
import os, sys, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CBL_HOME = os.environ.get('CBL_HOME', '/Users/arun/projects/CosmoBolognaLib')
sys.path.insert(0, os.path.join(CBL_HOME, 'Python'))
os.environ.setdefault('DYLD_LIBRARY_PATH', CBL_HOME)
import CosmoBolognaLib as cbl

import emcee, corner
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
FA   = os.path.join(HERE, '..', 'data', 'ForArun')
OUT_PLOTS  = os.path.join(HERE, '..', 'plots', 'cbl_mcmc')
OUT_CHAINS = os.path.join(HERE, '..', 'data', 'mcmc_chains')
os.makedirs(OUT_PLOTS, exist_ok=True)
os.makedirs(OUT_CHAINS, exist_ok=True)

# Aletheia cosmology
OMEGA_M_TRUE, SIGMA8_TRUE = 0.315, 0.811
H0, OMEGA_B, N_S, TAU = 0.6736, 0.0493, 0.9649, 0.0925
SCALAR_AMP, SCALAR_PIVOT = 2.194e-9, 0.05

# Fit setup
FRAC      = '2_5p'
MPS       = 2.33
R_MIN_FIT = 3.0 * MPS
DELTA_NL  = -0.7
B_EFF     = 1.0

# === Nuisance fixed at v1-MAP values ===========================
B_SLOPE_FIXED  = 0.7146   # from cbl_mcmc_v1.py MAP
B_OFFSET_FIXED = 0.6508
print(f"[setup] Nuisance fixed at v1-MAP: "
      f"B_slope={B_SLOPE_FIXED}, B_offset={B_OFFSET_FIXED}")

FA_MAP = {'2_5p':'025','5p':'05','10p':'10','20p':'20'}

def make_cosmo(om, s8):
    c = cbl.cosmology.LCDM(om, OMEGA_B, 0.0, H0,
                           SCALAR_AMP, SCALAR_PIVOT, N_S, TAU, True)
    c.set_parameter("sigma8", s8)
    return c

# Load data
fa_file = os.path.join(FA, f'vsf_dm_250cube_ss0p{FA_MAP[FRAC]}_NREC50_'
                          'SM0p00_CS0p79_TH-0p80_OV00.dat')
df = pd.read_csv(fa_file, sep=r'\s+', comment='#', names=['r','vsf','err'])
mask = (df['vsf'] > 0) & (df['r'] >= R_MIN_FIT)
R_data    = df.loc[mask, 'r'].values
vsf_data  = df.loc[mask, 'vsf'].values
err_data  = df.loc[mask, 'err'].values
LOG_VSF   = np.log10(vsf_data)
LOG_ERR   = err_data / (vsf_data * np.log(10))
print(f"[setup] {len(R_data)} bins from R={R_data[0]:.2f} to R={R_data[-1]:.2f}")

def theory(om, s8):
    try:
        c = make_cosmo(om, s8)
        sf = cbl.cosmology.SizeFunction(c)
        dc = c.deltac(0.0)
        return np.array([
            sf.size_function(float(R), 0.0, "Vdn",
                             B_EFF, B_SLOPE_FIXED, B_OFFSET_FIXED,
                             DELTA_NL, dc, "EisensteinHu")
            for R in R_data
        ])
    except Exception:
        return None

PRIORS = [('Omega_m', 0.10, 0.50), ('sigma8', 0.50, 1.20)]
def log_prior(theta):
    for v, (_, lo, hi) in zip(theta, PRIORS):
        if not (lo < v < hi): return -np.inf
    return 0.0

def log_likelihood(theta):
    th = theory(*theta)
    if th is None or np.any(~np.isfinite(th)) or np.any(th <= 0):
        return -np.inf
    return -0.5 * np.sum(((LOG_VSF - np.log10(th)) / LOG_ERR)**2)

def log_posterior(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp): return -np.inf
    return lp + log_likelihood(theta)

# MAP
print("\n[Stage 1] MAP estimation")
x0 = np.array([OMEGA_M_TRUE, SIGMA8_TRUE])
t0 = time.time()
res = minimize(lambda th: -log_posterior(th), x0, method='Nelder-Mead',
               options=dict(xatol=1e-4, fatol=1e-3, maxiter=200, disp=True))
print(f"  wall: {time.time()-t0:.1f}s")
x_map = res.x
print(f"  MAP: Ω_m={x_map[0]:.4f}, σ_8={x_map[1]:.4f}")
th_map = theory(*x_map)
if th_map is not None:
    chi2_map = np.sum(((LOG_VSF - np.log10(th_map)) / LOG_ERR)**2)
    dof = len(R_data) - 2
    print(f"  χ²={chi2_map:.2f}, χ²/ν={chi2_map/dof:.3f}  (ν=N-2={dof})")

# MCMC
NW, NS = 32, 1500
print(f"\n[Stage 2] MCMC: {NW} × {NS} steps, init from MAP")
pos = x_map + 1e-3 * np.random.randn(NW, 2)
for i, (_, lo, hi) in enumerate(PRIORS):
    pos[:, i] = np.clip(pos[:, i], lo+1e-3, hi-1e-3)
t0 = time.time()
sampler = emcee.EnsembleSampler(NW, 2, log_posterior)
sampler.run_mcmc(pos, NS, progress=True)
print(f"  wall: {time.time()-t0:.1f}s, acc={sampler.acceptance_fraction.mean():.3f}")
try:
    tau = sampler.get_autocorr_time(quiet=True)
    print(f"  tau: {tau}")
except emcee.autocorr.AutocorrError:
    tau = np.array([NS/50.] * 2)

burn = int(0.25 * NS)
thin = max(1, int(np.nanmin(tau)/2))
flat = sampler.get_chain(discard=burn, thin=thin, flat=True)
print(f"  burn={burn}, thin={thin}, samples={len(flat)}")

chain_path = os.path.join(OUT_CHAINS, f'cbl_chain_fixed_bias_{FRAC}.npy')
np.save(chain_path, flat)
print(f"  chain saved: {chain_path}")

print("\n[Posterior]")
for i, (name, truth) in enumerate(zip(['Omega_m', 'sigma8'],
                                       [OMEGA_M_TRUE, SIGMA8_TRUE])):
    q = np.percentile(flat[:, i], [16, 50, 84])
    print(f"  {name:>9s} = {q[1]:.4f}  (+{q[2]-q[1]:.4f} / -{q[1]-q[0]:.4f})  "
          f"(truth: {truth}, deviation: {(q[1]-truth)/((q[2]-q[0])/2):+.2f}σ)")

# Corner
fig = corner.corner(flat, labels=[r'$\Omega_m$', r'$\sigma_8$'],
                    truths=[OMEGA_M_TRUE, SIGMA8_TRUE],
                    quantiles=[0.16, 0.5, 0.84], show_titles=True,
                    levels=(1-np.exp(-0.5), 1-np.exp(-2)),
                    color='#2196F3', truth_color='#E53935')
fig.suptitle(f'CBL Vdn MCMC, nuisance FIXED at v1-MAP\n'
             f'NREC50 {FRAC} Simone, R > {R_MIN_FIT:.1f}', fontsize=12, y=1.02)
out = os.path.join(OUT_PLOTS, f'corner_fixed_bias_{FRAC}.png')
fig.savefig(out, dpi=180, bbox_inches='tight')
print(f"\nCorner plot: {out}")
print("Done.")
