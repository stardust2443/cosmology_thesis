#!/usr/bin/env python3
"""
cbl_mcmc_v1.py

First real CBL-based cosmological inference, following Contarini et al. 2023
"Cosmological constraints from the BOSS DR12 void size function" but adapted
to Aletheia DM-tracer subsamples.

Setup:
  - Data:        2.5% NREC50 Simone reference VSF (lowest χ²/ν in our regime)
  - Fit range:   R > 3 × MPS = 7 h⁻¹Mpc  (mimics Contarini's R > 3×MGS)
  - Theory:      CBL `Vdn` size_function with DM-tracer bias (b_eff=1, no bias)
  - Free:        Ω_m, σ_8, B_slope, B_offset  (4 params; nuisance pair per Eq. 6)
  - Fixed:       δ^NL_v = -0.7, b_eff = 1, del_c = cosmo.deltac(z=0)
  - Likelihood:  Gaussian log-VSF with Poisson errors (from ForArun .dat file)

Two stages:
  1. MAP: scipy.optimize.minimize on the log-posterior (cheap; supervisor's
     pre-MCMC sanity check). Confirms there IS a maximum interior to priors.
  2. MCMC: emcee EnsembleSampler from the MAP point. Note: we use emcee
     because building CBL's full Modelling_NumberCounts1D_Size pipeline
     would take longer than running 100 cluster jobs of this script. The
     theory call is CBL.

If the MAP-point χ²/ν is < 5 and the posteriors enclose Aletheia truth,
this is a publishable cosmological-inference result on DM tracers.

Run:
    /Users/arun/miniforge3/bin/python3.12 scripts/cbl_mcmc_v1.py
"""
import os, sys, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─── CBL bootstrap ────────────────────────────────────────────────────
CBL_HOME = os.environ.get('CBL_HOME', '/Users/arun/projects/CosmoBolognaLib')
sys.path.insert(0, os.path.join(CBL_HOME, 'Python'))
os.environ.setdefault('DYLD_LIBRARY_PATH', CBL_HOME)
import CosmoBolognaLib as cbl

import emcee
import corner
from scipy.optimize import minimize

# ─── Paths ────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
FA   = os.path.join(HERE, '..', 'data', 'ForArun')
OUT_PLOTS  = os.path.join(HERE, '..', 'plots', 'cbl_mcmc')
OUT_CHAINS = os.path.join(HERE, '..', 'data', 'mcmc_chains')
os.makedirs(OUT_PLOTS,  exist_ok=True)
os.makedirs(OUT_CHAINS, exist_ok=True)

# ─── Aletheia cosmology and fit setup ────────────────────────────────
OMEGA_M_TRUE, SIGMA8_TRUE = 0.315, 0.811
H0, OMEGA_B, N_S, TAU = 0.6736, 0.0493, 0.9649, 0.0925
SCALAR_AMP, SCALAR_PIVOT = 2.194e-9, 0.05

# Two nuisance parameter defaults from Contarini 2023 (eq 6, slope/offset of F)
# We let these vary; initial guesses are the CBL Vdn example values
B_SLOPE_DEFAULT  = 0.85676
B_OFFSET_DEFAULT = 0.42153

FRAC      = '2_5p'        # cleanest signal
MPS       = 2.33          # 2.5p subsample
R_MIN_FIT = 3.0 * MPS     # ≈ 7 h^-1 Mpc  (Contarini-equivalent cut)
DELTA_NL  = -0.7          # tracer-frame threshold; matches Contarini Sec 3.2
B_EFF     = 1.0           # DM tracers → unbiased

FA_MAP = {'2_5p':'025','5p':'05','10p':'10','20p':'20'}

# ─── Build cosmology (used for delta_c, then we recreate per-likelihood) ──
def make_cosmo(omega_m, sigma8):
    c = cbl.cosmology.LCDM(omega_m, OMEGA_B, 0.0, H0,
                           SCALAR_AMP, SCALAR_PIVOT, N_S, TAU, True)
    c.set_parameter("sigma8", sigma8)
    return c

# Precompute del_c at the truth (negligibly cosmology-dependent for ΛCDM)
DEL_C = make_cosmo(OMEGA_M_TRUE, SIGMA8_TRUE).deltac(0.0)
print(f"[setup] del_c(z=0) at truth = {DEL_C:.4f}")

# ─── Load data ────────────────────────────────────────────────────────
print(f"[setup] Loading Simone {FRAC} NREC50, R > {R_MIN_FIT:.2f}")
fa_file = os.path.join(FA, f'vsf_dm_250cube_ss0p{FA_MAP[FRAC]}_NREC50_'
                          'SM0p00_CS0p79_TH-0p80_OV00.dat')
df = pd.read_csv(fa_file, sep=r'\s+', comment='#', names=['r','vsf','err'])
mask = (df['vsf'] > 0) & (df['r'] >= R_MIN_FIT)
R_data, vsf_data, err_data = df.loc[mask, 'r'].values, df.loc[mask, 'vsf'].values, df.loc[mask, 'err'].values
print(f"[setup] {len(R_data)} bins from R={R_data[0]:.2f} to R={R_data[-1]:.2f}")

# Pre-compute log-data and log-errors (constant across MCMC)
LOG_VSF = np.log10(vsf_data)
LOG_ERR = err_data / (vsf_data * np.log(10))

# ─── Theory wrapper ──────────────────────────────────────────────────
def theory(omega_m, sigma8, slope, offset):
    """Return Vdn model evaluated at the data R values for these params."""
    try:
        c = make_cosmo(omega_m, sigma8)
        sf = cbl.cosmology.SizeFunction(c)
        del_c = c.deltac(0.0)
        out = np.array([
            sf.size_function(float(R), 0.0, "Vdn",
                             B_EFF, slope, offset,
                             DELTA_NL, del_c, "EisensteinHu")
            for R in R_data
        ])
        return out
    except Exception:
        return None

# ─── Bayesian framework ──────────────────────────────────────────────
PRIORS = [
    ('Omega_m',  0.10, 0.50),
    ('sigma8',   0.50, 1.20),
    ('B_slope',  0.30, 1.50),
    ('B_offset', 0.0,  1.0),
]

def log_prior(theta):
    for v, (_, lo, hi) in zip(theta, PRIORS):
        if not (lo < v < hi):
            return -np.inf
    return 0.0

def log_likelihood(theta):
    om, s8, sl, of = theta
    th = theory(om, s8, sl, of)
    if th is None or np.any(~np.isfinite(th)) or np.any(th <= 0):
        return -np.inf
    chi2 = np.sum(((LOG_VSF - np.log10(th)) / LOG_ERR)**2)
    return -0.5 * chi2

def log_posterior(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp): return -np.inf
    return lp + log_likelihood(theta)

# ─── Stage 1: MAP via scipy.optimize ─────────────────────────────────
print("\n[Stage 1] MAP estimation (scipy minimize on negative log-posterior)")
x0 = np.array([OMEGA_M_TRUE, SIGMA8_TRUE, B_SLOPE_DEFAULT, B_OFFSET_DEFAULT])
print(f"  Initial: Ω_m={x0[0]:.3f}, σ_8={x0[1]:.3f}, "
      f"B_slope={x0[2]:.3f}, B_offset={x0[3]:.3f}")
print(f"  log-posterior at init: {log_posterior(x0):.3f}")

t0 = time.time()
res = minimize(lambda th: -log_posterior(th), x0, method='Nelder-Mead',
               options=dict(xatol=1e-3, fatol=1e-2, maxiter=300, disp=True))
print(f"  Wall time: {time.time()-t0:.1f}s")

x_map = res.x
print(f"\n  MAP: Ω_m={x_map[0]:.4f}, σ_8={x_map[1]:.4f}, "
      f"B_slope={x_map[2]:.4f}, B_offset={x_map[3]:.4f}")
print(f"  -log-posterior at MAP: {res.fun:.3f}")
th_map = theory(*x_map)
if th_map is not None and np.all(th_map > 0):
    chi2_map = np.sum(((LOG_VSF - np.log10(th_map)) / LOG_ERR)**2)
    dof_map = len(R_data) - 4
    print(f"  χ²={chi2_map:.2f},  χ²/ν = {chi2_map / max(dof_map, 1):.3f}  (ν=N-4={dof_map})")

# ─── Stage 2: MCMC via emcee, initialized from MAP ────────────────────
NWALKERS = 32
NSTEPS   = 800
print(f"\n[Stage 2] MCMC: {NWALKERS} walkers × {NSTEPS} steps, "
      f"starting from MAP with 1e-3 ball")

pos = x_map + 1e-3 * np.random.randn(NWALKERS, 4)
# clip to priors
for i, (_, lo, hi) in enumerate(PRIORS):
    pos[:, i] = np.clip(pos[:, i], lo + 1e-3, hi - 1e-3)

t0 = time.time()
sampler = emcee.EnsembleSampler(NWALKERS, 4, log_posterior)
sampler.run_mcmc(pos, NSTEPS, progress=True)
print(f"  Wall time: {time.time()-t0:.1f}s")
print(f"  Acceptance fraction (mean): {sampler.acceptance_fraction.mean():.3f}")
try:
    tau = sampler.get_autocorr_time(quiet=True)
    print(f"  Autocorrelation times: {tau}")
except emcee.autocorr.AutocorrError as e:
    print(f"  Autocorr warning: {e}")
    tau = np.array([NSTEPS / 50.0] * 4)

burn_in = int(0.25 * NSTEPS)
thin    = max(1, int(np.nanmin(tau) / 2))
flat = sampler.get_chain(discard=burn_in, thin=thin, flat=True)
print(f"  Burn-in: {burn_in}, thin: {thin}, eff samples: {len(flat)}")

# Save chain
chain_path = os.path.join(OUT_CHAINS, f'cbl_chain_{FRAC}_NREC50.npy')
np.save(chain_path, flat)
print(f"  Chain saved: {chain_path}")

# ─── Posterior summary ────────────────────────────────────────────────
PARAM_NAMES = [n for n, _, _ in PRIORS]
TRUTHS      = [OMEGA_M_TRUE, SIGMA8_TRUE, None, None]
print("\n[Posterior]")
for i, (name, truth) in enumerate(zip(PARAM_NAMES, TRUTHS)):
    q = np.percentile(flat[:, i], [16, 50, 84])
    truth_str = f"(truth: {truth})" if truth is not None else ""
    print(f"  {name:>10s} = {q[1]:.4f}  (+{q[2]-q[1]:.4f} / -{q[1]-q[0]:.4f})  {truth_str}")

# ─── Plots ────────────────────────────────────────────────────────────
# Corner
fig = corner.corner(flat, labels=[r'$\Omega_m$', r'$\sigma_8$',
                                  r'$B_{slope}$', r'$B_{offset}$'],
                    truths=TRUTHS, quantiles=[0.16, 0.5, 0.84],
                    show_titles=True, levels=(1-np.exp(-0.5), 1-np.exp(-2)),
                    color='#2196F3', truth_color='#E53935')
fig.suptitle(f'CBL Vdn MCMC: NREC50 {FRAC} Simone, R > {R_MIN_FIT:.1f} h⁻¹Mpc',
             fontsize=13, y=1.02)
corner_path = os.path.join(OUT_PLOTS, f'corner_cbl_{FRAC}.png')
fig.savefig(corner_path, dpi=180, bbox_inches='tight')
print(f"\nCorner plot: {corner_path}")

# Best-fit overlay
om_med, s8_med, sl_med, of_med = np.median(flat, axis=0)
R_fine = np.logspace(np.log10(R_MIN_FIT*0.8), np.log10(R_data[-1]*1.1), 60)
c_med = make_cosmo(om_med, s8_med)
sf_med = cbl.cosmology.SizeFunction(c_med)
th_med = np.array([sf_med.size_function(float(R), 0.0, "Vdn",
                                         B_EFF, sl_med, of_med,
                                         DELTA_NL, c_med.deltac(0.0),
                                         "EisensteinHu")
                   for R in R_fine])
c_true = make_cosmo(OMEGA_M_TRUE, SIGMA8_TRUE)
sf_true = cbl.cosmology.SizeFunction(c_true)
th_true = np.array([sf_true.size_function(float(R), 0.0, "Vdn",
                                           B_EFF, sl_med, of_med,
                                           DELTA_NL, c_true.deltac(0.0),
                                           "EisensteinHu")
                    for R in R_fine])

fig, ax = plt.subplots(figsize=(9, 6.5))
ax.errorbar(R_data, vsf_data, yerr=err_data, fmt='ko', ms=5, capsize=3,
            label=f'Simone {FRAC} (R>{R_MIN_FIT:.1f})')
ax.plot(R_fine, th_med, '-', color='#2196F3', lw=2.4,
        label=f'MCMC median (Ω_m={om_med:.3f}, σ_8={s8_med:.3f})')
ax.plot(R_fine, th_true, '--', color='#E53935', lw=1.5,
        label=f'Truth (Ω_m={OMEGA_M_TRUE}, σ_8={SIGMA8_TRUE}), w/ MAP nuisance')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$R_v\ [h^{-1}\mathrm{Mpc}]$', fontsize=13)
ax.set_ylabel(r'$\mathrm{d}n/\mathrm{d}\ln R_v\ [h^3\mathrm{Mpc}^{-3}]$', fontsize=13)
ax.set_title(f'CBL Vdn best-fit: NREC50 {FRAC} Simone')
ax.grid(True, which='both', ls='--', alpha=0.3); ax.legend(fontsize=10)
bestfit_path = os.path.join(OUT_PLOTS, f'bestfit_cbl_{FRAC}.png')
fig.savefig(bestfit_path, dpi=180, bbox_inches='tight')
print(f"Best-fit plot: {bestfit_path}")
print("\nDone.")
