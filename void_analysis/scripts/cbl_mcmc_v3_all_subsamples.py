#!/usr/bin/env python3
"""
cbl_mcmc_v3_all_subsamples.py

Run the same v2-style (cosmology-only, nuisance fixed) MCMC on all four
NREC50 subsamples, then overlay the posteriors to test consistency.

For each subsample, we:
  - Restrict to R > 3*MPS
  - Fix nuisance (B_slope, B_offset) at per-subsample MAP from a joint fit
  - Run emcee on (Omega_m, sigma8)
  - Save chain and corner

Then produce a combined figure with all 4 (Omega_m, sigma8) posteriors and
the Aletheia truth marked.

This is the "headline" thesis figure: does the model recover truth at any
tracer density, or is the cosmology bias universal across subsamples?

Run:
    /Users/arun/miniforge3/bin/python3.12 scripts/cbl_mcmc_v3_all_subsamples.py
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

# Aletheia truth + cosmology
OMEGA_M_TRUE, SIGMA8_TRUE = 0.315, 0.811
H0, OMEGA_B, N_S, TAU = 0.6736, 0.0493, 0.9649, 0.0925
SCALAR_AMP, SCALAR_PIVOT = 2.194e-9, 0.05

DELTA_NL = -0.7
B_EFF    = 1.0

# Per-subsample MPS
MPS_OF = {'20p': 1.16, '10p': 1.47, '5p': 1.85, '2_5p': 2.33}
FA_MAP = {'2_5p':'025','5p':'05','10p':'10','20p':'20'}
COLORS = {'20p': '#9b59b6', '10p': '#3498db', '5p': '#2ecc71', '2_5p': '#f1c40f'}

NW, NS = 32, 1000

def make_cosmo(om, s8):
    c = cbl.cosmology.LCDM(om, OMEGA_B, 0.0, H0,
                           SCALAR_AMP, SCALAR_PIVOT, N_S, TAU, True)
    c.set_parameter("sigma8", s8)
    return c

def load_data(frac, r_min):
    fa_file = os.path.join(FA, f'vsf_dm_250cube_ss0p{FA_MAP[frac]}_NREC50_'
                              'SM0p00_CS0p79_TH-0p80_OV00.dat')
    df = pd.read_csv(fa_file, sep=r'\s+', comment='#', names=['r','vsf','err'])
    m = (df['vsf'] > 0) & (df['r'] >= r_min)
    return df.loc[m, 'r'].values, df.loc[m, 'vsf'].values, df.loc[m, 'err'].values

def theory(om, s8, slope, offset, R_data):
    try:
        c = make_cosmo(om, s8)
        sf = cbl.cosmology.SizeFunction(c)
        dc = c.deltac(0.0)
        return np.array([
            sf.size_function(float(R), 0.0, "Vdn",
                             B_EFF, slope, offset, DELTA_NL, dc, "EisensteinHu")
            for R in R_data
        ])
    except Exception:
        return None

PRIORS = [('Omega_m', 0.10, 0.50), ('sigma8', 0.50, 1.20)]

# ─── Per-subsample run ────────────────────────────────────────────────
chains  = {}
results = []

for frac in ['20p', '10p', '5p', '2_5p']:
    mps  = MPS_OF[frac]
    rmin = 3.0 * mps
    R_d, v_d, e_d = load_data(frac, rmin)
    print(f"\n=== {frac}: R>{rmin:.2f}, N_bins={len(R_d)} ===")
    if len(R_d) < 4:
        print(f"  SKIP — too few bins")
        continue

    LOG_V = np.log10(v_d)
    LOG_E = e_d / (v_d * np.log(10))

    # Stage A: 4-parameter MAP to calibrate per-subsample nuisance
    def neg_logp_4d(theta):
        om, s8, sl, of = theta
        if not (PRIORS[0][1] < om < PRIORS[0][2]): return 1e10
        if not (PRIORS[1][1] < s8 < PRIORS[1][2]): return 1e10
        if not (0.3 < sl < 1.5): return 1e10
        if not (0.0 < of < 1.0): return 1e10
        th = theory(om, s8, sl, of, R_d)
        if th is None or np.any(~np.isfinite(th)) or np.any(th <= 0): return 1e10
        return 0.5 * np.sum(((LOG_V - np.log10(th)) / LOG_E)**2)

    t0 = time.time()
    res4 = minimize(neg_logp_4d, [OMEGA_M_TRUE, SIGMA8_TRUE, 0.857, 0.42],
                    method='Nelder-Mead',
                    options=dict(xatol=1e-3, fatol=1e-2, maxiter=300))
    om_map, s8_map, sl_map, of_map = res4.x
    print(f"  Joint MAP ({time.time()-t0:.1f}s): "
          f"Ω_m={om_map:.3f}, σ_8={s8_map:.3f}, "
          f"B_slope={sl_map:.3f}, B_offset={of_map:.3f}")

    # Stage B: 2-parameter MCMC with nuisance fixed at joint MAP
    def log_post_2d(theta):
        om, s8 = theta
        if not (PRIORS[0][1] < om < PRIORS[0][2]): return -np.inf
        if not (PRIORS[1][1] < s8 < PRIORS[1][2]): return -np.inf
        th = theory(om, s8, sl_map, of_map, R_d)
        if th is None or np.any(~np.isfinite(th)) or np.any(th <= 0): return -np.inf
        return -0.5 * np.sum(((LOG_V - np.log10(th)) / LOG_E)**2)

    pos = np.array([om_map, s8_map]) + 1e-3 * np.random.randn(NW, 2)
    pos[:, 0] = np.clip(pos[:, 0], PRIORS[0][1]+1e-3, PRIORS[0][2]-1e-3)
    pos[:, 1] = np.clip(pos[:, 1], PRIORS[1][1]+1e-3, PRIORS[1][2]-1e-3)

    print(f"  MCMC ({NW}x{NS}) ...")
    t0 = time.time()
    sampler = emcee.EnsembleSampler(NW, 2, log_post_2d)
    sampler.run_mcmc(pos, NS, progress=True)
    print(f"  wall: {time.time()-t0:.1f}s, acc={sampler.acceptance_fraction.mean():.3f}")

    try:
        tau = sampler.get_autocorr_time(quiet=True)
    except emcee.autocorr.AutocorrError:
        tau = np.array([NS/50., NS/50.])
    burn = int(0.25*NS); thin = max(1, int(np.nanmin(tau)/2))
    flat = sampler.get_chain(discard=burn, thin=thin, flat=True)
    chains[frac] = flat
    np.save(os.path.join(OUT_CHAINS, f'cbl_chain_v3_{frac}.npy'), flat)

    # chi2 at MAP
    th_map = theory(om_map, s8_map, sl_map, of_map, R_d)
    chi2_map = np.sum(((LOG_V - np.log10(th_map)) / LOG_E)**2)
    dof = len(R_d) - 2
    qom = np.percentile(flat[:, 0], [16, 50, 84])
    qs8 = np.percentile(flat[:, 1], [16, 50, 84])
    sig_om = (qom[1] - OMEGA_M_TRUE) / ((qom[2]-qom[0])/2)
    sig_s8 = (qs8[1] - SIGMA8_TRUE) / ((qs8[2]-qs8[0])/2)

    print(f"  χ²/ν = {chi2_map/dof:.3f}")
    print(f"  Ω_m = {qom[1]:.3f} (+{qom[2]-qom[1]:.3f}/-{qom[1]-qom[0]:.3f})  "
          f"deviation: {sig_om:+.2f}σ")
    print(f"  σ_8 = {qs8[1]:.3f} (+{qs8[2]-qs8[1]:.3f}/-{qs8[1]-qs8[0]:.3f})  "
          f"deviation: {sig_s8:+.2f}σ")

    results.append({
        'frac': frac, 'nbins': len(R_d), 'rmin': rmin,
        'om_map': om_map, 's8_map': s8_map,
        'sl_map': sl_map, 'of_map': of_map,
        'om_med': qom[1], 's8_med': qs8[1],
        'sig_om': sig_om, 'sig_s8': sig_s8,
        'chi2_dof': chi2_map/dof,
    })

# ─── Summary table ────────────────────────────────────────────────────
print("\n\n=== Summary across subsamples ===")
print(f"{'frac':<6}{'Nbins':>6}{'χ²/ν':>8}"
      f"{'Ω_m':>10}{'(σ)':>8}{'σ_8':>10}{'(σ)':>8}")
print('-'*60)
for r in results:
    print(f"{r['frac']:<6}{r['nbins']:>6}{r['chi2_dof']:>8.3f}"
          f"{r['om_med']:>10.3f}{r['sig_om']:>+8.2f}"
          f"{r['s8_med']:>10.3f}{r['sig_s8']:>+8.2f}")

# ─── Combined corner plot ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
for frac in ['20p', '10p', '5p', '2_5p']:
    if frac not in chains: continue
    fl = chains[frac]
    # 2D contour
    from scipy.stats import gaussian_kde
    x, y = fl[:, 0], fl[:, 1]
    om_med = np.median(x); s8_med = np.median(y)
    ax.scatter(x, y, s=2, alpha=0.05, color=COLORS[frac], rasterized=True)
    ax.plot(om_med, s8_med, 'o', color=COLORS[frac], mec='k', mew=1, ms=10,
            label=f'{frac}  Ω={om_med:.3f}, σ_8={s8_med:.3f}')

ax.axvline(OMEGA_M_TRUE, color='red', ls='--', alpha=0.6, label='Aletheia truth')
ax.axhline(SIGMA8_TRUE,  color='red', ls='--', alpha=0.6)
ax.plot(OMEGA_M_TRUE, SIGMA8_TRUE, 'r*', ms=18, mec='k')
ax.set_xlabel(r'$\Omega_m$', fontsize=14)
ax.set_ylabel(r'$\sigma_8$', fontsize=14)
ax.set_title('CBL Vdn cosmology posteriors across 4 NREC50 subsamples\n'
             '(nuisance fixed per-subsample at 4-param MAP)', fontsize=12)
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, ls='--', alpha=0.3)
out = os.path.join(OUT_PLOTS, 'cbl_v3_all_subsamples_posterior.png')
plt.tight_layout()
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f"\nCombined posterior plot: {out}")
print("Done.")
