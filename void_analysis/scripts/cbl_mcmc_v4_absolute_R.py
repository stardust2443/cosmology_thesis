#!/usr/bin/env python3
"""
cbl_mcmc_v4_absolute_R.py

Test whether the Vdn model has a universal R-cutoff (e.g., R > 7 h^-1 Mpc)
across all four NREC50 subsamples, rather than an MPS-relative one.

Runs the same v3-style (cosmology MCMC, nuisance fixed at joint MAP) but
with the same ABSOLUTE R_min for every subsample. If chi^2/nu becomes
consistent (~1) across all four subsamples, the model-validity boundary
is a fixed R scale, not tied to tracer density.

Run:
    /Users/arun/miniforge3/bin/python3.12 scripts/cbl_mcmc_v4_absolute_R.py [R_min]
"""
import os, sys, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CBL_HOME = os.environ.get('CBL_HOME', '/Users/arun/projects/CosmoBolognaLib')
sys.path.insert(0, os.path.join(CBL_HOME, 'Python'))
os.environ.setdefault('DYLD_LIBRARY_PATH', CBL_HOME)
import CosmoBolognaLib as cbl

import emcee
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
FA   = os.path.join(HERE, '..', 'data', 'ForArun')
OUT_PLOTS  = os.path.join(HERE, '..', 'plots', 'cbl_mcmc')
OUT_CHAINS = os.path.join(HERE, '..', 'data', 'mcmc_chains')

OMEGA_M_TRUE, SIGMA8_TRUE = 0.315, 0.811
H0, OMEGA_B, N_S, TAU = 0.6736, 0.0493, 0.9649, 0.0925
SCALAR_AMP, SCALAR_PIVOT = 2.194e-9, 0.05
DELTA_NL, B_EFF = -0.7, 1.0

FA_MAP = {'2_5p':'025','5p':'05','10p':'10','20p':'20'}
COLORS = {'20p': '#9b59b6', '10p': '#3498db', '5p': '#2ecc71', '2_5p': '#f1c40f'}
PRIORS = [('Omega_m', 0.10, 0.50), ('sigma8', 0.50, 1.20)]

# CLI: [R_min] [field]  with field in {NREC50, BIT}, default NREC50
R_MIN = float(sys.argv[1]) if len(sys.argv) > 1 else 7.0
FIELD = sys.argv[2] if len(sys.argv) > 2 else 'NREC50'
assert FIELD in ('NREC50', 'BIT')
NW, NS = 32, 800
print(f"Field = {FIELD}, R_min = {R_MIN:.2f} h^-1 Mpc for ALL subsamples")
os.makedirs(OUT_CHAINS, exist_ok=True)
os.makedirs(OUT_PLOTS, exist_ok=True)

def make_cosmo(om, s8):
    c = cbl.cosmology.LCDM(om, OMEGA_B, 0.0, H0,
                           SCALAR_AMP, SCALAR_PIVOT, N_S, TAU, True)
    c.set_parameter("sigma8", s8)
    return c

def load_data(frac, r_min):
    fa = os.path.join(FA, f'vsf_dm_250cube_ss0p{FA_MAP[frac]}_{FIELD}_'
                          'SM0p00_CS0p79_TH-0p80_OV00.dat')
    df = pd.read_csv(fa, sep=r'\s+', comment='#', names=['r','vsf','err'])
    m = (df['vsf'] > 0) & (df['r'] >= r_min)
    return df.loc[m, 'r'].values, df.loc[m, 'vsf'].values, df.loc[m, 'err'].values

def theory(om, s8, sl, of, R_data):
    try:
        c = make_cosmo(om, s8); sf = cbl.cosmology.SizeFunction(c)
        dc = c.deltac(0.0)
        return np.array([sf.size_function(float(R), 0.0, "Vdn",
                                          B_EFF, sl, of, DELTA_NL, dc, "EisensteinHu")
                         for R in R_data])
    except Exception:
        return None

chains = {}; results = []

for frac in ['20p', '10p', '5p', '2_5p']:
    R_d, v_d, e_d = load_data(frac, R_MIN)
    if len(R_d) < 4:
        print(f"\n=== {frac}: skip (only {len(R_d)} bins) ==="); continue
    print(f"\n=== {frac}: R>{R_MIN:.1f}, N_bins={len(R_d)} ===")
    LOG_V, LOG_E = np.log10(v_d), e_d / (v_d * np.log(10))

    # Joint MAP for nuisance calibration
    def nlp4(theta):
        om, s8, sl, of = theta
        if not (PRIORS[0][1] < om < PRIORS[0][2]): return 1e10
        if not (PRIORS[1][1] < s8 < PRIORS[1][2]): return 1e10
        if not (0.3 < sl < 1.5): return 1e10
        if not (0.0 < of < 1.0): return 1e10
        th = theory(om, s8, sl, of, R_d)
        if th is None or np.any(~np.isfinite(th)) or np.any(th <= 0): return 1e10
        return 0.5 * np.sum(((LOG_V - np.log10(th))/LOG_E)**2)
    res = minimize(nlp4, [OMEGA_M_TRUE, SIGMA8_TRUE, 0.857, 0.42],
                   method='Nelder-Mead',
                   options=dict(xatol=1e-3, fatol=1e-2, maxiter=300))
    om_map, s8_map, sl_map, of_map = res.x
    print(f"  4D MAP: Ω_m={om_map:.3f}, σ_8={s8_map:.3f}, "
          f"slope={sl_map:.3f}, offset={of_map:.3f}")

    # 2D MCMC, nuisance fixed
    def lp2(theta):
        om, s8 = theta
        if not (PRIORS[0][1] < om < PRIORS[0][2]): return -np.inf
        if not (PRIORS[1][1] < s8 < PRIORS[1][2]): return -np.inf
        th = theory(om, s8, sl_map, of_map, R_d)
        if th is None or np.any(~np.isfinite(th)) or np.any(th <= 0): return -np.inf
        return -0.5 * np.sum(((LOG_V - np.log10(th))/LOG_E)**2)

    pos = np.array([om_map, s8_map]) + 1e-3 * np.random.randn(NW, 2)
    pos[:, 0] = np.clip(pos[:, 0], PRIORS[0][1]+1e-3, PRIORS[0][2]-1e-3)
    pos[:, 1] = np.clip(pos[:, 1], PRIORS[1][1]+1e-3, PRIORS[1][2]-1e-3)

    t0 = time.time()
    sampler = emcee.EnsembleSampler(NW, 2, lp2)
    sampler.run_mcmc(pos, NS, progress=True)
    try: tau = sampler.get_autocorr_time(quiet=True)
    except emcee.autocorr.AutocorrError: tau = np.array([NS/50., NS/50.])
    burn = int(0.25*NS); thin = max(1, int(np.nanmin(tau)/2))
    flat = sampler.get_chain(discard=burn, thin=thin, flat=True)
    chains[frac] = flat
    # Persist to disk so Figure X.3 / future scripts can read without rerunning
    chain_path = os.path.join(
        OUT_CHAINS, f'cbl_v4_R{R_MIN:.0f}_{FIELD}_{frac}.npy')
    np.save(chain_path, flat)
    print(f"  chain saved: {chain_path}")

    th_map = theory(om_map, s8_map, sl_map, of_map, R_d)
    chi2 = np.sum(((LOG_V - np.log10(th_map))/LOG_E)**2)
    dof = len(R_d) - 2
    qom = np.percentile(flat[:, 0], [16, 50, 84])
    qs8 = np.percentile(flat[:, 1], [16, 50, 84])
    sig_om = (qom[1] - OMEGA_M_TRUE) / max((qom[2]-qom[0])/2, 1e-6)
    sig_s8 = (qs8[1] - SIGMA8_TRUE) / max((qs8[2]-qs8[0])/2, 1e-6)
    print(f"  χ²/ν = {chi2/dof:.3f}  Ω_m={qom[1]:.3f} ({sig_om:+.1f}σ)  "
          f"σ_8={qs8[1]:.3f} ({sig_s8:+.1f}σ)")
    results.append({'frac':frac,'nbins':len(R_d),'chi2_dof':chi2/dof,
                    'om':qom[1],'s8':qs8[1],'sig_om':sig_om,'sig_s8':sig_s8})

# Summary
print(f"\n\n=== Summary at R > {R_MIN:.1f} h^-1 Mpc ===")
print(f"{'frac':<6}{'Nbins':>6}{'χ²/ν':>8}{'Ω_m':>10}{'(σ)':>8}{'σ_8':>10}{'(σ)':>8}")
print('-'*60)
for r in results:
    print(f"{r['frac']:<6}{r['nbins']:>6}{r['chi2_dof']:>8.3f}"
          f"{r['om']:>10.3f}{r['sig_om']:>+8.1f}"
          f"{r['s8']:>10.3f}{r['sig_s8']:>+8.1f}")

# Combined posterior plot
fig, ax = plt.subplots(figsize=(8, 7))
for frac in ['20p', '10p', '5p', '2_5p']:
    if frac not in chains: continue
    fl = chains[frac]
    om_med = np.median(fl[:,0]); s8_med = np.median(fl[:,1])
    ax.scatter(fl[:,0], fl[:,1], s=2, alpha=0.05, color=COLORS[frac], rasterized=True)
    ax.plot(om_med, s8_med, 'o', color=COLORS[frac], mec='k', mew=1, ms=10,
            label=f'{frac} (n={len(fl)//(NW*NS//NS):d})  Ω={om_med:.3f}, σ_8={s8_med:.3f}')
ax.axvline(OMEGA_M_TRUE, color='red', ls='--', alpha=0.6, label='truth')
ax.axhline(SIGMA8_TRUE,  color='red', ls='--', alpha=0.6)
ax.plot(OMEGA_M_TRUE, SIGMA8_TRUE, 'r*', ms=18, mec='k')
ax.set_xlabel(r'$\Omega_m$', fontsize=14)
ax.set_ylabel(r'$\sigma_8$', fontsize=14)
ax.set_title(f'CBL Vdn posteriors — {FIELD}, R > {R_MIN:.1f} h⁻¹Mpc (absolute), all 4 subsamples',
             fontsize=12)
ax.legend(fontsize=10, loc='best')
ax.grid(True, ls='--', alpha=0.3)
plt.tight_layout()
out = os.path.join(OUT_PLOTS, f'cbl_v4_R{R_MIN:.0f}_{FIELD}_all_subsamples.png')
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f"\nSaved: {out}")
