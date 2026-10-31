#!/usr/bin/env python3
"""
jackknife_vsf.py

Compute jackknife (leave-one-octant-out) error bars on the cleaned NREC50
VSFs, and save the resulting (R, dn/dlnR, sigma_JK) tables in the same
format as Simone's ForArun .dat files. These can then be fed directly
into run_mcmc_vsf.py as a drop-in replacement for the Simone data data,
giving honest cosmic-variance-aware errors instead of Poisson errors.

Method
------
1. Bin voids by which 125^3 octant their CENTER lies in (8 octants).
2. Compute the full-box VSF on V_eff = 250^3 -- this is the central value.
3. For each octant i, build a leave-one-out (LOO) VSF using the remaining
   7 octants on V_eff_LOO = 7 * 125^3.
4. Jackknife variance per bin: var = (N-1)/N * sum_i (VSF_LOO_i - mean_LOO)^2
5. Jackknife std = sqrt(var).

The N-1 factor is the standard jackknife correction for the fact that
LOO subsamples are positively correlated.

Outputs
-------
data/cleaned/vsf_jk/vsf_NREC50_{frac}_jk.dat   (3-column: r, vsf, sigma_jk)
plots/cleaning_validation/jackknife_vs_poisson.png  (visual comparison)
"""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
CLN  = os.path.join(HERE, '..', 'data', 'cleaned')
FA   = os.path.join(HERE, '..', 'data', 'ForArun')
OUT_DIR  = os.path.join(CLN, 'vsf_jk')
PLOT_DIR = os.path.join(HERE, '..', 'plots', 'cleaning_validation')
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

FORARUN_BIN_CENTERS = np.array([
    1.591446e+00, 1.791403e+00, 2.016484e+00, 2.269845e+00, 2.555040e+00,
    2.876068e+00, 3.237432e+00, 3.644199e+00, 4.102074e+00, 4.617479e+00,
    5.197642e+00, 5.850699e+00, 6.585810e+00, 7.413284e+00, 8.344726e+00,
    9.393199e+00, 1.057341e+01, 1.190190e+01, 1.339732e+01, 1.508062e+01,
])
log_c = np.log(FORARUN_BIN_CENTERS)
DLNR  = float(np.median(np.diff(log_c)))
EDGES = np.exp(np.append(log_c - DLNR/2, log_c[-1] + DLNR/2))

V_FULL = 250.0**3
V_OCT  = 125.0**3
V_LOO  = 7.0 * V_OCT  # leave-one-out volume

FA_MAP = {'2_5p':'025','5p':'05','10p':'10','20p':'20'}
FRACTIONS = ['20p','10p','5p','2_5p']

def counts_in_radii(radii):
    c, _ = np.histogram(radii, bins=EDGES)
    return c

def assign_octants(df):
    """Return integer octant id 0..7 for each void center."""
    x0 = 0.0 if df['x'].min() < 200 else 375.0
    half = 125.0
    ix = ((df['x'].values - x0) >= half).astype(int)
    iy = ((df['y'].values - x0) >= half).astype(int)
    iz = ((df['z'].values - x0) >= half).astype(int)
    return ix * 4 + iy * 2 + iz

def jackknife_vsf(df):
    """
    Returns (R, vsf_full, sigma_JK, counts_full).
    vsf_full uses V_eff = 250^3 (the natural single-realization estimate).
    sigma_JK is computed by leave-one-octant-out on V_eff_LOO = 7 * 125^3.
    """
    radii = df['r_ext'].values
    oct_id = assign_octants(df)

    # central value: full-box VSF
    counts_full = counts_in_radii(radii)
    vsf_full    = counts_full / (V_FULL * DLNR)

    # leave-one-out VSFs
    loo = np.zeros((8, len(FORARUN_BIN_CENTERS)))
    for k in range(8):
        mask = oct_id != k
        c_k = counts_in_radii(radii[mask])
        loo[k] = c_k / (V_LOO * DLNR)

    # standard jackknife variance (N-1)/N * Σ (θ_(i) - <θ_(i)>)²
    mean_loo = loo.mean(axis=0)
    var_jk   = (7.0 / 8.0) * np.sum((loo - mean_loo)**2, axis=0)
    sigma_jk = np.sqrt(var_jk)
    return FORARUN_BIN_CENTERS, vsf_full, sigma_jk, counts_full

def load_simone(frac):
    f = os.path.join(FA, f'vsf_dm_250cube_ss0p{FA_MAP[frac]}_NREC50_SM0p00_CS0p79_TH-0p80_OV00.dat')
    return pd.read_csv(f, sep=r'\s+', comment='#', names=['r','vsf','err'])

# ------------- main loop -------------

fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), sharey=True)

print(f"{'frac':<6}{'N_voids':>8}{'<σ_P/μ>':>11}{'<σ_JK/μ>':>11}{'JK/Poisson':>12}")
print("-"*52)

for ax, frac in zip(axes, FRACTIONS):
    df = pd.read_csv(os.path.join(CLN, f'cleaned_recon_cat_{frac}.csv'))
    R, vsf, sig_jk, counts = jackknife_vsf(df)

    # Poisson error for comparison
    sig_poisson = np.sqrt(counts) / (V_FULL * DLNR)
    sim = load_simone(frac)

    # Save .dat in Simone's 3-column format (r, vsf, err)
    out = pd.DataFrame({'r': R, 'vsf': vsf, 'err': sig_jk})
    out = out[out['vsf'] > 0]
    out_path = os.path.join(OUT_DIR, f'vsf_NREC50_{frac}_jk.dat')
    with open(out_path, 'w') as f:
        f.write('# r_v[h^-1 Mpc]    dn/dlnR[h^3 Mpc^-3]    sigma_JK\n')
        out.to_csv(f, sep='\t', index=False, header=False, float_format='%.6e')

    m = vsf > 0
    # plot: vsf with Poisson errors (light) vs jackknife errors (heavy)
    ax.errorbar(R[m], vsf[m], yerr=sig_poisson[m],
                fmt='o', color='gray', alpha=0.5, ms=3, label='Poisson')
    ax.errorbar(R[m], vsf[m], yerr=sig_jk[m],
                fmt='o', color='#3498db', ms=4, lw=1.7, label='Jackknife (this work)')
    ax.errorbar(sim['r'], sim['vsf'], yerr=sim['err'],
                fmt='k^', ms=4, label='Simone (Poisson)')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_title(f'NREC50 {frac}', fontsize=13)
    ax.set_xlabel(r'$R_v\ [h^{-1}\mathrm{Mpc}]$')
    ax.set_xlim(1.5, 16); ax.set_ylim(1e-7, 1e-2)
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.legend(fontsize=9, loc='lower left')

    # numerical summary at bins where vsf>0
    mvsf = vsf[m]
    ratio = sig_jk[m] / np.maximum(sig_poisson[m], 1e-30)
    print(f"{frac:<6}{len(df):>8}"
          f"{np.mean(sig_poisson[m]/mvsf):>11.2%}"
          f"{np.mean(sig_jk[m]/mvsf):>11.2%}"
          f"{np.mean(ratio):>12.2f}×")

axes[0].set_ylabel(r'$\mathrm{d}n/\mathrm{d}\ln R_v\ [h^{3}\mathrm{Mpc}^{-3}]$')
fig.suptitle('Jackknife vs Poisson errors on cleaned NREC50 VSFs', fontsize=14, y=1.02)
plt.tight_layout()
out_png = os.path.join(PLOT_DIR, 'jackknife_vs_poisson.png')
fig.savefig(out_png, dpi=200, bbox_inches='tight')
print(f"\nSaved files:")
for frac in FRACTIONS:
    print(f"  {OUT_DIR}/vsf_NREC50_{frac}_jk.dat")
print(f"  {out_png}")
