#!/usr/bin/env python3
"""
variance_octants.py

Empirical variance estimate for cleaned NREC50 VSFs.

Method
------
Subdivide the 250^3 box into 8 non-overlapping 125^3 octants. Bin voids by
the octant their CENTER lies in. Compute the VSF in each octant with
V_eff_oct = 125^3 on ForArun's bin scheme. Across the 8 sub-VSFs we have
a per-bin mean and std deviation -- a *lower bound* on the variance one
would see across independent 125^3 realizations (lower because shared
large-scale modes between octants reduce inter-octant scatter).

If the inter-octant scatter band overlaps the offset between our cleaned
250^3 VSF and ForArun's reference, the single-realization-variance
hypothesis is empirically supported.

Outputs
-------
plots/cleaning_validation/variance_octants_NREC50.png
"""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
CLN  = os.path.join(HERE, '..', 'data', 'cleaned')
FA   = os.path.join(HERE, '..', 'data', 'ForArun')
OUT  = os.path.join(HERE, '..', 'plots', 'cleaning_validation')
MULT = {'20p': 1.13, '10p': 1.27, '5p': 1.43, '2_5p': 1.27}

os.makedirs(OUT, exist_ok=True)

FORARUN_BIN_CENTERS = np.array([
    1.591446e+00, 1.791403e+00, 2.016484e+00, 2.269845e+00, 2.555040e+00,
    2.876068e+00, 3.237432e+00, 3.644199e+00, 4.102074e+00, 4.617479e+00,
    5.197642e+00, 5.850699e+00, 6.585810e+00, 7.413284e+00, 8.344726e+00,
    9.393199e+00, 1.057341e+01, 1.190190e+01, 1.339732e+01, 1.508062e+01,
])
log_c = np.log(FORARUN_BIN_CENTERS)
DLNR  = float(np.median(np.diff(log_c)))
EDGES = np.exp(np.append(log_c - DLNR/2, log_c[-1] + DLNR/2))

V_FULL = 250.0 ** 3
V_OCT  = 125.0 ** 3
FA_MAP = {'2_5p': '025', '5p': '05', '10p': '10', '20p': '20'}
FRACTIONS = ['20p', '10p', '5p', '2_5p']

def vsf_from_radii(radii, v_eff):
    counts, _ = np.histogram(radii, bins=EDGES)
    return counts / (v_eff * DLNR), counts

def load_clean(frac):
    return pd.read_csv(os.path.join(CLN, f'cleaned_v4_recon_cat_{frac}.csv'))

def load_forarun(frac):
    f = os.path.join(FA, f'vsf_dm_250cube_ss0p{FA_MAP[frac]}_NREC50_SM0p00_CS0p79_TH-0p80_OV00.dat')
    df = pd.read_csv(f, sep=r'\s+', comment='#', names=['r','vsf','err'])
    return df[df['vsf'] > 0]

fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), sharey=True)

print(f"{'frac':<6}{'N_voids':>9}{'<oct>':>9}{'σ/μ@peak':>12}{'gap_FA':>10}{'within band?':>15}")
print("-"*65)

for ax, frac in zip(axes, FRACTIONS):
    df = load_clean(frac)
    fa = load_forarun(frac)
    # Box origin: auto-detect (NREC50 cleaned outputs preserve raw 375-625 coords)
    x0 = 0.0 if df['x'].min() < 200 else 375.0
    half = 125.0

    # Build 8 octants by bisecting each axis at x0+half
    octant_vsfs = []
    for ix in (0, 1):
        for iy in (0, 1):
            for iz in (0, 1):
                m = (
                    (df['x'] >= x0 + ix*half) & (df['x'] < x0 + (ix+1)*half) &
                    (df['y'] >= x0 + iy*half) & (df['y'] < x0 + (iy+1)*half) &
                    (df['z'] >= x0 + iz*half) & (df['z'] < x0 + (iz+1)*half)
                )
                v, _ = vsf_from_radii(df.loc[m, 'R_resc'].values * MULT[frac], V_OCT)
                octant_vsfs.append(v)
    octant_vsfs = np.array(octant_vsfs)         # (8, n_bins)
    mean_oct = octant_vsfs.mean(axis=0)
    std_oct  = octant_vsfs.std(axis=0, ddof=1)

    full_vsf, _ = vsf_from_radii(df['R_resc'].values * MULT[frac], V_FULL)

    mask = mean_oct > 0
    cb = FORARUN_BIN_CENTERS

    ax.fill_between(cb[mask],
                    mean_oct[mask] - std_oct[mask],
                    mean_oct[mask] + std_oct[mask],
                    color='#3498db', alpha=0.30,
                    label='octant scatter (μ±σ)')
    ax.loglog(cb[mask], mean_oct[mask], color='#3498db', lw=1.5, ls='--',
              label='octant mean')
    ax.loglog(cb[full_vsf>0], full_vsf[full_vsf>0], color='#2c3e50', lw=2,
              label='full 250³ cleaned')
    ax.errorbar(fa['r'], fa['vsf'], yerr=fa['err'], fmt='ko', ms=4,
                label='Reference')

    # Quantitative summary
    full_pos = full_vsf[full_vsf>0]
    pk_idx = int(np.argmax(full_pos))
    pk_R = cb[full_vsf>0][pk_idx]
    # find the same bin in ForArun
    fa_dict = dict(zip(np.round(fa['r'].values,4), fa['vsf'].values))
    fa_at_pk = fa_dict.get(round(pk_R,4), np.nan)
    rel_scatter = (std_oct[mask][pk_idx] / mean_oct[mask][pk_idx]) if mean_oct[mask][pk_idx]>0 else np.nan
    rel_gap     = abs(full_pos[pk_idx] - fa_at_pk)/fa_at_pk if (fa_at_pk and fa_at_pk>0) else np.nan
    within = "YES" if (not np.isnan(rel_scatter) and rel_scatter > rel_gap) else "no"
    print(f"{frac:<6}{len(df):>9}{len(df)/8:>9.0f}{rel_scatter:>12.2%}{rel_gap:>10.2%}{within:>15}")

    ax.set_title(f'NREC50 {frac}', fontsize=14)
    ax.set_xlabel(r'$R_v\ [h^{-1}\mathrm{Mpc}]$')
    ax.set_xlim(1.5, 16); ax.set_ylim(1e-7, 1e-2)
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.legend(fontsize=9, loc='lower left')

axes[0].set_ylabel(r'$\mathrm{d}n/\mathrm{d}\ln R_v\ [h^{3}\mathrm{Mpc}^{-3}]$')
plt.tight_layout()
out = os.path.join(OUT, 'variance_octants_NREC50_v4.png')
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f"\nSaved: {out}")
print("\nIf σ/μ at peak ≳ |full−FA|/FA, single-realization variance explains the gap.")
