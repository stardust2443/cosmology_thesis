#!/usr/bin/env python3
"""
diagnose_peak_shift.py

Investigates why NREC50 5p and 2_5p subsamples show a VSF peak shifted to
larger radii compared to the ForArun Simone data.

Strategy: compare three VSF curves per subsample, all using ForArun's exact
20-bin scheme and V_eff = 250^3:
  (A) RAW catalog, no cuts (only r_ext > 0)
  (B) RAW after density threshold only (delta_avg <= -0.80)
  (C) FINAL cleaned catalog (post-overlap, post-border)
overlaid on the ForArun target.

If (A) or (B) already match ForArun's peak, the greedy overlap pass is
introducing the shift -> try size-stratified overlap.  If even (A) is shifted,
the shift is intrinsic to the raw catalog at low tracer density.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RAW = os.path.join(HERE, '..', 'data', 'raw')
FA  = os.path.join(HERE, '..', 'data', 'ForArun')
CLN = os.path.join(HERE, '..', 'data', 'cleaned')
OUT = os.path.join(HERE, '..', 'plots', 'cleaning_validation')
os.makedirs(OUT, exist_ok=True)

V_EFF = 250.0 ** 3

FORARUN_BIN_CENTERS = np.array([
    1.591446e+00, 1.791403e+00, 2.016484e+00, 2.269845e+00, 2.555040e+00,
    2.876068e+00, 3.237432e+00, 3.644199e+00, 4.102074e+00, 4.617479e+00,
    5.197642e+00, 5.850699e+00, 6.585810e+00, 7.413284e+00, 8.344726e+00,
    9.393199e+00, 1.057341e+01, 1.190190e+01, 1.339732e+01, 1.508062e+01,
])

def get_bins():
    log_c = np.log(FORARUN_BIN_CENTERS)
    dlnr  = np.median(np.diff(log_c))
    edges = np.exp(np.append(log_c - dlnr/2, log_c[-1] + dlnr/2))
    return edges, FORARUN_BIN_CENTERS, dlnr

def vsf(radii):
    edges, centers, dlnr = get_bins()
    counts, _ = np.histogram(radii, bins=edges)
    dn = counts / (V_EFF * dlnr)
    mask = counts > 0
    return centers[mask], dn[mask]

RAW_COLS = ['x', 'y', 'z', 'radius', 'r_ext', 'delta_2MPS', 'delta_avg']
FA_MAP   = {'2_5p': '025', '5p': '05', '10p': '10', '20p': '20'}
MULT     = {'20p': 1.13, '10p': 1.27, '5p': 1.43, '2_5p': 1.27}

def load_raw(frac):
    f = os.path.join(RAW, f'voids_recon_50_bitvf_aletheia_{frac}')
    return pd.read_csv(f, sep=r'\s+', comment='#', names=RAW_COLS)

def load_clean(frac):
    f = os.path.join(CLN, f'cleaned_v4_recon_cat_{frac}.csv')
    return pd.read_csv(f)

def load_forarun(frac):
    f = os.path.join(FA, f'vsf_dm_250cube_ss0p{FA_MAP[frac]}_NREC50_SM0p00_CS0p79_TH-0p80_OV00.dat')
    df = pd.read_csv(f, sep=r'\s+', comment='#', names=['r', 'vsf', 'err'])
    return df[df['vsf'] > 0]

def peak_r(centers, vsf_vals):
    if len(vsf_vals) == 0: return np.nan
    return centers[np.argmax(vsf_vals)]

fractions = ['20p', '10p', '5p', '2_5p']
fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), sharey=True)

print(f"{'frac':<6}{'N_raw':>8}{'N_TH':>8}{'N_clean':>9}{'pk_raw':>9}{'pk_TH':>9}{'pk_cln':>9}{'pk_FA':>9}")
print("-"*70)

for ax, frac in zip(axes, fractions):
    raw = load_raw(frac)
    raw = raw[raw['r_ext'] > 0]
    th  = raw[raw['delta_avg'] <= -0.80]
    cln = load_clean(frac)
    fa  = load_forarun(frac)

    cA, vA = vsf(raw['r_ext'].values)
    cB, vB = vsf(th['r_ext'].values)
    cC, vC = vsf(cln['R_resc'].values * MULT[frac])

    ax.loglog(cA, vA, color='gray',    lw=1.5, ls=':',  label=f'(A) raw  N={len(raw)}')
    ax.loglog(cB, vB, color='orange',  lw=1.5, ls='--', label=f'(B) +TH  N={len(th)}')
    ax.loglog(cC, vC, color='#3498db', lw=2.0,           label=f'(C) clean N={len(cln)}')
    ax.errorbar(fa['r'], fa['vsf'], yerr=fa['err'], fmt='ko', ms=4, label='Simone data (OV00)')

    ax.set_title(f'NREC50 {frac}', fontsize=14)
    ax.set_xlabel(r'$R_v\ [h^{-1}\mathrm{Mpc}]$')
    ax.set_xlim(1.5, 16)
    ax.set_ylim(1e-7, 1e-2)
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.legend(fontsize=9, loc='lower left')

    pk_fa = peak_r(fa['r'].values, fa['vsf'].values)
    print(f"{frac:<6}{len(raw):>8}{len(th):>8}{len(cln):>9}"
          f"{peak_r(cA,vA):>9.2f}{peak_r(cB,vB):>9.2f}"
          f"{peak_r(cC,vC):>9.2f}{pk_fa:>9.2f}")

axes[0].set_ylabel(r'$\mathrm{d}n/\mathrm{d}\ln R_v\ [h^{3}\mathrm{Mpc}^{-3}]$')
plt.tight_layout()
out = os.path.join(OUT, 'peak_shift_diagnosis_v4.png')
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f"\nSaved: {out}")
