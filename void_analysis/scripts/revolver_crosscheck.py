#!/usr/bin/env python3
"""
revolver_crosscheck.py

Sanity check: BitVF (NREC50, our cleaning) vs REVOLVER (cleaned) vs
ForArun Simone data, for all 4 subsamples.

Expected behaviour from the literature:
  - Both finders should agree near the VSF peak and at large R.
  - BitVF should resolve smaller voids (extends to smaller R) than
    REVOLVER, because OT reconstruction recovers structure below the
    REVOLVER tessellation floor.
  - REVOLVER's peak sits at slightly larger R because it under-resolves
    small voids.

Confirming this qualitative behaviour validates the BitVF + cleaning
pipeline against an independent void finder, independent of the
ForArun ensemble issue at sparse densities.
"""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
CLN  = os.path.join(HERE, '..', 'data', 'cleaned')
FA   = os.path.join(HERE, '..', 'data', 'ForArun')
OUT  = os.path.join(HERE, '..', 'plots', 'cleaning_validation')
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
V_EFF = 250.0 ** 3
FA_MAP = {'2_5p':'025','5p':'05','10p':'10','20p':'20'}
FRACTIONS = ['20p','10p','5p','2_5p']
MULT = {'20p': 1.13, '10p': 1.27, '5p': 1.43, '2_5p': 1.27}

def vsf(radii):
    counts, _ = np.histogram(radii, bins=EDGES)
    return counts / (V_EFF * DLNR), counts

def load_revolver(frac):
    f = os.path.join(CLN, f'cleaned_revolver_cat_{frac}.dat')
    return pd.read_csv(f, sep=r'\s+', comment='#', names=['x','y','z','R_new','ID'])

def load_bitvf(frac):
    return pd.read_csv(os.path.join(CLN, f'cleaned_v4_recon_cat_{frac}.csv'))

def load_fa(frac):
    f = os.path.join(FA, f'vsf_dm_250cube_ss0p{FA_MAP[frac]}_NREC50_SM0p00_CS0p79_TH-0p80_OV00.dat')
    df = pd.read_csv(f, sep=r'\s+', comment='#', names=['r','vsf','err'])
    return df[df['vsf']>0]

fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), sharey=True)

print(f"{'frac':<6}{'N_BitVF':>9}{'N_REV':>8}{'pk_BitVF':>11}{'pk_REV':>9}{'pk_FA':>8}")
print("-"*55)

for ax, frac in zip(axes, FRACTIONS):
    bv = load_bitvf(frac)
    rv = load_revolver(frac)
    fa = load_fa(frac)

    v_bv, c_bv = vsf(bv['R_resc'].values * MULT[frac])
    v_rv, c_rv = vsf(rv['R_new'].values)

    err_bv = np.sqrt(c_bv) / (V_EFF * DLNR)
    err_rv = np.sqrt(c_rv) / (V_EFF * DLNR)

    m_bv = v_bv > 0; m_rv = v_rv > 0
    cb = FORARUN_BIN_CENTERS

    # BitVF (this work)
    ax.errorbar(cb[m_bv], v_bv[m_bv], yerr=err_bv[m_bv],
                fmt='o-', color='#3498db', ms=4, lw=1.8, label='BitVF NREC50 (cleaned)')
    # REVOLVER
    ax.errorbar(cb[m_rv], v_rv[m_rv], yerr=err_rv[m_rv],
                fmt='s-', color='#e67e22', ms=4, lw=1.8, label='REVOLVER (cleaned)')
    # ForArun Simone data
    ax.errorbar(fa['r'], fa['vsf'], yerr=fa['err'],
                fmt='k^', ms=5, label='Target (BitVF NREC50, ensemble)')

    pk_bv = cb[m_bv][np.argmax(v_bv[m_bv])]
    pk_rv = cb[m_rv][np.argmax(v_rv[m_rv])]
    pk_fa = fa['r'].values[np.argmax(fa['vsf'].values)]
    print(f"{frac:<6}{len(bv):>9}{len(rv):>8}{pk_bv:>11.2f}{pk_rv:>9.2f}{pk_fa:>8.2f}")

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_title(f'{frac}', fontsize=14)
    ax.set_xlabel(r'$R_v\ [h^{-1}\mathrm{Mpc}]$')
    ax.set_xlim(1.5, 22); ax.set_ylim(1e-7, 1e-2)
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.legend(fontsize=9, loc='lower left')

axes[0].set_ylabel(r'$\mathrm{d}n/\mathrm{d}\ln R_v\ [h^{3}\mathrm{Mpc}^{-3}]$')
fig.suptitle('Cross-check: BitVF v4 (calibrated) vs REVOLVER vs Simone data',
             fontsize=15, y=1.02)
plt.tight_layout()
out = os.path.join(OUT, 'revolver_crosscheck_v4.png')
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f"\nSaved: {out}")
