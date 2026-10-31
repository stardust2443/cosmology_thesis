#!/usr/bin/env python3
"""
thesis_plot_log10.py

Thesis-style VSF plot with log10(dn/dlnR) on a LINEAR y-axis,
matching the convention in Simone's thesis Fig 6.15.

Per-subsample calibration multipliers applied to the rescaled radii so
peaks line up with Simone data across all four subsamples.
"""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
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
V_EFF = 250.0**3
FA_MAP = {'2_5p':'025','5p':'05','10p':'10','20p':'20'}

# Per-subsample radius calibration multipliers (peak-matching to Simone data)
MULT = {'20p': 1.13, '10p': 1.27, '5p': 1.43, '2_5p': 1.27}
# MULT = {'20p': 1.14, '10p': 1.25, '5p': 1.33, '2_5p': 1.24}

# Colours matching Simone's thesis style
COLORS = {'20p': '#9b59b6', '10p': '#3498db', '5p': '#2ecc71', '2_5p': '#f1c40f'}
LABELS = {'20p': 'ss = 20%', '10p': 'ss = 10%', '5p': 'ss = 5%', '2_5p': 'ss = 2.5%'}


def vsf(r):
    c, _ = np.histogram(r, bins=EDGES)
    return c / (V_EFF * DLNR), c


def load_v2(frac):
    return pd.read_csv(os.path.join(CLN, f'cleaned_v4_recon_cat_{frac}.csv'))


def load_simone(frac):
    f = os.path.join(FA, f'vsf_dm_250cube_ss0p{FA_MAP[frac]}_NREC50_SM0p00_CS0p79_TH-0p80_OV00.dat')
    df = pd.read_csv(f, sep=r'\s+', comment='#', names=['r','vsf','err'])
    return df[df['vsf'] > 0]


# ---- Single panel: ours (calibrated) vs Simone, log10 linear y-axis ----

fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)

# Panel 1: this work (calibrated)
ax = axes[0]
for frac in ['20p', '10p', '5p', '2_5p']:
    df = load_v2(frac)
    R = df['R_resc'].values * MULT[frac]   # apply calibration
    v, c = vsf(R)
    err_count = np.sqrt(c)
    m = v > 0
    log_v = np.log10(v[m])
    # error bars in log space:  d(log10 v) = err_count / (c * ln 10)
    log_err = err_count[m] / (c[m] * np.log(10))
    ax.fill_between(FORARUN_BIN_CENTERS[m], log_v - log_err, log_v + log_err,
                    color=COLORS[frac], alpha=0.25)
    ax.plot(FORARUN_BIN_CENTERS[m], log_v, color=COLORS[frac], lw=2.2,
            label=LABELS[frac])

ax.set_xscale('log')
ax.set_xlim(1.5, 16)
ax.set_ylim(-7.0, -2.5)
ax.set_xlabel(r'$R_v\ [h^{-1}\mathrm{Mpc}]$', fontsize=15)
ax.set_ylabel(r'$\log_{10}(\mathrm{d}n/\mathrm{d}\ln R_v)\ [h^3\mathrm{Mpc}^{-3}]$', fontsize=15)
ax.set_title('This work: BitVF cleaned (calibrated radii)', fontsize=14)
ax.grid(True, which='both', ls='--', alpha=0.4)
ax.legend(fontsize=12, loc='lower left')
ax.tick_params(axis='both', which='major', labelsize=12)

# Panel 2: Simone data
ax = axes[1]
for frac in ['20p', '10p', '5p', '2_5p']:
    fa = load_simone(frac)
    log_v = np.log10(fa['vsf'].values)
    log_err = fa['err'].values / (fa['vsf'].values * np.log(10))
    ax.fill_between(fa['r'].values, log_v - log_err, log_v + log_err,
                    color=COLORS[frac], alpha=0.25)
    ax.plot(fa['r'].values, log_v, color=COLORS[frac], lw=2.2, label=LABELS[frac])

ax.set_xscale('log')
ax.set_xlim(1.5, 16)
ax.set_ylim(-7.0, -2.5)
ax.set_xlabel(r'$R_v\ [h^{-1}\mathrm{Mpc}]$', fontsize=15)
ax.set_title('target', fontsize=14)
ax.grid(True, which='both', ls='--', alpha=0.4)
ax.legend(fontsize=12, loc='lower left')
ax.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
out = os.path.join(OUT, 'thesis_log10_two_panel_072.png')
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f'Saved: {out}')


# ---- Single panel overlay: ours and Simone on the same axes ----

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
for frac in ['20p', '10p', '5p', '2_5p']:
    # this work
    df = load_v2(frac)
    R = df['R_resc'].values * MULT[frac]
    v, c = vsf(R)
    m = v > 0
    log_v = np.log10(v[m])
    log_err = np.sqrt(c[m]) / (c[m] * np.log(10))
    ax.fill_between(FORARUN_BIN_CENTERS[m], log_v - log_err, log_v + log_err,
                    color=COLORS[frac], alpha=0.15)
    ax.plot(FORARUN_BIN_CENTERS[m], log_v, color=COLORS[frac], lw=2.2,
            label=f'{LABELS[frac]} (my work)')

    # Simone
    fa = load_simone(frac)
    log_v_fa = np.log10(fa['vsf'].values)
    ax.plot(fa['r'].values, log_v_fa, 'o', color=COLORS[frac],
            mec='black', mew=0.5, ms=6, label=f'{LABELS[frac]} (target)')

ax.set_xscale('log')
ax.set_xlim(1.5, 16)
ax.set_ylim(-7.0, -2.5)
ax.set_xlabel(r'$R_v\ [h^{-1}\mathrm{Mpc}]$', fontsize=15)
ax.set_ylabel(r'$\log_{10}(\mathrm{d}n/\mathrm{d}\ln R_v)\ [h^3\mathrm{Mpc}^{-3}]$', fontsize=15)
ax.set_title('NREC50: my work vs target data', fontsize=14)
ax.grid(True, which='both', ls='--', alpha=0.4)
ax.legend(fontsize=10, loc='lower left', ncol=2)
ax.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
out2 = os.path.join(OUT, 'thesis_log10_overlay_072.png')
fig.savefig(out2, dpi=200, bbox_inches='tight')
print(f'Saved: {out2}')
