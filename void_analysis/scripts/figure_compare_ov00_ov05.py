#!/usr/bin/env python3
"""
figure_compare_ov00_ov05.py

Compare our simple-cleaned TOUCH and CENTER against BOTH of Simone's
overlap variants: OV00 (every overlap rejected = strong) and OV05
(intermediate — some overlap allowed).

The hypothesis we're testing:
  - TOUCH  ↔  OV00   (already confirmed ~3% match in R≥7)
  - CENTER ↔  OV05   (to be tested)

Output:
  plots/diagnostics/touch_center_vs_ov00_ov05_5p.png
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from bin_v4_cleaned_to_vsf import FORARUN_BIN_CENTERS, EDGES, DLNR, V_EFF
from _vsf_plot_utils import (
    UNIFIED_RCPARAMS, XLIM, YLIM, R_MIN,
    SIMONE_COLOR, SIMONE_MARKER, SIMONE_MS,
)

DATA = os.path.join(HERE, '..', 'data')
OUT_DIR = os.path.join(HERE, '..', 'plots', 'diagnostics')

REFS = {
    'Simone OV00 (strong)':  os.path.join(DATA, 'ForArun',
        'vsf_dm_250cube_ss0p05_NREC50_SM0p00_CS0p79_TH-0p80_OV00.dat'),
    'Simone OV05 (weaker)':  os.path.join(DATA, 'ForArun',
        'vsf_dm_250cube_ss0p05_NREC50_SM0p00_CS0p79_TH-0p80_OV05.dat'),
}
VARIANTS = {
    'TOUCH δ=−0.80':   os.path.join(DATA, 'cleaned_simple0',
        'cleaned_simple0_nrec50_cat_5p_touch.csv'),
    'CENTER δ=−0.80':  os.path.join(DATA, 'cleaned_simple0',
        'cleaned_simple0_nrec50_cat_5p_center.csv'),
}

def bin_vsf(radii):
    counts, _ = np.histogram(radii, bins=EDGES)
    return counts / (V_EFF * DLNR), np.sqrt(counts) / (V_EFF * DLNR)

def med_ratio(vsf, sim_by_r, r_cut):
    rs = []
    for r, v in zip(FORARUN_BIN_CENTERS, vsf):
        if r < r_cut: continue
        vs = sim_by_r.get(round(r, 3))
        if vs and vs > 0 and v > 0:
            rs.append(v / vs)
    return float(np.median(rs)) if rs else float('nan')

# Load refs
refs = {}
for name, path in REFS.items():
    df = pd.read_csv(path, sep=r'\s+', comment='#', names=['r','vsf','err'])
    df = df[df['vsf'] > 0].reset_index(drop=True)
    refs[name] = dict(df=df, by_r=dict(zip(df['r'].round(3), df['vsf'])))

# Bin our variants
ours = {}
for name, path in VARIANTS.items():
    df = pd.read_csv(path)
    radii = df['R_resc'].values
    vsf, err = bin_vsf(radii)
    ours[name] = dict(vsf=vsf, err=err, n=len(df))

# Print ratios — each of ours vs each ref
print(f"\n{'='*80}")
print('CROSS-COMPARISON  (5% NREC50)')
print('='*80)
print(f"{'':<22}{'vs OV00 (R≥7)':>16}{'vs OV05 (R≥7)':>16}"
      f"{'vs OV00 (all)':>16}{'vs OV05 (all)':>16}")
print('-'*80)
for vname, v in ours.items():
    ov00 = med_ratio(v['vsf'], refs['Simone OV00 (strong)']['by_r'], R_MIN)
    ov05 = med_ratio(v['vsf'], refs['Simone OV05 (weaker)']['by_r'], R_MIN)
    ov00_all = med_ratio(v['vsf'], refs['Simone OV00 (strong)']['by_r'], 0.0)
    ov05_all = med_ratio(v['vsf'], refs['Simone OV05 (weaker)']['by_r'], 0.0)
    print(f"{vname:<22}{ov00:>16.3f}{ov05:>16.3f}{ov00_all:>16.3f}{ov05_all:>16.3f}")

# ─── Plot ────────────────────────────────────────────────────────
plt.rcParams.update(UNIFIED_RCPARAMS)
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
fig.subplots_adjust(top=0.86, wspace=0.05)
fig.suptitle('TOUCH ↔ OV00  vs  CENTER ↔ OV05   (5% NREC50)\n'
             "Testing whether our simple-cleaning variants map to Simone's "
             "two overlap settings",
             fontsize=13, fontweight='bold', y=0.97)

# LEFT: TOUCH vs OV00
ax_left.errorbar(refs['Simone OV00 (strong)']['df']['r'],
                 refs['Simone OV00 (strong)']['df']['vsf'],
                 yerr=refs['Simone OV00 (strong)']['df']['err'],
                 fmt='o', ls='none', ms=8, color=SIMONE_COLOR,
                 mec='k', mew=0.4, alpha=0.85, zorder=10,
                 label='Simone OV00 (strong)')
ax_left.errorbar(FORARUN_BIN_CENTERS, ours['TOUCH δ=−0.80']['vsf'],
                 yerr=ours['TOUCH δ=−0.80']['err'],
                 fmt='v', ls='none', ms=8, color='#c0392b',
                 mec='k', mew=0.4, zorder=5,
                 label=f"TOUCH δ=−0.80  (N={ours['TOUCH δ=−0.80']['n']})")
ax_left.set_xscale('log'); ax_left.set_yscale('log')
ax_left.set_xlim(XLIM); ax_left.set_ylim(YLIM)
ax_left.set_xlabel(r'$R_v$  [$h^{-1}$Mpc]')
ax_left.set_ylabel(r'$dn/d\ln R_v$  [$h^3$Mpc$^{-3}$]')
ax_left.set_title('TOUCH vs Simone OV00 (strong, every overlap rejected)',
                  fontsize=11)
ax_left.grid(True, ls='--', alpha=0.2, which='both')
ax_left.legend(loc='lower left', fontsize=10, framealpha=0.95)

# RIGHT: CENTER vs OV05
ax_right.errorbar(refs['Simone OV05 (weaker)']['df']['r'],
                  refs['Simone OV05 (weaker)']['df']['vsf'],
                  yerr=refs['Simone OV05 (weaker)']['df']['err'],
                  fmt='o', ls='none', ms=8, color=SIMONE_COLOR,
                  mec='k', mew=0.4, alpha=0.85, zorder=10,
                  label='Simone OV05 (weaker)')
ax_right.errorbar(FORARUN_BIN_CENTERS, ours['CENTER δ=−0.80']['vsf'],
                  yerr=ours['CENTER δ=−0.80']['err'],
                  fmt='^', ls='none', ms=8, color='#2980b9',
                  mec='k', mew=0.4, zorder=5,
                  label=f"CENTER δ=−0.80  (N={ours['CENTER δ=−0.80']['n']})")
ax_right.set_xscale('log'); ax_right.set_yscale('log')
ax_right.set_xlim(XLIM); ax_right.set_ylim(YLIM)
ax_right.set_xlabel(r'$R_v$  [$h^{-1}$Mpc]')
ax_right.set_title('CENTER vs Simone OV05 (weaker, some overlap allowed)',
                   fontsize=11)
ax_right.grid(True, ls='--', alpha=0.2, which='both')
ax_right.legend(loc='lower left', fontsize=10, framealpha=0.95)

out = os.path.join(OUT_DIR, 'touch_center_vs_ov00_ov05_5p.png')
fig.savefig(out, dpi=220, bbox_inches='tight')
print(f"\nSaved: {out}")
plt.close(fig)
