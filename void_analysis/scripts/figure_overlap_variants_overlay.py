#!/usr/bin/env python3
"""
figure_overlap_variants_overlay.py

Experiment B — overlap-rule variants on 5% NREC50, scatter + best-fit
overlay. Uses _vsf_plot_utils for consistent style with the ratio-sweep
and v4-noMULT plots.

Series shown:
  - Vdn theory at Aletheia truth                            (black dashed)
  - Simone reference 5% NREC50 (CBL-cleaned by Simone)      (black ●)
  - v4 cleaning, overlap on R_resc, with MULT (production)  (blue ■)
  - v4 cleaning, overlap on watershed R, with MULT          (orange ◆)
  - CBL native, ol_crit = central, ratio = 1.5              (green ▲)
  - CBL native, ol_crit = contrast, ratio = 1.5             (red ▼)

Two styles selectable via --style:
  fitR7 : scatter + deg-2 log-log polynomial fit restricted to R ≥ 7
          (the MCMC fit range, no extrapolation into the exclusion zone).
  nofit : pure scatter; only the Vdn-truth dashed line is a smooth curve.

Outputs:
  plots/diagnostics/cbl_cleaning_overlap_variants_5p_fitR7.png
  plots/diagnostics/cbl_cleaning_overlap_variants_5p_nofit.png
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CBL_HOME = os.environ.get('CBL_HOME', '/Users/arun/projects/CosmoBolognaLib')
sys.path.insert(0, os.path.join(CBL_HOME, 'Python'))
os.environ.setdefault('DYLD_LIBRARY_PATH', CBL_HOME)
import CosmoBolognaLib as cbl

from bin_v4_cleaned_to_vsf import FORARUN_BIN_CENTERS, EDGES, DLNR, V_EFF
from _vsf_plot_utils import (
    UNIFIED_RCPARAMS, XLIM, YLIM, R_MIN,
    SIMONE_COLOR, VDN_TRUTH_COLOR, SIMONE_MARKER, SIMONE_MS,
    fit_loglog_poly, plot_scatter_with_fit,
    draw_exclusion_zone, apply_axis_style,
)

DATA_DIR = os.path.join(HERE, '..', 'data')
SIMONE_REF   = os.path.join(DATA_DIR, 'ForArun',
    'vsf_dm_250cube_ss0p05_NREC50_SM0p00_CS0p79_TH-0p80_OV00.dat')
V4_RESC      = os.path.join(DATA_DIR, 'cleaned',
    'cleaned_v4_recon_cat_5p.csv')
V4_WS        = os.path.join(DATA_DIR, 'cleaned',
    'cleaned_v4_overlap_ws_recon_cat_5p.csv')
CBL_CENTRAL  = os.path.join(DATA_DIR, 'cleaned_cbl',
    'cleaned_cbl_nrec50_cat_5p.csv')
CBL_CONTRAST = os.path.join(DATA_DIR, 'cleaned_cbl',
    'cleaned_cbl_nrec50_cat_5p_olcontrast.csv')
OUT_DIR = os.path.join(HERE, '..', 'plots', 'diagnostics')
os.makedirs(OUT_DIR, exist_ok=True)

OMEGA_M_TRUE, SIGMA8_TRUE = 0.315, 0.811
H0, OMEGA_B, N_S, TAU = 0.6736, 0.0493, 0.9649, 0.0925
SCALAR_AMP, SCALAR_PIVOT = 2.194e-9, 0.05
V4_MULT_5P = 1.43


def vdn_truth_curve(R_grid):
    c = cbl.cosmology.LCDM(OMEGA_M_TRUE, OMEGA_B, 0.0, H0,
                           SCALAR_AMP, SCALAR_PIVOT, N_S, TAU, True)
    c.set_parameter("sigma8", SIGMA8_TRUE)
    sf = cbl.cosmology.SizeFunction(c)
    dc = c.deltac(0.0)
    return np.array([sf.size_function(float(R), 0.0, "Vdn",
                     1.0, 0.0, 1.0, -0.7, dc, "EisensteinHu")
                     for R in R_grid])


def bin_vsf(radii):
    counts, _ = np.histogram(radii, bins=EDGES)
    return counts, counts / (V_EFF * DLNR), np.sqrt(counts) / (V_EFF * DLNR)


def median_ratio(vsf, sim_by_r):
    rs = []
    for r, v in zip(FORARUN_BIN_CENTERS, vsf):
        if r < R_MIN: continue
        vs = sim_by_r.get(round(r, 3))
        if vs and vs > 0 and v > 0:
            rs.append(v / vs)
    return float(np.median(rs)) if rs else float('nan')


# ─── Load Simone ─────────────────────────────────────────────────────
sim = pd.read_csv(SIMONE_REF, sep=r'\s+', comment='#', names=['r','vsf','err'])
sim = sim[sim['vsf'] > 0].reset_index(drop=True)
sim_by_r = dict(zip(sim['r'].round(3), sim['vsf']))

# ─── Load variants ───────────────────────────────────────────────────
variants = []

# 1. v4 R_resc-overlap (production, ×MULT 1.43)
df = pd.read_csv(V4_RESC)
counts, vsf, err = bin_vsf(df['R_resc'].values * V4_MULT_5P)
variants.append(dict(
    label='v4 cleaning, overlap on $R_\\mathrm{resc}$ '
          f'(×MULT {V4_MULT_5P:.2f}, production)',
    color='#2980b9', marker='s',
    counts=counts, vsf=vsf, err=err,
    n_total=len(df), n_R7=int(((df['R_resc'] * V4_MULT_5P) >= R_MIN).sum()),
    med_ratio=median_ratio(vsf, sim_by_r),
))

# 2. v4 watershed-overlap (×MULT 1.43)
df = pd.read_csv(V4_WS)
counts, vsf, err = bin_vsf(df['R_resc'].values * V4_MULT_5P)
variants.append(dict(
    label='v4 cleaning, overlap on watershed $R$ '
          f'(×MULT {V4_MULT_5P:.2f})',
    color='#e67e22', marker='D',
    counts=counts, vsf=vsf, err=err,
    n_total=len(df), n_R7=int(((df['R_resc'] * V4_MULT_5P) >= R_MIN).sum()),
    med_ratio=median_ratio(vsf, sim_by_r),
))

# 3. CBL native, central
df = pd.read_csv(CBL_CENTRAL)
counts, vsf, err = bin_vsf(df['R'].values)
variants.append(dict(
    label='CBL native, ol_crit = CentralDensity (ratio = 1.5)',
    color='#27ae60', marker='^',
    counts=counts, vsf=vsf, err=err,
    n_total=len(df), n_R7=int((df['R'] >= R_MIN).sum()),
    med_ratio=median_ratio(vsf, sim_by_r),
))

# 4. CBL native, contrast
df = pd.read_csv(CBL_CONTRAST)
counts, vsf, err = bin_vsf(df['R'].values)
variants.append(dict(
    label='CBL native, ol_crit = DensityContrast (ratio = 1.5)',
    color='#c0392b', marker='v',
    counts=counts, vsf=vsf, err=err,
    n_total=len(df), n_R7=int((df['R'] >= R_MIN).sum()),
    med_ratio=median_ratio(vsf, sim_by_r),
))

# ─── CLI ─────────────────────────────────────────────────────────────
_ap = argparse.ArgumentParser()
_ap.add_argument('--style', default='fitR7', choices=['fitR7', 'nofit'],
                 help='fitR7: deg-2 log-log poly fit restricted to R≥7. '
                      'nofit: pure scatter, no fit curve.')
_args = _ap.parse_args()
USE_FITS = (_args.style == 'fitR7')

# ─── Plot ────────────────────────────────────────────────────────────
plt.rcParams.update(UNIFIED_RCPARAMS)

fig, ax = plt.subplots(figsize=(12, 8))
draw_exclusion_zone(ax)

# Vdn theory at truth (always shown)
R_grid = np.logspace(np.log10(XLIM[0]), np.log10(XLIM[1]), 200)
ax.plot(R_grid, vdn_truth_curve(R_grid),
        ls='--', color=VDN_TRUTH_COLOR, lw=2.2, zorder=3,
        label=(f'Vdn theory at truth '
               f'($\\Omega_m={OMEGA_M_TRUE}, \\sigma_8={SIGMA8_TRUE}$)'))

# Simone reference — scatter; fit restricted to R≥7 if --style fitR7
sim_R   = sim['r'].values
sim_vsf = sim['vsf'].values
sim_fit = (fit_loglog_poly(sim_R, sim_vsf, degree=2,
                           fit_mask=(sim_R >= R_MIN))
           if USE_FITS else None)
plot_scatter_with_fit(
    ax, sim_R, sim_vsf, sim['err'].values,
    fit_fn=sim_fit,
    color=SIMONE_COLOR, marker=SIMONE_MARKER, ms=SIMONE_MS + 1,
    fit_ls='-', fit_lw=2.2, fit_zorder=10, scatter_zorder=11,
    label=f'Simone reference 5% NREC50 (CBL-cleaned, production)',
)

# Variants — scatter; fit restricted to R≥7 if --style fitR7
for v in variants:
    fit = (fit_loglog_poly(FORARUN_BIN_CENTERS, v['vsf'], degree=2,
                           fit_mask=(FORARUN_BIN_CENTERS >= R_MIN))
           if USE_FITS else None)
    plot_scatter_with_fit(
        ax, FORARUN_BIN_CENTERS, v['vsf'], v['err'],
        fit_fn=fit,
        color=v['color'], marker=v['marker'], ms=7,
        fit_ls=':', fit_lw=1.5,
        label=(f"{v['label']}   "
               f"(N={v['n_total']}, $N_{{R\\geq 7}}$={v['n_R7']}, "
               f"$v/v_\\mathrm{{sim}}$ = {v['med_ratio']:.2f})"),
    )

style_caption = (
    'Smoothing lines: deg-2 log-log polynomial fit in $R \\geq 7$ h$^{-1}$Mpc only '
    '(MCMC fit range), not a physical model fit.'
    if USE_FITS else
    'Pure scatter (no per-variant fits). Only smooth curve is the Vdn-truth dashed line.'
)

apply_axis_style(
    ax,
    title=(
        'Overlap-rule variants — 5% NREC50\n'
        f'{style_caption}'
    ),
)

out = os.path.join(OUT_DIR,
                   f'cbl_cleaning_overlap_variants_5p_{_args.style}.png')
fig.savefig(out, dpi=220, bbox_inches='tight')
print(f"Saved: {out}")
plt.close(fig)

# Summary
print(f"\n{'='*92}")
print("OVERLAP-VARIANTS SUMMARY  (5% NREC50,  Simone reference)")
print('='*92)
print(f"{'variant':<58}{'N_tot':>8}{'N(R≥7)':>10}{'median v/v_sim':>16}")
print('-' * 92)
for v in variants:
    short = v['label'].replace('$R_\\mathrm{resc}$', 'R_resc')\
                      .replace('$R$', 'R')\
                      .replace('\\mathrm{', '{')
    print(f"{short[:58]:<58}{v['n_total']:>8}{v['n_R7']:>10}{v['med_ratio']:>16.3f}")
