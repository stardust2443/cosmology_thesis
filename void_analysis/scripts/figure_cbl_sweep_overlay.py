#!/usr/bin/env python3
"""
figure_cbl_sweep_overlay.py

Single-panel overlay plot for the CBL `clean_void_catalogue` ratio sweep
on 5% NREC50.

Two styles selectable via --style:
  fitR7 : scatter + deg-2 log-log polynomial fit restricted to R ≥ 7
          h⁻¹Mpc (the MCMC fit range). No extrapolation into the
          exclusion zone.
  nofit : pure scatter (markers + errorbars); only the Vdn-truth dashed
          line is a smooth curve.

Reads:
  - data/ForArun/vsf_dm_250cube_ss0p05_NREC50_*.dat        (Simone reference)
  - data/cleaned_cbl/sweep/cleaned_cbl_nrec50_cat_5p_r*.csv (sweep points)

Outputs:
  plots/diagnostics/cbl_cleaning_ratio_sweep_5p_fitR7.png
  plots/diagnostics/cbl_cleaning_ratio_sweep_5p_nofit.png

CLI:
  python scripts/figure_cbl_sweep_overlay.py --style fitR7
  python scripts/figure_cbl_sweep_overlay.py --style nofit
"""
import os, sys, glob, re, argparse
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
    VARIANT_MARKERS, VARIANT_COLORS,
    fit_loglog_poly, plot_scatter_with_fit,
    draw_exclusion_zone, apply_axis_style,
)

SWEEP   = os.path.join(HERE, '..', 'data', 'cleaned_cbl', 'sweep')
SIMONE_REF = os.path.join(HERE, '..', 'data', 'ForArun',
    'vsf_dm_250cube_ss0p05_NREC50_SM0p00_CS0p79_TH-0p80_OV00.dat')
OUT_DIR = os.path.join(HERE, '..', 'plots', 'diagnostics')
os.makedirs(OUT_DIR, exist_ok=True)

OMEGA_M_TRUE, SIGMA8_TRUE = 0.315, 0.811
H0, OMEGA_B, N_S, TAU = 0.6736, 0.0493, 0.9649, 0.0925
SCALAR_AMP, SCALAR_PIVOT = 2.194e-9, 0.05


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


def parse_ratio(fname):
    m = re.search(r'_r(\d+p\d+)\.csv$', fname)
    return float(m.group(1).replace('p', '.')) if m else None


# ─── Collect sweep inputs ────────────────────────────────────────────
# Two output dirs: the explicit sweep dir (with _r0p5, _r0p79 tags), and
# the parent cleaned_cbl dir (which holds the ratio=1.5 default run with
# no tag).
sweep_files = []
for f in glob.glob(os.path.join(SWEEP, 'cleaned_cbl_nrec50_cat_5p_r*.csv')):
    r = parse_ratio(f)
    if r is not None:
        sweep_files.append((r, f))
default_15 = os.path.join(HERE, '..', 'data', 'cleaned_cbl',
                          'cleaned_cbl_nrec50_cat_5p.csv')
if os.path.exists(default_15):
    sweep_files.append((1.5, default_15))
sweep_files = sorted(set(sweep_files), key=lambda x: x[0])
print(f"Found {len(sweep_files)} sweep outputs:")
for r, p in sweep_files:
    print(f"  ratio = {r:.2f}  ← {os.path.relpath(p, HERE)}")

# ─── Load Simone ─────────────────────────────────────────────────────
sim = pd.read_csv(SIMONE_REF, sep=r'\s+', comment='#', names=['r','vsf','err'])
sim = sim[sim['vsf'] > 0].reset_index(drop=True)
sim_by_r = dict(zip(sim['r'].round(3), sim['vsf']))

# ─── Bin each sweep output and compute the ratio v_cbl/v_simone ──────
variants = []
for r, path in sweep_files:
    df = pd.read_csv(path)
    counts, vsf, err = bin_vsf(df['R'].values)
    fit_ratios = []
    for rb, c, v in zip(FORARUN_BIN_CENTERS, counts, vsf):
        if rb < R_MIN: continue
        vs = sim_by_r.get(round(rb, 3))
        if vs and vs > 0 and v > 0:
            fit_ratios.append(v / vs)
    med = float(np.median(fit_ratios)) if fit_ratios else float('nan')
    variants.append(dict(
        ratio=r, n_total=len(df),
        n_R7=int((df['R'] >= R_MIN).sum()),
        counts=counts, vsf=vsf, err=err,
        med_ratio=med,
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

# Vdn theory at truth (always shown; it's theory, not a fit)
R_grid = np.logspace(np.log10(XLIM[0]), np.log10(XLIM[1]), 200)
ax.plot(R_grid, vdn_truth_curve(R_grid),
        ls='--', color=VDN_TRUTH_COLOR, lw=2.2, zorder=3,
        label=(f'Vdn theory at Aletheia truth '
               f'($\\Omega_m={OMEGA_M_TRUE}, \\sigma_8={SIGMA8_TRUE}$)'))

# Simone reference — scatter; fit only if --style fitR7 and restricted to R≥7
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
    label=f'Simone reference 5% NREC50 ({len(sim)} bins)',
)

# Sweep variants — scatter; fit (R≥7 only) if --style fitR7
for i, v in enumerate(variants):
    color  = VARIANT_COLORS[i % len(VARIANT_COLORS)]
    marker = VARIANT_MARKERS[i % len(VARIANT_MARKERS)]
    fit = (fit_loglog_poly(FORARUN_BIN_CENTERS, v['vsf'], degree=2,
                           fit_mask=(FORARUN_BIN_CENTERS >= R_MIN))
           if USE_FITS else None)
    plot_scatter_with_fit(
        ax, FORARUN_BIN_CENTERS, v['vsf'], v['err'],
        fit_fn=fit,
        color=color, marker=marker, ms=7,
        fit_ls=':', fit_lw=1.5,
        label=(f'CBL clean, ratio = {v["ratio"]:.2f}   '
               f'(N={v["n_total"]}, $N_{{R\\geq 7}}$={v["n_R7"]}, '
               f'$v/v_\\mathrm{{sim}}$ = {v["med_ratio"]:.2f})'),
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
        'CBL `clean_void_catalogue` ratio sweep — 5% NREC50\n'
        f'threshold = 0.20, ol_crit = central, radius_col = r_ext\n'
        f'{style_caption}'
    ),
)

out = os.path.join(OUT_DIR, f'cbl_cleaning_ratio_sweep_5p_{_args.style}.png')
fig.savefig(out, dpi=220, bbox_inches='tight')
print(f"Saved: {out}")
plt.close(fig)

# Summary
print(f"\n{'='*70}")
print("RATIO SWEEP SUMMARY  (5% NREC50,  Simone reference)")
print('='*70)
print(f"{'ratio':<8}{'N_tot':>8}{'N(R≥7)':>10}{'median v_cbl/v_simone':>26}")
print('-' * 52)
for v in variants:
    print(f"{v['ratio']:<8.2f}{v['n_total']:>8}{v['n_R7']:>10}{v['med_ratio']:>26.3f}")
