#!/usr/bin/env python3
"""
figure_touch_ov00_vdn_bestfit.py

Best-fit comparison: TOUCH vs Simone OV00 vs Vdn(δ_v=−0.80).

For each dataset (TOUCH, OV00), fits Vdn-with-free-δ_v at R≥7 by
1-D χ² minimisation, then overlays:
  - The data (scatter)
  - The best-fit Vdn curve for each dataset
  - The fiducial Vdn at δ_v = −0.80 (reference)
  - Per-bin residuals against each dataset's own best-fit

Also produces a χ²(δ_v) scan plot for both datasets.

Output:
  plots/diagnostics/bestfit_touch_ov00_vdn_5p.png
  plots/diagnostics/bestfit_deltav_scan_5p.png
"""
import os, sys
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
    SIMONE_COLOR, SIMONE_MARKER, SIMONE_MS, VDN_TRUTH_COLOR,
    fit_loglog_poly,
)

DATA = os.path.join(HERE, '..', 'data')
OUT_DIR = os.path.join(HERE, '..', 'plots', 'diagnostics')
os.makedirs(OUT_DIR, exist_ok=True)

TOUCH_CSV  = os.path.join(DATA, 'cleaned_simple0',
    'cleaned_simple0_nrec50_cat_5p_touch.csv')
OV00_REF   = os.path.join(DATA, 'ForArun',
    'vsf_dm_250cube_ss0p05_NREC50_SM0p00_CS0p79_TH-0p80_OV00.dat')

# Simone's prescription
BOX        = 250.0
BORDER_PAD = 20.0   # void center must be ≥ 20 h⁻¹Mpc from each face
                    # (V_EFF already = 210³ in bin_v4_cleaned_to_vsf)

# Aletheia truth cosmology
OMEGA_M_TRUE, SIGMA8_TRUE = 0.315, 0.811
H0, OMEGA_B, N_S, TAU = 0.6736, 0.0493, 0.9649, 0.0925
SCALAR_AMP, SCALAR_PIVOT = 2.194e-9, 0.05

DELTA_V_FIDUCIAL = -0.80           # matches cleaning threshold
DELTA_V_SCAN = np.linspace(-0.95, -0.55, 41)


# ─── CBL Vdn setup (done once) ─────────────────────────────────────
print("Setting up CBL Vdn evaluator…")
_cosmo = cbl.cosmology.LCDM(OMEGA_M_TRUE, OMEGA_B, 0.0, H0,
                            SCALAR_AMP, SCALAR_PIVOT, N_S, TAU, True)
_cosmo.set_parameter("sigma8", SIGMA8_TRUE)
_sf = cbl.cosmology.SizeFunction(_cosmo)
_dc = _cosmo.deltac(0.0)


def vdn_at(R_array, delta_v):
    """Vdn theory at one δ_v, evaluated at an array of R values."""
    return np.array([_sf.size_function(float(R), 0.0, "Vdn",
                     1.0, 0.0, 1.0, float(delta_v), _dc, "EisensteinHu")
                     for R in R_array])


def bin_vsf(radii):
    counts, _ = np.histogram(radii, bins=EDGES)
    return counts, counts / (V_EFF * DLNR), np.sqrt(counts) / (V_EFF * DLNR)


def apply_border_cut(df, pad=BORDER_PAD):
    """Keep voids whose CENTER is ≥ `pad` from every face."""
    x, y, z = df['x'].values, df['y'].values, df['z'].values
    lo = min(x.min(), y.min(), z.min())
    hi = lo + BOX
    keep = (
        (x - lo >= pad) & (hi - x >= pad) &
        (y - lo >= pad) & (hi - y >= pad) &
        (z - lo >= pad) & (hi - z >= pad)
    )
    return df[keep].reset_index(drop=True)


# ─── Load TOUCH ────────────────────────────────────────────────────
# The CSV is already produced by clean_voids_simple.py with the proper
# 20 h⁻¹Mpc center-based border cut applied at cleaning time. No need
# to re-apply it here (would be double-cutting).
print("Loading TOUCH catalogue…")
df_touch_cut = pd.read_csv(TOUCH_CSV)
touch_R = df_touch_cut['R_resc'].values
touch_counts, touch_vsf, touch_err = bin_vsf(touch_R)
print(f"  TOUCH N = {len(df_touch_cut)}  (border-cut already applied at cleaning)")

# ─── Load Simone OV00 ──────────────────────────────────────────────
print("Loading Simone OV00 reference…")
ov00_df = pd.read_csv(OV00_REF, sep=r'\s+', comment='#',
                      names=['r', 'vsf', 'err'])
ov00_df = ov00_df[ov00_df['vsf'] > 0].reset_index(drop=True)
# Align OV00 onto FORARUN bin centers (they're identical to within rounding)
ov00_R = ov00_df['r'].values
ov00_vsf = ov00_df['vsf'].values
ov00_err = ov00_df['err'].values


# ─── δ_v scan ──────────────────────────────────────────────────────
def chi2_at_deltav(data_R, data_vsf, data_err, delta_v, r_min=R_MIN):
    vdn = vdn_at(data_R, delta_v)
    ok = (data_vsf > 0) & (vdn > 0) & (data_err > 0) & (data_R >= r_min)
    if not ok.any():
        return np.inf, 0
    chi2 = float(np.sum(((data_vsf[ok] - vdn[ok]) / data_err[ok]) ** 2))
    return chi2, int(ok.sum())


def scan_deltav(data_R, data_vsf, data_err, deltavs, label):
    print(f"  scanning δ_v for {label} ({len(deltavs)} steps)…")
    chi2s = np.zeros(len(deltavs))
    ndof = 0
    for i, dv in enumerate(deltavs):
        c, n = chi2_at_deltav(data_R, data_vsf, data_err, dv)
        chi2s[i] = c
        ndof = n
    # Best δ_v with quadratic interpolation around the minimum
    imin = int(np.argmin(chi2s))
    if 0 < imin < len(deltavs) - 1:
        # Fit parabola through (dv[imin-1..imin+1], chi2[imin-1..imin+1])
        x0, x1, x2 = deltavs[imin-1], deltavs[imin], deltavs[imin+1]
        y0, y1, y2 = chi2s[imin-1], chi2s[imin], chi2s[imin+1]
        denom = (y0 - 2*y1 + y2)
        if abs(denom) > 1e-30:
            best_dv = x1 - 0.5 * (y2 - y0) / denom * (x1 - x0)
        else:
            best_dv = x1
    else:
        best_dv = deltavs[imin]
    best_chi2, _ = chi2_at_deltav(data_R, data_vsf, data_err, best_dv)
    return chi2s, ndof, float(best_dv), float(best_chi2)


print("\nFitting Vdn with free δ_v (R ≥ 7, MCMC range)…")
touch_chi2s, touch_ndof, touch_best_dv, touch_best_chi2 = scan_deltav(
    FORARUN_BIN_CENTERS, touch_vsf, touch_err, DELTA_V_SCAN, 'TOUCH')
ov00_chi2s, ov00_ndof, ov00_best_dv, ov00_best_chi2 = scan_deltav(
    ov00_R, ov00_vsf, ov00_err, DELTA_V_SCAN, 'OV00')

# Repeat for ALL R (no R_MIN cut)
print("\nFitting Vdn with free δ_v (ALL R, for comparison)…")
touch_chi2s_all, touch_ndof_all, touch_best_dv_all, touch_best_chi2_all = scan_deltav(
    FORARUN_BIN_CENTERS, touch_vsf, touch_err, DELTA_V_SCAN, 'TOUCH (all)')
# Override r_min=0 for the all-R version
def chi2_all(R, vsf, err, dv):
    return chi2_at_deltav(R, vsf, err, dv, r_min=0.0)
def scan_all(R, vsf, err, deltavs, label):
    print(f"  scanning δ_v for {label} (all R, {len(deltavs)} steps)…")
    chi2s = np.zeros(len(deltavs)); ndof = 0
    for i, dv in enumerate(deltavs):
        c, n = chi2_all(R, vsf, err, dv); chi2s[i] = c; ndof = n
    imin = int(np.argmin(chi2s))
    if 0 < imin < len(deltavs) - 1:
        x0, x1, x2 = deltavs[imin-1], deltavs[imin], deltavs[imin+1]
        y0, y1, y2 = chi2s[imin-1], chi2s[imin], chi2s[imin+1]
        denom = (y0 - 2*y1 + y2)
        best = x1 - 0.5 * (y2 - y0) / denom * (x1 - x0) if abs(denom) > 1e-30 else x1
    else:
        best = deltavs[imin]
    return chi2s, ndof, float(best), float(chi2_all(R, vsf, err, best)[0])

touch_chi2s_all, touch_ndof_all, touch_best_dv_all, touch_best_chi2_all = scan_all(
    FORARUN_BIN_CENTERS, touch_vsf, touch_err, DELTA_V_SCAN, 'TOUCH')
ov00_chi2s_all, ov00_ndof_all, ov00_best_dv_all, ov00_best_chi2_all = scan_all(
    ov00_R, ov00_vsf, ov00_err, DELTA_V_SCAN, 'OV00')

# χ² at fiducial δ_v=−0.80 for both
touch_chi2_fid, _ = chi2_at_deltav(FORARUN_BIN_CENTERS, touch_vsf, touch_err,
                                    DELTA_V_FIDUCIAL)
ov00_chi2_fid,  _ = chi2_at_deltav(ov00_R, ov00_vsf, ov00_err,
                                    DELTA_V_FIDUCIAL)
touch_chi2_fid_all, _ = chi2_all(FORARUN_BIN_CENTERS, touch_vsf, touch_err,
                                  DELTA_V_FIDUCIAL)
ov00_chi2_fid_all,  _ = chi2_all(ov00_R, ov00_vsf, ov00_err, DELTA_V_FIDUCIAL)

# ─── Polynomial trend lines (deg-3 log-log) ───────────────────────
# Fit BOTH at R≥7 (matching the Vdn fit range) AND on all R, for
# a fair side-by-side comparison.
print("\nFitting deg-3 log-log polynomial trend lines…")
poly_touch_R7  = fit_loglog_poly(FORARUN_BIN_CENTERS, touch_vsf, degree=3,
                                  fit_mask=(FORARUN_BIN_CENTERS >= R_MIN))
poly_touch_all = fit_loglog_poly(FORARUN_BIN_CENTERS, touch_vsf, degree=3)
poly_ov00_R7   = fit_loglog_poly(ov00_R, ov00_vsf, degree=3,
                                  fit_mask=(ov00_R >= R_MIN))
poly_ov00_all  = fit_loglog_poly(ov00_R, ov00_vsf, degree=3)

print(f"\n{'='*100}")
print('Vdn-with-free-δ_v fit results — R ≥ 7 (science range) vs ALL R')
print('='*100)
print(f"{'variant':<10}{'range':<10}{'dof':>5}"
      f"{'χ²(δ_v=−0.80)':>16}{'χ²_red':>10}"
      f"{'best δ_v':>12}{'χ²(best)':>12}{'χ²_red(best)':>15}")
print('-'*100)
for label, ndof_R7, ndof_all, c_fid, c_fid_all, best_R7, best_chi2_R7, \
    best_all, best_chi2_all in [
    ('TOUCH', touch_ndof, touch_ndof_all, touch_chi2_fid, touch_chi2_fid_all,
     touch_best_dv, touch_best_chi2, touch_best_dv_all, touch_best_chi2_all),
    ('OV00',  ov00_ndof,  ov00_ndof_all,  ov00_chi2_fid,  ov00_chi2_fid_all,
     ov00_best_dv,  ov00_best_chi2,  ov00_best_dv_all,  ov00_best_chi2_all),
]:
    print(f"{label:<10}{'R≥7':<10}{ndof_R7:>5}"
          f"{c_fid:>16.1f}{c_fid/max(ndof_R7-1,1):>10.2f}"
          f"{best_R7:>12.4f}{best_chi2_R7:>12.1f}"
          f"{best_chi2_R7/max(ndof_R7-1,1):>15.2f}")
    print(f"{'':<10}{'all R':<10}{ndof_all:>5}"
          f"{c_fid_all:>16.0f}{c_fid_all/max(ndof_all-1,1):>10.0f}"
          f"{best_all:>12.4f}{best_chi2_all:>12.0f}"
          f"{best_chi2_all/max(ndof_all-1,1):>15.0f}")

# ─── Best-fit curves on a smooth grid ──────────────────────────────
R_grid = np.logspace(np.log10(XLIM[0]), np.log10(XLIM[1]), 200)
vdn_fid_smooth   = vdn_at(R_grid, DELTA_V_FIDUCIAL)
vdn_touch_smooth = vdn_at(R_grid, touch_best_dv)
vdn_ov00_smooth  = vdn_at(R_grid, ov00_best_dv)

# Best-fit Vdn evaluated at the bin centers (for residuals)
touch_bestfit_at_bins = vdn_at(FORARUN_BIN_CENTERS, touch_best_dv)
ov00_bestfit_at_bins  = vdn_at(ov00_R, ov00_best_dv)


# ═══════════════════════════════════════════════════════════════════
# FIGURE A: scatter + best-fit curves + residual sub-panel
# ═══════════════════════════════════════════════════════════════════
plt.rcParams.update(UNIFIED_RCPARAMS)
fig, (ax_top, ax_res) = plt.subplots(
    2, 1, figsize=(13, 10),
    sharex=True, gridspec_kw={'height_ratios': [2.4, 1], 'hspace': 0.06})
fig.subplots_adjust(top=0.92)
fig.suptitle('TOUCH vs Simone OV00 vs Vdn: ss5%, 50 OTs\n'
             'Best-fit Vdn (free δ_v) at R ≥ 7,  V$_{eff}$ = 210³ h$^{-3}$Mpc³',
             fontsize=13, fontweight='bold', y=0.99)

# ── Top: data + best-fit Vdn curves + fiducial Vdn ──
TOUCH_COLOR = '#c0392b'
ax_top.plot(R_grid, vdn_fid_smooth, ls='--', color='#888', lw=2, zorder=3,
            label=f'Vdn fiducial  ($\\delta_v = {DELTA_V_FIDUCIAL:.2f}$)')

# TOUCH scatter + its best-fit Vdn
ok_t = touch_vsf > 0
ax_top.errorbar(FORARUN_BIN_CENTERS[ok_t], touch_vsf[ok_t],
                yerr=touch_err[ok_t],
                fmt='v', ls='none', ms=8, color=TOUCH_COLOR,
                ecolor=TOUCH_COLOR, elinewidth=1.2, capsize=2.5,
                mec='k', mew=0.4, zorder=5,
                label=f'TOUCH δ=−0.80  (N={len(df_touch_cut)})')
ax_top.plot(R_grid, vdn_touch_smooth, ls=':', color=TOUCH_COLOR, lw=2,
            zorder=4,
            label=f'Vdn best-fit (TOUCH)  $\\delta_v = {touch_best_dv:.3f}$')

# OV00 scatter + its best-fit Vdn
ax_top.errorbar(ov00_R, ov00_vsf, yerr=ov00_err,
                fmt=SIMONE_MARKER, ls='none', ms=SIMONE_MS,
                color=SIMONE_COLOR, ecolor=SIMONE_COLOR,
                elinewidth=1.2, capsize=2.5, mec='k', mew=0.4,
                alpha=0.85, zorder=10,
                label='Simone OV00 (strong)')
ax_top.plot(R_grid, vdn_ov00_smooth, ls=':', color=SIMONE_COLOR, lw=2,
            zorder=9,
            label=f'Vdn best-fit (OV00)  $\\delta_v = {ov00_best_dv:.3f}$')

# Polynomial trend lines (deg-3 log-log, R≥7 fit, drawn only in R≥7)
R_grid_R7 = np.logspace(np.log10(R_MIN), np.log10(XLIM[1]), 200)
if poly_touch_R7 is not None:
    ax_top.plot(R_grid_R7, poly_touch_R7(R_grid_R7),
                ls='-.', color=TOUCH_COLOR, lw=1.6, alpha=0.85, zorder=6,
                label='poly trend (TOUCH, deg-3, R≥7)')
if poly_ov00_R7 is not None:
    ax_top.plot(R_grid_R7, poly_ov00_R7(R_grid_R7),
                ls='-.', color=SIMONE_COLOR, lw=1.6, alpha=0.85, zorder=8,
                label='poly trend (OV00, deg-3, R≥7)')

ax_top.axvline(R_MIN, color='#aaa', ls='-', lw=1.2, zorder=1)
ax_top.set_xscale('log'); ax_top.set_yscale('log')
ax_top.set_xlim(XLIM); ax_top.set_ylim(YLIM)
ax_top.set_ylabel(r'$dn/d\ln R_v$  [$h^3$Mpc$^{-3}$]')
ax_top.grid(True, ls='--', alpha=0.25, which='both')
ax_top.legend(loc='lower left', fontsize=9.5, framealpha=0.95,
              edgecolor='#bbb')
ax_top.tick_params(labelbottom=False)

# ── Bottom: residuals (data − own best-fit) / σ ──
ax_res.axhspan(-1, 1, color='#27ae60', alpha=0.12, zorder=0)
ax_res.axhspan(-3, 3, color='#f39c12', alpha=0.07, zorder=0)
ax_res.axhline(0, color='k', lw=0.8, zorder=1)
ax_res.axvline(R_MIN, color='#aaa', ls='-', lw=1.2, zorder=1)

# TOUCH residuals
ok_tr = (touch_vsf > 0) & (touch_bestfit_at_bins > 0) & (touch_err > 0)
resid_t = np.full(len(FORARUN_BIN_CENTERS), np.nan)
resid_t[ok_tr] = ((touch_vsf[ok_tr] - touch_bestfit_at_bins[ok_tr])
                   / touch_err[ok_tr])
fit_t = FORARUN_BIN_CENTERS >= R_MIN
ax_res.scatter(FORARUN_BIN_CENTERS[ok_tr & ~fit_t],
               resid_t[ok_tr & ~fit_t],
               marker='v', s=35, color=TOUCH_COLOR, alpha=0.3,
               edgecolors='none', zorder=4)
ax_res.scatter(FORARUN_BIN_CENTERS[ok_tr & fit_t],
               resid_t[ok_tr & fit_t],
               marker='v', s=60, color=TOUCH_COLOR, alpha=0.9,
               edgecolors='k', linewidths=0.4, zorder=5, label='TOUCH')

# OV00 residuals
ok_or = (ov00_vsf > 0) & (ov00_bestfit_at_bins > 0) & (ov00_err > 0)
resid_o = np.full(len(ov00_R), np.nan)
resid_o[ok_or] = ((ov00_vsf[ok_or] - ov00_bestfit_at_bins[ok_or])
                   / ov00_err[ok_or])
fit_o = ov00_R >= R_MIN
ax_res.scatter(ov00_R[ok_or & ~fit_o], resid_o[ok_or & ~fit_o],
               marker=SIMONE_MARKER, s=30, color=SIMONE_COLOR, alpha=0.3,
               edgecolors='none', zorder=8)
ax_res.scatter(ov00_R[ok_or & fit_o], resid_o[ok_or & fit_o],
               marker=SIMONE_MARKER, s=55, color=SIMONE_COLOR, alpha=0.85,
               edgecolors='k', linewidths=0.4, zorder=9, label='OV00')

ax_res.set_xscale('log')
ax_res.set_xlim(XLIM); ax_res.set_ylim(-6, 6)
ax_res.set_xlabel(r'$R_v$  [$h^{-1}$Mpc]')
ax_res.set_ylabel(r'(data − best-fit) / $\sigma$')
ax_res.grid(True, ls='--', alpha=0.25, which='both')
ax_res.legend(loc='upper right', fontsize=9, framealpha=0.95)
ax_res.text(0.02, 0.05, 'Green band: ±1σ,  orange band: ±3σ',
            transform=ax_res.transAxes, fontsize=8, color='#666')

outA = os.path.join(OUT_DIR, 'bestfit_touch_ov00_vdn_5p.png')
fig.savefig(outA, dpi=220, bbox_inches='tight')
print(f"\nSaved: {outA}")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# FIGURE B: χ²(δ_v) scan curves
# ═══════════════════════════════════════════════════════════════════
fig, (ax_t, ax_o) = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
fig.subplots_adjust(top=0.86, wspace=0.20)
fig.suptitle('χ²(δ_v) scan: best-fit δ_v from Vdn fit at R ≥ 7\n'
             f'TOUCH best δ_v = {touch_best_dv:.3f}     '
             f'OV00 best δ_v = {ov00_best_dv:.3f}',
             fontsize=13, fontweight='bold', y=0.99)

for ax, chi2s, best_dv, label, color in [
    (ax_t, touch_chi2s, touch_best_dv, 'TOUCH',  TOUCH_COLOR),
    (ax_o, ov00_chi2s,  ov00_best_dv,  'OV00',   SIMONE_COLOR),
]:
    ax.plot(DELTA_V_SCAN, chi2s, '-o', ms=5, color=color, lw=1.5,
            mec='k', mew=0.3)
    ax.axvline(best_dv, color=color, ls=':', lw=1.8,
               label=f'best δ_v = {best_dv:.3f}')
    ax.axvline(DELTA_V_FIDUCIAL, color='#888', ls='--', lw=1.5,
               label=f'fiducial δ_v = {DELTA_V_FIDUCIAL:.2f}')
    ax.set_xlabel(r'$\delta_v$  (Vdn threshold)')
    ax.set_ylabel(r'$\chi^2_{R\geq 7}$')
    ax.set_yscale('log')
    ax.set_title(label, fontsize=11)
    ax.grid(True, ls='--', alpha=0.25, which='both')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)

outB = os.path.join(OUT_DIR, 'bestfit_deltav_scan_5p.png')
fig.savefig(outB, dpi=220, bbox_inches='tight')
print(f"Saved: {outB}")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# FIGURE C: all-R trend lines + comparison of R≥7 vs all-R fits
# Top panel: data + Vdn(δ_v=−0.80) + poly trends fitted on ALL R
# Bottom panel: same but poly trends fitted on R≥7 only
# Lets you see how dramatically the polynomial diverges if you
# include small-R bins.
# ═══════════════════════════════════════════════════════════════════
fig, (ax_all, ax_R7) = plt.subplots(2, 1, figsize=(13, 11),
                                     sharex=True, sharey=True,
                                     gridspec_kw={'hspace': 0.08})
fig.subplots_adjust(top=0.92)
fig.suptitle('Trend-line fits: ALL R  vs  R ≥ 7 only — 5% NREC50\n'
             'Solid Vdn (δ_v=−0.80 fiducial) shown for reference; '
             'poly trends are deg-3 log-log fits',
             fontsize=13, fontweight='bold', y=0.99)

R_grid_full = np.logspace(np.log10(XLIM[0]), np.log10(XLIM[1]), 200)
R_grid_R7   = np.logspace(np.log10(R_MIN),   np.log10(XLIM[1]), 200)

for ax, fit_label, poly_t, poly_o, dv_t, dv_o, grid in [
    (ax_all, 'ALL R',
     poly_touch_all, poly_ov00_all,
     touch_best_dv_all, ov00_best_dv_all, R_grid_full),
    (ax_R7,  'R ≥ 7 only (MCMC range)',
     poly_touch_R7,  poly_ov00_R7,
     touch_best_dv, ov00_best_dv, R_grid_R7),
]:
    # Fiducial Vdn
    ax.plot(R_grid_full, vdn_fid_smooth, ls='--', color='#888', lw=2,
            zorder=3, label=f'Vdn fiducial (δ_v=−0.80)')

    # TOUCH data + its best-fit Vdn from THIS range + poly trend
    ok_t = touch_vsf > 0
    ax.errorbar(FORARUN_BIN_CENTERS[ok_t], touch_vsf[ok_t],
                yerr=touch_err[ok_t],
                fmt='v', ls='none', ms=8, color=TOUCH_COLOR,
                ecolor=TOUCH_COLOR, elinewidth=1.2, capsize=2.5,
                mec='k', mew=0.4, zorder=5,
                label=f'TOUCH  (N={len(df_touch_cut)})')
    ax.plot(R_grid_full, vdn_at(R_grid_full, dv_t),
            ls=':', color=TOUCH_COLOR, lw=2, zorder=4,
            label=f'Vdn best-fit (TOUCH, {fit_label})  δ_v={dv_t:.3f}')
    if poly_t is not None:
        ax.plot(grid, poly_t(grid),
                ls='-.', color=TOUCH_COLOR, lw=1.6, alpha=0.85, zorder=6,
                label=f'poly trend (TOUCH, deg-3, {fit_label})')

    # OV00 data + its best-fit Vdn + poly trend
    ax.errorbar(ov00_R, ov00_vsf, yerr=ov00_err,
                fmt=SIMONE_MARKER, ls='none', ms=SIMONE_MS,
                color=SIMONE_COLOR, ecolor=SIMONE_COLOR,
                elinewidth=1.2, capsize=2.5, mec='k', mew=0.4,
                alpha=0.85, zorder=10, label='Simone OV00')
    ax.plot(R_grid_full, vdn_at(R_grid_full, dv_o),
            ls=':', color=SIMONE_COLOR, lw=2, zorder=9,
            label=f'Vdn best-fit (OV00, {fit_label})  δ_v={dv_o:.3f}')
    if poly_o is not None:
        ax.plot(grid, poly_o(grid),
                ls='-.', color=SIMONE_COLOR, lw=1.6, alpha=0.85, zorder=8,
                label=f'poly trend (OV00, deg-3, {fit_label})')

    ax.axvline(R_MIN, color='#aaa', ls='-', lw=1.2, zorder=1)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(XLIM); ax.set_ylim(YLIM)
    ax.set_ylabel(r'$dn/d\ln R_v$  [$h^3$Mpc$^{-3}$]')
    ax.grid(True, ls='--', alpha=0.25, which='both')
    ax.set_title(f'Trend-line + Vdn fits using {fit_label}',
                 fontsize=11, pad=6)
    ax.legend(loc='lower left', fontsize=8.5, framealpha=0.95,
              edgecolor='#bbb', ncol=2)

ax_R7.set_xlabel(r'$R_v$  [$h^{-1}$Mpc]')

outC = os.path.join(OUT_DIR, 'bestfit_trends_allR_vs_R7_5p.png')
fig.savefig(outC, dpi=220, bbox_inches='tight')
print(f"Saved: {outC}")
plt.close(fig)
