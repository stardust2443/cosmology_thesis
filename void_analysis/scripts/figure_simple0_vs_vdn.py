#!/usr/bin/env python3
"""
figure_simple0_vs_vdn.py

Two plots for the simple0 cleaning vs Vdn theory AND Simone reference:
  1. Log-log VSF overlay:  simple0 touch & center vs Vdn truth + Simone
  2. Chi-squared per bin:  χ² against Vdn (top) and ratio vs Simone (bottom)

Output:
  plots/diagnostics/simple0_vsf_vs_vdn_5p.png
  plots/diagnostics/simple0_chi2_vs_vdn_5p.png
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
    VDN_TRUTH_COLOR, SIMONE_COLOR, SIMONE_MARKER, SIMONE_MS,
    plot_scatter_with_fit, apply_axis_style,
)

DATA = os.path.join(HERE, '..', 'data', 'cleaned_simple0')
OUT_DIR = os.path.join(HERE, '..', 'plots', 'diagnostics')
os.makedirs(OUT_DIR, exist_ok=True)

SIMPLE_TOUCH  = os.path.join(DATA, 'cleaned_simple0_nrec50_cat_5p_touch.csv')
SIMPLE_CENTER = os.path.join(DATA, 'cleaned_simple0_nrec50_cat_5p_center.csv')
SIMONE_REF    = os.path.join(HERE, '..', 'data', 'ForArun',
    'vsf_dm_250cube_ss0p05_NREC50_SM0p00_CS0p79_TH-0p80_OV00.dat')

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
                     1.0, 0.0, 1.0, -0.8, dc, "EisensteinHu")
                     for R in R_grid])


def bin_vsf(radii):
    counts, _ = np.histogram(radii, bins=EDGES)
    vsf = counts / (V_EFF * DLNR)
    err = np.sqrt(counts) / (V_EFF * DLNR)
    return counts, vsf, err


def load_radii(path):
    df = pd.read_csv(path)
    col = 'R_resc' if 'R_resc' in df.columns else 'R'
    return df[col].values, len(df)


# ─── Load variants ──────────────────────────────────────────────────
variants = []
for path, label, color, marker in [
    (SIMPLE_CENTER, 'simple0, overlap = CENTER', '#16a085', 'D'),
    (SIMPLE_TOUCH,  'simple0, overlap = TOUCH',  '#d35400', 's'),
]:
    radii, n_total = load_radii(path)
    counts, vsf, err = bin_vsf(radii)
    variants.append(dict(
        label=label, color=color, marker=marker,
        vsf=vsf, err=err, counts=counts,
        n_total=n_total, n_R7=int((radii >= R_MIN).sum()),
    ))

# ─── Load Simone reference ──────────────────────────────────────────
sim = pd.read_csv(SIMONE_REF, sep=r'\s+', comment='#',
                  names=['r', 'vsf', 'err'])
sim = sim[sim['vsf'] > 0].reset_index(drop=True)
sim_by_r = dict(zip(sim['r'].round(3), sim['vsf']))
sim_err_by_r = dict(zip(sim['r'].round(3), sim['err']))

# ─── Vdn truth at bin centers ───────────────────────────────────────
vdn_at_bins = vdn_truth_curve(FORARUN_BIN_CENTERS)

# ─── PLOT 1: Log-log VSF overlay ────────────────────────────────────
plt.rcParams.update(UNIFIED_RCPARAMS)
fig, ax = plt.subplots(figsize=(12, 8))

# Vdn theory — smooth curve
R_grid = np.logspace(np.log10(XLIM[0]), np.log10(XLIM[1]), 200)
ax.plot(R_grid, vdn_truth_curve(R_grid),
        ls='--', color=VDN_TRUTH_COLOR, lw=2.2, zorder=3,
        label=(f'Vdn theory at truth '
               f'($\\Omega_m={OMEGA_M_TRUE}, \\sigma_8={SIGMA8_TRUE}$)'))

# Simone reference — scatter
plot_scatter_with_fit(
    ax, sim['r'].values, sim['vsf'].values, sim['err'].values,
    fit_fn=None,
    color=SIMONE_COLOR, marker=SIMONE_MARKER, ms=SIMONE_MS + 1,
    scatter_zorder=11,
    label='Simone reference 5% NREC50 (CBL-cleaned)',
)

# Each variant — pure scatter, no fit lines
for v in variants:
    plot_scatter_with_fit(
        ax, FORARUN_BIN_CENTERS, v['vsf'], v['err'],
        fit_fn=None,
        color=v['color'], marker=v['marker'], ms=8,
        label=(f"{v['label']}  "
               f"(N={v['n_total']}, $N_{{R\\geq 7}}$={v['n_R7']})"),
    )

apply_axis_style(
    ax,
    title=('simple0 cleaning vs Vdn theory & Simone — 5% NREC50\n'
           'Scatter only; dashed line = Vdn at Aletheia truth.'),
)

out1 = os.path.join(OUT_DIR, 'simple0_vsf_vs_vdn_5p.png')
fig.savefig(out1, dpi=220, bbox_inches='tight')
print(f"Saved: {out1}")
plt.close(fig)

# ─── PLOT 2: Chi-squared vs Vdn (top) + ratio vs Simone (bottom) ───
fig, (ax_chi, ax_rat) = plt.subplots(2, 1, figsize=(12, 10),
                                      sharex=True,
                                      gridspec_kw={'height_ratios': [1, 1],
                                                   'hspace': 0.12})

for v in variants:
    # ── Top panel: per-bin chi2 vs Vdn ──
    ok = (v['vsf'] > 0) & (vdn_at_bins > 0)
    chi2_per_bin = np.full(len(FORARUN_BIN_CENTERS), np.nan)
    chi2_per_bin[ok] = (v['vsf'][ok] - vdn_at_bins[ok]) ** 2 / v['err'][ok] ** 2

    fit_mask = FORARUN_BIN_CENTERS >= R_MIN
    n_dof = int((ok & fit_mask).sum())
    chi2_total = float(np.nansum(chi2_per_bin[fit_mask]))

    ax_chi.bar(FORARUN_BIN_CENTERS[ok], chi2_per_bin[ok],
               width=FORARUN_BIN_CENTERS[ok] * (10**DLNR - 1) * 0.4,
               color=v['color'], alpha=0.6, edgecolor=v['color'],
               label=(f"{v['label']}  "
                      f"($\\chi^2_{{R\\geq7}}$ = {chi2_total:.1f}, "
                      f"dof = {n_dof})"),
               zorder=3)

    # ── Bottom panel: ratio vs Simone ──
    ratio_r, ratio_v, ratio_err = [], [], []
    for j, r in enumerate(FORARUN_BIN_CENTERS):
        vs = sim_by_r.get(round(r, 3))
        ve = sim_err_by_r.get(round(r, 3))
        if vs and vs > 0 and v['vsf'][j] > 0:
            rat = v['vsf'][j] / vs
            # Propagate errors from both
            rel_err = np.sqrt((v['err'][j] / v['vsf'][j]) ** 2 +
                              (ve / vs) ** 2) if ve and ve > 0 else 0
            ratio_r.append(r)
            ratio_v.append(rat)
            ratio_err.append(rat * rel_err)
    ratio_r = np.array(ratio_r)
    ratio_v = np.array(ratio_v)
    ratio_err = np.array(ratio_err)

    in_fit = ratio_r >= R_MIN
    med_ratio = float(np.median(ratio_v[in_fit])) if in_fit.any() else float('nan')

    # Faded points below R_MIN, solid above
    if (~in_fit).any():
        ax_rat.errorbar(ratio_r[~in_fit], ratio_v[~in_fit],
                        yerr=ratio_err[~in_fit],
                        fmt=v['marker'], ls='none', ms=5,
                        color=v['color'], alpha=0.3,
                        ecolor=v['color'], elinewidth=0.8,
                        capsize=2, zorder=4)
    if in_fit.any():
        ax_rat.errorbar(ratio_r[in_fit], ratio_v[in_fit],
                        yerr=ratio_err[in_fit],
                        fmt=v['marker'], ls='none', ms=7,
                        color=v['color'],
                        ecolor=v['color'], elinewidth=1.2,
                        capsize=2.5, mec='k', mew=0.4, zorder=5,
                        label=(f"{v['label']}  "
                               f"(median $v/v_\\mathrm{{sim}}$ = "
                               f"{med_ratio:.2f})"))

# Simone's own chi2 vs Vdn
sim_vdn = vdn_truth_curve(sim['r'].values)
sim_ok = (sim['vsf'].values > 0) & (sim_vdn > 0)
sim_chi2 = np.full(len(sim), np.nan)
sim_chi2[sim_ok] = (sim['vsf'].values[sim_ok] - sim_vdn[sim_ok]) ** 2 / sim['err'].values[sim_ok] ** 2
sim_fit = sim['r'].values >= R_MIN
sim_chi2_total = float(np.nansum(sim_chi2[sim_ok & sim_fit]))
sim_ndof = int((sim_ok & sim_fit).sum())
ax_chi.bar(sim['r'].values[sim_ok], sim_chi2[sim_ok],
           width=sim['r'].values[sim_ok] * (10**DLNR - 1) * 0.4,
           color=SIMONE_COLOR, alpha=0.35, edgecolor=SIMONE_COLOR,
           label=(f"Simone reference  "
                  f"($\\chi^2_{{R\\geq7}}$ = {sim_chi2_total:.1f}, "
                  f"dof = {sim_ndof})"),
           zorder=2)

# Style — top panel (chi2)
ax_chi.axhline(1, color='k', ls=':', lw=1, alpha=0.5, zorder=1)
ax_chi.axvline(R_MIN, color='#aaa', ls='-', lw=1.2, zorder=1)
ax_chi.set_xscale('log')
ax_chi.set_ylabel(r'$\chi^2_i = (n_i^{\mathrm{data}} - n_i^{\mathrm{Vdn}})^2 \,/\, \sigma_i^2$')
ax_chi.set_title('Per-bin $\\chi^2$ vs Vdn + ratio vs Simone — simple0, 5% NREC50',
                 fontweight='bold', pad=10)
ax_chi.legend(fontsize=9, framealpha=0.95, edgecolor='#bbb')
ax_chi.grid(True, ls='--', alpha=0.25, which='both')
ax_chi.set_xlim(XLIM)
ax_chi.tick_params(labelbottom=False)

# Style — bottom panel (ratio vs Simone)
ax_rat.axhline(1, color='k', lw=0.8, zorder=2)
ax_rat.fill_between(list(XLIM), 0.8, 1.2,
                    color='#27ae60', alpha=0.10, zorder=0)
ax_rat.fill_between(list(XLIM), 0.5, 2.0,
                    color='#f39c12', alpha=0.07, zorder=0)
ax_rat.axvline(R_MIN, color='#aaa', ls='-', lw=1.2, zorder=1)
ax_rat.set_xscale('log')
ax_rat.set_yscale('log')
ax_rat.set_xlim(XLIM)
ax_rat.set_ylim(0.2, 5)
ax_rat.set_yticks([0.25, 0.5, 1.0, 2.0, 4.0])
ax_rat.set_yticklabels(['0.25', '0.5', '1', '2', '4'])
ax_rat.set_xlabel(r'$R_v$  [$h^{-1}$Mpc]')
ax_rat.set_ylabel(r'simple0 / Simone')
ax_rat.legend(fontsize=9, framealpha=0.95, edgecolor='#bbb',
              loc='upper left')
ax_rat.grid(True, ls='--', alpha=0.25, which='both')
ax_rat.text(0.97, 0.05, 'Green band: ±20%,  orange band: factor 2',
            transform=ax_rat.transAxes, ha='right', va='bottom',
            fontsize=8, color='#666')

out2 = os.path.join(OUT_DIR, 'simple0_chi2_vs_vdn_5p.png')
fig.savefig(out2, dpi=220, bbox_inches='tight')
print(f"Saved: {out2}")
plt.close(fig)

# ─── Summary table ──────────────────────────────────────────────────
print(f"\n{'='*72}")
print('SUMMARY  (simple0 vs Vdn truth & Simone, R >= 7 h⁻¹Mpc)')
print('='*72)
fit_mask = FORARUN_BIN_CENTERS >= R_MIN
# Simone vs Vdn
sim_fit_ok = sim_ok & sim_fit
sim_chi2_r = float(sim_chi2_total) / max(sim_ndof - 1, 1)
print(f"  {'Simone reference':<40}  χ²={sim_chi2_total:8.1f}  "
      f"dof={sim_ndof}  χ²_red={sim_chi2_r:.2f}  "
      f"v/v_sim=1.00")
for v in variants:
    # chi2 vs Vdn
    ok = (v['vsf'] > 0) & (vdn_at_bins > 0) & fit_mask
    chi2_bins = (v['vsf'][ok] - vdn_at_bins[ok]) ** 2 / v['err'][ok] ** 2
    n_dof = int(ok.sum())
    chi2_red = float(chi2_bins.sum()) / max(n_dof - 1, 1)
    # median ratio vs Simone
    rs = []
    for j, r in enumerate(FORARUN_BIN_CENTERS):
        if r < R_MIN: continue
        vs = sim_by_r.get(round(r, 3))
        if vs and vs > 0 and v['vsf'][j] > 0:
            rs.append(v['vsf'][j] / vs)
    med = float(np.median(rs)) if rs else float('nan')
    print(f"  {v['label']:<40}  χ²={chi2_bins.sum():8.1f}  "
          f"dof={n_dof}  χ²_red={chi2_red:.2f}  "
          f"v/v_sim={med:.2f}")
