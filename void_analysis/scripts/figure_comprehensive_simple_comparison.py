#!/usr/bin/env python3
"""
figure_comprehensive_simple_comparison.py

Comprehensive comparison of simple-cleaned VSFs across:
  - Two density thresholds: δ_th = -0.79 (cleaned_simple) and -0.80 (cleaned_simple0)
  - Two overlap criteria:   CENTER and TOUCH
  - Calibrated against:     Vdn theory at truth AND Simone's reference

Generates 4 plots:
  1. 2×2 log-log VSF panels (one per variant), each showing data + Vdn + Simone
  2. 2×2 residual panels: (data - Vdn) / σ  per bin
  3. 2×2 ratio-vs-Simone panels: data / Simone per bin
  4. Summary bar chart: χ² and v/v_sim across all 4 variants

Output:
  plots/diagnostics/comprehensive_vsf_4panel.png
  plots/diagnostics/comprehensive_residuals_vdn.png
  plots/diagnostics/comprehensive_ratio_simone.png
  plots/diagnostics/comprehensive_summary_bars.png
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
)

DATA = os.path.join(HERE, '..', 'data')
OUT_DIR = os.path.join(HERE, '..', 'plots', 'diagnostics')
os.makedirs(OUT_DIR, exist_ok=True)

SIMONE_REF = os.path.join(DATA, 'ForArun',
    'vsf_dm_250cube_ss0p05_NREC50_SM0p00_CS0p79_TH-0p80_OV00.dat')

# All 4 variants: (path, short_label, long_label, color, marker)
VARIANTS = [
    # dict(path=os.path.join(DATA, 'cleaned_simple',
    #          'cleaned_simple_nrec50_cat_5p_center.csv'),
    #      short='CENTER δ=−0.79', long='δ_th = −0.79, overlap = CENTER',
    #      color='#16a085', marker='D'),
    # dict(path=os.path.join(DATA, 'cleaned_simple',
    #          'cleaned_simple_nrec50_cat_5p_touch.csv'),
    #      short='TOUCH δ=−0.79', long='δ_th = −0.79, overlap = TOUCH',
    #      color='#d35400', marker='s'),
    dict(path=os.path.join(DATA, 'cleaned_simple0',
             'cleaned_simple0_nrec50_cat_5p_center.csv'),
         short='CENTER δ=−0.80', long='δ_th = −0.80, overlap = CENTER',
         color='#2980b9', marker='^'),
    dict(path=os.path.join(DATA, 'cleaned_simple0',
             'cleaned_simple0_nrec50_cat_5p_touch.csv'),
         short='TOUCH δ=−0.80', long='δ_th = −0.80, overlap = TOUCH',
         color='#c0392b', marker='v'),
]

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


# ─── Load everything ────────────────────────────────────────────────
sim = pd.read_csv(SIMONE_REF, sep=r'\s+', comment='#',
                  names=['r', 'vsf', 'err'])
sim = sim[sim['vsf'] > 0].reset_index(drop=True)
sim_by_r = dict(zip(sim['r'].round(3), sim['vsf']))
sim_err_by_r = dict(zip(sim['r'].round(3), sim['err']))

R_grid = np.logspace(np.log10(XLIM[0]), np.log10(XLIM[1]), 200)
vdn_smooth = vdn_truth_curve(R_grid)
vdn_at_bins = vdn_truth_curve(FORARUN_BIN_CENTERS)

variants = []
for v in VARIANTS:
    df = pd.read_csv(v['path'])
    col = 'R_resc' if 'R_resc' in df.columns else 'R'
    radii = df[col].values
    counts, vsf, err = bin_vsf(radii)

    # Ratio vs Simone at each bin
    ratio_sim = np.full(len(FORARUN_BIN_CENTERS), np.nan)
    ratio_sim_err = np.full(len(FORARUN_BIN_CENTERS), np.nan)
    for j, r in enumerate(FORARUN_BIN_CENTERS):
        vs = sim_by_r.get(round(r, 3))
        ve = sim_err_by_r.get(round(r, 3))
        if vs and vs > 0 and vsf[j] > 0:
            ratio_sim[j] = vsf[j] / vs
            if ve and ve > 0:
                ratio_sim_err[j] = ratio_sim[j] * np.sqrt(
                    (err[j] / vsf[j]) ** 2 + (ve / vs) ** 2)

    # Residual vs Vdn: (data - theory) / sigma
    ok_vdn = (vsf > 0) & (vdn_at_bins > 0) & (err > 0)
    resid_vdn = np.full(len(FORARUN_BIN_CENTERS), np.nan)
    resid_vdn[ok_vdn] = (vsf[ok_vdn] - vdn_at_bins[ok_vdn]) / err[ok_vdn]

    # Chi2 vs Vdn
    chi2_vdn = np.full(len(FORARUN_BIN_CENTERS), np.nan)
    chi2_vdn[ok_vdn] = resid_vdn[ok_vdn] ** 2

    fit_mask = FORARUN_BIN_CENTERS >= R_MIN
    chi2_R7 = float(np.nansum(chi2_vdn[fit_mask]))
    ndof_R7 = int((ok_vdn & fit_mask).sum())
    chi2_all = float(np.nansum(chi2_vdn))
    ndof_all = int(ok_vdn.sum())

    # Median ratios
    ratio_vals_R7 = ratio_sim[fit_mask & np.isfinite(ratio_sim)]
    ratio_vals_all = ratio_sim[np.isfinite(ratio_sim)]
    med_ratio_R7 = float(np.median(ratio_vals_R7)) if len(ratio_vals_R7) else float('nan')
    med_ratio_all = float(np.median(ratio_vals_all)) if len(ratio_vals_all) else float('nan')

    variants.append(dict(
        **v, n_total=len(df), n_R7=int((radii >= R_MIN).sum()),
        vsf=vsf, err=err, counts=counts,
        ratio_sim=ratio_sim, ratio_sim_err=ratio_sim_err,
        resid_vdn=resid_vdn, chi2_vdn=chi2_vdn,
        chi2_R7=chi2_R7, ndof_R7=ndof_R7,
        chi2_all=chi2_all, ndof_all=ndof_all,
        med_ratio_R7=med_ratio_R7, med_ratio_all=med_ratio_all,
    ))

# Simone's own residuals/chi2 vs Vdn
sim_vdn = vdn_truth_curve(sim['r'].values)
sim_ok = (sim['vsf'].values > 0) & (sim_vdn > 0) & (sim['err'].values > 0)
sim_resid = np.full(len(sim), np.nan)
sim_resid[sim_ok] = (sim['vsf'].values[sim_ok] - sim_vdn[sim_ok]) / sim['err'].values[sim_ok]
sim_chi2 = np.full(len(sim), np.nan)
sim_chi2[sim_ok] = sim_resid[sim_ok] ** 2
sim_fit = sim['r'].values >= R_MIN
sim_chi2_R7 = float(np.nansum(sim_chi2[sim_ok & sim_fit]))
sim_ndof_R7 = int((sim_ok & sim_fit).sum())
sim_chi2_all = float(np.nansum(sim_chi2[sim_ok]))
sim_ndof_all = int(sim_ok.sum())

plt.rcParams.update(UNIFIED_RCPARAMS)

# ═══════════════════════════════════════════════════════════════════
# PLOT 1: log-log VSF panels (layout adapts to number of variants)
# ═══════════════════════════════════════════════════════════════════
n_var = len(variants)
n_cols = min(n_var, 2)
n_rows = (n_var + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols,
                          figsize=(8 * n_cols, 7 * n_rows),
                          sharex=True, sharey=True, squeeze=False)
fig.subplots_adjust(top=0.82, wspace=0.15)
fig.suptitle('VSF vs Vdn theory & Simone for ss5%, 50 OTs',
             fontsize=14, fontweight='bold', y=0.95)

for idx, (v, ax) in enumerate(zip(variants, axes.flat)):
    # Vdn theory
    ax.plot(R_grid, vdn_smooth,
            ls='--', color=VDN_TRUTH_COLOR, lw=2, zorder=3,
            label='Vdn at truth' if idx == 0 else None)

    # Simone
    ax.errorbar(sim['r'], sim['vsf'], yerr=sim['err'],
                fmt=SIMONE_MARKER, ls='none', ms=7,
                color=SIMONE_COLOR, ecolor=SIMONE_COLOR,
                elinewidth=1, capsize=2, mec='k', mew=0.3,
                alpha=0.7, zorder=10,
                label='Simone ref.' if idx == 0 else None)

    # This variant
    ok = v['vsf'] > 0
    ax.errorbar(FORARUN_BIN_CENTERS[ok], v['vsf'][ok], yerr=v['err'][ok],
                fmt=v['marker'], ls='none', ms=7,
                color=v['color'], ecolor=v['color'],
                elinewidth=1.2, capsize=2.5, mec='k', mew=0.4,
                zorder=5)

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(XLIM); ax.set_ylim(YLIM)
    ax.grid(True, ls='--', alpha=0.2, which='both')
    ax.set_title(f"overlap = {v['short'].split()[0]}\n"
                 f"N={v['n_total']},  N(R≥7)={v['n_R7']},  "
                 f"v/v$_{{sim}}$: R≥7={v['med_ratio_R7']:.2f}, "
                 f"all={v['med_ratio_all']:.2f}",
                 fontsize=9.5, pad=8)

    if idx >= n_cols * (n_rows - 1):
        ax.set_xlabel(r'$R_v$  [$h^{-1}$Mpc]')
    if idx % n_cols == 0:
        ax.set_ylabel(r'$dn/d\ln R_v$  [$h^3$Mpc$^{-3}$]')

# Hide unused axes
for j in range(n_var, n_rows * n_cols):
    axes.flat[j].set_visible(False)
axes[0, 0].legend(loc='lower left', fontsize=9, framealpha=0.95)

out1 = os.path.join(OUT_DIR, 'comprehensive_vsf_4panel.png')
fig.savefig(out1, dpi=220, bbox_inches='tight')
print(f"Saved: {out1}")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════
# PLOT 2: residual panels (data - Vdn) / σ
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(n_rows, n_cols,
                          figsize=(8 * n_cols, 7 * n_rows),
                          sharex=True, sharey=True, squeeze=False)
fig.subplots_adjust(top=0.82, wspace=0.15)
fig.suptitle('Residuals vs Vdn:  $(n^{\\rm data} - n^{\\rm Vdn}) / \\sigma$ on ss5%, 50 OTs\n'
             'Green band: ±1σ,  orange band: ±3σ',
             fontsize=14, fontweight='bold', y=0.95)

for idx, (v, ax) in enumerate(zip(variants, axes.flat)):
    # Bands
    ax.axhspan(-1, 1, color='#27ae60', alpha=0.12, zorder=0)
    ax.axhspan(-3, 3, color='#f39c12', alpha=0.07, zorder=0)
    ax.axhline(0, color='k', lw=0.8, zorder=1)
    ax.axvline(R_MIN, color='#aaa', ls='-', lw=1.2, zorder=1)

    # Simone residuals
    sim_in = sim['r'].values >= R_MIN
    if sim_ok.any():
        ax.scatter(sim['r'].values[sim_ok & ~sim_in],
                   sim_resid[sim_ok & ~sim_in],
                   marker=SIMONE_MARKER, s=30, color=SIMONE_COLOR,
                   alpha=0.25, zorder=8, edgecolors='none')
        ax.scatter(sim['r'].values[sim_ok & sim_in],
                   sim_resid[sim_ok & sim_in],
                   marker=SIMONE_MARKER, s=50, color=SIMONE_COLOR,
                   alpha=0.6, zorder=9, edgecolors='k', linewidths=0.3,
                   label='Simone' if idx == 0 else None)

    # This variant
    fit_mask = FORARUN_BIN_CENTERS >= R_MIN
    ok = np.isfinite(v['resid_vdn'])
    # Faded below R_MIN
    ax.scatter(FORARUN_BIN_CENTERS[ok & ~fit_mask],
               v['resid_vdn'][ok & ~fit_mask],
               marker=v['marker'], s=35, color=v['color'],
               alpha=0.3, zorder=4, edgecolors='none')
    # Solid at R >= 7
    ax.scatter(FORARUN_BIN_CENTERS[ok & fit_mask],
               v['resid_vdn'][ok & fit_mask],
               marker=v['marker'], s=60, color=v['color'],
               alpha=0.9, zorder=5, edgecolors='k', linewidths=0.4,
               label=v['short'] if idx == 0 else None)

    ax.set_xscale('log')
    ax.set_xlim(XLIM)
    ax.set_ylim(-15, 25)
    ax.grid(True, ls='--', alpha=0.2, which='both')
    ax.set_title(f"overlap = {v['short'].split()[0]}   "
                 f"($\\chi^2_{{R≥7}}$ = {v['chi2_R7']:.0f}/{v['ndof_R7']})",
                 fontsize=10, pad=8)

    if idx >= n_cols * (n_rows - 1):
        ax.set_xlabel(r'$R_v$  [$h^{-1}$Mpc]')
    if idx % n_cols == 0:
        ax.set_ylabel(r'$(n^{\rm data} - n^{\rm Vdn}) \,/\, \sigma$')

for j in range(n_var, n_rows * n_cols):
    axes.flat[j].set_visible(False)
axes[0, 0].legend(loc='upper right', fontsize=9, framealpha=0.95)

out2 = os.path.join(OUT_DIR, 'comprehensive_residuals_vdn.png')
fig.savefig(out2, dpi=220, bbox_inches='tight')
print(f"Saved: {out2}")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════
# PLOT 3: ratio-vs-Simone panels
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(n_rows, n_cols,
                          figsize=(8 * n_cols, 7 * n_rows),
                          sharex=True, sharey=True, squeeze=False)
fig.subplots_adjust(top=0.82, wspace=0.15)
fig.suptitle('Ratio vs Simone:  simple-cleaned / Simone: ss5%, 50 OTs\n'
             'Green band: ±20%,  orange band: factor 2',
             fontsize=14, fontweight='bold', y=0.95)

for idx, (v, ax) in enumerate(zip(variants, axes.flat)):
    # Bands
    ax.axhline(1, color='k', lw=0.8, zorder=2)
    ax.fill_between(XLIM, 0.8, 1.2, color='#27ae60', alpha=0.12, zorder=0)
    ax.fill_between(XLIM, 0.5, 2.0, color='#f39c12', alpha=0.07, zorder=0)
    ax.axvline(R_MIN, color='#aaa', ls='-', lw=1.2, zorder=1)

    ok = np.isfinite(v['ratio_sim'])
    fit_mask = FORARUN_BIN_CENTERS >= R_MIN

    # Faded below R_MIN
    below = ok & ~fit_mask
    if below.any():
        ax.errorbar(FORARUN_BIN_CENTERS[below], v['ratio_sim'][below],
                    yerr=v['ratio_sim_err'][below],
                    fmt=v['marker'], ls='none', ms=5,
                    color=v['color'], alpha=0.3,
                    ecolor=v['color'], elinewidth=0.8,
                    capsize=2, zorder=4)

    # Solid at R >= 7
    above = ok & fit_mask
    if above.any():
        ax.errorbar(FORARUN_BIN_CENTERS[above], v['ratio_sim'][above],
                    yerr=v['ratio_sim_err'][above],
                    fmt=v['marker'], ls='none', ms=7,
                    color=v['color'],
                    ecolor=v['color'], elinewidth=1.2,
                    capsize=2.5, mec='k', mew=0.4, zorder=5)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(XLIM)
    ax.set_ylim(0.15, 6)
    ax.set_yticks([0.25, 0.5, 1.0, 2.0, 4.0])
    ax.set_yticklabels(['0.25', '0.5', '1', '2', '4'])
    ax.grid(True, ls='--', alpha=0.2, which='both')
    ax.set_title(f"overlap = {v['short'].split()[0]}   "
                 f"v/v$_{{sim}}$: all={v['med_ratio_all']:.2f}, "
                 f"R≥7={v['med_ratio_R7']:.2f}",
                 fontsize=10, pad=8)

    if idx >= n_cols * (n_rows - 1):
        ax.set_xlabel(r'$R_v$  [$h^{-1}$Mpc]')
    if idx % n_cols == 0:
        ax.set_ylabel('simple-cleaned / Simone')

for j in range(n_var, n_rows * n_cols):
    axes.flat[j].set_visible(False)

out3 = os.path.join(OUT_DIR, 'comprehensive_ratio_simone.png')
fig.savefig(out3, dpi=220, bbox_inches='tight')
print(f"Saved: {out3}")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════
# PLOT 4: Summary bar chart — χ² and v/v_sim across all variants
# ═══════════════════════════════════════════════════════════════════
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Summary across all simple-cleaning variants: ss5% 50 OTs\n',
             fontsize=13, fontweight='bold', y=1.02)

labels = [v['short'] for v in variants] + ['Simone']
colors = [v['color'] for v in variants] + [SIMONE_COLOR]
x = np.arange(len(labels))
bar_w = 0.6

# Panel A: χ²_red vs Vdn (R≥7)
chi2_reds = [v['chi2_R7'] / max(v['ndof_R7'] - 1, 1) for v in variants]
chi2_reds.append(sim_chi2_R7 / max(sim_ndof_R7 - 1, 1))
bars = ax1.bar(x, chi2_reds, bar_w, color=colors, edgecolor='k', linewidth=0.5, alpha=0.8)
for i, val in enumerate(chi2_reds):
    ax1.text(i, val + 1, f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax1.set_ylabel(r'$\chi^2_{\rm red}$  (vs Vdn, $R \geq 7$)')
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
ax1.axhline(1, color='k', ls=':', lw=1, alpha=0.4)
ax1.set_title('χ² vs Vdn theory', fontsize=11)
ax1.grid(axis='y', ls='--', alpha=0.2)

# Panel B: median v/v_sim (R>=7)
ratios_R7 = [v['med_ratio_R7'] for v in variants] + [1.0]
bars = ax2.bar(x, ratios_R7, bar_w, color=colors, edgecolor='k', linewidth=0.5, alpha=0.8)
for i, val in enumerate(ratios_R7):
    ax2.text(i, val + 0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax2.axhline(1.0, color='k', ls='-', lw=1)
ax2.axhspan(0.8, 1.2, color='#27ae60', alpha=0.12)
ax2.set_ylabel(r'median $v / v_{\rm sim}$  ($R \geq 7$)')
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
ax2.set_ylim(0, 1.8)
ax2.set_title('Ratio vs Simone (R ≥ 7)', fontsize=11)
ax2.grid(axis='y', ls='--', alpha=0.2)

# Panel C: median v/v_sim (all R)
ratios_all = [v['med_ratio_all'] for v in variants] + [1.0]
bars = ax3.bar(x, ratios_all, bar_w, color=colors, edgecolor='k', linewidth=0.5, alpha=0.8)
for i, val in enumerate(ratios_all):
    ax3.text(i, val + 0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax3.axhline(1.0, color='k', ls='-', lw=1)
ax3.axhspan(0.8, 1.2, color='#27ae60', alpha=0.12)
ax3.set_ylabel(r'median $v / v_{\rm sim}$  (all $R$)')
ax3.set_xticks(x)
ax3.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
ax3.set_ylim(0, 2.0)
ax3.set_title('Ratio vs Simone (all R)', fontsize=11)
ax3.grid(axis='y', ls='--', alpha=0.2)

fig.tight_layout()
out4 = os.path.join(OUT_DIR, 'comprehensive_summary_bars.png')
fig.savefig(out4, dpi=220, bbox_inches='tight')
print(f"Saved: {out4}")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════
# PLOT 4b: Per-bin χ² vs Vdn — all variants + Simone on one plot
#          Top: per-bin χ², Bottom: cumulative χ²
#          Both show all bins, with R<7 and R≥7 visually separated
# ═══════════════════════════════════════════════════════════════════
fig, (ax_bin, ax_cum) = plt.subplots(2, 1, figsize=(14, 10),
                                      sharex=True,
                                      gridspec_kw={'height_ratios': [1, 1],
                                                   'hspace': 0.10})

fig.suptitle('Per-bin $\\chi^2$ vs Vdn theory on both criteria + Simone\n'
             'ss5% 50 OTs.  Vertical line at $R = 7$ separates MCMC fit range.',
             fontsize=13, fontweight='bold', y=0.98)

bar_offset = np.linspace(-0.35, 0.35, len(variants) + 1)
total_width = 0.7 / (len(variants) + 1)

# Simone per-bin chi2
sim_bin_R = sim['r'].values
for k in range(len(sim)):
    if not sim_ok[k]:
        continue
    ax_bin.bar(sim_bin_R[k] * (1 + bar_offset[-1] * 0.03),
               sim_chi2[k],
               width=sim_bin_R[k] * (10**DLNR - 1) * total_width,
               color=SIMONE_COLOR, alpha=0.4, edgecolor=SIMONE_COLOR,
               zorder=2,
               label='Simone' if k == 0 else None)

# Variant per-bin chi2
for vi, v in enumerate(variants):
    ok = np.isfinite(v['chi2_vdn'])
    for k in np.where(ok)[0]:
        ax_bin.bar(FORARUN_BIN_CENTERS[k] * (1 + bar_offset[vi] * 0.03),
                   v['chi2_vdn'][k],
                   width=FORARUN_BIN_CENTERS[k] * (10**DLNR - 1) * total_width,
                   color=v['color'], alpha=0.5, edgecolor=v['color'],
                   zorder=3,
                   label=v['short'] if k == np.where(ok)[0][0] else None)

ax_bin.axhline(1, color='k', ls=':', lw=1, alpha=0.5, zorder=1)
ax_bin.axvline(R_MIN, color='#aaa', ls='-', lw=1.5, zorder=1)
ax_bin.set_xscale('log')
ax_bin.set_ylabel(r'$\chi^2_i$  per bin (vs Vdn)')
ax_bin.legend(fontsize=8.5, framealpha=0.95, edgecolor='#bbb',
              ncol=3, loc='upper right')
ax_bin.grid(True, ls='--', alpha=0.2, which='both')
ax_bin.set_xlim(XLIM)
ax_bin.tick_params(labelbottom=False)

# Add text annotations for total chi2
txt_y = 0.95
for v in variants:
    ax_bin.text(0.01, txt_y,
                f"{v['short']}:  χ²(all)={v['chi2_all']:.0f}/{v['ndof_all']}  "
                f"χ²(R≥7)={v['chi2_R7']:.0f}/{v['ndof_R7']}",
                transform=ax_bin.transAxes, fontsize=7.5,
                color=v['color'], fontweight='bold', va='top')
    txt_y -= 0.05
ax_bin.text(0.01, txt_y,
            f"Simone:  χ²(all)={sim_chi2_all:.0f}/{sim_ndof_all}  "
            f"χ²(R≥7)={sim_chi2_R7:.0f}/{sim_ndof_R7}",
            transform=ax_bin.transAxes, fontsize=7.5,
            color=SIMONE_COLOR, fontweight='bold', va='top')

# Bottom: cumulative chi2 across ALL bins (left to right)
# Simone cumulative
sim_sorted_idx = np.argsort(sim_bin_R)
sim_cum_chi2 = []
running = 0
sim_r_sorted = []
for k in sim_sorted_idx:
    if sim_ok[k]:
        running += sim_chi2[k]
        sim_cum_chi2.append(running)
        sim_r_sorted.append(sim_bin_R[k])
ax_cum.plot(sim_r_sorted, sim_cum_chi2,
            color=SIMONE_COLOR, marker=SIMONE_MARKER, ms=5, lw=1.5,
            zorder=10, label=f'Simone (total={sim_chi2_all:.0f})')

# Variant cumulative
for v in variants:
    ok = np.isfinite(v['chi2_vdn'])
    idx_sorted = np.argsort(FORARUN_BIN_CENTERS[ok])
    r_sorted = FORARUN_BIN_CENTERS[ok][idx_sorted]
    chi2_sorted = v['chi2_vdn'][ok][idx_sorted]
    cum = np.cumsum(chi2_sorted)
    ax_cum.plot(r_sorted, cum,
                color=v['color'], marker=v['marker'], ms=5, lw=1.5,
                zorder=5,
                label=f"{v['short']} (total={v['chi2_all']:.0f})")

ax_cum.axvline(R_MIN, color='#aaa', ls='-', lw=1.5, zorder=1)
ax_cum.set_xscale('log')
ax_cum.set_xlim(XLIM)
ax_cum.set_xlabel(r'$R_v$  [$h^{-1}$Mpc]')
ax_cum.set_ylabel(r'Cumulative $\chi^2$ (left to right)')
ax_cum.legend(fontsize=8.5, framealpha=0.95, edgecolor='#bbb',
              ncol=2, loc='upper left')
ax_cum.grid(True, ls='--', alpha=0.2, which='both')
ax_cum.text(0.97, 0.05,
            'Left of grey line: R < 7 (all bins)\n'
            'Right of grey line: R ≥ 7 (MCMC range)',
            transform=ax_cum.transAxes, ha='right', va='bottom',
            fontsize=8, color='#666',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc', alpha=0.9))

out4b = os.path.join(OUT_DIR, 'comprehensive_chi2_perbin.png')
fig.savefig(out4b, dpi=220, bbox_inches='tight')
print(f"Saved: {out4b}")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*100}")
print('COMPREHENSIVE SUMMARY  (5% NREC50)')
print('='*100)
print(f"{'variant':<22}{'N':>7}{'N(R≥7)':>8}"
      f"{'χ²(all)':>10}{'dof':>5}{'χ²_r':>7}"
      f"{'χ²(R≥7)':>10}{'dof':>5}{'χ²_r':>7}"
      f"{'v/v all':>9}{'v/v R≥7':>9}")
print('-'*110)
for v in variants:
    chi2r_all = v['chi2_all'] / max(v['ndof_all'] - 1, 1)
    chi2r_R7 = v['chi2_R7'] / max(v['ndof_R7'] - 1, 1)
    print(f"{v['short']:<22}{v['n_total']:>7}{v['n_R7']:>8}"
          f"{v['chi2_all']:>10.0f}{v['ndof_all']:>5}{chi2r_all:>7.1f}"
          f"{v['chi2_R7']:>10.0f}{v['ndof_R7']:>5}{chi2r_R7:>7.1f}"
          f"{v['med_ratio_all']:>9.3f}{v['med_ratio_R7']:>9.3f}")
sim_chi2r_all = sim_chi2_all / max(sim_ndof_all - 1, 1)
sim_chi2r_R7 = sim_chi2_R7 / max(sim_ndof_R7 - 1, 1)
print(f"{'Simone reference':<22}{len(sim):>7}{int(sim_fit.sum()):>8}"
      f"{sim_chi2_all:>10.0f}{sim_ndof_all:>5}{sim_chi2r_all:>7.1f}"
      f"{sim_chi2_R7:>10.0f}{sim_ndof_R7:>5}{sim_chi2r_R7:>7.1f}"
      f"{'1.000':>9}{'1.000':>9}")

# ═══════════════════════════════════════════════════════════════════
# PLOT 5: Single-panel overlay — all 4 variants + Simone + Vdn
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 9))

# Vdn theory
ax.plot(R_grid, vdn_smooth,
        ls='--', color=VDN_TRUTH_COLOR, lw=2.2, zorder=3,
        label=(f'Vdn at truth '
               f'($\\Omega_m={OMEGA_M_TRUE}, \\sigma_8={SIGMA8_TRUE}$)'))

# Simone reference
ax.errorbar(sim['r'], sim['vsf'], yerr=sim['err'],
            fmt=SIMONE_MARKER, ls='none', ms=SIMONE_MS + 1,
            color=SIMONE_COLOR, ecolor=SIMONE_COLOR,
            elinewidth=1.2, capsize=2.5, mec='k', mew=0.4,
            alpha=0.8, zorder=11,
            label='Simone reference (ss5%, 50 OTs)')

# All 4 variants
for v in variants:
    ok = v['vsf'] > 0
    ax.errorbar(FORARUN_BIN_CENTERS[ok], v['vsf'][ok], yerr=v['err'][ok],
                fmt=v['marker'], ls='none', ms=7,
                color=v['color'], ecolor=v['color'],
                elinewidth=1, capsize=2, mec='k', mew=0.3,
                zorder=5,
                label=(f"{v['short']}  "
                       f"(N={v['n_total']}, v/v$_{{sim}}$: "
                       f"all={v['med_ratio_all']:.2f}, "
                       f"R≥7={v['med_ratio_R7']:.2f})"))

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(XLIM); ax.set_ylim(YLIM)
ax.set_xlabel(r'$R_v$  [$h^{-1}$Mpc]')
ax.set_ylabel(r'$dn/d\ln R_v$  [$h^3$Mpc$^{-3}$]')
ax.set_title('All simple-cleaning variants vs Vdn & Simone ss5% 50 OTs\n'
             'δ_th = {−0.80} × overlap = {CENTER, TOUCH}',
             fontweight='bold', pad=10)
ax.grid(True, ls='--', alpha=0.2, which='both')
ax.legend(loc='lower left', fontsize=9, framealpha=0.95, edgecolor='#bbb')

out5 = os.path.join(OUT_DIR, 'comprehensive_overlay_all.png')
fig.savefig(out5, dpi=220, bbox_inches='tight')
print(f"Saved: {out5}")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════
# PLOT 6: Cumulative void count N(>R) — all variants + Simone
# ═══════════════════════════════════════════════════════════════════
fig, (ax_cum, ax_rat) = plt.subplots(2, 1, figsize=(14, 10),
                                      sharex=True,
                                      gridspec_kw={'height_ratios': [2, 1],
                                                   'hspace': 0.08})

# Build cumulative N(>R) from each variant's raw radii
R_eval = np.logspace(np.log10(1.5), np.log10(20), 300)

# Simone: reconstruct cumulative from her differential VSF
# N(>R) = integral of dn/dlnR * V_eff * dlnR from R to inf
# Approximate: sum the VSF bins from right to left
sim_R = sim['r'].values
sim_vsf_vals = sim['vsf'].values
sim_cum_at_bins = np.zeros(len(sim_R))
for j in range(len(sim_R) - 1, -1, -1):
    sim_cum_at_bins[j] = sim_vsf_vals[j] * V_EFF * DLNR
    if j < len(sim_R) - 1:
        sim_cum_at_bins[j] += sim_cum_at_bins[j + 1]

ax_cum.plot(sim_R, sim_cum_at_bins,
            color=SIMONE_COLOR, marker=SIMONE_MARKER, ms=6, lw=1.5,
            zorder=11, label='Simone reference')

# Each variant: direct cumulative from raw radii
for v in variants:
    df = pd.read_csv(v['path'])
    col = 'R_resc' if 'R_resc' in df.columns else 'R'
    radii = df[col].values
    cum_counts = np.array([np.sum(radii >= R) for R in R_eval])

    ax_cum.plot(R_eval, cum_counts,
                color=v['color'], lw=1.8, zorder=5,
                label=f"{v['short']}  (N_tot={v['n_total']})")

ax_cum.axvline(R_MIN, color='#aaa', ls='-', lw=1.2, zorder=1)
ax_cum.set_yscale('log')
ax_cum.set_xscale('log')
ax_cum.set_xlim(XLIM)
ax_cum.set_ylim(1, 2e4)
ax_cum.set_ylabel(r'$N(> R)$  [cumulative count]')
ax_cum.set_title('Cumulative void count $N(>R)$ on different criteria vs Simone\n'
                 'ss5% 50 OTs, Aletheia 250 $h^{-1}$Mpc box',
                 fontweight='bold', pad=10)
ax_cum.grid(True, ls='--', alpha=0.2, which='both')
ax_cum.legend(loc='upper right', fontsize=9, framealpha=0.95, edgecolor='#bbb')
ax_cum.tick_params(labelbottom=False)

# Bottom panel: ratio of cumulative N(>R) to Simone's cumulative
# Interpolate Simone's cumulative onto R_eval
sim_cum_eval = np.interp(R_eval, sim_R, sim_cum_at_bins,
                         left=np.nan, right=np.nan)

ax_rat.axhline(1, color='k', lw=0.8, zorder=2)
ax_rat.fill_between(R_eval, 0.8, 1.2, color='#27ae60', alpha=0.12, zorder=0)
ax_rat.axvline(R_MIN, color='#aaa', ls='-', lw=1.2, zorder=1)

for v in variants:
    df = pd.read_csv(v['path'])
    col = 'R_resc' if 'R_resc' in df.columns else 'R'
    radii = df[col].values
    cum_counts = np.array([np.sum(radii >= R) for R in R_eval], dtype=float)
    ratio = cum_counts / sim_cum_eval
    ok = np.isfinite(ratio) & (sim_cum_eval > 5)
    ax_rat.plot(R_eval[ok], ratio[ok],
                color=v['color'], lw=1.8, zorder=5,
                label=v['short'])

ax_rat.set_xscale('log')
ax_rat.set_xlim(XLIM)
ax_rat.set_ylim(0.3, 3.0)
ax_rat.set_yscale('log')
ax_rat.set_yticks([0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0])
ax_rat.set_yticklabels(['0.5', '0.8', '1', '1.2', '1.5', '2', '3'])
ax_rat.set_xlabel(r'$R_v$  [$h^{-1}$Mpc]')
ax_rat.set_ylabel(r'$N(>R)$ / $N_{\rm Simone}(>R)$')
ax_rat.grid(True, ls='--', alpha=0.2, which='both')
ax_rat.legend(loc='upper left', fontsize=9, framealpha=0.95, edgecolor='#bbb')
ax_rat.text(0.97, 0.05, 'Green band: ±20%',
            transform=ax_rat.transAxes, ha='right', va='bottom',
            fontsize=8, color='#666')

out6 = os.path.join(OUT_DIR, 'comprehensive_cumulative_NgtR.png')
fig.savefig(out6, dpi=220, bbox_inches='tight')
print(f"Saved: {out6}")
plt.close(fig)
