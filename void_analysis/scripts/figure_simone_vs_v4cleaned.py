#!/usr/bin/env python3
"""
figure_simone_vs_v4cleaned.py

Side-by-side comparison of Simone's reference NREC50 VSFs and the
v4-cleaned VSFs that we ourselves bin from cleaned_v4_recon_cat_*.csv.

Layout: 2×2 panels (one per subsample), each with a main panel showing
both VSFs as scatter + best-fit (deg-3 log-log polynomial), and a residual
sub-panel showing the v4 / Simone ratio across R.

Styling is shared with `figure_cbl_sweep_overlay.py` and
`figure_overlap_variants_overlay.py` via `_vsf_plot_utils.py`.

Output:
  plots/diagnostics/simone_vs_v4cleaned_vsfs.png            (default — with MULT)
  plots/diagnostics/simone_vs_v4cleaned_noMULT_vsfs.png     (with --data-dir noMULT)

CLI:
  python scripts/figure_simone_vs_v4cleaned.py
  python scripts/figure_simone_vs_v4cleaned.py \
      --data-dir data/ForArun_v4cleaned_noMULT --out-suffix _noMULT \
      --label "v4 cleaned (no MULT)"
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from _vsf_plot_utils import (
    UNIFIED_RCPARAMS, XLIM, YLIM, R_MIN,
    SIMONE_COLOR, SIMONE_MARKER, SIMONE_MS,
    SUBSAMPLE_COLORS, MPS,
    fit_loglog_poly, plot_scatter_with_fit,
    draw_exclusion_zone, apply_axis_style,
)

FA_REF = os.path.join(HERE, '..', 'data', 'ForArun')
DEFAULT_FA_V4 = os.path.join(HERE, '..', 'data', 'ForArun_v4cleaned')
OUT = os.path.join(HERE, '..', 'plots', 'diagnostics')
os.makedirs(OUT, exist_ok=True)

SUBSAMPLES = ['20p', '10p', '5p', '2_5p']
FA_MAP   = {'2_5p': '025', '5p': '05', '10p': '10', '20p': '20'}
LABELS   = {'20p': '20%', '10p': '10%', '5p': '5%', '2_5p': '2.5%'}


def load(field_dir, frac):
    fpath = os.path.join(field_dir,
        f'vsf_dm_250cube_ss0p{FA_MAP[frac]}_NREC50_'
        'SM0p00_CS0p79_TH-0p80_OV00.dat')
    df = pd.read_csv(fpath, sep=r'\s+', comment='#', names=['r', 'vsf', 'err'])
    return df[df['vsf'] > 0].reset_index(drop=True)


_ap = argparse.ArgumentParser()
_ap.add_argument('--data-dir', default=DEFAULT_FA_V4,
                 help='v4-cleaned VSF dir (default: data/ForArun_v4cleaned).')
_ap.add_argument('--out-suffix', default='',
                 help='Data-side suffix on output filename. E.g. "_noMULT". '
                      'The style suffix ("_fitR7"/"_nofit") is appended after.')
_ap.add_argument('--label', default='v4 cleaned (this work)',
                 help='Legend label for the v4 series.')
_ap.add_argument('--style', default='fitR7', choices=['fitR7', 'nofit'],
                 help='fitR7: deg-2 log-log poly fit restricted to R≥7. '
                      'nofit: pure scatter, no fit curve.')
_args = _ap.parse_args()
USE_FITS = (_args.style == 'fitR7')

plt.rcParams.update(UNIFIED_RCPARAMS)

# Panel y-range narrower than the unified YLIM — these comparisons sit in
# the 5e-7 to 4e-3 range too, but tight to the data:
PANEL_YLIM = (5e-7, 4e-3)

fig = plt.figure(figsize=(15, 12))
outer = gridspec.GridSpec(2, 2, hspace=0.30, wspace=0.24,
                          left=0.07, right=0.97, top=0.91, bottom=0.06)

for idx, frac in enumerate(SUBSAMPLES):
    inner = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[idx], height_ratios=[3, 1], hspace=0.0)
    ax_main = fig.add_subplot(inner[0])
    ax_rat  = fig.add_subplot(inner[1], sharex=ax_main)

    ref = load(FA_REF, frac)
    v4  = load(_args.data_dir, frac)

    # ── Main panel ──
    draw_exclusion_zone(ax_main)

    # Simone reference — scatter; fit restricted to R≥7 if --style fitR7
    ref_R   = ref['r'].values
    ref_vsf = ref['vsf'].values
    ref_fit = (fit_loglog_poly(ref_R, ref_vsf, degree=2,
                               fit_mask=(ref_R >= R_MIN))
               if USE_FITS else None)
    plot_scatter_with_fit(
        ax_main, ref_R, ref_vsf, ref['err'].values,
        fit_fn=ref_fit,
        color=SIMONE_COLOR, marker=SIMONE_MARKER, ms=SIMONE_MS,
        fit_ls='-', fit_lw=2.0, fit_zorder=10, scatter_zorder=11,
        label='Simone reference',
    )

    # v4-cleaned — scatter; fit restricted to R≥7 if --style fitR7
    v4_R   = v4['r'].values
    v4_vsf = v4['vsf'].values
    v4_fit = (fit_loglog_poly(v4_R, v4_vsf, degree=2,
                              fit_mask=(v4_R >= R_MIN))
              if USE_FITS else None)
    plot_scatter_with_fit(
        ax_main, v4_R, v4_vsf, v4['err'].values,
        fit_fn=v4_fit,
        color=SUBSAMPLE_COLORS[frac], marker='s', ms=8,
        fit_ls=':', fit_lw=1.6,
        label=_args.label,
    )

    ax_main.set_xscale('log'); ax_main.set_yscale('log')
    ax_main.set_xlim(XLIM); ax_main.set_ylim(PANEL_YLIM)
    ax_main.set_ylabel(r'$dn/d\ln R_v$  [$h^3$Mpc$^{-3}$]')
    ax_main.tick_params(labelbottom=False)
    ax_main.grid(True, ls='--', alpha=0.25, which='both')
    ax_main.set_title(
        f'NREC50  {LABELS[frac]}   (MPS = {MPS[frac]} $h^{{-1}}$Mpc)',
        pad=8,
    )
    if idx == 0:
        ax_main.legend(loc='lower left', framealpha=0.95, edgecolor='#bbb')

    # ── Ratio v4 / Simone on shared bins ──
    merged = pd.merge(
        ref[['r', 'vsf', 'err']].rename(columns={'vsf': 'vsf_ref',
                                                 'err': 'err_ref'}),
        v4[['r', 'vsf', 'err']].rename(columns={'vsf': 'vsf_v4',
                                                'err': 'err_v4'}),
        on='r', how='inner').sort_values('r').reset_index(drop=True)
    ratio = merged['vsf_v4'] / merged['vsf_ref']
    ratio_err = ratio * np.sqrt(
        (merged['err_v4'] / merged['vsf_v4']) ** 2 +
        (merged['err_ref'] / merged['vsf_ref']) ** 2)
    fit_mask = merged['r'] >= R_MIN
    med_ratio = (float(np.median(ratio[fit_mask]))
                 if fit_mask.any() else float('nan'))

    # Badge — bottom-right (avoids overlap with data which rises top-left)
    ax_main.text(
        0.97, 0.05,
        f'v4 / Simone (R≥7, median):  {med_ratio:.2f}$\\times$',
        transform=ax_main.transAxes, ha='right', va='bottom', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.35', fc='white',
                  ec='#999', alpha=0.92))

    # ── Ratio sub-panel ──
    draw_exclusion_zone(ax_rat)
    ax_rat.axhline(1, color='k', lw=0.8, zorder=2)
    ax_rat.fill_between(list(XLIM) + [XLIM[1] + 1], 0.8, 1.2,
                        color='#27ae60', alpha=0.10, zorder=0)
    ax_rat.fill_between(list(XLIM) + [XLIM[1] + 1], 0.5, 2.0,
                        color='#f39c12', alpha=0.07, zorder=0)

    in_fit = merged['r'] >= R_MIN
    if (~in_fit).any():
        ax_rat.errorbar(merged['r'][~in_fit], ratio[~in_fit],
                        yerr=ratio_err[~in_fit],
                        fmt='s', ls='none', ms=5,
                        color=SUBSAMPLE_COLORS[frac], alpha=0.3,
                        ecolor=SUBSAMPLE_COLORS[frac], elinewidth=0.8,
                        capsize=2, zorder=4)
    if in_fit.any():
        ax_rat.errorbar(merged['r'][in_fit], ratio[in_fit],
                        yerr=ratio_err[in_fit],
                        fmt='s', ls='none', ms=6.5,
                        color=SUBSAMPLE_COLORS[frac],
                        ecolor=SUBSAMPLE_COLORS[frac], elinewidth=1.2,
                        capsize=2.5, mec='k', mew=0.4, zorder=5)

    ax_rat.set_xscale('log'); ax_rat.set_yscale('log')
    ax_rat.set_xlim(XLIM); ax_rat.set_ylim(0.2, 5)
    ax_rat.set_yticks([0.25, 0.5, 1.0, 2.0, 4.0])
    ax_rat.set_yticklabels(['0.25', '0.5', '1', '2', '4'])
    ax_rat.set_xlabel(r'$R_v$  [$h^{-1}$Mpc]')
    ax_rat.set_ylabel(r'$v4 / $Simone', fontsize=10)
    ax_rat.grid(True, ls='--', alpha=0.25, which='both')

style_caption = (
    'Smoothing lines: deg-2 log-log polynomial fit in $R \\geq 7$ h$^{-1}$Mpc only '
    '(MCMC fit range), not a physical model fit.'
    if USE_FITS else
    'Pure scatter (no per-variant fits). Only smooth curves shown elsewhere are Vdn theory.'
)
fig.suptitle(
    f'NREC50 VSF: Simone reference vs v4-cleaned\n'
    f'Grey zone: $R < 7$ $h^{{-1}}$Mpc (excluded from MCMC).   '
    f'Green band: $\\pm 20$%, orange band: factor 2.\n'
    f'{style_caption}',
    fontsize=12, y=0.98)

out = os.path.join(OUT,
    f'simone_vs_v4cleaned{_args.out_suffix}_vsfs_{_args.style}.png')
fig.savefig(out, dpi=220, bbox_inches='tight')
print(f"Saved: {out}")
plt.close(fig)
