#!/usr/bin/env python3
"""
figure_v5_calibration.py

Publication-quality calibration figure for the v5 joint MCMC result.
4-panel layout (one per subsample), each showing:
  - Data points (faded in exclusion zone, solid in fit range)
  - Vdn at v5 joint MAP (best fit) — solid dark line
  - Vdn at Aletheia truth + v5 MAP nuisance — dashed red
  - Shaded grey: R < 7 exclusion zone
  - Bottom sub-panel: normalised residuals in fit range only

Run:
    python scripts/figure_v5_calibration.py
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import LogLocator, NullFormatter

CBL_HOME = os.environ.get('CBL_HOME', '/Users/arun/projects/CosmoBolognaLib')
sys.path.insert(0, os.path.join(CBL_HOME, 'Python'))
os.environ.setdefault('DYLD_LIBRARY_PATH', CBL_HOME)
import CosmoBolognaLib as cbl

HERE = os.path.dirname(os.path.abspath(__file__))
FA   = os.path.join(HERE, '..', 'data', 'ForArun')
CHAIN_DIR = os.path.join(HERE, '..', 'data', 'mcmc_chains')
OUT_DIR   = os.path.join(HERE, '..', 'plots', 'diagnostics')
os.makedirs(OUT_DIR, exist_ok=True)

OMEGA_M_TRUE, SIGMA8_TRUE = 0.315, 0.811
H0, OMEGA_B, N_S, TAU = 0.6736, 0.0493, 0.9649, 0.0925
SCALAR_AMP, SCALAR_PIVOT = 2.194e-9, 0.05
DELTA_NL, B_EFF = -0.7, 1.0

FA_MAP = {'2_5p': '025', '5p': '05', '10p': '10', '20p': '20'}
SUBSAMPLES = ['20p', '10p', '5p', '2_5p']
MPS = {'20p': 1.16, '10p': 1.47, '5p': 1.85, '2_5p': 2.33}
COLORS = {'20p': '#8e44ad', '10p': '#2980b9', '5p': '#27ae60', '2_5p': '#d4ac0d'}
LABELS = {'20p': '20%', '10p': '10%', '5p': '5%', '2_5p': '2.5%'}
R_MIN = 7.0

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 8.5,
    'figure.facecolor': 'white',
})


def make_cosmo(om, s8):
    c = cbl.cosmology.LCDM(om, OMEGA_B, 0.0, H0,
                           SCALAR_AMP, SCALAR_PIVOT, N_S, TAU, True)
    c.set_parameter("sigma8", s8)
    return c


def vdn_curve(R_grid, om, s8, slope, offset):
    c = make_cosmo(om, s8)
    sf = cbl.cosmology.SizeFunction(c)
    dc = c.deltac(0.0)
    return np.array([sf.size_function(float(R), 0.0, "Vdn",
                     B_EFF, slope, offset, DELTA_NL, dc, "EisensteinHu")
                     for R in R_grid])


def load_data(frac, field, data_dir=None):
    base = data_dir if data_dir else FA
    fpath = os.path.join(base, f'vsf_dm_250cube_ss0p{FA_MAP[frac]}_{field}_'
                                'SM0p00_CS0p79_TH-0p80_OV00.dat')
    df = pd.read_csv(fpath, sep=r'\s+', comment='#', names=['r', 'vsf', 'err'])
    m = df['vsf'] > 0
    return df.loc[m, 'r'].values, df.loc[m, 'vsf'].values, df.loc[m, 'err'].values


def load_meta(field, tag='cluster'):
    for t in [tag, '']:
        for stem in [f'cbl_v5_joint_R{R_MIN:.0f}_{field}_{t}_meta.json'
                     if t else f'cbl_v5_joint_R{R_MIN:.0f}_{field}_meta.json']:
            path = os.path.join(CHAIN_DIR, stem)
            if os.path.exists(path):
                with open(path) as f:
                    return json.load(f)
    return None


def make_figure(field, meta, tag_suffix='', data_dir=None, data_label=''):
    xmap = meta['x_map']
    om_map, s8_map = xmap['Omega_m'], xmap['sigma8']
    sl_map, of_map = xmap['B_slope'], xmap['B_offset']
    chi2_dof = meta['chi2_per_dof_map']

    summary = meta['summary']
    s8_med = summary['sigma8']['median']
    s8_dev = summary['sigma8']['sigma_dev']
    om_med = summary['Omega_m']['median']
    om_dev = summary['Omega_m']['sigma_dev']

    fig = plt.figure(figsize=(15, 11))
    outer = gridspec.GridSpec(2, 2, hspace=0.28, wspace=0.25,
                              left=0.07, right=0.97, top=0.90, bottom=0.06)

    for idx, frac in enumerate(SUBSAMPLES):
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[idx], height_ratios=[3, 1], hspace=0.0)
        ax_main = fig.add_subplot(inner[0])
        ax_res  = fig.add_subplot(inner[1], sharex=ax_main)

        R_data, vsf_data, err_data = load_data(frac, field, data_dir=data_dir)
        fit_mask = R_data >= R_MIN

        # Crop x-axis to where the data actually lives — avoids the visually
        # confusing region where Vdn diverges upward at R → 0 while the data
        # peaks and drops toward the resolution limit.
        x_lo = max(2.5, R_data.min() * 0.95)
        x_hi = max(20.0, R_data.max() * 1.05)

        # Two model grids: one for fit range (drawn solid), one for exclusion
        # zone (drawn dotted, less prominent). Stops at where data starts.
        R_grid_fit = np.logspace(np.log10(R_MIN), np.log10(x_hi), 120)
        R_grid_exc = np.logspace(np.log10(x_lo), np.log10(R_MIN), 80)

        vsf_map_fit   = vdn_curve(R_grid_fit, om_map, s8_map, sl_map, of_map)
        vsf_truth_fit = vdn_curve(R_grid_fit, OMEGA_M_TRUE, SIGMA8_TRUE, sl_map, of_map)
        vsf_map_exc   = vdn_curve(R_grid_exc, om_map, s8_map, sl_map, of_map)
        vsf_truth_exc = vdn_curve(R_grid_exc, OMEGA_M_TRUE, SIGMA8_TRUE, sl_map, of_map)

        # ── Main panel ──
        ax_main.axvspan(0.1, R_MIN, color='#f0f0f0', zorder=0)
        ax_main.axvline(R_MIN, color='#aaa', ls='-', lw=1.2, zorder=1)

        # Excluded data (faded)
        exc = ~fit_mask
        if np.any(exc):
            ax_main.errorbar(R_data[exc], vsf_data[exc], yerr=err_data[exc],
                             fmt='o', ms=4, color=COLORS[frac], alpha=0.25,
                             ecolor=COLORS[frac], elinewidth=1, capsize=2, zorder=3)
        # Fit-range data (solid)
        ax_main.errorbar(R_data[fit_mask], vsf_data[fit_mask], yerr=err_data[fit_mask],
                         fmt='o', ms=7, color=COLORS[frac], ecolor=COLORS[frac],
                         elinewidth=1.5, capsize=3, mec='k', mew=0.5, zorder=5,
                         label=f'Data {LABELS[frac]} (MPS={MPS[frac]})')

        # Model curves: solid in fit range, dotted (light) in exclusion zone
        ax_main.plot(R_grid_exc, vsf_map_exc, ':', color='#2c3e50',
                     lw=1.2, alpha=0.5, zorder=2)
        ax_main.plot(R_grid_exc, vsf_truth_exc, ':', color='#c0392b',
                     lw=1.2, alpha=0.5, zorder=2)
        ax_main.plot(R_grid_fit, vsf_map_fit, '-', color='#2c3e50', lw=2.5, zorder=4,
                     label=f'Best fit ($\\Omega_m$={om_map:.3f}, $\\sigma_8$={s8_map:.3f})')
        ax_main.plot(R_grid_fit, vsf_truth_fit, '--', color='#c0392b', lw=2, zorder=3,
                     label=f'Truth ($\\Omega_m$={OMEGA_M_TRUE}, $\\sigma_8$={SIGMA8_TRUE})')

        ax_main.set_yscale('log')
        ax_main.set_xscale('log')
        ax_main.set_ylim(5e-7, 5e-4)
        ax_main.set_xlim(x_lo, x_hi)
        ax_main.set_ylabel(r'dn/dln$R_v$ [$h^3$Mpc$^{-3}$]')
        ax_main.tick_params(labelbottom=False)
        ax_main.grid(True, ls='--', alpha=0.2, which='major')
        ax_main.legend(fontsize=8, loc='upper right',
                       framealpha=0.92, edgecolor='#ccc')
        ax_main.set_title(f'{field}  {LABELS[frac]}  (MPS = {MPS[frac]} $h^{{-1}}$Mpc)',
                          fontsize=12, fontweight='bold', pad=8)

        # Per-subsample chi2 badge
        vsf_map_at_data = vdn_curve(R_data[fit_mask], om_map, s8_map, sl_map, of_map)
        log_d = np.log10(vsf_data[fit_mask])
        log_e = err_data[fit_mask] / (vsf_data[fit_mask] * np.log(10))
        res_map_fit = (log_d - np.log10(vsf_map_at_data)) / log_e
        chi2_sub = np.sum(res_map_fit**2)
        nbins_sub = len(res_map_fit)
        ax_main.text(0.03, 0.06,
                     f'$\\chi^2$/bin = {chi2_sub/nbins_sub:.2f}  ({nbins_sub} bins)',
                     transform=ax_main.transAxes, fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#999', alpha=0.9))

        # ── Residual panel (fit range only) ──
        vsf_truth_at_data = vdn_curve(R_data[fit_mask], OMEGA_M_TRUE, SIGMA8_TRUE,
                                       sl_map, of_map)
        res_truth_fit = (log_d - np.log10(vsf_truth_at_data)) / log_e

        ax_res.axvspan(1.0, R_MIN, color='#f0f0f0', zorder=0)
        ax_res.axvline(R_MIN, color='#aaa', ls='-', lw=1.2, zorder=1)
        ax_res.axhline(0, color='k', lw=0.8, zorder=2)
        ax_res.fill_between([1, 25], -1, 1, color='#27ae60', alpha=0.08, zorder=0)
        ax_res.fill_between([1, 25], -2, 2, color='#f39c12', alpha=0.06, zorder=0)

        ax_res.plot(R_data[fit_mask], res_map_fit, 's', ms=6, color='#2c3e50',
                    mec='k', mew=0.4, zorder=5, label='Best fit')
        ax_res.plot(R_data[fit_mask], res_truth_fit, 'D', ms=5, color='#c0392b',
                    mec='#c0392b', mew=0.4, alpha=0.8, zorder=4, label='Truth')

        ax_res.set_xscale('log')
        ax_res.set_xlim(x_lo, x_hi)
        ax_res.set_ylim(-3.5, 3.5)
        ax_res.set_yticks([-3, -2, -1, 0, 1, 2, 3])
        ax_res.set_xlabel(r'$R_v$ [$h^{-1}$Mpc]')
        ax_res.set_ylabel(r'$(d - m)/\sigma$', fontsize=10)
        ax_res.grid(True, ls='--', alpha=0.2, which='major')
        if idx == 0:
            ax_res.legend(fontsize=7.5, loc='upper left', ncol=2)

    fig.suptitle(
        f'{field}:  v5 joint best fit  vs  Aletheia truth\n'
        f'$\\sigma_8$ = {s8_med:.3f} ({s8_dev:+.1f}$\\sigma$),   '
        f'$\\Omega_m$ = {om_med:.3f} ({om_dev:+.1f}$\\sigma$),   '
        f'$\\chi^2/\\nu$ = {chi2_dof:.2f}   '
        f'(grey = excluded, R < {R_MIN:.0f} $h^{{-1}}$Mpc)',
        fontsize=13, y=0.97, va='top')

    # tag_suffix is e.g. "_simone" (default) or "_v4cleaned"
    suffix = tag_suffix if tag_suffix else '_simone'
    outpath = os.path.join(OUT_DIR, f'joint_calibration_{field}{suffix}.png')
    fig.savefig(outpath, dpi=220, bbox_inches='tight')
    print(f'Saved: {outpath}')
    plt.close(fig)
    return outpath


# ─── Main ────────────────────────────────────────────────────────────
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--data-dir', default=None,
                help='Override data dir. Use data/ForArun_v4cleaned for v4-cleaned VSFs.')
ap.add_argument('--meta-tag', default='cluster',
                help='Meta-file tag; default `cluster`. For the v4-cleaned smoke '
                     'run use `v4cleaned_smoke`.')
ap.add_argument('--out-suffix', default='',
                help='Suffix appended to output filename. e.g. `_v4cleaned`.')
args = ap.parse_args()

print("Loading v5 joint metadata...")
meta_nrec50 = load_meta('NREC50', tag=args.meta_tag)
meta_bit    = load_meta('BIT',    tag=args.meta_tag)

if meta_nrec50:
    print(f"\nNREC50 MAP: Om={meta_nrec50['x_map']['Omega_m']:.4f}, "
          f"s8={meta_nrec50['x_map']['sigma8']:.4f}")
    make_figure('NREC50', meta_nrec50,
                tag_suffix=args.out_suffix, data_dir=args.data_dir)
else:
    print(f"No NREC50 meta found for tag={args.meta_tag} — skip")

if meta_bit:
    print(f"\nBIT MAP: Om={meta_bit['x_map']['Omega_m']:.4f}, "
          f"s8={meta_bit['x_map']['sigma8']:.4f}")
    make_figure('BIT', meta_bit,
                tag_suffix=args.out_suffix, data_dir=args.data_dir)
else:
    print(f"No BIT meta found for tag={args.meta_tag} — skip")

print("\nDone.")
