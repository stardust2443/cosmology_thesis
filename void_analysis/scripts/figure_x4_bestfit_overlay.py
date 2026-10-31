#!/usr/bin/env python3
"""
figure_x4_bestfit_overlay.py

Thesis Figure X.4: best-fit theory vs data overlay for a representative
subsample, including the 1σ posterior band sampled from the v5 joint chain.

Single panel, focused, publication-quality. Visual proof that the MCMC
posterior median model passes through the data within Poisson errors.

Default subsample: NREC50 5%. CLI: --field, --frac, --nsamples.

Run:
    python scripts/figure_x4_bestfit_overlay.py
    python scripts/figure_x4_bestfit_overlay.py --frac 2_5p
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CBL_HOME = os.environ.get('CBL_HOME', '/Users/arun/projects/CosmoBolognaLib')
sys.path.insert(0, os.path.join(CBL_HOME, 'Python'))
os.environ.setdefault('DYLD_LIBRARY_PATH', CBL_HOME)
import CosmoBolognaLib as cbl

HERE = os.path.dirname(os.path.abspath(__file__))
FA   = os.path.join(HERE, '..', 'data', 'ForArun')
CHAIN_DIR = os.path.join(HERE, '..', 'data', 'mcmc_chains')
OUT_DIR   = os.path.join(HERE, '..', 'plots', 'cbl_mcmc')
os.makedirs(OUT_DIR, exist_ok=True)

OMEGA_M_TRUE, SIGMA8_TRUE = 0.315, 0.811
H0, OMEGA_B, N_S, TAU = 0.6736, 0.0493, 0.9649, 0.0925
SCALAR_AMP, SCALAR_PIVOT = 2.194e-9, 0.05
DELTA_NL, B_EFF = -0.7, 1.0
R_MIN = 7.0

FA_MAP = {'2_5p': '025', '5p': '05', '10p': '10', '20p': '20'}
MPS = {'20p': 1.16, '10p': 1.47, '5p': 1.85, '2_5p': 2.33}
LABELS = {'20p': '20%', '10p': '10%', '5p': '5%', '2_5p': '2.5%'}
COLORS = {'20p': '#8e44ad', '10p': '#2980b9', '5p': '#27ae60', '2_5p': '#d4ac0d'}


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


def load_data(frac, field):
    fpath = os.path.join(FA, f'vsf_dm_250cube_ss0p{FA_MAP[frac]}_{field}_'
                              'SM0p00_CS0p79_TH-0p80_OV00.dat')
    df = pd.read_csv(fpath, sep=r'\s+', comment='#', names=['r', 'vsf', 'err'])
    m = df['vsf'] > 0
    return df.loc[m, 'r'].values, df.loc[m, 'vsf'].values, df.loc[m, 'err'].values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--field', default='NREC50', choices=['NREC50', 'BIT'])
    ap.add_argument('--frac', default='5p', choices=['20p', '10p', '5p', '2_5p'])
    ap.add_argument('--nsamples', type=int, default=200,
                    help='number of chain samples to draw the posterior band')
    args = ap.parse_args()

    chain_path = os.path.join(CHAIN_DIR,
        f'cbl_v5_joint_R{R_MIN:.0f}_{args.field}_cluster_flat.npy')
    meta_path  = os.path.join(CHAIN_DIR,
        f'cbl_v5_joint_R{R_MIN:.0f}_{args.field}_cluster_meta.json')
    chain = np.load(chain_path)
    with open(meta_path) as f:
        meta = json.load(f)

    om_map = meta['x_map']['Omega_m']
    s8_map = meta['x_map']['sigma8']
    sl_map = meta['x_map']['B_slope']
    of_map = meta['x_map']['B_offset']

    s_summary = meta['summary']
    s8_med, s8_dev = s_summary['sigma8']['median'], s_summary['sigma8']['sigma_dev']
    om_med, om_dev = s_summary['Omega_m']['median'], s_summary['Omega_m']['sigma_dev']

    R_data, vsf_data, err_data = load_data(args.frac, args.field)
    fit_mask = R_data >= R_MIN

    R_grid = np.logspace(np.log10(1.5), np.log10(20), 150)

    # MAP and truth curves
    vsf_map = vdn_curve(R_grid, om_map, s8_map, sl_map, of_map)
    vsf_truth = vdn_curve(R_grid, OMEGA_M_TRUE, SIGMA8_TRUE, sl_map, of_map)

    # Posterior band: sample N draws from the chain, evaluate curve, take 16/84
    print(f"Drawing {args.nsamples} posterior samples for the 1σ band...")
    rng = np.random.default_rng(42)
    idx = rng.choice(len(chain), size=args.nsamples, replace=False)
    band = np.zeros((args.nsamples, len(R_grid)))
    for i, j in enumerate(idx):
        om, s8, sl, of = chain[j]
        try:
            band[i] = vdn_curve(R_grid, om, s8, sl, of)
        except Exception:
            band[i] = np.nan
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{args.nsamples}")

    band_lo = np.nanpercentile(band, 16, axis=0)
    band_hi = np.nanpercentile(band, 84, axis=0)
    band_med = np.nanpercentile(band, 50, axis=0)

    # Per-subsample chi2 in fit range (using MAP)
    vsf_map_at_data = vdn_curve(R_data[fit_mask], om_map, s8_map, sl_map, of_map)
    log_d = np.log10(vsf_data[fit_mask])
    log_e = err_data[fit_mask] / (vsf_data[fit_mask] * np.log(10))
    res = (log_d - np.log10(vsf_map_at_data)) / log_e
    chi2 = np.sum(res**2)
    nbins = len(res)

    # ── Plot ──
    plt.rcParams.update({
        'font.size': 12, 'axes.labelsize': 14, 'xtick.labelsize': 12,
        'ytick.labelsize': 12, 'legend.fontsize': 11, 'figure.facecolor': 'white',
    })

    fig, (ax_main, ax_res) = plt.subplots(
        2, 1, figsize=(9.5, 7.5), sharex=True,
        gridspec_kw=dict(height_ratios=[3, 1], hspace=0.05))

    color = COLORS[args.frac]

    # Exclusion zone
    ax_main.axvspan(1.0, R_MIN, color='#f0f0f0', zorder=0)
    ax_main.axvline(R_MIN, color='#aaa', ls='-', lw=1.5, zorder=1)

    # Posterior band
    ax_main.fill_between(R_grid, band_lo, band_hi, color='#3498db', alpha=0.25,
                          zorder=2, label='1$\\sigma$ posterior band')

    # MAP curve
    ax_main.plot(R_grid, vsf_map, '-', color='#1c2833', lw=2.5, zorder=4,
                  label=f'Best fit ($\\Omega_m$={om_map:.3f}, $\\sigma_8$={s8_map:.3f})')

    # Truth curve
    ax_main.plot(R_grid, vsf_truth, '--', color='#c0392b', lw=2, zorder=3,
                  label=f'Aletheia truth ($\\Omega_m$={OMEGA_M_TRUE}, $\\sigma_8$={SIGMA8_TRUE})')

    # Data — faded outside fit range
    exc = ~fit_mask
    if np.any(exc):
        ax_main.errorbar(R_data[exc], vsf_data[exc], yerr=err_data[exc],
                         fmt='o', ms=5, color=color, alpha=0.25,
                         ecolor=color, elinewidth=1.2, capsize=2.5, zorder=5)
    ax_main.errorbar(R_data[fit_mask], vsf_data[fit_mask], yerr=err_data[fit_mask],
                     fmt='o', ms=9, color=color, ecolor=color,
                     elinewidth=2, capsize=4, mec='k', mew=0.7, zorder=6,
                     label=f'Data: {args.field} {LABELS[args.frac]}')

    ax_main.set_yscale('log')
    ax_main.set_xscale('log')
    ax_main.set_xlim(1.5, 20)
    ax_main.set_ylim(5e-7, 5e-4)
    ax_main.set_ylabel(r'$dn / d\ln R_v$  [$h^3$ Mpc$^{-3}$]')
    ax_main.grid(True, ls='--', alpha=0.2, which='major')
    ax_main.legend(loc='upper right', fontsize=10.5, framealpha=0.95, edgecolor='#ccc')

    ax_main.set_title(
        f'{args.field} {LABELS[args.frac]} subsample  (MPS = {MPS[args.frac]} $h^{{-1}}$Mpc)\n'
        f'v5 joint posterior:  $\\sigma_8$ = {s8_med:.3f} ({s8_dev:+.2f}$\\sigma$),  '
        f'$\\Omega_m$ = {om_med:.3f} ({om_dev:+.2f}$\\sigma$),  '
        f'$\\chi^2/\\nu_{{sub}}$ = {chi2/nbins:.2f}',
        fontsize=12, fontweight='bold', pad=10)

    # ── Residual panel ──
    res_truth_at_data = vdn_curve(R_data[fit_mask], OMEGA_M_TRUE, SIGMA8_TRUE,
                                   sl_map, of_map)
    res_t = (log_d - np.log10(res_truth_at_data)) / log_e

    ax_res.axvspan(1.0, R_MIN, color='#f0f0f0', zorder=0)
    ax_res.axvline(R_MIN, color='#aaa', ls='-', lw=1.5, zorder=1)
    ax_res.axhline(0, color='k', lw=0.8, zorder=2)
    ax_res.fill_between([1, 25], -1, 1, color='#27ae60', alpha=0.10, zorder=0)
    ax_res.fill_between([1, 25], -2, 2, color='#f39c12', alpha=0.06, zorder=0)

    ax_res.plot(R_data[fit_mask], res, 's', ms=8, color='#1c2833',
                mec='k', mew=0.5, zorder=5, label='Best fit')
    ax_res.plot(R_data[fit_mask], res_t, 'D', ms=7, color='#c0392b',
                mec='#c0392b', mew=0.5, alpha=0.85, zorder=4, label='Truth')

    ax_res.set_xscale('log')
    ax_res.set_ylim(-3.5, 3.5)
    ax_res.set_yticks([-3, -2, -1, 0, 1, 2, 3])
    ax_res.set_xlabel(r'$R_v$  [$h^{-1}$ Mpc]')
    ax_res.set_ylabel(r'$(d-m)/\sigma$')
    ax_res.grid(True, ls='--', alpha=0.2, which='major')
    ax_res.legend(loc='lower right', fontsize=10, ncol=2)

    out = os.path.join(OUT_DIR,
        f'figure_x4_bestfit_{args.field}_{args.frac}.png')
    fig.savefig(out, dpi=240, bbox_inches='tight')
    print(f"\nSaved: {out}")
    plt.close(fig)


if __name__ == '__main__':
    main()
