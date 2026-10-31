#!/usr/bin/env python3
"""
figure_sanity_check_cbl.py

CBL-pipeline rebuild of the original Phase-2 sanity-check plot:
  2 panels (BIT left, NREC50 right), all four subsamples overlaid,
  with Vdn at Aletheia truth (from cbl.cosmology.SizeFunction) as the
  single dashed reference curve. No fitting, no MAP — purely model
  prediction at the known cosmology versus the binned data.

Outputs:
  plots/diagnostics/sanity_check_cbl_simone.png       (default, on Simone refs)
  plots/diagnostics/sanity_check_cbl_v4cleaned.png    (--data-dir ForArun_v4cleaned)

CLI:
  python scripts/figure_sanity_check_cbl.py
  python scripts/figure_sanity_check_cbl.py --data-dir data/ForArun_v4cleaned \
      --out-suffix _v4cleaned
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CBL_HOME = os.environ.get('CBL_HOME', '/Users/arun/projects/CosmoBolognaLib')
sys.path.insert(0, os.path.join(CBL_HOME, 'Python'))
os.environ.setdefault('DYLD_LIBRARY_PATH', CBL_HOME)
import CosmoBolognaLib as cbl

HERE = os.path.dirname(os.path.abspath(__file__))
FA   = os.path.join(HERE, '..', 'data', 'ForArun')
OUT  = os.path.join(HERE, '..', 'plots', 'diagnostics')
os.makedirs(OUT, exist_ok=True)

# Aletheia truth + cosmology constants
OMEGA_M_TRUE, SIGMA8_TRUE = 0.315, 0.811
H0, OMEGA_B, N_S, TAU = 0.6736, 0.0493, 0.9649, 0.0925
SCALAR_AMP, SCALAR_PIVOT = 2.194e-9, 0.05
DELTA_NL, B_EFF = -0.7, 1.0
# Bias-correction OFF: slope=0 short-circuits the Contarini rescaling, giving the
# DM-only Vdn prediction. This matches the curve used in the original sanity check.
B_SLOPE, B_OFFSET = 0.0, 1.0

FA_MAP = {'2_5p': '025', '5p': '05', '10p': '10', '20p': '20'}
SUBSAMPLES = ['20p', '10p', '5p', '2_5p']
COLORS = {'20p': '#9b59b6', '10p': '#3498db', '5p': '#2ecc71', '2_5p': '#f1c40f'}
LABELS = {'20p': 'ss = 20%', '10p': 'ss = 10%',
          '5p':  'ss = 5%',  '2_5p': 'ss = 2.5%'}


def vdn_curve_truth(R_grid):
    c = cbl.cosmology.LCDM(OMEGA_M_TRUE, OMEGA_B, 0.0, H0,
                           SCALAR_AMP, SCALAR_PIVOT, N_S, TAU, True)
    c.set_parameter("sigma8", SIGMA8_TRUE)
    sf = cbl.cosmology.SizeFunction(c)
    dc = c.deltac(0.0)
    return np.array([sf.size_function(float(R), 0.0, "Vdn",
                     B_EFF, B_SLOPE, B_OFFSET, DELTA_NL, dc, "EisensteinHu")
                     for R in R_grid])


def load_data(frac, field, data_dir):
    fpath = os.path.join(data_dir,
        f'vsf_dm_250cube_ss0p{FA_MAP[frac]}_{field}_'
        'SM0p00_CS0p79_TH-0p80_OV00.dat')
    df = pd.read_csv(fpath, sep=r'\s+', comment='#', names=['r','vsf','err'])
    m = df['vsf'] > 0
    return df.loc[m, 'r'].values, df.loc[m, 'vsf'].values, df.loc[m, 'err'].values


def make_plot(data_dir, out_suffix, title_suffix):
    R_grid = np.logspace(np.log10(2.5), np.log10(20), 200)
    vdn_truth = vdn_curve_truth(R_grid)

    plt.rcParams.update({
        'font.size': 12, 'axes.labelsize': 14, 'xtick.labelsize': 11,
        'ytick.labelsize': 11, 'legend.fontsize': 10, 'figure.facecolor': 'white',
    })

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6),
                             gridspec_kw=dict(wspace=0.18))
    titles = ['True Displacement Field (BIT)', 'OT Reconstructed (NREC50)']
    fields = ['BIT', 'NREC50']

    for ax, field, ttl in zip(axes, fields, titles):
        # Theory curve
        ax.plot(R_grid, vdn_truth, '--', color='black', lw=2.2, zorder=3,
                label=f'Vdn Theory (CBL, $\\Omega_m$={OMEGA_M_TRUE}, '
                      f'$\\sigma_8$={SIGMA8_TRUE})')

        # Data, all 4 subsamples
        for frac in SUBSAMPLES:
            try:
                R, vsf, err = load_data(frac, field, data_dir)
            except FileNotFoundError:
                continue
            ax.errorbar(R, vsf, yerr=err, fmt='o', ms=6,
                        color=COLORS[frac], ecolor=COLORS[frac],
                        elinewidth=1.3, capsize=2.5, mec='k', mew=0.4,
                        label=LABELS[frac], zorder=5)

        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(2.5, 20); ax.set_ylim(1e-6, 4e-3)
        ax.set_xlabel(r'$R_v$  [$h^{-1}$Mpc]')
        ax.set_ylabel(r'$dn/d\ln R_v$  [$h^3$Mpc$^{-3}$]')
        ax.set_title(ttl, fontsize=14, fontweight='bold', pad=8)
        ax.grid(True, ls='--', alpha=0.25, which='both')
        ax.legend(fontsize=10, loc='upper right', framealpha=0.95)

    fig.suptitle(f'CBL pipeline sanity check  —  Vdn at Aletheia truth vs binned VSF{title_suffix}',
                 fontsize=15, y=1.02)
    out_path = os.path.join(OUT, f'sanity_check_cbl{out_suffix}.png')
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=None,
                    help='Override data dir. Default: data/ForArun (Simone refs).')
    ap.add_argument('--out-suffix', default='_simone',
                    help='Suffix on output filename. Default: _simone.')
    ap.add_argument('--title-suffix', default='  (Simone reference)',
                    help='Suffix on plot suptitle.')
    args = ap.parse_args()

    data_dir = args.data_dir if args.data_dir else FA
    out = make_plot(data_dir, args.out_suffix, args.title_suffix)
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
