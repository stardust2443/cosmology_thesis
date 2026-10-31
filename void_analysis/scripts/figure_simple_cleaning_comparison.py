#!/usr/bin/env python3
"""
figure_simple_cleaning_comparison.py

Compare the Simone's prescribed simple two-step cleaning (clean_voids_simple.py)
against Simone's reference VSF and against our prior cleaning variants on
5% NREC50.

Series shown (scatter; no per-variant smoothing fits — only the Vdn-truth
dashed line is a smooth curve):
  - Vdn theory at Aletheia truth
  - Simone reference
  - simple cleaning, overlap = center
  - simple cleaning, overlap = touch
  - (optional: v4 production, CBL native — toggled via CLI)

Output:
  plots/diagnostics/simple_cleaning_vs_simone_5p.png       (simple-only)
  plots/diagnostics/all_cleaning_variants_vs_simone_5p.png (all variants)
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
    plot_scatter_with_fit, apply_axis_style,
)

DATA = os.path.join(HERE, '..', 'data')
SIMONE_REF = os.path.join(DATA, 'ForArun',
    'vsf_dm_250cube_ss0p05_NREC50_SM0p00_CS0p79_TH-0p80_OV00.dat')
SIMPLE_TOUCH  = os.path.join(DATA, 'cleaned_simple',
    'cleaned_simple_nrec50_cat_5p_touch.csv')
SIMPLE_CENTER = os.path.join(DATA, 'cleaned_simple',
    'cleaned_simple_nrec50_cat_5p_center.csv')
#V4_PROD       = os.path.join(DATA, 'cleaned', 'cleaned_v4_recon_cat_5p.csv')
#V4_WS         = os.path.join(DATA, 'cleaned',
#    'cleaned_v4_overlap_ws_recon_cat_5p.csv')
CBL_CENTRAL   = os.path.join(DATA, 'cleaned_cbl',
    'cleaned_cbl_nrec50_cat_5p.csv')
OUT_DIR = os.path.join(HERE, '..', 'plots', 'diagnostics')

#V4_MULT_5P = 1.43

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
    return counts, counts / (V_EFF * DLNR), np.sqrt(counts) / (V_EFF * DLNR)


def load_radii(path, col=None):
    df = pd.read_csv(path)
    if col is None:
        col = 'R_resc' if 'R_resc' in df.columns else 'R'
    return df[col].values, len(df)


def median_ratio(vsf, sim_by_r, r_cut=R_MIN):
    rs = []
    for r, v in zip(FORARUN_BIN_CENTERS, vsf):
        if r < r_cut: continue
        vs = sim_by_r.get(round(r, 3))
        if vs and vs > 0 and v > 0:
            rs.append(v / vs)
    return float(np.median(rs)) if rs else float('nan')


def make_plot(variants, out_path, title_extra=''):
    plt.rcParams.update(UNIFIED_RCPARAMS)
    fig, ax = plt.subplots(figsize=(12, 8))

    # Vdn theory
    R_grid = np.logspace(np.log10(XLIM[0]), np.log10(XLIM[1]), 200)
    ax.plot(R_grid, vdn_truth_curve(R_grid),
            ls='--', color=VDN_TRUTH_COLOR, lw=2.2, zorder=3,
            label=(f'Vdn at Aletheia true cosmology:'
                   f'($\\Omega_m={OMEGA_M_TRUE}, \\sigma_8={SIGMA8_TRUE}$)'))

    # Simone
    sim = pd.read_csv(SIMONE_REF, sep=r'\s+', comment='#',
                      names=['r', 'vsf', 'err'])
    sim = sim[sim['vsf'] > 0]
    plot_scatter_with_fit(
        ax, sim['r'].values, sim['vsf'].values, sim['err'].values,
        fit_fn=None,
        color=SIMONE_COLOR, marker=SIMONE_MARKER, ms=SIMONE_MS + 1,
        scatter_zorder=10,
        label=f'Simone Reference 5% data')

    # Each variant — pure scatter
    for v in variants:
        plot_scatter_with_fit(
            ax, FORARUN_BIN_CENTERS, v['vsf'], v['err'],
            fit_fn=None,
            color=v['color'], marker=v['marker'], ms=8,
            label=(f"{v['label']}   "
                   f"(N={v['n_total']}, $N_{{R\\geq 7}}$={v['n_R7']}, "
                   f"$v/v_\\mathrm{{sim}}$: all={v['med_ratio_all']:.2f}, "
                   f"$R\\geq 7$={v['med_ratio']:.2f})"),
        )

    apply_axis_style(
        ax,
        title=(
            f'5% 50 OTs cleaning comparison vs Simone reference{title_extra}\n'
            'Pure scatter;  the dashed line: Vdn at Aletheia true cosmology. DELTA_TH= -0.80'
            #'Grey: $R < 7$ $h^{-1}$Mpc (excluded from MCMC).'
        ),
    )

    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


# ─── Build variant list ──────────────────────────────────────────────
sim = pd.read_csv(SIMONE_REF, sep=r'\s+', comment='#',
                  names=['r', 'vsf', 'err'])
sim = sim[sim['vsf'] > 0]
sim_by_r = dict(zip(sim['r'].round(3), sim['vsf']))


def make_variant(path, label, color, marker, mult=1.0, col=None):
    radii, n_total = load_radii(path, col=col)
    radii = radii * mult
    counts, vsf, err = bin_vsf(radii)
    return dict(label=label, color=color, marker=marker,
                vsf=vsf, err=err, counts=counts, n_total=n_total,
                n_R7=int((radii >= R_MIN).sum()),
                med_ratio=median_ratio(vsf, sim_by_r),
                med_ratio_all=median_ratio(vsf, sim_by_r, r_cut=0.0))


variants_simple_only = [
    make_variant(SIMPLE_CENTER,
                 'simple cleaning, overlap = CENTER',
                 "#1692a0", 'D'),
    make_variant(SIMPLE_TOUCH,
                 'simple cleaning, overlap = TOUCH',
                 '#d35400', '^'),
]

variants_all = [
    #make_variant(V4_PROD,
    #             f'v4 production: R_resc ×MULT {V4_MULT_5P} (old)',
    #             '#2980b9', 'P', mult=V4_MULT_5P, col='R_resc'),
    #make_variant(V4_WS,
    #             f'v4 watershed-overlap ×MULT {V4_MULT_5P} (old)',
    #             '#e67e22', 'X', mult=V4_MULT_5P, col='R_resc'),
    make_variant(CBL_CENTRAL,
                 'CBL native, ol_crit = central, ratio = 1.5',
                 '#27ae60', '*'),
    make_variant(SIMPLE_CENTER,
                 'SIMPLE, overlap = CENTER (option i)',
                 '#1692a0', 'D'),
    make_variant(SIMPLE_TOUCH,
                 'SIMPLE, overlap = TOUCH (option ii)',
                 '#d35400', '^'),
]

# ─── Generate plots ──────────────────────────────────────────────────
make_plot(variants_simple_only,
          os.path.join(OUT_DIR, 'simple_cleaning_vs_simone_5p3.png'),
          title_extra=': simple algorithm only')
# make_plot(variants_all,
#           os.path.join(OUT_DIR, 'all_cleaning_variants_vs_simone_5p0.png'),
#           title_extra=': all cleaning variants')

# ─── Summary table ───────────────────────────────────────────────────
print(f"\n{'='*88}")
print('CLEANING COMPARISON SUMMARY  (5% NREC50, Simone reference)')
print('='*88)
print(f"{'variant':<50}{'N_tot':>8}{'v/v_sim all':>14}")
#print(f"{'variant':<50}{'N_tot':>8}{'N(R≥7)':>10}{'v/v_sim all':>14}{'v/v_sim R≥7':>14}")
print('-' * 96)
for v in variants_all:
    print(f"{v['label']:<50}{v['n_total']:>8}{v['n_R7']:>10}"
          f"{v['med_ratio_all']:>14.3f}{v['med_ratio']:>14.3f}")
