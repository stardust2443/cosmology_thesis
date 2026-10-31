#!/usr/bin/env python3
"""
explore_cleaning.py

Interactive parameter-exploration tool for the cleaning pipeline.
Runs the pipeline with multiple parameter values and produces an
overlay plot showing how each choice affects the VSF.

Usage examples:

  # Sweep delta_th (rescaling threshold) at default values
  python explore_cleaning.py --param dth --values -0.80,-0.70,-0.60,-0.50

  # Sweep the TH cut on delta_avg
  python explore_cleaning.py --param th_cut --values -0.90,-0.80,-0.70

  # Sweep border cut size
  python explore_cleaning.py --param border --values 10,20,30,40

  # Compare radius columns (no rescaling, just choice of radius)
  python explore_cleaning.py --param radius_col --values radius,r_ext

  # Compare overlap radius scale factors
  python explore_cleaning.py --param overlap_factor --values 0.8,1.0,1.2,1.5

  # Compare a global radius multiplier (for "artificial shift" experiments)
  python explore_cleaning.py --param radius_mult --values 1.0,1.10,1.13,1.20

Pick the subsample with --frac (default 5p). Defaults are the v2 production
values: dth=-0.70, th_cut=-0.80, border=20, radius_col=R_resc, overlap=1.0.
"""
import os, sys, argparse, numpy as np, pandas as pd
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RAW  = os.path.join(HERE, '..', 'data', 'raw')
SUBS = os.path.join(HERE, '..', 'data', 'subsamples')
FA   = os.path.join(HERE, '..', 'data', 'ForArun')
OUT  = os.path.join(HERE, '..', 'plots', 'cleaning_validation')
os.makedirs(OUT, exist_ok=True)

RAW_COLS  = ['x','y','z','radius','r_ext','delta_2MPS','delta_avg']
BOX       = 250.0

FORARUN_BIN_CENTERS = np.array([
    1.591446e+00, 1.791403e+00, 2.016484e+00, 2.269845e+00, 2.555040e+00,
    2.876068e+00, 3.237432e+00, 3.644199e+00, 4.102074e+00, 4.617479e+00,
    5.197642e+00, 5.850699e+00, 6.585810e+00, 7.413284e+00, 8.344726e+00,
    9.393199e+00, 1.057341e+01, 1.190190e+01, 1.339732e+01, 1.508062e+01,
])
log_c = np.log(FORARUN_BIN_CENTERS)
DLNR  = float(np.median(np.diff(log_c)))
EDGES = np.exp(np.append(log_c - DLNR/2, log_c[-1] + DLNR/2))
V_EFF = BOX**3
FA_MAP = {'2_5p':'025','5p':'05','10p':'10','20p':'20'}

# ---------- core pipeline ---------- #

def rescale(tree, parts, center, R_ws, rho_mean, dth):
    R_max = max(R_ws * 2.0, 8.0)
    idx = tree.query_ball_point(center, R_max)
    if len(idx) < 5:
        return R_ws
    dists = np.sort(np.linalg.norm(parts[idx] - center, axis=1))
    R_grid = np.linspace(0.5, R_max, 80)
    counts = np.searchsorted(dists, R_grid)
    delta_cum = (counts / (rho_mean * (4.0/3) * np.pi * R_grid**3)) - 1.0
    above = delta_cum >= dth
    cross_up = np.where(np.diff(above.astype(int)) == 1)[0]
    if len(cross_up) == 0:
        return R_ws
    i = cross_up[-1]
    d0, d1 = delta_cum[i], delta_cum[i+1]
    if d1 == d0:
        return R_grid[i]
    return R_grid[i] + (R_grid[i+1] - R_grid[i]) * (dth - d0) / (d1 - d0)


def remove_overlaps(coords, radii):
    if len(coords) == 0:
        return np.array([], dtype=bool)
    keep = np.ones(len(coords), dtype=bool)
    tree = cKDTree(coords)
    rmax = radii.max() * 2.1
    for i in np.argsort(-radii):
        if not keep[i]: continue
        for j in tree.query_ball_point(coords[i], rmax):
            if j == i or not keep[j] or radii[j] >= radii[i]: continue
            if np.linalg.norm(coords[i]-coords[j]) < radii[i] + radii[j]:
                keep[j] = False
    return keep


def vsf(r):
    c, _ = np.histogram(r, bins=EDGES)
    return c / (V_EFF * DLNR), c


def run_pipeline(df_raw, parts, tree, rho_mean, box_origin, params):
    """Run cleaning with a specific parameter set, return cleaned df."""
    df = df_raw.copy()

    # Compute rescaled radius if needed
    if params['radius_col'] == 'R_resc':
        centers = df[['x','y','z']].values
        radii_ws = df['radius'].values
        R_resc = np.array([
            rescale(tree, parts, centers[i], radii_ws[i], rho_mean, params['dth'])
            for i in range(len(df))
        ])
        df['R_resc'] = R_resc

    # TH cut on delta_avg
    df = df[df['delta_avg'] <= params['th_cut']].copy()

    # Pick the radius column for cleaning + VSF, optionally multiply
    rad = df[params['radius_col']].values * params['radius_mult']
    df['_rad_for_clean'] = rad * params['overlap_factor']
    df['_rad_for_vsf']   = rad

    # Overlap removal
    keep = remove_overlaps(df[['x','y','z']].values, df['_rad_for_clean'].values)
    df = df[keep]

    # Border cut
    lo, hi = box_origin + params['border'], box_origin + BOX - params['border']
    df = df[(df['x']>lo)&(df['x']<hi)&(df['y']>lo)&(df['y']<hi)&(df['z']>lo)&(df['z']<hi)]
    return df


def load_fa(frac):
    f = os.path.join(FA, f'vsf_dm_250cube_ss0p{FA_MAP[frac]}_NREC50_SM0p00_CS0p79_TH-0p80_OV00.dat')
    df = pd.read_csv(f, sep=r'\s+', comment='#', names=['r','vsf','err'])
    return df[df['vsf'] > 0]


# ---------- main ---------- #

DEFAULTS = {
    'dth':            -0.70,
    'th_cut':         -0.80,
    'border':         20.0,
    'radius_col':     'R_resc',
    'overlap_factor': 1.0,
    'radius_mult':    1.0,
}

PARAM_TYPE = {
    'dth':            float,
    'th_cut':         float,
    'border':         float,
    'radius_col':     str,
    'overlap_factor': float,
    'radius_mult':    float,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--frac', default='5p',
                   help='Subsample (20p,10p,5p,2_5p)')
    p.add_argument('--param', required=True,
                   choices=list(DEFAULTS.keys()),
                   help='Parameter to vary')
    p.add_argument('--values', required=True,
                   help='Comma-separated list of values')
    p.add_argument('--out',   default=None, help='Output PNG path')
    args = p.parse_args()

    cast = PARAM_TYPE[args.param]
    values = [cast(v) for v in args.values.split(',')]

    print(f'Loading raw catalog and particles for {args.frac}...')
    raw_file = os.path.join(RAW, f'voids_recon_50_bitvf_aletheia_{args.frac}')
    df_raw = pd.read_csv(raw_file, sep=r'\s+', comment='#', names=RAW_COLS)
    df_raw = df_raw[df_raw['radius'] > 0].reset_index(drop=True)
    box_origin = 375.0 if df_raw['x'].min() > 300 else 0.0

    sub_file = os.path.join(SUBS, f'DM_particles_ss{args.frac}_250cube.dat')
    parts = np.loadtxt(sub_file, usecols=(0,1,2))
    rho_mean = len(parts) / BOX**3
    tree = cKDTree(parts) if any(args.param in ('dth',) or DEFAULTS['radius_col']=='R_resc' for _ in [0]) else None
    fa = load_fa(args.frac)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(values)))

    print(f'\nSweeping {args.param} ∈ {values}')
    print(f'Defaults: {DEFAULTS}')
    print(f'\n{"value":>10}{"N":>8}{"peak":>8}{"SSR":>10}')
    print('-'*40)

    fa_dict = dict(zip(np.round(fa['r'].values,4), fa['vsf'].values))
    cb = FORARUN_BIN_CENTERS

    for value, color in zip(values, cmap):
        params = DEFAULTS.copy()
        params[args.param] = value

        df_clean = run_pipeline(df_raw, parts, tree, rho_mean, box_origin, params)
        v, c = vsf(df_clean['_rad_for_vsf'].values)
        e = np.sqrt(c) / (V_EFF * DLNR)
        m = v > 0

        if not m.any():
            print(f'{str(value):>10}{0:>8}{"-":>8}{"-":>10}')
            continue

        peak = cb[m][np.argmax(v[m])]
        ssr_sum = 0; n = 0
        for i, ce in enumerate(cb):
            if v[i] <= 0: continue
            vf = fa_dict.get(round(ce, 4))
            if vf is None or vf <= 0: continue
            ssr_sum += (np.log10(v[i]) - np.log10(vf))**2
            n += 1
        ssr = ssr_sum / max(n, 1)

        ax.errorbar(cb[m], v[m], yerr=e[m], fmt='-o',
                    color=color, ms=4, lw=1.7,
                    label=f'{args.param}={value}  N={len(df_clean)}  pk={peak:.2f}  SSR={ssr:.3f}')

        print(f'{str(value):>10}{len(df_clean):>8}{peak:>8.2f}{ssr:>10.4f}')

    # Simone reference
    ax.errorbar(fa['r'], fa['vsf'], yerr=fa['err'], fmt='k^', ms=6,
                lw=2, label='Simone data (OV00)')
    pk_fa = fa['r'].values[np.argmax(fa['vsf'].values)]

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$R_v\ [h^{-1}\mathrm{Mpc}]$', fontsize=13)
    ax.set_ylabel(r'$\mathrm{d}n/\mathrm{d}\ln R_v\ [h^3 \mathrm{Mpc}^{-3}]$', fontsize=13)
    ax.set_xlim(1.5, 16); ax.set_ylim(1e-7, 1e-2)
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.legend(fontsize=10, loc='lower left')
    ax.set_title(f'NREC50 {args.frac}: sweep over {args.param}    '
                 f'(Simone peak={pk_fa:.2f})', fontsize=13)

    plt.tight_layout()
    out = args.out or os.path.join(OUT, f'sweep_{args.param}_{args.frac}.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
