#!/usr/bin/env python3
"""
empirical_rescale_sweep.py

Empirical rescaling: for each void, compute the cumulative Eulerian
density profile from the DM particles, find R where <delta>(R) = delta_th.
Sweep delta_th ∈ {-0.90, ..., -0.30} and find the value that best matches
Simone's VSF (peak position + amplitude) across all 4 subsamples.

This bypasses the question of "what density field BitVF used internally"
and asks instead: "what Eulerian threshold reproduces Simone's gold
standard, using only data we have?"
"""
import os, numpy as np, pandas as pd
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
RAW  = os.path.join(HERE, '..', 'data', 'raw')
SUBS = os.path.join(HERE, '..', 'data', 'subsamples')
FA   = os.path.join(HERE, '..', 'data', 'ForArun')
CLN  = os.path.join(HERE, '..', 'data', 'cleaned')
os.makedirs(CLN, exist_ok=True)

RAW_COLS  = ['x','y','z','radius','r_ext','delta_2MPS','delta_avg']
BOX       = 250.0
BORDER    = 20.0

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


def rescale_one(tree, parts, center, R_ws, rho_mean, delta_th, R_max=None):
    """
    Eulerian rescaling: find R where cumulative <delta>(R) = delta_th.
    Search in [0.5, R_max]; if no clean crossing returns R_ws (fallback).
    """
    if R_max is None:
        R_max = max(R_ws * 2.0, 8.0)
    idx = tree.query_ball_point(center, R_max)
    if len(idx) < 5:
        return R_ws  # fallback
    dists = np.sort(np.linalg.norm(parts[idx] - center, axis=1))

    R_grid = np.linspace(0.5, R_max, 80)
    vol = (4.0/3) * np.pi * R_grid**3
    counts = np.searchsorted(dists, R_grid)
    delta_cum = (counts / (rho_mean * vol)) - 1.0

    # OUTERMOST crossing from below to above delta_th
    above = delta_cum >= delta_th
    transitions = np.diff(above.astype(int))
    cross_up = np.where(transitions == 1)[0]
    if len(cross_up) == 0:
        # Either always above (shallow void) or always below (very deep)
        return R_ws  # fallback to watershed
    i = cross_up[-1]
    d0, d1 = delta_cum[i], delta_cum[i+1]
    if d1 == d0:
        return R_grid[i]
    return R_grid[i] + (R_grid[i+1] - R_grid[i]) * (delta_th - d0) / (d1 - d0)


def remove_overlaps(coords, radii):
    if len(coords) == 0:
        return np.array([], dtype=bool)
    order = np.argsort(-radii)  # descending
    keep = np.ones(len(coords), dtype=bool)
    tree = cKDTree(coords)
    for k, i in enumerate(order):
        if not keep[i]:
            continue
        rmax = radii[i] + radii.max()
        for j in tree.query_ball_point(coords[i], rmax):
            if j == i or not keep[j]:
                continue
            if radii[j] >= radii[i]:
                continue  # j was kept earlier (larger), shouldn't trigger
            d = np.linalg.norm(coords[i] - coords[j])
            if d < radii[i] + radii[j]:
                keep[j] = False
    return keep


def vsf(radii, v_eff=V_EFF):
    counts, _ = np.histogram(radii, bins=EDGES)
    return counts / (v_eff * DLNR)


def load_fa(frac):
    f = os.path.join(FA, f'vsf_dm_250cube_ss0p{FA_MAP[frac]}_NREC50_SM0p00_CS0p79_TH-0p80_OV00.dat')
    df = pd.read_csv(f, sep=r'\s+', comment='#', names=['r','vsf','err'])
    return df[df['vsf'] > 0]


def process(frac, delta_th_list):
    print(f'\n=== {frac} ===')
    df = pd.read_csv(os.path.join(RAW, f'voids_recon_50_bitvf_aletheia_{frac}'),
                     sep=r'\s+', comment='#', names=RAW_COLS)
    df = df[df['radius'] > 0].reset_index(drop=True)
    box_origin = 375.0 if df['x'].min() > 300 else 0.0
    print(f'  N raw voids: {len(df)}')

    parts = np.loadtxt(os.path.join(SUBS, f'DM_particles_ss{frac}_250cube.dat'),
                       usecols=(0,1,2))
    rho_mean = len(parts) / BOX**3
    tree = cKDTree(parts)

    fa = load_fa(frac)

    centers = df[['x','y','z']].values

    results = {}
    for dth in delta_th_list:
        # Rescale all voids to delta_th
        R_resc = np.array([
            rescale_one(tree, parts, centers[i], df['radius'].iloc[i], rho_mean, dth)
            for i in range(len(df))
        ])

        # Apply cleaning: TH cut on BitVF reported delta_avg, then overlap, then border
        # NB: cleaning happens AFTER rescaling, on the rescaled radii
        df_th = df.copy()
        df_th['R_resc'] = R_resc
        df_th = df_th[df_th['delta_avg'] <= -0.80]   # original TH cut on reported

        keep = remove_overlaps(df_th[['x','y','z']].values, df_th['R_resc'].values)
        df_th = df_th[keep]

        lo, hi = box_origin + BORDER, box_origin + BOX - BORDER
        df_th = df_th[
            (df_th['x']>lo)&(df_th['x']<hi)&
            (df_th['y']>lo)&(df_th['y']<hi)&
            (df_th['z']>lo)&(df_th['z']<hi)
        ]

        v = vsf(df_th['R_resc'].values)
        if v.max() == 0:
            continue
        peak_R = FORARUN_BIN_CENTERS[np.argmax(v)]

        # Score: log-residual against Simone
        fa_dict = dict(zip(np.round(fa['r'].values, 4), fa['vsf'].values))
        ssr = 0.0; n = 0
        for k, c in enumerate(FORARUN_BIN_CENTERS):
            if v[k] <= 0: continue
            v_fa = fa_dict.get(round(c, 4))
            if v_fa is None or v_fa <= 0: continue
            ssr += (np.log10(v[k]) - np.log10(v_fa))**2
            n += 1
        results[dth] = (peak_R, ssr/max(n,1), len(df_th))

    fa_pk = fa['r'].values[np.argmax(fa['vsf'].values)]
    print(f'  Simone peak: {fa_pk:.2f}')
    print(f'  {"delta_th":>10}{"peak":>8}{"SSR":>10}{"N_clean":>10}')
    for dth, (pk, ssr, n) in results.items():
        print(f'  {dth:>10.2f}{pk:>8.2f}{ssr:>10.4f}{n:>10}')
    return results, fa_pk


def main():
    delta_th_list = [-0.90, -0.80, -0.70, -0.60, -0.50, -0.40, -0.30]
    all_results = {}
    for frac in ['20p','10p','5p','2_5p']:
        res, fa_pk = process(frac, delta_th_list)
        all_results[frac] = (res, fa_pk)

    # Summary
    print('\n=== Summary: best delta_th per subsample ===')
    print(f'{"frac":<6}{"best_dth":>10}{"peak":>8}{"Simone_pk":>10}{"SSR":>10}')
    for frac, (res, fa_pk) in all_results.items():
        best_dth = min(res, key=lambda d: res[d][1])
        pk, ssr, n = res[best_dth]
        print(f'{frac:<6}{best_dth:>10.2f}{pk:>8.2f}{fa_pk:>10.2f}{ssr:>10.4f}')


if __name__ == '__main__':
    main()
