#!/usr/bin/env python3
"""
clean_voids_v2.py

Production cleaning pipeline with Eulerian density rescaling.

Pipeline:
  1. Read raw BitVF catalog
  2. For each void, compute Eulerian cumulative density profile
     and find R where <delta>(R) = DELTA_TH (effective threshold)
  3. Apply original TH filter on BitVF's reported delta_avg
  4. Strict overlap removal (OV00) on RESCALED radii
  5. 20 Mpc/h center-based border cut

The effective threshold DELTA_TH = -0.70 was determined by sweep against
Simone's VSFs. It is ~10-15% less negative than the canonical
-0.80 because we measure density in Eulerian (real-space) coordinates, while
the canonical threshold is defined in BitVF's internal Lagrangian-like field.
"""
import os, sys, argparse, numpy as np, pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
RAW  = os.path.join(HERE, '..', 'data', 'raw')
SUBS = os.path.join(HERE, '..', 'data', 'subsamples')
CLN  = os.path.join(HERE, '..', 'data', 'cleaned')
os.makedirs(CLN, exist_ok=True)

RAW_COLS  = ['x','y','z','radius','r_ext','delta_2MPS','delta_avg']
BOX       = 250.0
BORDER    = 20.0
DELTA_TH_RESCALE = -0.80   # Empirical Eulerian threshold for rescaling
DELTA_TH_CUT     = -0.80   # Original BitVF TH cut on reported delta_avg

RAW_FILES = {
    'NREC50': 'voids_recon_50_bitvf_aletheia_{frac}',
    'BIT':    '0_voids_bitvf_aletheia_{frac}.txt',
}


def rescale_one(tree, parts, center, R_ws, rho_mean):
    R_max = max(R_ws * 2.0, 8.0)
    idx = tree.query_ball_point(center, R_max)
    if len(idx) < 5:
        return R_ws
    dists = np.sort(np.linalg.norm(parts[idx] - center, axis=1))

    R_grid = np.linspace(0.5, R_max, 80)
    vol = (4.0/3) * np.pi * R_grid**3
    counts = np.searchsorted(dists, R_grid)
    delta_cum = (counts / (rho_mean * vol)) - 1.0

    above = delta_cum >= DELTA_TH_RESCALE
    transitions = np.diff(above.astype(int))
    cross_up = np.where(transitions == 1)[0]
    if len(cross_up) == 0:
        return R_ws
    i = cross_up[-1]
    d0, d1 = delta_cum[i], delta_cum[i+1]
    if d1 == d0:
        return R_grid[i]
    return R_grid[i] + (R_grid[i+1] - R_grid[i]) * (DELTA_TH_RESCALE - d0) / (d1 - d0)


def remove_overlaps(coords, radii):
    if len(coords) == 0:
        return np.array([], dtype=bool)
    order = np.argsort(-radii)
    keep = np.ones(len(coords), dtype=bool)
    tree = cKDTree(coords)
    rmax_global = radii.max() * 2.1
    for i in tqdm(order, desc='  overlap', leave=False):
        if not keep[i]:
            continue
        for j in tree.query_ball_point(coords[i], rmax_global):
            if j == i or not keep[j] or radii[j] >= radii[i]:
                continue
            d = np.linalg.norm(coords[i] - coords[j])
            if d < radii[i] + radii[j]:
                keep[j] = False
    return keep


def process(frac, field='NREC50'):
    print(f'\n=== {field} {frac} ===')
    raw_file = os.path.join(RAW, RAW_FILES[field].format(frac=frac))
    if not os.path.exists(raw_file):
        print(f'  SKIP: {raw_file} not found')
        return

    df = pd.read_csv(raw_file, sep=r'\s+', comment='#', names=RAW_COLS)
    df = df[df['radius'] > 0].reset_index(drop=True)
    box_origin = 375.0 if df['x'].min() > 300 else 0.0
    print(f'  N raw voids: {len(df)}')

    sub_file = os.path.join(SUBS, f'DM_particles_ss{frac}_250cube.dat')
    if not os.path.exists(sub_file):
        print(f'  SKIP: subsample {sub_file} not found')
        return
    print(f'  Loading particles...')
    parts = np.loadtxt(sub_file, usecols=(0,1,2))
    rho_mean = len(parts) / BOX**3
    tree = cKDTree(parts)

    centers = df[['x','y','z']].values
    radii_ws = df['radius'].values
    print(f'  Rescaling to <delta>(R) = {DELTA_TH_RESCALE}...')
    R_resc = np.array([
        rescale_one(tree, parts, centers[i], radii_ws[i], rho_mean)
        for i in tqdm(range(len(df)), desc='  rescale', leave=False)
    ])
    df['R_resc'] = R_resc
    print(f'  Median R_resc / R_watershed: {(R_resc/radii_ws).mean():.3f}')

    # Step 1: TH cut on BitVF's reported delta_avg
    df = df[df['delta_avg'] <= DELTA_TH_CUT].copy()
    print(f'  After TH cut (delta_avg<={DELTA_TH_CUT}): {len(df)}')

    # Step 2: overlap removal on rescaled radii
    keep = remove_overlaps(df[['x','y','z']].values, df['R_resc'].values)
    df = df[keep]
    print(f'  After overlap removal: {len(df)}')

    # Step 3: border cut
    lo, hi = box_origin + BORDER, box_origin + BOX - BORDER
    df = df[
        (df['x']>lo)&(df['x']<hi)&(df['y']>lo)&(df['y']<hi)&(df['z']>lo)&(df['z']<hi)
    ]
    print(f'  After border cut: {len(df)}')

    # Step 4: resolution-floor cut at R_min ~ MPS (matches Simone convention)
    mps = (BOX**3 / len(parts))**(1./3)
    R_min = 1.33*mps  # could relax to mps/2 or tighten to 1.4*mps
    df = df[df['R_resc'] >= R_min]
    print(f'  After R_min cut (R_resc >= MPS = {mps:.2f}): {len(df)}')

    tag = 'recon' if field == 'NREC50' else 'tdf'
    out_file = os.path.join(CLN, f'cleaned_v4_{tag}_cat_{frac}.csv')
    df.to_csv(out_file, index=False)
    print(f'  Saved: {out_file}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frac', default=None)
    parser.add_argument('--field', default='NREC50', choices=['NREC50','BIT'])
    args = parser.parse_args()
    fractions = [args.frac] if args.frac else ['20p','10p','5p','2_5p']
    for frac in fractions:
        process(frac, args.field)


if __name__ == '__main__':
    main()
