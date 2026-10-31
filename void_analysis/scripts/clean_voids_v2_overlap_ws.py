#!/usr/bin/env python3
"""
clean_voids_v2_overlap_ws.py

Variant of `clean_voids_v2.py` for Experiment B2: keep all cleaning steps
identical, but use the *watershed* radius (`df['radius']`, the raw BitVF
column before rescaling) for the overlap comparison instead of `R_resc`.

Single-purpose driver — re-imports rescaling and overlap utilities from
`clean_voids_v2.py` unchanged. Runs cleaning on 5p only by default.

Output: data/cleaned/cleaned_v4_overlap_ws_recon_cat_{frac}.csv
"""
import os, sys, argparse
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Reuse the unchanged utilities from the original cleaning script
from clean_voids_v2 import (
    rescale_one, remove_overlaps,
    RAW, SUBS, CLN, BOX, BORDER,
    DELTA_TH_RESCALE, DELTA_TH_CUT, RAW_FILES, RAW_COLS,
)


def process_overlap_ws(frac, field='NREC50'):
    print(f'\n=== {field} {frac}  [variant: watershed-radius overlap] ===')
    raw_file = os.path.join(RAW, RAW_FILES[field].format(frac=frac))
    if not os.path.exists(raw_file):
        print(f'  SKIP: {raw_file} not found'); return

    df = pd.read_csv(raw_file, sep=r'\s+', comment='#', names=RAW_COLS)
    df = df[df['radius'] > 0].reset_index(drop=True)
    box_origin = 375.0 if df['x'].min() > 300 else 0.0
    print(f'  N raw voids: {len(df)}')

    sub_file = os.path.join(SUBS, f'DM_particles_ss{frac}_250cube.dat')
    if not os.path.exists(sub_file):
        print(f'  SKIP: subsample {sub_file} not found'); return
    print(f'  Loading particles...')
    parts = np.loadtxt(sub_file, usecols=(0, 1, 2))
    rho_mean = len(parts) / BOX**3
    tree = cKDTree(parts)

    centers = df[['x', 'y', 'z']].values
    radii_ws = df['radius'].values
    print(f'  Rescaling to <delta>(R) = {DELTA_TH_RESCALE} ...')
    R_resc = np.array([
        rescale_one(tree, parts, centers[i], radii_ws[i], rho_mean)
        for i in tqdm(range(len(df)), desc='  rescale', leave=False)
    ])
    df['R_resc'] = R_resc
    print(f'  Median R_resc / R_watershed: {(R_resc / radii_ws).mean():.3f}')

    # Step 1: TH cut on BitVF's reported delta_avg (unchanged)
    df = df[df['delta_avg'] <= DELTA_TH_CUT].copy()
    print(f'  After TH cut: {len(df)}')

    # Step 2 (CHANGED): overlap removal using the WATERSHED radius (df['radius'])
    # rather than the rescaled R_resc. Geometric rule "larger wins" is unchanged.
    keep = remove_overlaps(df[['x', 'y', 'z']].values, df['radius'].values)
    df = df[keep]
    print(f'  After watershed-radius overlap removal: {len(df)}')

    # Step 3: border cut (unchanged)
    lo, hi = box_origin + BORDER, box_origin + BOX - BORDER
    df = df[
        (df['x'] > lo) & (df['x'] < hi) &
        (df['y'] > lo) & (df['y'] < hi) &
        (df['z'] > lo) & (df['z'] < hi)
    ]
    print(f'  After border cut: {len(df)}')

    # Step 4: R_min cut on R_resc (unchanged)
    mps = (BOX**3 / len(parts))**(1./3)
    R_min = 1.33 * mps
    df = df[df['R_resc'] >= R_min]
    print(f'  After R_min cut (R_resc >= {R_min:.2f}): {len(df)}')

    tag = 'recon' if field == 'NREC50' else 'tdf'
    out_file = os.path.join(
        CLN, f'cleaned_v4_overlap_ws_{tag}_cat_{frac}.csv')
    df.to_csv(out_file, index=False)
    print(f'  Saved: {out_file}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frac', default='5p',
                    choices=['20p', '10p', '5p', '2_5p'])
    ap.add_argument('--field', default='NREC50', choices=['NREC50', 'BIT'])
    args = ap.parse_args()
    process_overlap_ws(args.frac, args.field)


if __name__ == '__main__':
    main()
