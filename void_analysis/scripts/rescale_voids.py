#!/usr/bin/env python3
"""
rescale_voids.py

Implements the missing density rescaling step: for each BitVF void,
find the sphere centered on the void whose average interior density
contrast equals δ = -0.80. This is done on a grid-based density field
(cell_size = 1.0 h^-1 Mpc), matching the BitVF internal resolution.

Pipeline:
  1. Load DM tracer particles, build 3D overdensity grid
  2. For each void center, compute cumulative <δ>(R) on the grid
  3. Find R where <δ>(R) = -0.80 → r_rescaled
  4. Apply cleaning: overlap removal (OV00) + border cut
  5. Output cleaned catalog with rescaled radii

Usage:
  python rescale_voids.py [--frac 5p] [--field NREC50]
"""
import os, sys, argparse, numpy as np, pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
RAW  = os.path.join(HERE, '..', 'data', 'raw')
SUBS = os.path.join(HERE, '..', 'data', 'subsamples')
OUT  = os.path.join(HERE, '..', 'data', 'cleaned')
os.makedirs(OUT, exist_ok=True)

RAW_COLS = ['x','y','z','radius','r_ext','delta_2MPS','delta_avg']
BOX_SIZE = 250.0
CELL     = 1.0         # Grid cell size in Mpc/h (matches BitVF)
DELTA_TH = -0.80       # Target density contrast for rescaling
BORDER   = 20.0        # Border cut in Mpc/h

# Raw catalog filename patterns
RAW_FILES = {
    'NREC50': 'voids_recon_50_bitvf_aletheia_{frac}',
    'BIT':    '0_voids_bitvf_aletheia_{frac}.txt',
}
SUB_FILE = 'DM_particles_ss{frac}_250cube.dat'


def build_density_grid(particle_file, box_origin):
    """Load particles, bin into a 3D grid, return overdensity field."""
    print(f'  Loading particles from {os.path.basename(particle_file)}...')
    parts = np.loadtxt(particle_file, usecols=(0,1,2))
    N_total = len(parts)
    rho_mean = N_total / BOX_SIZE**3
    print(f'  {N_total:,} particles loaded, ρ_mean = {rho_mean:.6f} / (Mpc/h)³')

    n_cells = int(BOX_SIZE / CELL)
    grid = np.zeros((n_cells, n_cells, n_cells), dtype=np.float64)

    # NGP (Nearest Grid Point) assignment
    idx = np.floor((parts - box_origin) / CELL).astype(int)
    idx = np.clip(idx, 0, n_cells - 1)
    # Use np.add.at for unbuffered in-place addition
    np.add.at(grid, (idx[:,0], idx[:,1], idx[:,2]), 1)

    # Convert to overdensity: δ = ρ/ρ_mean - 1
    expected_per_cell = rho_mean * CELL**3
    grid = grid / expected_per_cell - 1.0
    print(f'  Density grid built: {n_cells}³ cells, <δ> = {grid.mean():.4f}')
    return grid, rho_mean


def rescale_one_void(grid, cx, cy, cz, R_ws, box_origin, R_max_factor=1.5):
    """
    Find the radius where the cumulative average density = DELTA_TH.

    Uses the grid cells within R_max of the void center. Evaluates
    the cumulative <δ>(R) at each cell-shell boundary and interpolates
    to find the crossing.

    Returns: r_rescaled (float), or np.nan if no crossing found.
    """
    n_cells = grid.shape[0]
    R_max = max(R_ws * R_max_factor, R_ws + 3.0)  # At least 3 cells beyond watershed

    # Sub-grid extraction
    ic = int((cx - box_origin) / CELL)
    jc = int((cy - box_origin) / CELL)
    kc = int((cz - box_origin) / CELL)
    margin = int(np.ceil(R_max / CELL)) + 1

    i0 = max(ic - margin, 0);  i1 = min(ic + margin + 1, n_cells)
    j0 = max(jc - margin, 0);  j1 = min(jc + margin + 1, n_cells)
    k0 = max(kc - margin, 0);  k1 = min(kc + margin + 1, n_cells)

    # Cell center coordinates
    ii = np.arange(i0, i1) * CELL + box_origin + CELL/2
    jj = np.arange(j0, j1) * CELL + box_origin + CELL/2
    kk = np.arange(k0, k1) * CELL + box_origin + CELL/2
    gi, gj, gk = np.meshgrid(ii, jj, kk, indexing='ij')

    dists = np.sqrt((gi - cx)**2 + (gj - cy)**2 + (gk - cz)**2).ravel()
    deltas = grid[i0:i1, j0:j1, k0:k1].ravel()

    # Keep only cells within R_max
    mask = dists <= R_max
    dists  = dists[mask]
    deltas = deltas[mask]

    if len(dists) < 5:
        return np.nan

    # Sort by distance
    order = np.argsort(dists)
    dists  = dists[order]
    deltas = deltas[order]

    # Cumulative average δ
    cum_delta = np.cumsum(deltas) / np.arange(1, len(deltas)+1)

    # Find the OUTERMOST crossing from below -0.80 to above -0.80
    # (cumulative avg starts very negative near center, rises toward 0)
    crossings = np.where((cum_delta[:-1] <= DELTA_TH) & (cum_delta[1:] > DELTA_TH))[0]

    if len(crossings) > 0:
        i = crossings[-1]  # outermost
        d0, r0 = cum_delta[i], dists[i]
        d1, r1 = cum_delta[i+1], dists[i+1]
        if r1 > r0:
            return r0 + (r1 - r0) * (DELTA_TH - d0) / (d1 - d0)
        else:
            return (r0 + r1) / 2

    # If cumulative stays below -0.80 all the way to R_max → very deep, drop it
    # If cumulative is always above -0.80 → too shallow, drop it
    return np.nan


def remove_overlaps(df, radius_col='r_rescaled'):
    """Strict OV00 overlap removal: D < R_i + R_j → remove smaller."""
    if df.empty:
        return df
    df_s = df.sort_values(by=radius_col, ascending=False).reset_index(drop=True)
    coords = df_s[['x','y','z']].values
    radii  = df_s[radius_col].values
    keep   = np.ones(len(df_s), dtype=bool)
    tree   = cKDTree(coords)
    max_r  = radii.max() * 2.1

    for i in tqdm(range(len(df_s)), desc='  Overlap removal', leave=False):
        if not keep[i]:
            continue
        neighbors = tree.query_ball_point(coords[i], r=max_r)
        for j in neighbors:
            if j <= i or not keep[j]:
                continue
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < (radii[i] + radii[j]):
                keep[j] = False
    return df_s[keep].copy()


def apply_border_cut(df, box_origin, radius_col='r_rescaled'):
    """Center-based border cut of BORDER Mpc/h."""
    lo = box_origin + BORDER
    hi = box_origin + BOX_SIZE - BORDER
    mask = (
        (df['x'] > lo) & (df['x'] < hi) &
        (df['y'] > lo) & (df['y'] < hi) &
        (df['z'] > lo) & (df['z'] < hi)
    )
    return df[mask].copy()


def process_one(frac, field='NREC50'):
    print(f'\n{"="*60}')
    print(f'Processing {field} {frac}')
    print(f'{"="*60}')

    # Load raw void catalog
    raw_file = os.path.join(RAW, RAW_FILES[field].format(frac=frac))
    if not os.path.exists(raw_file):
        print(f'  SKIP: {raw_file} not found')
        return
    df = pd.read_csv(raw_file, sep=r'\s+', comment='#', names=RAW_COLS)
    df = df[df['radius'] > 0].copy()
    print(f'  Raw voids: {len(df)}')

    # Detect box origin
    box_origin = 375.0 if df['x'].min() > 300 else 0.0
    print(f'  Box origin: {box_origin}')

    # Load particles and build density grid
    sub_file = os.path.join(SUBS, SUB_FILE.format(frac=frac))
    if not os.path.exists(sub_file):
        print(f'  SKIP: {sub_file} not found')
        return
    grid, rho_mean = build_density_grid(sub_file, box_origin)

    # Rescale each void
    print(f'  Rescaling {len(df)} voids to δ_avg = {DELTA_TH}...')
    r_rescaled = np.full(len(df), np.nan)
    for i in tqdm(range(len(df)), desc='  Rescaling', leave=False):
        row = df.iloc[i]
        r_rescaled[i] = rescale_one_void(
            grid, row['x'], row['y'], row['z'], row['radius'], box_origin)

    df['r_rescaled'] = r_rescaled
    valid = df['r_rescaled'].notna() & (df['r_rescaled'] > 0)
    print(f'  Successfully rescaled: {valid.sum()} / {len(df)} '
          f'({100*valid.mean():.1f}%)')
    print(f'  Median R_rescaled / R_watershed: '
          f'{(df.loc[valid,"r_rescaled"]/df.loc[valid,"radius"]).median():.3f}')

    df = df[valid].copy()

    # Clean: overlap removal on rescaled radii
    df_clean = remove_overlaps(df, radius_col='r_rescaled')
    print(f'  After overlap removal: {len(df_clean)}')

    # Clean: border cut
    df_clean = apply_border_cut(df_clean, box_origin)
    print(f'  After border cut: {len(df_clean)}')

    # Save
    tag = 'recon' if field == 'NREC50' else 'tdf'
    out_file = os.path.join(OUT, f'rescaled_{tag}_cat_{frac}.csv')
    df_clean.to_csv(out_file, index=False)
    print(f'  Saved: {out_file} ({len(df_clean)} voids)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frac', default=None, help='Single fraction (e.g. 5p)')
    parser.add_argument('--field', default='NREC50', choices=['NREC50','BIT'])
    args = parser.parse_args()

    fractions = [args.frac] if args.frac else ['20p','10p','5p','2_5p']
    for frac in fractions:
        process_one(frac, args.field)

    # Also do BIT if raw catalogs exist
    if args.field == 'NREC50' and args.frac is None:
        for frac in ['2_5p','5p']:
            process_one(frac, 'BIT')


if __name__ == '__main__':
    main()
