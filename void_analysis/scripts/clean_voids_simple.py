#!/usr/bin/env python3
"""
clean_voids_simple.py

Two-step cleaning algorithm as prescribed by Simone:

Step 1 — Rescale each watershed void to a sphere with mean density
contrast = −0.80 (default), using the "midpoint between every tracer"
discretisation:
  - sort tracer distances {d_0, d_1, ...} from the void center
  - evaluate the cumulative density contrast at each midpoint
    r_i = (d_i + d_{i+1}) / 2 with cumulative count (i + 1)
  - take R_resc as the largest r where δ ≤ −0.80 (linearly interpolated
    between the bracketing midpoints)

Step 2 — Overlap removal on R_resc with one of two volume-based criteria:
  --overlap center  : drop a void if its center lies inside any bigger void
  --overlap touch   : drop a void if it touches any bigger void
                      (i.e. d_centers < R_big + R_small) — strict

Plus a border cut (sphere must fit entirely inside the box).

What is *deliberately not done* (per the advisor):
  - No `delta_avg` TH cut
  - No per-subsample MULT factor at binning
  - No empirical R_min floor — the rescaling itself sets the smallest R
  - No reproducing CBL's older more complicated cleaning

Output: data/cleaned_simple/cleaned_simple_recon_cat_{frac}_{criterion}.csv
        (columns: x, y, z, R_resc)

CLI:
  python scripts/clean_voids_simple.py --frac 5p --overlap center
  python scripts/clean_voids_simple.py --frac 5p --overlap touch
  python scripts/clean_voids_simple.py --all --overlap touch
"""
import os, sys, time, argparse
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
RAW  = os.path.join(HERE, '..', 'data', 'raw')
SUBS = os.path.join(HERE, '..', 'data', 'subsamples')
OUT  = os.path.join(HERE, '..', 'data', 'cleaned_simple0')
os.makedirs(OUT, exist_ok=True)

BOX = 250.0       # Aletheia cube side, h⁻¹Mpc
# Simone (2026-05): keep voids whose CENTER is ≥ 20 h⁻¹Mpc from every face,
# and use V_eff = (BOX − 2·BORDER_PAD)³ at binning. Centers-only cut (NOT
# sphere-fits) so the geometric volume matches the survey volume exactly.
BORDER_PAD = 20.0

RAW_COLS = ['x', 'y', 'z', 'radius', 'r_ext', 'delta_2MPS', 'delta_avg']
RAW_FILES = {
    'NREC50': 'voids_recon_50_bitvf_aletheia_{frac}',
    'BIT':    '0_voids_bitvf_aletheia_{frac}.txt',
}
SUB_FILE = 'DM_particles_ss{frac}_250cube.dat'

DELTA_TH = -0.79  # advisor default
R_MAX_PROBE = 25.0  # cap how far we walk outward from each void center
# default R_MAX_PROBE was 25.

# ─── Step 1: rescale via midpoint-between-tracers method ─────────────
def rescale_void(center, kdtree, particles, rho_mean,
                 delta_th=DELTA_TH, r_max=R_MAX_PROBE):
    """
    Compute the rescaled radius R for one void.

    Sort the tracer distances from the void center, evaluate the cumulative
    density at each midpoint (r_i = (d_i + d_{i+1}) / 2 with cumulative
    count i+1), and return the largest r where the running density contrast
    δ ≤ delta_th. Linear interpolation between the bracketing midpoints.

    Returns None if the void has no underdense interior (i.e. the running
    δ never drops below delta_th within r_max).
    """
    idx = kdtree.query_ball_point(center, r_max)
    if len(idx) < 4:
        return None
    dists = np.sort(np.linalg.norm(particles[idx] - center, axis=1))
    # Midpoints: r_mid[i] = (d_i + d_{i+1}) / 2 with cum_count = i + 1
    r_mid     = 0.5 * (dists[:-1] + dists[1:])
    cum_count = np.arange(1, len(r_mid) + 1, dtype=float)
    vol       = (4.0 / 3.0) * np.pi * r_mid ** 3
    delta     = cum_count / (rho_mean * vol) - 1.0

    # Largest r where δ ≤ threshold. The running δ generally rises with r
    # (mean density approaches the cosmic mean) — we want the last index
    # where we're still safely below the threshold.
    below = delta <= delta_th
    if not below.any():
        return None
    last_below = np.where(below)[0][-1]
    if last_below == len(r_mid) - 1:
        # Stayed below all the way to r_max — return the largest midpoint
        return r_mid[last_below]
    # Linearly interpolate between (r_mid[last_below], delta[last_below])
    # and (r_mid[last_below + 1], delta[last_below + 1])
    r0, r1 = r_mid[last_below], r_mid[last_below + 1]
    d0, d1 = delta[last_below], delta[last_below + 1]
    if d1 == d0:
        return r0
    return r0 + (r1 - r0) * (delta_th - d0) / (d1 - d0)


# ─── Step 2: overlap removal ─────────────────────────────────────────
def remove_overlaps(coords, radii, criterion):
    """
    Volume-based overlap removal. `criterion`:
      'center' — drop a void if its center is inside any bigger void's sphere
      'touch'  — drop a void if it touches any bigger void (any overlap → drop)
    """
    n = len(coords)
    keep = np.ones(n, dtype=bool)
    if n == 0:
        return keep
    order = np.argsort(-radii)   # process largest first
    tree = cKDTree(coords)
    r_max = radii.max()

    for i in tqdm(order, desc=f'  overlap ({criterion})', leave=False):
        if not keep[i]:
            continue
        # Search radius: enough to catch any potential overlap candidate
        if criterion == 'center':
            search_R = radii[i]
        else:  # 'touch'
            search_R = radii[i] + r_max
        for j in tree.query_ball_point(coords[i], search_R):
            if j == i or not keep[j] or radii[j] >= radii[i]:
                continue
            d = np.linalg.norm(coords[i] - coords[j])
            if criterion == 'center':
                # Drop j if its center is inside void i
                if d < radii[i]:
                    keep[j] = False
            else:  # 'touch'
                if d < radii[i] + radii[j]:
                    keep[j] = False
    return keep


# ─── Border cut ──────────────────────────────────────────────────────
def border_cut(coords, radii, box_lo, box_hi, pad=BORDER_PAD):
    """Keep voids whose CENTER is at least `pad` from every face.

    Simone's prescription: hard 20 h⁻¹Mpc center-distance cut (NOT sphere-fits).
    The effective survey volume is then (box_side − 2·pad)³ exactly.
    The `radii` argument is kept for API compatibility but ignored.
    """
    keep = (
        (coords[:, 0] - box_lo >= pad) & (box_hi - coords[:, 0] >= pad) &
        (coords[:, 1] - box_lo >= pad) & (box_hi - coords[:, 1] >= pad) &
        (coords[:, 2] - box_lo >= pad) & (box_hi - coords[:, 2] >= pad)
    )
    return keep


# ─── Top-level cleaning driver ───────────────────────────────────────
def process(frac, field, overlap, delta_th=DELTA_TH, r_max=R_MAX_PROBE):
    print(f'\n{"="*60}\n[CLEAN simple] {field} {frac}  overlap={overlap}\n{"="*60}')
    t0 = time.time()

    raw_path = os.path.join(RAW, RAW_FILES[field].format(frac=frac))
    if not os.path.exists(raw_path):
        print(f'  SKIP: {raw_path} not found'); return

    df = pd.read_csv(raw_path, sep=r'\s+', comment='#', names=RAW_COLS)
    df = df[df['radius'] > 0].reset_index(drop=True)
    print(f'  Raw voids: {len(df)}')

    sub_path = os.path.join(SUBS, SUB_FILE.format(frac=frac))
    if not os.path.exists(sub_path):
        print(f'  SKIP: {sub_path} not found'); return
    parts = np.loadtxt(sub_path, usecols=(0, 1, 2))
    print(f'  DM tracers: {len(parts)}')

    # Detect coordinate frame: Aletheia raws come at [375, 625]; particles too.
    # Keep both in the same frame (no shift).
    box_lo = float(min(parts[:, 0].min(), df['x'].min()))
    box_hi = box_lo + BOX
    rho_mean = len(parts) / BOX ** 3
    print(f'  Box frame:  [{box_lo:.1f}, {box_hi:.1f}]   '
          f'ρ_mean = {rho_mean:.5f} h³Mpc⁻³')

    # Build a KD-tree on the tracers (fast spatial query)
    tree = cKDTree(parts)

    # Step 1: rescale each void
    centers = df[['x', 'y', 'z']].values
    print(f'  Step 1: rescaling each watershed void to ⟨δ⟩ = {delta_th} '
          f'(midpoint-between-tracers method, r_max = {r_max})')
    R_resc = np.full(len(df), np.nan, dtype=float)
    for i in tqdm(range(len(df)), desc='  rescale', leave=False):
        r = rescale_void(centers[i], tree, parts, rho_mean,
                         delta_th=delta_th, r_max=r_max)
        if r is not None:
            R_resc[i] = r
    n_resc = int(np.isfinite(R_resc).sum())
    print(f'    {n_resc} voids successfully rescaled '
          f'(dropped {len(df) - n_resc} with no underdense interior)')
    df['R_resc'] = R_resc
    df = df[np.isfinite(df['R_resc'])].reset_index(drop=True)

    # Border cut — voids whose rescaled sphere must fit in the box
    keep = border_cut(df[['x', 'y', 'z']].values, df['R_resc'].values,
                      box_lo, box_hi, pad=BORDER_PAD)
    n_before = len(df)
    df = df[keep].reset_index(drop=True)
    print(f'  Border cut (centers ≥ {BORDER_PAD:.0f} h⁻¹Mpc from each face '
          f'of [{box_lo:.0f}, {box_hi:.0f}]³):  {n_before} → {len(df)}')

    # Step 2: overlap removal
    print(f'  Step 2: overlap removal ({overlap})')
    keep = remove_overlaps(df[['x', 'y', 'z']].values,
                            df['R_resc'].values, overlap)
    n_before = len(df)
    df = df[keep].reset_index(drop=True)
    print(f'    {n_before} → {len(df)} after overlap removal')

    # Output
    out_path = os.path.join(
        OUT, f'cleaned_simple0_{field.lower()}_cat_{frac}_{overlap}.csv')
    df[['x', 'y', 'z', 'R_resc']].to_csv(out_path, index=False)
    print(f'  Saved: {out_path}')
    r_arr = df['R_resc'].values
    print(f'  R range:  [{r_arr.min():.2f}, {r_arr.max():.2f}]  '
          f'median {np.median(r_arr):.2f}  N(R≥7) = {int((r_arr >= 7).sum())}')
    print(f'  Wall time: {time.time() - t0:.1f}s')
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frac', default='5p',
                    choices=['20p', '10p', '5p', '2_5p'])
    ap.add_argument('--field', default='NREC50', choices=['NREC50', 'BIT'])
    ap.add_argument('--all', action='store_true',
                    help='Run all 4 NREC50 subsamples.')
    ap.add_argument('--overlap', default='touch',
                    choices=['center', 'touch'],
                    help="(i) center: drop if center inside a bigger void; "
                         "(ii) touch: drop if any overlap with a bigger void.")
    ap.add_argument('--delta-th', type=float, default=DELTA_TH,
                    help='Rescaling density-contrast threshold (default -0.79).')
    ap.add_argument('--r-max', type=float, default=R_MAX_PROBE,
                    help='Cap on rescaling search radius (default 25 h⁻¹Mpc).')
    args = ap.parse_args()

    fracs = ['20p', '10p', '5p', '2_5p'] if args.all else [args.frac]
    for f in fracs:
        process(f, args.field, args.overlap,
                delta_th=args.delta_th, r_max=args.r_max)


if __name__ == '__main__':
    main()
