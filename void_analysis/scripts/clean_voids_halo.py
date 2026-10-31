#!/usr/bin/env python3
"""
clean_voids_halo.py

Cleans raw halo voids produced by BitVF:
1. Rescales each void using the halo spatial KDTree to delta_th = -0.79
   with periodic minimum image convention.
2. Removes overlaps (center / touch).
3. Applies border cut (centers >= BORDER_PAD from faces, per Simone's protocol).
4. Saves cleaned catalogs to data/voids_halo/cleaned/
"""
import os, sys, time, argparse
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, '..', 'data', 'voids_halo', 'raw')
HALO_DIR = os.path.join(HERE, '..', 'data', 'halos', 'z0p00')
OUT_DIR = os.path.join(HERE, '..', 'data', 'voids_halo', 'cleaned')
os.makedirs(OUT_DIR, exist_ok=True)

BOX = 1000.0 # h^-1 Mpc
BORDER_PAD = 20.0
DELTA_TH = -0.79
R_MAX_PROBE = 45.0

def rescale_void(center, kdtree, halos, rho_mean, delta_th=DELTA_TH, r_max=R_MAX_PROBE):
    idx = kdtree.query_ball_point(center, r_max)
    if len(idx) < 4:
        return None
    # Periodic minimum image convention
    diff = halos[idx] - center
    diff -= BOX * np.round(diff / BOX)
    dists = np.sort(np.linalg.norm(diff, axis=1))
    
    r_mid = 0.5 * (dists[:-1] + dists[1:])
    cum_count = np.arange(1, len(r_mid) + 1, dtype=float)
    vol = (4.0 / 3.0) * np.pi * r_mid ** 3
    delta = cum_count / (rho_mean * vol) - 1.0

    below = delta <= delta_th
    if not below.any():
        return None
    last_below = np.where(below)[0][-1]
    if last_below == len(r_mid) - 1:
        return r_mid[last_below]
    r0, r1 = r_mid[last_below], r_mid[last_below + 1]
    d0, d1 = delta[last_below], delta[last_below + 1]
    if d1 == d0:
        return r0
    return r0 + (r1 - r0) * (delta_th - d0) / (d1 - d0)

def remove_overlaps(coords, radii, criterion='center'):
    n = len(coords)
    keep = np.ones(n, dtype=bool)
    if n == 0:
        return keep
    order = np.argsort(-radii) # largest first
    kdtree = cKDTree(coords, boxsize=BOX)
    r_max = float(radii.max())
    
    for i in order:
        if not keep[i]:
            continue
        ci = coords[i]
        ri = radii[i]
        search_R = ri if criterion == 'center' else (ri + r_max)
        cand = kdtree.query_ball_point(ci, search_R)
        
        for j in cand:
            if j == i or not keep[j] or radii[j] >= ri:
                continue
            diff = coords[j] - ci
            diff -= BOX * np.round(diff / BOX)
            d = np.linalg.norm(diff)
            
            if criterion == 'center':
                if d < ri:
                    keep[j] = False
            elif criterion == 'touch':
                if d < ri + radii[j]:
                    keep[j] = False
    return keep

def clean_catalog(sample='100P', criterion='center'):
    raw_fname = f"voids_raw_bitvf_{sample}_Nrec1.dat"
    raw_path = os.path.join(RAW_DIR, raw_fname)
    if not os.path.exists(raw_path):
        print(f"Raw void file not found: {raw_path}")
        return
        
    halo_path = os.path.join(HALO_DIR, f"halos_{sample}_xyz.dat")
    print(f"Loading haloes for {sample} from {halo_path}...")
    halos = np.loadtxt(halo_path)
    n_halos = len(halos)
    rho_mean = n_halos / (BOX**3)
    print(f"  Loaded {n_halos} haloes. Mean density: {rho_mean:.5e} h^3 Mpc^-3. Building KDTree...")
    kdtree = cKDTree(halos, boxsize=BOX)
    
    print(f"Loading raw voids from {raw_path}...")
    raw_df = pd.read_csv(raw_path, sep=r'\s+', comment='#', header=None,
                         names=['x', 'y', 'z', 'radius', 'r_ext', 'delta_2MPS', 'delta_avg'])
    print(f"  Loaded {len(raw_df)} raw voids.")
    
    # 1. Border cut on raw centers (per Simone: keep centers >= BORDER_PAD from faces)
    mask_box = (
        (raw_df['x'] >= BORDER_PAD) & (raw_df['x'] <= BOX - BORDER_PAD) &
        (raw_df['y'] >= BORDER_PAD) & (raw_df['y'] <= BOX - BORDER_PAD) &
        (raw_df['z'] >= BORDER_PAD) & (raw_df['z'] <= BOX - BORDER_PAD)
    )
    raw_box = raw_df[mask_box].copy()
    print(f"  After border cut (center >= {BORDER_PAD} h^-1 Mpc): {len(raw_box)} voids (retained {len(raw_box)/len(raw_df)*100:.1f}%).")
    
    # 2. Rescale radii
    print("  Rescaling void radii to delta_th = -0.79...")
    t0 = time.time()
    rescaled_radii = []
    valid_indices = []
    
    coords = raw_box[['x', 'y', 'z']].values
    for idx, center in enumerate(coords):
        r_resc = rescale_void(center, kdtree, halos, rho_mean)
        if r_resc is not None and r_resc > 0.0:
            rescaled_radii.append(r_resc)
            valid_indices.append(idx)
            
    print(f"  Rescaling complete in {time.time() - t0:.1f}s. Valid underdense voids: {len(valid_indices)} ({len(valid_indices)/len(coords)*100:.1f}%).")
    
    coords_valid = coords[valid_indices]
    radii_valid = np.array(rescaled_radii)
    
    # 3. Overlap removal
    print(f"  Removing overlaps using '{criterion}' criterion...")
    t0 = time.time()
    keep_mask = remove_overlaps(coords_valid, radii_valid, criterion=criterion)
    coords_clean = coords_valid[keep_mask]
    radii_clean = radii_valid[keep_mask]
    print(f"  Overlap removal complete in {time.time() - t0:.1f}s. Final cleaned voids: {len(radii_clean)} ({len(radii_clean)/len(radii_valid)*100:.1f}% kept).")
    print(f"  R_resc stats: min={radii_clean.min():.2f}, median={np.median(radii_clean):.2f}, mean={radii_clean.mean():.2f}, max={radii_clean.max():.2f} h^-1 Mpc")
    
    # Save cleaned catalog
    out_df = pd.DataFrame({
        'x': coords_clean[:, 0],
        'y': coords_clean[:, 1],
        'z': coords_clean[:, 2],
        'R_resc': radii_clean
    })
    out_fname = f"cleaned_voids_halo_{sample}_{criterion}.csv"
    out_path = os.path.join(OUT_DIR, out_fname)
    out_df.to_csv(out_path, index=False)
    print(f"  Saved cleaned voids to {out_path}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=str, default='100P')
    parser.add_argument('--criterion', type=str, default='center', choices=['center', 'touch'])
    args = parser.parse_args()
    clean_catalog(args.sample, args.criterion)
