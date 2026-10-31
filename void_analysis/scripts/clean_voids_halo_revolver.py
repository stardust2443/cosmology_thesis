#!/usr/bin/env python3
"""
clean_voids_halo_revolver.py

Cleans and rescales raw halo voids produced by REVOLVER (ZOBOV):
1. Reads raw REVOLVER void catalog (zobov-Voids_cat.txt).
2. Applies border cut: centers >= BORDER_PAD (20.0 h^-1 Mpc) from faces.
3. Rescales each void using the halo spatial KDTree to delta_th = -0.79
   with periodic minimum image convention. Also computes native REVOLVER rescaling.
4. Removes overlaps (touch: isolated spheres; center: concentric centers).
5. Saves cleaned catalogs to data/voids_halo_revolver/cleaned/.
"""
import os, sys, time, argparse
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_BASE = os.path.join(HERE, '..', 'data', 'voids_halo_revolver', 'raw')
HALO_DIR = os.path.join(HERE, '..', 'data', 'halos', 'z0p00')
OUT_DIR = os.path.join(HERE, '..', 'data', 'voids_halo_revolver', 'cleaned')
os.makedirs(OUT_DIR, exist_ok=True)

BOX = 1000.0  # h^-1 Mpc
BORDER_PAD = 20.0
DELTA_TH = -0.79
R_MAX_PROBE = 50.0

def rescale_single_void(args):
    center, kdtree, halos, rho_mean, delta_th, r_max = args
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

def remove_overlaps(coords, radii, criterion='touch'):
    n = len(coords)
    keep = np.ones(n, dtype=bool)
    if n == 0:
        return keep
    order = np.argsort(-radii)  # largest first
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

def clean_revolver_catalog(sample='100P', criterion='touch', n_workers=8):
    raw_path = os.path.join(RAW_BASE, sample, "zobov-Voids_cat.txt")
    if not os.path.exists(raw_path):
        print(f"Error: REVOLVER catalog not found: {raw_path}")
        return
        
    halo_npy = os.path.join(HALO_DIR, f"halos_{sample}_xyz.npy")
    print(f"[{sample}] Loading haloes from {halo_npy}...")
    t0 = time.time()
    halos = np.load(halo_npy)
    n_halos = len(halos)
    rho_mean = n_halos / (BOX ** 3)
    print(f"  Loaded {n_halos} haloes in {time.time()-t0:.2f}s. Building KDTree in RAM...")
    t0 = time.time()
    kdtree = cKDTree(halos, boxsize=BOX)
    print(f"  KDTree built in {time.time()-t0:.2f}s.")
    
    print(f"[{sample}] Loading raw REVOLVER voids from {raw_path}...")
    # REVOLVER catalog columns: VoidID, x, y, z, radius (R_eff), delta_min, delta_avg, lambda_v, DensRatio
    cols = ['VoidID', 'x', 'y', 'z', 'R_eff', 'delta_min', 'delta_avg', 'lambda_v', 'DensRatio']
    raw_df = pd.read_csv(raw_path, sep=r'\s+', comment='#', header=None, names=cols)
    print(f"  Loaded {len(raw_df)} raw REVOLVER voids.")
    
    # 1. Border cut on raw centers (keep centers >= BORDER_PAD from faces)
    mask_box = (
        (raw_df['x'] >= BORDER_PAD) & (raw_df['x'] <= BOX - BORDER_PAD) &
        (raw_df['y'] >= BORDER_PAD) & (raw_df['y'] <= BOX - BORDER_PAD) &
        (raw_df['z'] >= BORDER_PAD) & (raw_df['z'] <= BOX - BORDER_PAD)
    )
    raw_box = raw_df[mask_box].copy()
    print(f"  After border padding (center >= {BORDER_PAD} h^-1 Mpc): {len(raw_box)} voids ({len(raw_box)/len(raw_df)*100:.1f}%).")
    
    # 2. Rescale radii to delta_th = -0.79 using KDTree (Apples-to-apples with BitVF)
    print(f"  Rescaling void radii to delta_th = {DELTA_TH} using 3D periodic KDTree ({n_workers} workers)...")
    t0 = time.time()
    coords = raw_box[['x', 'y', 'z']].values
    r_eff = raw_box['R_eff'].values
    d_avg = raw_box['delta_avg'].values
    
    # Native REVOLVER rescaling: R_native = R_eff * ((1 + delta_avg)/(1 + delta_th))^(1/3)
    # Filter unphysical delta_avg <= -1 or delta_avg > 0
    valid_davg = (d_avg > -1.0) & (d_avg < 0.0)
    r_native = np.zeros(len(raw_box))
    r_native[valid_davg] = r_eff[valid_davg] * ((1.0 + d_avg[valid_davg]) / (1.0 + DELTA_TH)) ** (1.0 / 3.0)
    
    rescaled_radii = []
    valid_indices = []
    
    for idx, center in enumerate(coords):
        r_resc = rescale_single_void((center, kdtree, halos, rho_mean, DELTA_TH, R_MAX_PROBE))
        if r_resc is not None and r_resc > 0.0:
            rescaled_radii.append(r_resc)
            valid_indices.append(idx)
            
    dt = time.time() - t0
    print(f"  Rescaling complete in {dt:.1f}s. Valid underdense voids: {len(valid_indices)} ({len(valid_indices)/len(coords)*100:.1f}%).")
    
    coords_valid = coords[valid_indices]
    radii_valid = np.array(rescaled_radii)
    r_native_valid = r_native[valid_indices]
    
    # 3. Overlap removal
    print(f"  Removing overlaps using '{criterion}' criterion...")
    t0 = time.time()
    keep_mask = remove_overlaps(coords_valid, radii_valid, criterion=criterion)
    coords_clean = coords_valid[keep_mask]
    radii_clean = radii_valid[keep_mask]
    r_native_clean = r_native_valid[keep_mask]
    dt = time.time() - t0
    print(f"  Overlap removal complete in {dt:.1f}s. Final cleaned voids: {len(radii_clean)} ({len(radii_clean)/len(radii_valid)*100:.1f}% kept).")
    print(f"  R_resc stats: min={radii_clean.min():.2f}, median={np.median(radii_clean):.2f}, mean={radii_clean.mean():.2f}, max={radii_clean.max():.2f} h^-1 Mpc")
    
    # 4. Save cleaned catalog
    out_df = pd.DataFrame({
        'x': coords_clean[:, 0],
        'y': coords_clean[:, 1],
        'z': coords_clean[:, 2],
        'R_resc': radii_clean,
        'R_native': r_native_clean
    })
    out_fname = f"cleaned_voids_halo_{sample}_{criterion}.csv"
    out_path = os.path.join(OUT_DIR, out_fname)
    out_df.to_csv(out_path, index=False)
    print(f"  Saved cleaned REVOLVER voids to {out_path}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=str, default='100P')
    parser.add_argument('--criterion', type=str, default='touch', choices=['center', 'touch'])
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    
    clean_revolver_catalog(args.sample, args.criterion, args.workers)
