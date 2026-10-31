#!/usr/bin/env python3
"""
clean_voids.py
Void cleaning pipeline for bitVF Aletheia catalogs.
Methodology: TH-0.80 threshold -> OV00 overlap removal (r_ext) -> 20 Mpc/h border cut.
"""
import os
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

def remove_overlaps_strict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes overlaps using the STRICT 'No Touching' criterion (D < R_i + R_j).
    We evaluate overlaps directly on the physical rescaled radius (`r_ext`),
    ensuring that the final physical voids do not overlap in volume.
    """
    if df.empty:
        return df
    
    # 1. Sort by r_ext descending (Guarantees we keep the biggest physical void)
    df_sorted = df.sort_values(by='r_ext', ascending=False).reset_index(drop=True)
    
    coords = df_sorted[['x', 'y', 'z']].values
    radii = df_sorted['r_ext'].values
    keep_mask = np.ones(len(df_sorted), dtype=bool)
    
    max_search_radius = radii.max() * 2.1 
    tree = cKDTree(coords)
    
    for i in tqdm(range(len(df_sorted)), desc="  Resolving Overlaps", leave=False):
        if not keep_mask[i]:
            continue
            
        neighbors = tree.query_ball_point(coords[i], r=max_search_radius)
        
        for j in neighbors:
            if j <= i or not keep_mask[j]:
                continue
                
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < (radii[i] + radii[j]):
                keep_mask[j] = False
                
    return df_sorted[keep_mask].copy()

def clean_catalog(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw[df_raw['r_ext'] > 0].copy()
    print(f"  -> Raw valid bitVF voids: {len(df)}")

    # STEP 1: Density threshold (TH-0.80).
    # Applied BEFORE overlap removal so spurious shallow voids
    # don't eliminate real deep voids during the greedy pass.
    df = df[df['delta_avg'] <= -0.80].copy()
    print(f"  -> After deltaAv <= -0.80 threshold: {len(df)}")

    # STEP 2: Strict Overlap Removal (OV00) using r_ext
    df_clean = remove_overlaps_strict(df)
    print(f"  -> Surviving Overlap Cut: {len(df_clean)}")

    # STEP 3: Center Border Pruning (20 h^-1 Mpc)
    x_min = df_clean['x'].min()
    box_lower = 375.0 if x_min > 300 else 0.0
    box_upper = box_lower + 250.0

    border_cut = 20.0
    mask_border = (
        (df_clean['x'] > box_lower + border_cut) &
        (df_clean['x'] < box_upper - border_cut) &
        (df_clean['y'] > box_lower + border_cut) &
        (df_clean['y'] < box_upper - border_cut) &
        (df_clean['z'] > box_lower + border_cut) &
        (df_clean['z'] < box_upper - border_cut)
    )
    df_clean = df_clean[mask_border].copy()
    print(f"  -> Surviving Boundary Cuts ({border_cut} Mpc/h): {len(df_clean)}")

    return df_clean

def main():
    raw_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    clean_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'cleaned')
    os.makedirs(clean_dir, exist_ok=True)
    
    fractions = ['20p', '10p', '5p', '2_5p']
    cols = ['x', 'y', 'z', 'radius', 'r_ext', 'delta_2MPS', 'delta_avg']
    
    print("=== Processing True Displacement Field Catalogs ===")
    for frac in fractions:
        raw_file = os.path.join(raw_dir, f"0_voids_bitvf_aletheia_{frac}.txt")
        if not os.path.exists(raw_file): continue
        print(f"\nFraction: {frac} (TDF)")
        df_raw = pd.read_csv(raw_file, sep=r'\s+', comment='#', names=cols)
        df_final = clean_catalog(df_raw)
        out_path = os.path.join(clean_dir, f"cleaned_tdf_cat_{frac}.csv")
        df_final.to_csv(out_path, index=False)
        print(f"  -> Saved {len(df_final)} voids to {out_path}")

    print("\n=== Processing Optimal Transport Reconstructed Catalogs ===")
    for frac in fractions:
        raw_file = os.path.join(raw_dir, f"voids_recon_50_bitvf_aletheia_{frac}")
        if not os.path.exists(raw_file): continue
        print(f"\nFraction: {frac} (OT Reconstructed)")
        df_raw = pd.read_csv(raw_file, sep=r'\s+', comment='#', names=cols)
        df_final = clean_catalog(df_raw)
        out_path = os.path.join(clean_dir, f"cleaned_recon_cat_{frac}.csv")
        df_final.to_csv(out_path, index=False)
        print(f"  -> Saved {len(df_final)} voids to {out_path}")

if __name__ == "__main__":
    main()
