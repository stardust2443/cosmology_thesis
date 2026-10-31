#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

def remove_overlaps(df):
    if len(df) == 0: return df
    df = df.sort_values('radius', ascending=False).reset_index(drop=True)
    keep = np.ones(len(df), dtype=bool)
    coords = df[['x', 'y', 'z']].values
    radii = df['radius'].values
    tree = cKDTree(coords)
    
    for i in range(len(df)):
        if not keep[i]: continue
        idx = tree.query_ball_point(coords[i], radii[i])
        for j in idx:
            if j > i: keep[j] = False
    return df[keep].copy()

def clean_catalog(df):
    print(f"  -> Starting with {len(df)} raw voids")
    
    # 1. KDTree Overlap Removal (MUST BE FIRST)
    # Uses the true, raw boundaries found by bitVF to kill nested sub-voids
    df = remove_overlaps(df)
    print(f"     [Surviving KDTree] {len(df)}")

    # 2. Border Pruning 
    # Use the true raw radii so boundary voids don't accidentally slip in
    box_min, box_max = 375.0, 625.0
    mask_border = (
        (df['x'] - df['radius'] > box_min) & (df['x'] + df['radius'] < box_max) &
        (df['y'] - df['radius'] > box_min) & (df['y'] + df['radius'] < box_max) &
        (df['z'] - df['radius'] > box_min) & (df['z'] + df['radius'] < box_max)
    )
    df = df[mask_border].copy()
    print(f"     [Surviving Border] {len(df)}")
    
    # 3. Density Rescaling (Safe to do now)
    density_ratio = np.clip(df['delta_avg'] + 1.0, 1e-4, None)
    df['radius'] = df['radius'] * np.cbrt(density_ratio / 0.2)
    print(f"     [Applied Density Rescaling]")
    
    # 4. Final Radius Bounds
    mask_radius = (df['radius'] > 1.5) & (df['radius'] < 50.0)
    df = df[mask_radius].copy()
    
    return df

def main():
    raw_dir = '../data/raw'
    clean_dir = '../data/cleaned'
    os.makedirs(clean_dir, exist_ok=True)
    fractions = ['20p', '10p', '5p', '2_5p']
    
    for frac in fractions:
        raw_file = os.path.join(raw_dir, f"voids_bitvf_aletheia_{frac}.txt")
        if not os.path.exists(raw_file):
            raw_file = os.path.join(raw_dir, f"voids_bitvf_aletheia_{frac}")
            
        if not os.path.exists(raw_file):
            print(f"Skipping {frac}: Raw file not found.")
            continue
            
        print(f"\nProcessing fraction: {frac}")
        
        # CORRECTED COLUMNS TO MATCH BITVF OUTPUT
        cols = ['x', 'y', 'z', 'radius', 'r_ext', 'delta_min', 'delta_avg']
        
        df_raw = pd.read_csv(raw_file, sep=r'\s+', names=cols, comment='#')
        
        df_clean = clean_catalog(df_raw)
        
        out_path = os.path.join(clean_dir, f"cleaned_bitvf_aletheia_cat_1_{frac}.csv")
        df_clean.to_csv(out_path, index=False)
        print(f"  -> Saved {len(df_clean)} pure voids.")

if __name__ == "__main__":
    main()