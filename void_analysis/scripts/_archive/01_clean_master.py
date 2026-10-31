#!/usr/bin/env python3
"""
01_clean_master.py
The definitive void cleaning pipeline for bitVF Aletheia catalogs.
"""
import os
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

def remove_overlaps_no_touching(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes overlaps using the STRICT 'No Touching' criterion (D < R_large + R_small).
    Processes voids sequentially from largest to smallest, ensuring the biggest survives.
    """
    if df.empty:
        return df
    
    # 1. Sort by radius descending (Guarantees we keep the biggest void)
    df_sorted = df.sort_values(by='radius', ascending=False).reset_index(drop=True)
    
    coords = df_sorted[['x', 'y', 'z']].values
    radii = df_sorted['radius'].values
    keep_mask = np.ones(len(df_sorted), dtype=bool)
    
    from scipy.spatial import cKDTree
    # We must search far enough to catch touching boundaries (R_max + R_max)
    max_search_radius = radii.max() * 2.1 
    tree = cKDTree(coords)
    
    for i in range(len(df_sorted)):
        if not keep_mask[i]:
            continue
            
        # Find all voids near the current LARGE void (void 'i')
        neighbors = tree.query_ball_point(coords[i], r=max_search_radius)
        
        for j in neighbors:
            if j <= i or not keep_mask[j]:
                continue
                
            dist = np.linalg.norm(coords[i] - coords[j])
            
            # THE ADVISOR'S RULE: If they touch at all, kill the smaller one (void 'j')
            if dist < (radii[i] + radii[j]):
                keep_mask[j] = False
                
    return df_sorted[keep_mask].copy()

def clean_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Applies the physical pipeline in the strict, correct order."""
    print(f"  -> Raw bitVF voids: {len(df)}")
    
    # STEP 1: Strict Overlap Removal (Must use raw bitVF geometry)
    df_clean = remove_overlaps_no_touching(df)
    print(f"  -> Surviving No-Touching Cut: {len(df_clean)}")
    
    # STEP 2: Border Pruning 
    box_min, box_max = 375.0, 625.0
    mask_border = (
        (df_clean['x'] - df_clean['radius'] > box_min) & 
        (df_clean['x'] + df_clean['radius'] < box_max) &
        (df_clean['y'] - df_clean['radius'] > box_min) & 
        (df_clean['y'] + df_clean['radius'] < box_max) &
        (df_clean['z'] - df_clean['radius'] > box_min) & 
        (df_clean['z'] + df_clean['radius'] < box_max)
    )
    df_clean = df_clean[mask_border].copy()
    print(f"  -> Surviving Boundary Cuts: {len(df_clean)}")
    
    # STEP 3: Density Rescaling
    density_ratio = np.clip(df_clean['delta_avg'] + 1.0, 1e-4, None)
    df_clean['radius'] = df_clean['radius'] * np.cbrt(density_ratio / 0.2)
    print(f"  -> Applied Volume Rescaling.")
    
    # STEP 4: Final Physical Radius Bounds
    mask_radius = (df_clean['radius'] >= 2.0) & (df_clean['radius'] <= 50.0)
    df_final = df_clean[mask_radius].copy()
    print(f"  -> Final Valid Voids: {len(df_final)}")
    
    return df_final

def main():
    raw_dir = '../data/raw'
    clean_dir = '../data/cleaned'
    os.makedirs(clean_dir, exist_ok=True)
    fractions = ['20p', '10p', '5p', '2_5p']
    
    for frac in fractions:
        raw_file = os.path.join(raw_dir, f"voids_bitvf_aletheia_{frac}.txt")
        if not os.path.exists(raw_file):
            continue
            
        print(f"\nProcessing fraction: {frac}")
        
        cols = ['x', 'y', 'z', 'radius', 'r_ext', 'delta_min', 'delta_avg']
        df_raw = pd.read_csv(raw_file, sep=r'\s+', names=cols, comment='#')
        
        df_final = clean_catalog(df_raw)
        
        out_path = os.path.join(clean_dir, f"cleaned_bitvf_aletheia_cat_{frac}.csv")
        df_final.to_csv(out_path, index=False)
        print(f"  -> Saved {len(df_final)} voids to {out_path}")

if __name__ == "__main__":
    main()