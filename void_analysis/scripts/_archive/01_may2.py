#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

def clean_catalog_hybrid(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: 
        return df

    # 1. Density Threshold
    df = df[df['delta_avg'] <= -0.8].copy()

    # 2. Sort by RAW radii
    df_sorted = df.sort_values(by='radius', ascending=False).reset_index(drop=True)
    coords = df_sorted[['x', 'y', 'z']].values
    raw_radii = df_sorted['radius'].values
    keep_mask = np.ones(len(df_sorted), dtype=bool)
    
    tree = cKDTree(coords)
    
    # 3. Overlap Removal (using RAW radii exclusion zones)
    for i in range(len(df_sorted)):
        if not keep_mask[i]:
            continue
            
        neighbors = tree.query_ball_point(coords[i], r=raw_radii[i] * 2.01)
        for j in neighbors:
            if j > i and keep_mask[j]:
                if np.linalg.norm(coords[i] - coords[j]) < (raw_radii[i] + raw_radii[j]):
                    keep_mask[j] = False
                    
    df_clean = df_sorted[keep_mask].copy()
    
    # 4. SvdW Volume Rescaling
    density_ratio = np.clip(df_clean['delta_avg'] + 1.0, 1e-4, None)
    df_clean['radius'] *= np.cbrt(density_ratio / 0.2)
    
    # 5. Border Pruning
    box_min, box_max = 375.0, 625.0
    mask_border = (
        (df_clean['x'] - df_clean['radius'] > box_min) & 
        (df_clean['x'] + df_clean['radius'] < box_max) &
        (df_clean['y'] - df_clean['radius'] > box_min) & 
        (df_clean['y'] + df_clean['radius'] < box_max) &
        (df_clean['z'] - df_clean['radius'] > box_min) & 
        (df_clean['z'] + df_clean['radius'] < box_max)
    )
    
    # Note: Radius masking (2.0 to 50.0) has been removed.
    return df_clean[mask_border].copy()

def main():
    raw_dir = '../data/raw'
    clean_dir = '../data/cleaned'
    os.makedirs(clean_dir, exist_ok=True)
    
    cols = ['x', 'y', 'z', 'radius', 'r_ext', 'delta_min', 'delta_avg']
    
    for frac in ['20p', '10p', '5p', '2_5p']:
        raw_file = os.path.join(raw_dir, f"voids_recon_50_bitvf_aletheia_{frac}")
        if not os.path.exists(raw_file):
            continue
            
        df_raw = pd.read_csv(raw_file, sep=r'\s+', names=cols, comment='#')
        df_final = clean_catalog_hybrid(df_raw)
        
        out_path = os.path.join(clean_dir, f"cleaned_recon_50_bitvf_hybrid_{frac}.csv")
        df_final.to_csv(out_path, index=False)
        print(f"[{frac}] Success. Saved {len(df_final)} voids.")

if __name__ == "__main__":
    main()