import os
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

def compute_vsf(radii, v_eff):
    bins = np.logspace(0, 2.0, 30)
    counts, _ = np.histogram(radii, bins=bins)
    bin_centers = np.sqrt(bins[:-1] * bins[1:])
    dlnr = np.log(bins[1:] / bins[:-1])
    dn_dlnr = counts / (v_eff * dlnr)
    return bin_centers, dn_dlnr

def process_catalog(df, ov_mode):
    df_sorted = df.sort_values(by='r_ext', ascending=False).reset_index(drop=True)
    coords = df_sorted[['x', 'y', 'z']].values
    radii = df_sorted['r_ext'].values
    keep_mask = np.ones(len(df_sorted), dtype=bool)
    
    tree = cKDTree(coords)
    max_search = radii.max() * 2.1
    
    for i in range(len(df_sorted)):
        if not keep_mask[i]: continue
        neighbors = tree.query_ball_point(coords[i], r=max_search)
        for j in neighbors:
            if j <= i or not keep_mask[j]: continue
            dist = np.linalg.norm(coords[i] - coords[j])
            if ov_mode == 'OV00':
                if dist < (radii[i] + radii[j]):
                    keep_mask[j] = False
            elif ov_mode == 'OV05':
                if dist < radii[i]:
                    keep_mask[j] = False
    return df_sorted[keep_mask]

def compare_to_truth(frac, mode):
    raw_file = f'/Users/arun/projects/my_cosmo_analysis/void_analysis/data/raw/0_voids_bitvf_aletheia_{frac}.txt'
    if not os.path.exists(raw_file): return
    
    cols = ['x', 'y', 'z', 'radius', 'r_ext', 'delta_2MPS', 'delta_avg']
    df = pd.read_csv(raw_file, sep=r'\s+', comment='#', names=cols)
    # Try border cut of 20 (V_eff = 210^3)
    x_min, x_max = df['x'].min(), df['x'].max()
    box_lower = 375.0 if x_min > 300 else 0.0
    box_upper = box_lower + 250.0
    mask_border = (
        (df['x'] > box_lower + 20) & (df['x'] < box_upper - 20) &
        (df['y'] > box_lower + 20) & (df['y'] < box_upper - 20) &
        (df['z'] > box_lower + 20) & (df['z'] < box_upper - 20)
    )
    df = df[mask_border].copy()
    
    df_clean = process_catalog(df, mode)
    v_eff = 210.0**3
    r_c, vsf_calc = compute_vsf(df_clean['r_ext'].values, v_eff)
    
    # Load truth
    frac_map = {'2_5p': '025', '5p': '05', '10p': '10', '20p': '20'}
    truth_file = f'/Users/arun/projects/my_cosmo_analysis/void_analysis/data/ForArun/vsf_dm_250cube_ss0p{frac_map[frac]}_BIT_SM0p00_CS0p79_TH-0p80_{mode}.dat'
    df_truth = pd.read_csv(truth_file, sep=r'\s+', comment='#', names=['r', 'vsf', 'err'])
    
    print(f"\n--- {frac} {mode} ---")
    print(f"Calculated  vs  Truth (at R~4.1)")
    idx = np.abs(r_c - 4.1).argmin()
    idx_t = np.abs(df_truth['r'] - 4.1).argmin()
    print(f"{vsf_calc[idx]:.2e}  |  {df_truth['vsf'].iloc[idx_t]:.2e}")
    
    print(f"Calculated  vs  Truth (at R~8.3)")
    idx = np.abs(r_c - 8.3).argmin()
    idx_t = np.abs(df_truth['r'] - 8.3).argmin()
    print(f"{vsf_calc[idx]:.2e}  |  {df_truth['vsf'].iloc[idx_t]:.2e}")

compare_to_truth('2_5p', 'OV00')
compare_to_truth('2_5p', 'OV05')
compare_to_truth('5p', 'OV00')
compare_to_truth('5p', 'OV05')
