#!/usr/bin/env python3
"""
Temporary version without density cut to diagnose the issue.
"""
import os
import yaml
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

def load_config(config_path="../configs/pipeline_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def remove_overlaps_prefer_larger(df_clean):
    """Removes overlapping voids, keeping the larger one in each pair."""
    if len(df_clean) == 0:
        return df_clean.copy()

    coords = df_clean[['x', 'y', 'z']].values
    radii = df_clean['radius'].values

    tree = cKDTree(coords)
    to_remove = set()

    pairs = tree.query_pairs(r=np.max(radii) * 2)

    for i, j in pairs:
        if i in to_remove or j in to_remove:
            continue

        dist = np.linalg.norm(coords[i] - coords[j])
        if dist < radii[i] + radii[j]:
            if radii[i] > radii[j]:
                to_remove.add(j)
            else:
                to_remove.add(i)

    keep_mask = np.array([i not in to_remove for i in range(len(df_clean))])
    return df_clean[keep_mask].copy()

def main():
    print("=== Testing WITHOUT density cut ===\n")
    config = load_config()
    raw_dir = config['paths']['raw_dir']
    c = config['cleaning']

    for frac in ['5p']:
        raw_file = os.path.join(raw_dir, f"voids_bitvf_aletheia_{frac}.txt")

        if not os.path.exists(raw_file):
            print(f"Skipping {frac}: Raw file not found")
            continue

        cols = ['x', 'y', 'z', 'radius', 'radius_ext', 'delta_min', 'delta_avg']
        df_raw = pd.read_csv(raw_file, sep=r'\s+', names=cols, comment='#', dtype=float)

        print(f"Raw data: {len(df_raw)} voids")
        print(f"  Radius range: {df_raw['radius'].min():.2f} - {df_raw['radius'].max():.2f}")
        print(f"  Delta_min range: {df_raw['delta_min'].min():.2f} - {df_raw['delta_min'].max():.2f}")

        # Try: NO density cut, just radius
        df_no_cut = df_raw.copy()
        mask = (df_no_cut['radius'] > c['rmin']) & (df_no_cut['radius'] < c['rmax'])
        df_no_cut = df_no_cut[mask].copy()
        print(f"\nAfter radius cut only: {len(df_no_cut)}")
        print(f"  Radius histogram:")
        hist, edges = np.histogram(df_no_cut['radius'].values, bins=20)
        centers = (edges[:-1] + edges[1:]) / 2
        peak_idx = np.argmax(hist)
        print(f"  Peak at: {centers[peak_idx]:.2f} Mpc ({hist[peak_idx]} voids)")

        # Now try overlapping
        df_no_overlap = remove_overlaps_prefer_larger(df_no_cut)
        print(f"\nAfter overlap removal: {len(df_no_overlap)}")
        hist2, _ = np.histogram(df_no_overlap['radius'].values, bins=20)
        peak_idx2 = np.argmax(hist2)
        print(f"  Peak at: {centers[peak_idx2]:.2f} Mpc ({hist2[peak_idx2]} voids)")

        # Try WITH density cut
        df_with_cut = df_raw.copy()
        mask = df_with_cut['delta_min'] < c['delta_nl_threshold']
        df_with_cut = df_with_cut[mask].copy()
        mask = (df_with_cut['radius'] > c['rmin']) & (df_with_cut['radius'] < c['rmax'])
        df_with_cut = df_with_cut[mask].copy()
        print(f"\nWith density cut: {len(df_with_cut)}")
        print(f"  Radius histogram:")
        hist3, _ = np.histogram(df_with_cut['radius'].values, bins=20)
        peak_idx3 = np.argmax(hist3)
        print(f"  Peak at: {centers[peak_idx3]:.2f} Mpc ({hist3[peak_idx3]} voids)")

if __name__ == "__main__":
    main()
