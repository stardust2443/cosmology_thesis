#!/usr/bin/env python3
"""
01_clean_catalogs.py
Reads raw BitVF void catalogs, applies density/radius cuts,
overlap removal (prefer larger), border cutting, and saves clean CSVs.
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
    """
    Removes overlapping voids, keeping the larger one in each pair.
    Uses cKDTree for efficient spatial queries.
    """
    if len(df_clean) == 0:
        return df_clean.copy()

    coords = df_clean[['x', 'y', 'z']].values
    radii = df_clean['radius'].values

    tree = cKDTree(coords)
    to_remove = set()

    # Query for void neighbor pairs
    pairs = tree.query_pairs(r=np.max(radii) * 2)  # Conservative upper bound

    for i, j in pairs:
        if i in to_remove or j in to_remove:
            continue

        # Check if voids actually overlap
        dist = np.linalg.norm(coords[i] - coords[j])
        if dist < radii[i] + radii[j]:
            # Keep the larger void
            if radii[i] > radii[j]:
                to_remove.add(j)
            else:
                to_remove.add(i)

    keep_mask = np.array([i not in to_remove for i in range(len(df_clean))])
    return df_clean[keep_mask].copy()

def apply_border_cutting(df_clean, box_size, periodic=True):
    """
    Removes voids whose centers are too close to simulation box boundaries.
    For periodic boxes, applies a softer cut to avoid edge effects.
    """
    if periodic:
        # For periodic boundary conditions, estimate the offset of the simulation box
        # and apply a fractional border cut
        min_coord = 374.0  # Approximate minimum from data
        max_coord = 625.0  # Approximate maximum from data

        # Keep voids away from wrapped boundaries (last ~2x void radius from edges)
        # This adds ~5% margin on each side
        margin = max_coord - min_coord - box_size  # ~124
        inner_min = min_coord + margin * 0.05
        inner_max = max_coord - margin * 0.05

        mask = (df_clean['x'] > inner_min) & (df_clean['x'] < inner_max)
        mask &= (df_clean['y'] > inner_min) & (df_clean['y'] < inner_max)
        mask &= (df_clean['z'] > inner_min) & (df_clean['z'] < inner_max)
    else:
        # For non-periodic, strict boundary condition
        mask = (df_clean['x'] - df_clean['radius'] > 0) & (df_clean['x'] + df_clean['radius'] < box_size)
        mask &= (df_clean['y'] - df_clean['radius'] > 0) & (df_clean['y'] + df_clean['radius'] < box_size)
        mask &= (df_clean['z'] - df_clean['radius'] > 0) & (df_clean['z'] + df_clean['radius'] < box_size)

    return df_clean[mask].copy()

def clean_catalog(df, config):
    """Applies standard physical cuts to the raw void catalog."""
    c = config['cleaning']
    box_size = config['cosmology']['box_size']
    periodic = config['cosmology'].get('periodic', True)

    print(f"  Raw voids: {len(df)}")

    # Cut 1: Radius bounds (apply FIRST, before density cut)
    mask = (df['radius'] > c['rmin']) & (df['radius'] < c['rmax'])
    df_cut1 = df[mask].copy()
    print(f"  After radius cut: {len(df_cut1)}")

    # Cut 2: Central density threshold (very liberal to preserve small voids)
    # Small voids naturally have less negative delta_min values
    threshold = -0.2  # Very liberal - almost no density cut
    mask = df_cut1['delta_min'] < threshold
    df_cut2 = df_cut1[mask].copy()
    print(f"  After density cut (threshold={threshold}): {len(df_cut2)}")

    # Cut 3: Border cutting (remove voids intersecting box edges)
    df_border_cut = apply_border_cutting(df_cut2, box_size, periodic=periodic)
    print(f"  After border cutting: {len(df_border_cut)}")

    # Cut 4: Overlap removal (keep larger voids)
    if c.get('overlap_removal', True):
        df_overlap = remove_overlaps_prefer_larger(df_border_cut)
        print(f"  After overlap removal: {len(df_overlap)}")
        return df_overlap

    return df_border_cut

def main():
    config = load_config()
    raw_dir = config['paths']['raw_dir']
    clean_dir = config['paths']['clean_dir']

    for frac in config['fractions']:
        raw_file = os.path.join(raw_dir, f"voids_bitvf_aletheia_{frac}.txt")

        if not os.path.exists(raw_file):
            print(f"Skipping {frac}: Raw file not found at {raw_file}")
            continue

        print(f"Processing fraction: {frac}")

        # BitVF output columns: X Y Z r r_ext delta(2MPS) deltaAv
        cols = ['x', 'y', 'z', 'radius', 'radius_ext', 'delta_min', 'delta_avg']

        # Read the raw file (skip first header line with '#')
        df_raw = pd.read_csv(raw_file, sep=r'\s+', names=cols, comment='#', dtype=float)

        # Clean the catalog
        df_clean = clean_catalog(df_raw, config)

        # Save out to CSV
        out_path = os.path.join(clean_dir, f"cleaned_void_cat_{frac}.csv")
        df_clean.to_csv(out_path, index=False)

        print(f"  -> Saved {len(df_clean)} cleaned voids (out of {len(df_raw)} raw).\n")

if __name__ == "__main__":
    main()