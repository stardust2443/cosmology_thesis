#!/usr/bin/env python3
"""
param_exploration.py
A testing loop to evaluate how BitVF's internal parameters impact the resulting VSF.
Specifically tests variations in:
- N_rec (number of random realizations)
- th (stopping threshold)
- grid cell size
- Gaussian smoothing scale (sigma_sm)

This script assumes that BitVF output catalogs are named with their parameters, e.g.
raw/voids_Nrec50_th1e-4_grid256_sig2.0.txt
It runs the cleaning pipeline and computes/plots the VSF for comparison.
"""
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess

# Import the cleaning and vsf tools we built
from clean_voids import load_void_catalog, clean_catalog
from compute_vsf import compute_vsf

def main():
    data_dir = '/Users/arun/projects/my_cosmo_analysis/void_analysis/data'
    raw_dir = os.path.join(data_dir, 'raw')
    clean_dir = os.path.join(data_dir, 'cleaned')
    plot_dir = '/Users/arun/projects/my_cosmo_analysis/void_analysis/plots'
    os.makedirs(plot_dir, exist_ok=True)
    
    # We define the parameter sets we expect to explore
    files = glob.glob(os.path.join(raw_dir, "*_th*_*")) + glob.glob(os.path.join(raw_dir, "voids_recon_*"))
    
    # Try to load a baseline REVOLVER catalog for residual comparison
    baseline_file = os.path.join(raw_dir, "cleaned_revolver_cat_20p_R0.dat") # Example baseline
    baseline_vsf = None
    if not os.path.exists(baseline_file):
        baseline_file = os.path.join(clean_dir, "cleaned_revolver_cat_20p_R0.csv")
    
    if os.path.exists(baseline_file):
        try:
            print(f"Loading REVOLVER baseline: {baseline_file}")
            df_base = pd.read_csv(baseline_file) if baseline_file.endswith('.csv') else pd.read_csv(baseline_file, sep=r'\s+', header=None)
            r_col = 'radius' if 'radius' in df_base.columns else 3
            baseline_centers, baseline_abundance = compute_vsf(df_base[r_col].values, box_size=250.0)
            baseline_vsf = (baseline_centers, baseline_abundance)
        except Exception as e:
            print(f"Failed to load baseline: {e}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    if baseline_vsf is not None:
        valid_b = baseline_vsf[1] > 0
        ax1.plot(baseline_vsf[0][valid_b], baseline_vsf[1][valid_b], marker='o', linestyle='--', color='black', label='REVOLVER Baseline')
    
    for raw_file in files:
        base_name = os.path.basename(raw_file)
        print(f"Evaluating parameter config from file: {base_name}")
        
        # 1. Clean the catalog
        try:
            df_raw = load_void_catalog(raw_file)
            df_clean = clean_catalog(df_raw, border_cut=20.0)
            
            # Save cleaned
            clean_path = os.path.join(clean_dir, f"cleaned_{base_name}.csv")
            df_clean.to_csv(clean_path, index=False)
            
            # 2. Compute VSF
            bin_centers, abundance = compute_vsf(df_clean['radius'].values, box_size=250.0)
            valid = abundance > 0
            
            # Extract parameters for label (simplistic extraction for demonstration)
            label = base_name.replace('voids_', '').replace('.txt', '')
            
            # 3. Plot VSF
            ax1.plot(bin_centers[valid], abundance[valid], marker='s', linestyle='-', alpha=0.7, label=label)
            
            # 4. Plot Residuals if baseline exists
            if baseline_vsf is not None:
                # Interpolate baseline to BitVF bin centers for direct comparison
                base_interp = np.interp(bin_centers[valid], baseline_vsf[0], baseline_vsf[1])
                residual = (abundance[valid] - base_interp) / np.clip(base_interp, 1e-10, None)
                ax2.plot(bin_centers[valid], residual, marker='s', linestyle='-', alpha=0.7)
            
        except Exception as e:
            print(f"Failed on {base_name}: {e}")
            
    ax1.set_yscale('log')
    ax1.set_ylabel(r'Abundance $n(R) \ [h^3\rm{Mpc}^{-3}(\Delta\ln R)^{-1}]$', fontsize=14)
    ax1.set_title('BitVF Parameter Exploration: VSF Impact', fontsize=16)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    
    ax2.set_xscale('log')
    ax2.set_xlabel(r'Void Radius $R_v \ [h^{-1}\rm{Mpc}]$', fontsize=14)
    ax2.set_ylabel(r'Fractional Diff.', fontsize=14)
    ax2.axhline(0, color='black', linestyle='--', alpha=0.7)
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout()
    
    out_plot = os.path.join(plot_dir, 'param_exploration_vsf.png')
    plt.savefig(out_plot, dpi=300)
    plt.close()
    
    print(f"\nSaved parameter exploration VSF plot to {out_plot}")

if __name__ == "__main__":
    main()
