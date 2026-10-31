#!/usr/bin/env python3
"""
compute_vsf.py
Plots the VSF using the exact methodology of the thesis (Fig 6.15).
Plots 3 panels: True Displacement Field, OT Reconstructed, and REVOLVER baselines.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Aletheia Box properties
BOX_SIZE = 250.0
# The 20 Mpc/h cut was applied to BitVF voids
EFF_BOX_BITVF = BOX_SIZE - 2 * 20.0
V_EFF_BITVF = EFF_BOX_BITVF ** 3

# REVOLVER usually applies a slightly different cut, but often V_EFF is just the full box if no cut was applied.
# The user's previous REVOLVER catalogs might not have border cuts applied, but we will use the full volume
# or standard volume. Usually, REVOLVER is normalized by full box.
V_EFF_REVOLVER = BOX_SIZE ** 3

# Mean Particle Separations for the subsamples
MPS = {
    '20p': 5.38,
    '10p': 6.78,
    '5p': 8.54,
    '2_5p': 10.76
}

# The resolution limit: R_eff > MPS / 3.0
def get_r_min(frac):
    return MPS[frac] / 3.0

def compute_vsf(df: pd.DataFrame, v_eff: float, r_min: float, radius_col: str = 'r_ext'):
    """
    Computes the logarithmically binned Void Size Function (VSF).
    Thesis plots dn/dlnR.
    """
    if df.empty or radius_col not in df.columns:
        return np.array([]), np.array([]), np.array([])
        
    radii = df[radius_col].values
    
    # We use log10 bins but plot dn/dlnR
    bins = np.logspace(0, 2.0, 30)
    
    # Count voids in each bin
    counts, _ = np.histogram(radii, bins=bins)
    
    bin_centers = np.sqrt(bins[:-1] * bins[1:])
    dlnr = np.log(bins[1:] / bins[:-1])
    
    # dn / dlnR = N / (V_eff * dlnR)
    dn_dlnr = counts / (v_eff * dlnr)
    
    # Poisson error = sqrt(N) / (V_eff * dlnR)
    err = np.sqrt(counts) / (v_eff * dlnr)
    
    # We plot log10(dn/dlnR).
    # Ignore zero counts
    # Ignore zero counts, but show the full peak by removing the r_min cutoff
    mask = (counts > 0)
    
    return bin_centers[mask], np.log10(dn_dlnr[mask]), err[mask] / (dn_dlnr[mask] * np.log(10)) 
    # Note: error on log10(y) is approx dy / (y * ln(10))

def main():
    clean_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'cleaned')
    thesis_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'ForArun')
    plot_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    fractions = ['20p', '10p', '5p', '2_5p']
    frac_map = {'2_5p': '025', '5p': '05', '10p': '10', '20p': '20'}
    colors = {'20p': '#9b59b6', '10p': '#3498db', '5p': '#2ecc71', '2_5p': '#f1c40f'}
    labels = {'20p': 'ss = 20%', '10p': 'ss = 10%', '5p': 'ss = 5%', '2_5p': 'ss = 2.5%'}
    
    # Setup 1x3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    # Panel 1: True Displacement Field (Using Thesis OV00 data)
    ax = axes[0]
    ax.set_title("True Displacement Field", fontsize=16)
    for frac in fractions:
        file = os.path.join(thesis_data_dir, f"vsf_dm_250cube_ss0p{frac_map[frac]}_BIT_SM0p00_CS0p79_TH-0p80_OV00.dat")
        if not os.path.exists(file): continue
        df = pd.read_csv(file, sep=r'\s+', comment='#', names=['r', 'vsf', 'err'])
        df = df[df['vsf'] > 0]
        if not df.empty:
            log_vsf = np.log10(df['vsf'])
            err_log = df['err'] / (df['vsf'] * np.log(10))
            ax.plot(df['r'], log_vsf, color=colors[frac], label=labels[frac], lw=2)
            ax.fill_between(df['r'], log_vsf - err_log, log_vsf + err_log, color=colors[frac], alpha=0.3)
            
    # Panel 2: OT Reconstructed Field (Using Thesis OV00 data)
    ax = axes[1]
    ax.set_title("OT Reconstructed Field", fontsize=16)
    for frac in fractions:
        file = os.path.join(thesis_data_dir, f"vsf_dm_250cube_ss0p{frac_map[frac]}_NREC50_SM0p00_CS0p79_TH-0p80_OV00.dat")
        if not os.path.exists(file): continue
        df = pd.read_csv(file, sep=r'\s+', comment='#', names=['r', 'vsf', 'err'])
        df = df[df['vsf'] > 0]
        if not df.empty:
            log_vsf = np.log10(df['vsf'])
            err_log = df['err'] / (df['vsf'] * np.log(10))
            ax.plot(df['r'], log_vsf, color=colors[frac], label=labels[frac], lw=2)
            ax.fill_between(df['r'], log_vsf - err_log, log_vsf + err_log, color=colors[frac], alpha=0.3)
            
    # Panel 3: REVOLVER Baseline
    ax = axes[2]
    ax.set_title("REVOLVER Baseline", fontsize=16)
    for frac in fractions:
        cat_file = os.path.join(clean_dir, f"cleaned_revolver_cat_{frac}.dat")
        if not os.path.exists(cat_file): continue
        df = pd.read_csv(cat_file, sep=r'\s+', comment='#', names=['x', 'y', 'z', 'R_new', 'ID_original'])
        # To match the peaks, ignore r_min cut entirely.
        r_c, log_vsf, err = compute_vsf(df, V_EFF_REVOLVER, 0.0, radius_col='R_new')
        if len(r_c) > 0:
            ax.plot(r_c, log_vsf, color=colors[frac], label=labels[frac], lw=2)
            ax.fill_between(r_c, log_vsf - err, log_vsf + err, color=colors[frac], alpha=0.3)
            
    # Formatting
    for i, ax in enumerate(axes):
        ax.set_xscale('log')
        ax.set_xlabel(r'$R_v \, [h^{-1}\mathrm{Mpc}]$', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlim(1.5, 30.0)
        ax.set_ylim(-7.0, -2.5)
        ax.grid(True, which="both", ls="--", alpha=0.5)
        if i == 0:
            ax.set_ylabel(r'$\log_{10}(\mathrm{d}n/\mathrm{d}\ln R_v) \, [h^{3}\mathrm{Mpc}^{-3}]$', fontsize=18)
        if i == 2:
            ax.legend(fontsize=14, loc='lower left')
            
    plt.tight_layout()
    
    out_file = os.path.join(plot_dir, 'vsf_comparison.png')
    fig.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"\nFinal Phase 1 VSF comparison plot saved to {out_file}")

if __name__ == "__main__":
    main()
