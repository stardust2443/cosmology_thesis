import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Configuration
BOX_SIZE = 250.0
V_eff = BOX_SIZE**3
FRACTIONS = ['2_5p', '5p', '10p', '20p']
TITLES = ['2.5% subsample', '5% subsample', '10% subsample', '20% subsample']

# Colors matching the thesis aesthetic
COLOR_REV = '#1f77b4' # Muted Blue
COLOR_BIT = '#ff7f0e' # Vibrant Orange

def get_vsf(radii_array, r_min=5.0, r_max=50.0, bins_per_decade=10):
    """Calculates VSF identically for any input catalog."""
    bins = np.exp(np.arange(np.log(r_min), np.log(r_max) + 1e-12, np.log(10)/bins_per_decade))
    bin_centers = np.sqrt(bins[:-1] * bins[1:])
    dlnR = np.log(bins[1:] / bins[:-1])
    
    counts, _ = np.histogram(radii_array, bins=bins)
    number_density = counts / (V_eff * dlnR)
    errors = np.sqrt(counts) / (V_eff * dlnR)
    
    return bin_centers, number_density, errors

def main():
    # 2. Setup the 2x2 Figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten() # Flattens the 2x2 matrix into a 1D array for easy looping
    
    for idx, frac in enumerate(FRACTIONS):
        ax = axes[idx]
        
        bitvf_file = f"../data/cleaned/cleaned_void_cat_{frac}.csv"
        revolver_file = f"../data/cleaned/cleaned_revolver_cat_{frac}.dat"
        
        # Check if files exist before plotting
        if not os.path.exists(bitvf_file) or not os.path.exists(revolver_file):
            ax.text(0.5, 0.5, f'Data missing for {frac}', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(TITLES[idx])
            continue

        # Load data
        df_bitvf = pd.read_csv(bitvf_file)
        df_revolver = pd.read_csv(revolver_file, sep=r'\s+', comment='#', names=['VoidID', 'x', 'y', 'z', 'radius', 'delta_min', 'delta_avg', 'lambda_v', 'DensRatio'])

        # Compute VSFs
        r_bitvf, n_bitvf, err_bitvf = get_vsf(df_bitvf['radius'].values)
        r_rev, n_rev, err_rev = get_vsf(df_revolver['radius'].values)

        # Plot REVOLVER Baseline
        ax.errorbar(r_rev, n_rev, yerr=err_rev, fmt='-s', color=COLOR_REV, alpha=0.8,
                     label='REVOLVER (z=0)', capsize=3, markersize=5)
                     
        # Plot BitVF
        ax.errorbar(r_bitvf, n_bitvf, yerr=err_bitvf, fmt='-o', color=COLOR_BIT, 
                     label='BitVF (z=0)', capsize=3, markersize=5)

        # Formatting each subplot
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(TITLES[idx], fontsize=14)
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend(fontsize=11, frameon=False)
        
        # Only add outer axis labels to keep the grid clean
        if idx >= 2: # Bottom row
            ax.set_xlabel(r'$R_{\rm eff} \ [h^{-1}{\rm Mpc}]$', fontsize=14)
        if idx % 2 == 0: # Left column
            ax.set_ylabel(r'$dn/d\ln R \ [h^3{\rm Mpc}^{-3}]$', fontsize=14)

    plt.tight_layout()
    plt.savefig("../plots/Figure_6_15_Recreation.pdf")
    print("Multi-panel comparison plot saved to plots/plot_all_0.pdf")

if __name__ == "__main__":
    main()