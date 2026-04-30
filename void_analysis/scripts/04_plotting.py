import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Configuration
BOX_SIZE = 250.0
V_eff = BOX_SIZE**3

FRACTIONS = ['20p', '10p', '5p', '2_5p'] 
LABELS = {'20p': 'ss = 20%', '10p': 'ss = 10%', '5p': 'ss = 5%', '2_5p': 'ss = 2.5%'}

COLORS = {'20p': '#9400D3',   
          '10p': '#0066CC',   
          '5p': '#2E8B57',    
          '2_5p': '#FFD700'}  

# Corrected Plotting Parameters to match Thesis
Rmin = 1.5 
Rmax = 20.0
bins_per_decade = 15 # Increased resolution for smaller range

def get_vsf(radii_array):
    """Calculates VSF and Poisson errors for shading."""
    bins = np.exp(np.arange(np.log(Rmin), np.log(Rmax) + 1e-12, np.log(10)/bins_per_decade))
    bin_centers = np.sqrt(bins[:-1] * bins[1:])
    dlnR = np.log(bins[1:] / bins[:-1])
    
    counts, _ = np.histogram(radii_array, bins=bins)
    
    number_density = counts / (V_eff * dlnR)
    errors = np.sqrt(counts) / (V_eff * dlnR)
    
    return bin_centers, number_density, errors

def main():
    plt.figure(figsize=(8, 8))
    
    for frac in FRACTIONS:
        file_path = f"../data/cleaned/cleaned_bitvf_aletheia_cat_new_{frac}.csv"
        
        if not os.path.exists(file_path):
            print(f"Skipping {frac}: Data not yet available.")
            continue
            
        df = pd.read_csv(file_path)
        r, n, err = get_vsf(df['radius'].values)
        
        valid = n > 0
        r = r[valid]
        n = n[valid]
        err = err[valid]
        
        y_plot = np.log10(n)
        y_upper = np.log10(n + err)
        y_lower = np.log10(np.clip(n - err, 1e-12, np.inf)) 
        
        plt.plot(r, y_plot, '-', lw=2.5, color=COLORS[frac], label=LABELS[frac])
        plt.fill_between(r, y_lower, y_upper, color=COLORS[frac], alpha=0.2)

    # Corrected Formatting
    plt.xscale('log')
    plt.xlabel(r"$R_v \ [h^{-1}\mathrm{Mpc}]$", fontsize=16)
    plt.ylabel(r"$\log_{10}(dn/d\ln R_v) \ [h^3\mathrm{Mpc}^{-3}]$", fontsize=16)
    
    # Locked axis limits to match the thesis
    plt.xlim(1.5, 20)
    plt.ylim(-8.0, -3.0)
    
    plt.grid(True, which='both', ls=':', alpha=0.5)
    plt.legend(loc='lower left', fontsize=14, frameon=True, edgecolor='black')
    
    plt.tight_layout()
    out_path = "../plots/BitVF_New.png"
    plt.savefig(out_path, dpi=300)
    print(f"\nSuccess! Plot saved to {out_path}")

if __name__ == "__main__":
    main()