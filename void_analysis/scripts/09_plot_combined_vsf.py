import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def plot_combined_vsf(file_5p, file_2_5p, box_size=250):
    print("Calculating Combined VSF with Logarithmic Binning...")
    
    # 1. Logarithmic Binning (from 2 to 25 to safely cover the "20ish" request)
    r_bins = np.geomspace(2.0, 25.0, 12)
    r_centers = np.sqrt(r_bins[:-1] * r_bins[1:]) # Geometric centers
    dlnR = np.log(r_bins[1:] / r_bins[:-1])       # Log bin width
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    def add_to_plot(file_path, label, color, marker):
        try:
            df = pd.read_csv(file_path)
            # radii = df['radius'].values
            radii = df.iloc[:, 3].values

            counts, _ = np.histogram(radii, bins=r_bins)
            vol = box_size**3
            
            # Abundance: dn/dlnR
            dn_dlnR = counts / vol / dlnR
            errors = np.sqrt(np.where(counts > 0, counts, 1.0)) / vol / dlnR
            
            # Mask out zero-count bins to prevent log-scale crash
            mask = counts > 0
            ax.errorbar(r_centers[mask], dn_dlnR[mask], yerr=errors[mask], 
                        fmt=marker, color=color, capsize=4, markersize=7, 
                        label=label, alpha=0.85)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # 2. Add both datasets
    add_to_plot(file_5p, '5% Subsample', 'blue', 'o')
    add_to_plot(file_2_5p, '2.5% Subsample', 'red', 's')
    
    # 3. Formatting to supervisor's exact specifications
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1.8, 25) # Locks the view to ~2 to 20ish
    
    # Force x-axis to show normal numbers (2, 10, 20) instead of 10^1
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks([2, 5, 10, 20])
    
    ax.set_xlabel(r'Void Radius $R$ $[h^{-1}\mathrm{Mpc}]$', fontsize=14)
    ax.set_ylabel(r'Abundance $dn/d\ln R$ $[h^3 \mathrm{Mpc}^{-3}]$', fontsize=14)
    ax.set_title('BitVF VSF (Single Realization) 0 cat', fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    
    plt.tight_layout()
    plt.savefig('../plots/cleaned_bitvf_catog.pdf')
    print("Success! Plot saved to ../plots/cleaned_bitvf_catog.pdf")

if __name__ == "__main__":
    # Ensure these point to your files
    FILE_5P = "../data/cleaned/cleaned_bitvf_aletheia_cat_5p.csv"
    FILE_2_5P = "../data/cleaned/cleaned_bitvf_aletheia_cat_2_5p.csv"
    
    plot_combined_vsf(FILE_5P, FILE_2_5P)