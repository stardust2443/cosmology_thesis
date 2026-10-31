import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --- 1. Global Aesthetic Settings (Publication Style) ---
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'axes.linewidth': 1.5,
    'errorbar.capsize': 4,
    'lines.markeredgewidth': 1.5
})

def plot_combined_vsf(file_20p, file_10p, file_5p, file_2_5p, box_size=250):
    print("Generating VSF Plot...")
    
    # Logarithmic Binning
    r_bins = np.geomspace(2.0, 25.0, 12)
    r_centers = np.sqrt(r_bins[:-1] * r_bins[1:]) 
    dlnR = np.log(r_bins[1:] / r_bins[:-1])       
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    def add_to_plot(file_path, label, color, marker):
        try:
            df = pd.read_csv(file_path)
            df = df[df.iloc[:, 6] < -0.8] # Apply delta_avg < -0.8 cut
            radii = df.iloc[:, 3].values

            counts, _ = np.histogram(radii, bins=r_bins)
            vol = box_size**3
            
            dn_dlnR = counts / vol / dlnR
            
            # --- FIX: Smarter Error Bar Bounding ---
            # Standard upper error
            err_up = np.sqrt(counts) / vol / dlnR
            
            # If standard error pushes the lower bound below 0, cap it at a 50% drop
            # so it doesn't plunge infinitely downward on the log scale.
            err_down_raw = np.sqrt(counts) / vol / dlnR
            err_down = np.where(dn_dlnR - err_down_raw <= 0, 
                                dn_dlnR * 0.5,  # Max drop is half a decade
                                err_down_raw)
            
            mask = counts > 0
            
            # Plot with thicker lines and modern colors
            ax.errorbar(r_centers[mask], dn_dlnR[mask], 
                        yerr=[err_down[mask], err_up[mask]], 
                        fmt=marker, color=color, 
                        elinewidth=1.8, capthick=1.8, markersize=8, 
                        label=label, alpha=0.9)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # Using a high-contrast, professional palette
    add_to_plot(file_20p, '20% Subsample', '#4B0092', 'x') # Deep Purple
    add_to_plot(file_10p, '10% Subsample', "#0032B2", 'o') # Ocean Blue
    add_to_plot(file_5p, '5% Subsample', "#009E1D", '^')   # Emerald Green
    add_to_plot(file_2_5p, '2.5% Subsample', "#D53C00", 's') # Deep Coral
        
    # --- 3. Fixing the Axis Ticks ---
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1.8, 25) 
    
    # Explicitly define major and minor ticks to force matplotlib's hand
    major_ticks = [2, 5, 10, 20]
    minor_ticks = [3, 4, 6, 7, 8, 9, 12, 15, 25]
    
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    
    # Use FuncFormatter to keep standard numbers without breaking minor ticks
    ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.xaxis.set_minor_formatter(plt.NullFormatter()) # Hide minor tick text
    
    ax.set_xlabel(r'Void Radius $R$ $[h^{-1}\mathrm{Mpc}]$')
    ax.set_ylabel(r'Abundance $dn/d\ln R$ $[h^3 \mathrm{Mpc}^{-3}]$')
    ax.set_title('BitVF VSF (01/05) order 1', pad=15)
    
    ax.legend(frameon=True, edgecolor='black', framealpha=1.0)
    
    # Clean, subtle grid
    ax.grid(True, which="major", ls="-", alpha=0.2, color='gray')
    ax.grid(True, which="minor", ls=":", alpha=0.15, color='gray')
    
    plt.tight_layout()
    plt.savefig('../plots/cleaned_recon_50_bitvf_v1.png', dpi=300)
    print("Success! Plot saved at cleaned_recon_50_bitvf_v1.png.")

if __name__ == "__main__":
    FILE_20P = "../data/cleaned/cleaned_recon_50_bitvf_hybrid_20p.csv"
    FILE_10P = "../data/cleaned/cleaned_recon_50_bitvf_hybrid_10p.csv"
    FILE_5P = "../data/cleaned/cleaned_recon_50_bitvf_hybrid_5p.csv"
    FILE_2_5P = "../data/cleaned/cleaned_recon_50_bitvf_hybrid_2_5p.csv"

    plot_combined_vsf(FILE_20P, FILE_10P, FILE_5P, FILE_2_5P)