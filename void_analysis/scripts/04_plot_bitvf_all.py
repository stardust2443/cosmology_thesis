import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Configuration
BOX_SIZE = 250.0
V_eff = BOX_SIZE**3

# Ordered highest to lowest density to match your image legend
FRACTIONS = ['20p', '10p', '5p', '2_5p']
LABELS = {'20p': 'ss = 20%', '10p': 'ss = 10%', '5p': 'ss = 5%', '2_5p': 'ss = 2.5%'}

# Colors matching your reference image
COLORS = {'20p': '#9400D3',   # Purple
          '10p': '#0066CC',   # Blue
          '5p': '#2E8B57',    # Green
          '2_5p': '#FFD700'}  # Yellow/Gold

# Plotting parameters aligned with your reference image
Rmin = 2.0  # Start from 2 to capture the peak at 3-4
Rmax = 20.0  # Match reference image scale
bins_per_decade = 15  # Fine binning to resolve the peak

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
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    for frac in FRACTIONS:
        # Target cleaned data
        file_path = f"../data/cleaned/cleaned_bitvf_aletheia_cat_{frac}.csv"

        if not os.path.exists(file_path):
            print(f"Skipping {frac}: Data not yet available at {file_path}")
            continue

        df = pd.read_csv(file_path)
        print(f"Processing {frac}: {len(df)} voids")

        r, n, err = get_vsf(df['radius'].values)

        # Filter out empty bins to avoid log10(0) math errors
        valid = n > 0
        r = r[valid]
        n = n[valid]
        err = err[valid]

        # Convert to log10 space for plotting (matching your reference image)
        y_plot = np.log10(n)

        # Calculate logarithmic errors
        err_log = err / (n * np.log(10))

        y_upper = y_plot + err_log
        y_lower = y_plot - err_log

        # Plot solid line and shaded error region
        ax.plot(r, y_plot, '-', lw=2.5, color=COLORS[frac], label=LABELS[frac])
        ax.fill_between(r, y_lower, y_upper, color=COLORS[frac], alpha=0.2)

    # Formatting to match the provided image
    ax.set_xscale('log')
    ax.set_xlabel(r"$R_v \ [h^{-1}\mathrm{Mpc}]$", fontsize=14)
    ax.set_ylabel(r"$\log(dn/d\ln R_v) \ [h^3\mathrm{Mpc}^{-3}]$", fontsize=14)
    ax.set_title("BitVF - Real v.", fontsize=14)

    # Set limits to match the visual framing of your reference
    ax.set_xlim(2, 15)
    ax.set_ylim(-7, -3.5)

    ax.grid(True, which='both', ls=':', alpha=0.8)
    ax.legend(fontsize=12, frameon=True, edgecolor='black', loc='upper right')

    # Save
    plt.tight_layout()
    out_path = "../plots/BitVF_Recon_Cleaned.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nSuccess! Plot saved to {out_path}")

    # Also save as PNG for preview
    out_path_png = "../plots/BitVF_Recon_Cleaned.png"
    plt.savefig(out_path_png, dpi=150, bbox_inches='tight')
    print(f"Also saved PNG to {out_path_png}")
    plt.show()

if __name__ == "__main__":
    main()