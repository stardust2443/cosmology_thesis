import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm', 
    'font.size': 16,
    'axes.labelsize': 20,
    'legend.fontsize': 15,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'axes.linewidth': 1.2,
})

# 2. Configuration
BOX_SIZE = 250.0
V_eff = BOX_SIZE**3

FRACTIONS = ['20p', '10p', '5p', '2_5p']
LABELS = {'20p': 'ss = 20%', '10p': 'ss = 10%', '5p': 'ss = 5%', '2_5p': 'ss = 2.5%'}

COLORS = {'20p': '#6A3399',   # Deep Purple
          '10p': '#ADD8E6',   # Light Blue
          '5p':  '#3CB371',   # Sea Green
          '2_5p': '#FFD700'}  # Yellow

Rmin = 1.8  
Rmax = 20.0  
bins_per_decade = 10  

def get_vsf(radii_array):
    bins = np.exp(np.arange(np.log(Rmin), np.log(Rmax) + 1e-12, np.log(10)/bins_per_decade))
    bin_centers = np.sqrt(bins[:-1] * bins[1:])
    dlnR = np.log(bins[1:] / bins[:-1])

    counts, _ = np.histogram(radii_array, bins=bins)
    number_density = counts / (V_eff * dlnR)
    errors = np.sqrt(counts) / (V_eff * dlnR)

    return bin_centers, number_density, errors

def main():
    fig, ax = plt.subplots(figsize=(8, 6))

    for frac in FRACTIONS:
        file_path = f"../data/cleaned/cleaned_bitvf_standard_{frac}.csv"

        if not os.path.exists(file_path):
            print(f"Skipping {frac}: Data not found")
            continue

        df = pd.read_csv(file_path)
        r, n, err = get_vsf(df['radius'].values)

        valid = n > 0
        r = r[valid]
        n = n[valid]
        err = err[valid]

        # NO SHIFTS applied. Plotting the pure physical data.
        y_plot = np.log10(n)
        err_log = err / (n * np.log(10))

        y_upper = y_plot + err_log
        y_lower = y_plot - err_log

        ax.plot(r, y_plot, '-', lw=2.5, color=COLORS[frac], label=LABELS[frac])
        ax.fill_between(r, y_lower, y_upper, color=COLORS[frac], alpha=0.3, edgecolor='none')

    # --- 3. Strict Formatting to Match Target Image ---
    
    # Axes limits
    ax.set_xlim(1.5, 20)
    ax.set_ylim(-8.0, -2.8)
    
    # Log scale for X
    ax.set_xscale('log')
    
    # Tick Formatting (Inward pointing, top and right enabled)
    ax.tick_params(axis='both', which='major', direction='in', length=8, top=True, right=True)
    ax.tick_params(axis='both', which='minor', direction='in', length=4, top=True, right=True)
    
    # Custom X Ticks to match image exactly
    ax.set_xticks([2, 5, 10, 15])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=20))
    ax.xaxis.set_minor_formatter(NullFormatter())

    # Labels
    ax.set_xlabel(r"$R_v \ [h^{-1}\mathrm{Mpc}]$")
    ax.set_ylabel(r"$\log(\mathrm{d}n/\mathrm{d}{\ln R_v}) \ [h^3 \mathrm{Mpc}^{-3}]$")
    
    # Subtle Grid
    ax.grid(True, which='major', ls=':', alpha=0.4, color='gray')

    # Add the Text Box ("BitVF - Real v.")
    ax.text(0.12, 0.40, "BitVF 01/05", transform=ax.transAxes, 
            fontsize=18, verticalalignment='center',
            bbox=dict(facecolor='white', edgecolor='black', alpha=1.0, pad=10))

    # Add Legend (No frame, 2 columns, located at the bottom left)
    ax.legend(loc='lower left', bbox_to_anchor=(0.05, 0.1), ncol=2, frameon=False, 
              handlelength=2.0, columnspacing=1.5)

    plt.tight_layout()
    plt.savefig("../plots/plots_0105/cleaned_bitvf_standard_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()