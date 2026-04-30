import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration
BOX_SIZE = 250.0
fractions = ['2_5p'] # Add '5p', '10p', '20p' here tomorrow!
colors = {'2_5p': 'blue', '5p': 'orange', '10p': 'green', '20p': 'red'}

plt.figure(figsize=(8, 6))

for frac in fractions:
    # 1. Load the data
    file_path = f"data/cleaned/cleaned_bitvf_aletheia_cat_{frac}.csv"
    if not os.path.exists(file_path):
        continue
    
    df = pd.read_csv(file_path)
    radii = df['radius'].values
    
    # 2. Setup Logarithmic Bins (Rmin=5, Rmax=50)
    bins = np.exp(np.arange(np.log(5.0), np.log(50.0) + 1e-12, 0.15))
    bin_centers = np.sqrt(bins[:-1] * bins[1:])
    dlnR = np.log(bins[1:] / bins[:-1])
    V_eff = BOX_SIZE**3
    
    # 3. Calculate VSF
    counts, _ = np.histogram(radii, bins=bins)
    number_density = counts / (V_eff * dlnR)
    
    # Calculate Poisson error bars
    errors = np.sqrt(counts) / (V_eff * dlnR)
    
    # 4. Plot
    plt.errorbar(bin_centers, number_density, yerr=errors, fmt='-o', 
                 color=colors[frac], label=f'{frac.replace("_", ".")} Tracers', 
                 capsize=3)

# Formatting
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Effective Radius $R_{\rm eff} \ [h^{-1}{\rm Mpc}]$', fontsize=14)
plt.ylabel(r'$dn/d\ln R \ [h^3{\rm Mpc}^{-3}]$', fontsize=14)
plt.title('Convergence of the Void Size Function', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.5)

plt.tight_layout()
plt.savefig("plots/VSF_Convergence_1.pdf")
print("Plot saved to plots/VSF_Convergence_1.pdf")