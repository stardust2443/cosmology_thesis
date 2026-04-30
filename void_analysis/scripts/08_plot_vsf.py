import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_single_vsf(void_file, box_size=250):
    print(f"Calculating VSF for {void_file}...")
    
    # Load the cleaned void catalog
    df = pd.read_csv(void_file)
    radii = df['radius'].values
    
    # Define bins
    r_bins = np.linspace(10, 50, 20)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    
    # Calculate Volume-normalized abundance (dn/dlnR or dn/dR)
    counts, _ = np.histogram(radii, bins=r_bins)
    vol = box_size**3
    dn_dr = counts / vol / np.diff(r_bins)
    
    # Calculate standard Poisson error bars for the visual
    errors = np.sqrt(np.where(counts > 0, counts, 1)) / vol / np.diff(r_bins)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(r_centers, dn_dr, yerr=errors, fmt='o', color='black', 
                capsize=3, markersize=6, label='Measured VSF (bitVF)')
    
    ax.set_yscale('log')
    ax.set_xlabel('Void Radius R [Mpc/h]', fontsize=14)
    ax.set_ylabel(r'Abundance $dn/dR$ $[h/Mpc]^4$', fontsize=14)
    ax.set_title('Measured Void Size Function (Single Realization)', fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('../plots/single_vsf_check_2_5p.pdf')
    print("Success! VSF plot saved to ../plots/single_vsf_check_2_5p.pdf")

if __name__ == "__main__":
    VOID_DATA = "../data/cleaned/cleaned_void_cat_2_5p.csv" 
    plot_single_vsf(VOID_DATA)