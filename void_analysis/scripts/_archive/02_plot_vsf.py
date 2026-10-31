#!/usr/bin/env python3
"""
02_plot_vsf.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree as KDTree
from tqdm import tqdm
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

# --- Publication Aesthetics ---
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

BOX_SIZE = 250.0
OUT_DIR = "../data/cleaned/Thesis_Exact"
os.makedirs(OUT_DIR, exist_ok=True)

# Point these to your RAW, untouched Aletheia bitVF catalogs
samples = {
    "20p": "../data/raw/voids_recon_50_bitvf_aletheia_20p",
    "10p": "../data/raw/voids_recon_50_bitvf_aletheia_10p",
    "5p":  "../data/raw/voids_recon_50_bitvf_aletheia_5p",
    "2_5p": "../data/raw/voids_recon_50_bitvf_aletheia_2_5p"
}

def process_svdw(df):
    if df.empty: return df
    
    # 1. Analytic SvdW Shrinking (This creates the bump!)
    # No density threshold is used, allowing the math to handle the noise.
    density_ratio = np.clip(df['delta_avg'] + 1.0, 1e-4, None)
    df['radius'] = df['radius'] * np.cbrt(density_ratio / 0.2)
    
    # 2. Top-Down Sort by the new theoretical radii
    df_sorted = df.sort_values(by='radius', ascending=False).reset_index(drop=True)
    coords = df_sorted[['x', 'y', 'z']].values
    radii = df_sorted['radius'].values
    keep_mask = np.ones(len(df_sorted), dtype=bool)
    
    tree = KDTree(coords)
    
    # 3. Strict Touching Overlap Removal (dist < R1 + R2)
    for i in tqdm(range(len(df_sorted)), desc="  Pruning"):
        if not keep_mask[i]: continue
            
        search_radius = radii[i] * 2.01 
        neighbors = tree.query_ball_point(coords[i], r=search_radius)
        
        for j in neighbors:
            if j > i and keep_mask[j]:
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < (radii[i] + radii[j]):
                    keep_mask[j] = False
                    
    df_clean = df_sorted[keep_mask].copy()
    
    # 4. Standard Physical Cut (Keep everything R >= 2.0 to show the bump)
    return df_clean[df_clean['radius'] >= 2.0].copy()

# --- Execution ---
cleaned_results = {}
for key, path in samples.items():
    if not os.path.exists(path): continue
    print(f"\n=== Processing SvdW Analytic Sample: {key} ===")
    
    cols = ['x', 'y', 'z', 'radius', 'r_ext', 'delta_min', 'delta_avg']
    df_raw = pd.read_csv(path, sep=r'\s+', comment='#', names=cols)
    
    df_clean = process_svdw(df_raw)
    cleaned_results[key] = df_clean
    
    outcsv = os.path.join(OUT_DIR, f"bitvf_svdw_exact_{key}.csv")
    df_clean.to_csv(outcsv, index=False)

# --- Plotting ---
if cleaned_results:
    print("\nGenerating Thesis Exact VSF plot...")
    fig, ax = plt.subplots(figsize=(9, 9))
    
    bins = np.exp(np.arange(np.log(1.5), np.log(25.0) + 1e-12, np.log(10)/15))
    bin_centers = np.sqrt(bins[:-1] * bins[1:])
    dlnR = np.log(bins[1:] / bins[:-1])
    V_eff = BOX_SIZE**3

    colors = {'20p': '#6A3399', '10p': '#ADD8E6', '5p': '#3CB371', '2_5p': '#FFD700'}
    
    for key, cat in cleaned_results.items():
        if cat.empty: continue
        
        y, _ = np.histogram(cat['radius'].values, bins=bins)
        dn_dlnR = y / V_eff / dlnR
        err = np.sqrt(y) / V_eff / dlnR
        
        # We plot all bins R >= 2.0 to capture the upward slope
        mask = (dn_dlnR > 0) & (bin_centers >= 2.0)
        y_vals = dn_dlnR[mask]
        
        y_plot = np.log10(y_vals)
        err_log = err[mask] / (y_vals * np.log(10))
        label_str = f'ss = {key.replace("p", "%").replace("_", ".")}'
        
        ax.plot(bin_centers[mask], y_plot, '-', lw=2.5, color=colors[key], label=label_str)
        ax.fill_between(bin_centers[mask], y_plot - err_log, y_plot + err_log, color=colors[key], alpha=0.3, edgecolor='none')

    ax.set_xscale('log')
    ax.set_xlim(1.5, 20)
    ax.set_ylim(-8.0, -3.2)
    
    ax.tick_params(axis='both', which='major', direction='in', length=8, top=True, right=True)
    ax.tick_params(axis='both', which='minor', direction='in', length=4, top=True, right=True)
    ax.set_xticks([2, 5, 10, 15])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=20))
    ax.xaxis.set_minor_formatter(NullFormatter())

    ax.set_xlabel(r"$R_v \ [h^{-1}\mathrm{Mpc}]$")
    ax.set_ylabel(r"$\log(\mathrm{d}n/\mathrm{d}{\ln R_v}) \ [h^3 \mathrm{Mpc}^{-3}]$")
    
    ax.text(0.12, 0.40, "BitVF VSF", transform=ax.transAxes, 
            fontsize=18, verticalalignment='center',
            bbox=dict(facecolor='white', edgecolor='black', alpha=1.0, pad=10))

    ax.legend(loc='lower left', bbox_to_anchor=(0.05, 0.1), ncol=2, frameon=False, columnspacing=1.5)
    ax.grid(True, which='major', ls=':', alpha=0.4, color='gray')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "bitvf_vsf_0405.pdf"), dpi=300, bbox_inches='tight')
    print("Success! Thesis Match Plot saved.")