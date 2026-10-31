#!/usr/bin/env python3
"""
03_plot_comparison.py
Plots the bitVF empirical benchmark against the original REVOLVER catalogs,
including a bottom sub-panel for the statistical residuals (bitVF - REV) / sigma.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from matplotlib.lines import Line2D

# --- Publication Aesthetics ---
plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.size': 16,
    'axes.labelsize': 18,
    'legend.fontsize': 13,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'axes.linewidth': 1.2,
})

BOX_SIZE = 250.0
OUT_DIR = "../data/cleaned/benchmark_revolver_style"

# 1. Your BITVF Benchmark outputs (from our previous script)
bitvf_files = {
    "20p": os.path.join(OUT_DIR, "bitvf_revolver_benchmark_20p.csv"),
    "10p": os.path.join(OUT_DIR, "bitvf_revolver_benchmark_10p.csv"),
    "5p":  os.path.join(OUT_DIR, "bitvf_revolver_benchmark_5p.csv"),
    "2_5p": os.path.join(OUT_DIR, "bitvf_revolver_benchmark_2_5p.csv")
}

# 2. Your ORIGINAL PURE REVOLVER outputs (Update these paths!)
revolver_files = {
    "20p": "/Users/arun/projects/Revolver/cleaned/REVOLVER_dm_ss0p20_-0p80_NOov.dat",
    "10p": "/Users/arun/projects/Revolver/cleaned/REVOLVER_dm_ss0p10_-0p80_NOov.dat",
    "5p":  "/Users/arun/projects/Revolver/cleaned/REVOLVER_dm_ss0p05_-0p80_NOov.dat",
    "2_5p": "/Users/arun/projects/Revolver/cleaned/REVOLVER_dm_ss0p025_-0p80_NOov.dat"
}

def read_catalog(path, is_revolver=False):
    if not os.path.exists(path): return pd.DataFrame()
    
    if is_revolver:
        # REVOLVER format from your older snippet
        cols = ['VoidID','x','y','z','radius','delta_min','delta_avg','lambda_v','DensRatio']
        df = pd.read_csv(path, sep=r'\s+', comment='#', names=cols, header=None, engine='python')
    else:
        # bitVF CSV format
        df = pd.read_csv(path)
    return df

# --- Plotting Setup ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 11), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
fig.subplots_adjust(hspace=0.05)

bins = np.exp(np.arange(np.log(1.5), np.log(25.0) + 1e-12, np.log(10)/15))
bin_centers = np.sqrt(bins[:-1] * bins[1:])
dlnR = np.log(bins[1:] / bins[:-1])
V_eff = BOX_SIZE**3

colors = {'20p': '#6A3399', '10p': '#ADD8E6', '5p': '#3CB371', '2_5p': '#FFD700'}
mps_dict = {'20p': 5.4, '10p': 6.8, '5p': 8.6, '2_5p': 10.7}

# --- Data Processing & Plotting ---
for key in ['20p', '10p', '5p', '2_5p']:
    df_bitvf = read_catalog(bitvf_files[key], is_revolver=False)
    df_rev = read_catalog(revolver_files[key], is_revolver=True)
    
    if df_bitvf.empty or df_rev.empty:
        print(f"Skipping {key}: Missing one of the catalogs.")
        continue

    # Histograms
    N_bitvf, _ = np.histogram(df_bitvf['radius'].values, bins=bins)
    N_rev, _ = np.histogram(df_rev['radius'].values, bins=bins)
    
    dn_bitvf = N_bitvf / V_eff / dlnR
    dn_rev = N_rev / V_eff / dlnR
    
    # Poisson Error based on REVOLVER (Reference)
    err_rev_n = np.sqrt(N_rev)
    err_rev_dn = err_rev_n / V_eff / dlnR
    
    # Physical Cutoff Mask
    min_R = max(2.0, mps_dict.get(key, 5.0) / 3.0)
    mask = (N_rev > 0) & (N_bitvf > 0) & (bin_centers >= min_R * 0.95)
    
    x_plot = bin_centers[mask]
    
    # --- Top Panel: VSF ---
    y_bitvf = np.log10(dn_bitvf[mask])
    y_rev = np.log10(dn_rev[mask])
    err_log = err_rev_dn[mask] / (dn_rev[mask] * np.log(10))
    
    c = colors[key]
    ax1.plot(x_plot, y_bitvf, '-', lw=2.5, color=c)
    ax1.plot(x_plot, y_rev, '--', lw=2.5, color=c)
    ax1.fill_between(x_plot, y_rev - err_log, y_rev + err_log, color=c, alpha=0.2, edgecolor='none')
    
    # --- Bottom Panel: Residuals (bitVF - REV) / err_REV ---
    # This measures how many standard deviations bitVF is away from REVOLVER
    residual = (dn_bitvf[mask] - dn_rev[mask]) / err_rev_dn[mask]
    ax2.plot(x_plot, residual, 'o-', color=c, lw=2.0, markersize=6)

# --- Top Panel Formatting ---
ax1.set_xscale('log')
ax1.set_ylim(-8.0, -3.2)
ax1.set_ylabel(r"$\log(\mathrm{d}n/\mathrm{d}{\ln R_v}) \ [h^3 \mathrm{Mpc}^{-3}]$")
ax1.grid(True, which='major', ls=':', alpha=0.4, color='gray')

# Custom Legend to avoid 8 separate labels
legend_elements = [
    Line2D([0], [0], color='k', lw=2.5, linestyle='-', label='BitVF (Empirical Scaled)'),
    Line2D([0], [0], color='k', lw=2.5, linestyle='--', label='Original REVOLVER')
]
for k, c in colors.items():
    label_str = f'ss = {k.replace("p", "%").replace("_", ".")}'
    legend_elements.append(Line2D([0], [0], color=c, lw=6, label=label_str))

ax1.legend(handles=legend_elements, loc='lower left', ncol=2, frameon=False, columnspacing=1.5)

# --- Bottom Panel Formatting ---
ax2.axhline(0, color='black', lw=1.5, ls='--')
ax2.set_ylim(-3.5, 3.5) # Sets the bound to +/- 3 Sigma
ax2.set_ylabel(r"$\frac{\Delta n}{\sigma_{\mathrm{REV}}}$", fontsize=22)
ax2.set_xlabel(r"$R_v \ [h^{-1}\mathrm{Mpc}]$")
ax2.grid(True, which='major', ls=':', alpha=0.4, color='gray')

# Tick Formatting
for ax in [ax1, ax2]:
    ax.tick_params(axis='both', which='major', direction='in', length=8, top=True, right=True)
    ax.tick_params(axis='both', which='minor', direction='in', length=4, top=True, right=True)

ax2.set_xticks([2, 5, 10, 15])
ax2.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
ax2.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=20))
ax2.xaxis.set_minor_formatter(NullFormatter())

plt.savefig(os.path.join(OUT_DIR, "BitVF_vs_Revolver_Comparison.pdf"), dpi=300, bbox_inches='tight')
print("Comparison Plot Generated Successfully!")