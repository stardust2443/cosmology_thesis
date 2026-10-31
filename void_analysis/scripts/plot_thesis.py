import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_data(ov_type):
    data_dir = '/Users/arun/projects/my_cosmo_analysis/void_analysis/data/ForArun'
    plot_dir = '/Users/arun/projects/my_cosmo_analysis/void_analysis/plots'
    os.makedirs(plot_dir, exist_ok=True)
    
    fractions = ['20', '10', '05', '025']
    colors = {'20': '#9b59b6', '10': '#3498db', '05': '#2ecc71', '025': '#f1c40f'}
    labels = {'20': 'ss = 20%', '10': 'ss = 10%', '05': 'ss = 5%', '025': 'ss = 2.5%'}
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    
    # BIT = True Displacement Field
    ax = axes[0]
    ax.set_title(f"True Displacement Field ({ov_type})", fontsize=16)
    for frac in fractions:
        file = os.path.join(data_dir, f"vsf_dm_250cube_ss0p{frac}_BIT_SM0p00_CS0p79_TH-0p80_{ov_type}.dat")
        if os.path.exists(file):
            df = pd.read_csv(file, sep=r'\s+', comment='#', names=['r', 'vsf', 'err'])
            mask = df['vsf'] > 0
            df = df[mask]
            ax.plot(df['r'], np.log10(df['vsf']), color=colors[frac], label=labels[frac], lw=2)
            err_log = df['err'] / (df['vsf'] * np.log(10))
            ax.fill_between(df['r'], np.log10(df['vsf']) - err_log, np.log10(df['vsf']) + err_log, color=colors[frac], alpha=0.3)
            
    # NREC50 = OT Reconstructed Field
    ax = axes[1]
    ax.set_title(f"OT Reconstructed Field ({ov_type})", fontsize=16)
    for frac in fractions:
        file = os.path.join(data_dir, f"vsf_dm_250cube_ss0p{frac}_NREC50_SM0p00_CS0p79_TH-0p80_{ov_type}.dat")
        if os.path.exists(file):
            df = pd.read_csv(file, sep=r'\s+', comment='#', names=['r', 'vsf', 'err'])
            mask = df['vsf'] > 0
            df = df[mask]
            ax.plot(df['r'], np.log10(df['vsf']), color=colors[frac], label=labels[frac], lw=2)
            err_log = df['err'] / (df['vsf'] * np.log(10))
            ax.fill_between(df['r'], np.log10(df['vsf']) - err_log, np.log10(df['vsf']) + err_log, color=colors[frac], alpha=0.3)
            
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xlabel(r'$R_v \, [h^{-1}\mathrm{Mpc}]$', fontsize=18)
        ax.set_xlim(1.5, 30.0)
        ax.set_ylim(-7.0, -2.5)
        ax.grid(True, which="both", ls="--", alpha=0.5)
    
    axes[0].set_ylabel(r'$\log_{10}(\mathrm{d}n/\mathrm{d}\ln R_v) \, [h^{3}\mathrm{Mpc}^{-3}]$', fontsize=18)
    axes[0].legend(fontsize=14)
    
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, f'thesis_truth_{ov_type}.png'), dpi=300)
    print(f"Saved thesis_truth_{ov_type}.png")

if __name__ == '__main__':
    plot_data('OV00')
    plot_data('OV05')
