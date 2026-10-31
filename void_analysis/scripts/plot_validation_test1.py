#!/usr/bin/env python3
"""
plot_validation_test1.py
Plot the comparison between Poisson errors and Jackknife errors for both TOUCH and CENTER criteria at R_min = 7.0 Mpc/h.
Shows whether jackknife errors (which capture cosmic variance and cleaning covariance) open up the contours.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
CHAIN_DIR = os.path.join(HERE, '..', 'data', 'mcmc_chains')
OUT_DIR   = os.path.join(HERE, '..', 'plots', 'cbl_mcmc')
os.makedirs(OUT_DIR, exist_ok=True)

OMEGA_M_TRUE, SIGMA8_TRUE = 0.315, 0.811
C_POISSON = '#1A237E' # Indigo
C_JK      = '#1B5E20' # Emerald Green

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.facecolor': 'white',
})

def load_chain(rmin, overlap, tag='', field='NREC50'):
    suffix = f"_{tag}" if tag else ""
    path = os.path.join(CHAIN_DIR, f'cbl_v5_joint_R{rmin:.0f}_{field}_{overlap}{suffix}_flat.npy')
    if os.path.exists(path):
        return np.load(path)
    else:
        print(f"Warning: chain not found at {path}")
        return None

def contour_2d(ax, om, s8, color, label, filled=True):
    xmin, xmax = 0.08, 0.48
    ymin, ymax = 0.55, 0.95
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])

    try:
        kernel = gaussian_kde(np.vstack([om, s8]), bw_method=0.2)
        zz = np.reshape(kernel(positions), xx.shape)
    except Exception as e:
        print(f"Failed to compute KDE: {e}")
        return

    zz = gaussian_filter(zz, sigma=1.0)
    zz = zz / zz.max()

    levels_fill = [0.05, 0.32, 1.0]
    levels_line = [0.05, 0.32]

    if filled:
        ax.contourf(xx, yy, zz, levels=levels_fill, colors=[color], alpha=[0.08, 0.20])
    ax.contour(xx, yy, zz, levels=levels_line, colors=[color], linewidths=[1.2, 2.0], linestyles=['--', '-'])
    ax.plot([], [], '-', color=color, lw=2, label=label)

def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    field = 'NREC50'
    rmin = 7.0

    # 1. Left Panel: TOUCH (OV00) comparisons
    ax = axes[0]
    print("Plotting TOUCH comparisons...")
    chain_pois = load_chain(rmin, 'OV00', tag='', field=field)
    chain_jk   = load_chain(rmin, 'OV00', tag='jk', field=field)
    
    if chain_pois is not None:
        contour_2d(ax, chain_pois[:, 0], chain_pois[:, 1], C_POISSON, 'Poisson Errors')
    if chain_jk is not None:
        contour_2d(ax, chain_jk[:, 0], chain_jk[:, 1], C_JK, 'Jackknife Errors (Cosmic Var.)')
        
    ax.plot(OMEGA_M_TRUE, SIGMA8_TRUE, '*', color='#B58900', ms=15, mec='k', mew=1, zorder=10, label='Truth')
    ax.set_title('TOUCH (OV00, Strict Overlap)', fontsize=14)
    ax.set_xlabel(r'$\Omega_m$')
    ax.set_ylabel(r'$\sigma_8$')
    ax.set_xlim(0.08, 0.48)
    ax.set_ylim(0.55, 0.95)
    ax.grid(True, ls='--', alpha=0.2)
    ax.legend(loc='upper right')

    # 2. Right Panel: CENTER (OV05) comparisons
    ax = axes[1]
    print("Plotting CENTER comparisons...")
    chain_pois_ctr = load_chain(rmin, 'OV05', tag='', field=field)
    chain_jk_ctr   = load_chain(rmin, 'OV05', tag='jk', field=field)
    
    if chain_pois_ctr is not None:
        contour_2d(ax, chain_pois_ctr[:, 0], chain_pois_ctr[:, 1], C_POISSON, 'Poisson Errors')
    if chain_jk_ctr is not None:
        contour_2d(ax, chain_jk_ctr[:, 0], chain_jk_ctr[:, 1], C_JK, 'Jackknife Errors (Cosmic Var.)')
        
    ax.plot(OMEGA_M_TRUE, SIGMA8_TRUE, '*', color='#B58900', ms=15, mec='k', mew=1, zorder=10, label='Truth')
    ax.set_title('CENTER (OV05, Allowed Overlap)', fontsize=14)
    ax.set_xlabel(r'$\Omega_m$')
    ax.set_xlim(0.08, 0.48)
    ax.set_ylim(0.55, 0.95)
    ax.grid(True, ls='--', alpha=0.2)
    ax.legend(loc='upper right')

    plt.suptitle(f'Error Estimation Impact: Poisson vs Jackknife (Joint MCMC, R > 7.0 $h^{{-1}}$Mpc)', fontsize=16, y=1.02)
    plt.tight_layout()
    
    outpath = os.path.join(OUT_DIR, 'validation_test1_covariance_impact.png')
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    print(f"\nSaved plot to: {outpath}")

if __name__ == '__main__':
    main()
