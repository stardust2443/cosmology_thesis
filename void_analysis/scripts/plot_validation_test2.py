#!/usr/bin/env python3
"""
plot_validation_test2.py
Plot the MCMC contours for the scale-cut sensitivity scan (Test 2).
Loads chains for OV00 and OV05 at R_min = 7, 8, 9 Mpc/h and plots them side-by-side.
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
C_OV00 = '#9A031E'  # Cherry Red for TOUCH (OV00)
C_OV05 = '#1A237E'  # Indigo for CENTER (OV05)

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.facecolor': 'white',
})

def load_chain(rmin, overlap, field='NREC50'):
    path = os.path.join(CHAIN_DIR, f'cbl_v5_joint_R{rmin:.0f}_{field}_{overlap}_flat.npy')
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
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    rmins = [7.0, 8.0, 9.0]
    field = 'NREC50'

    for i, rmin in enumerate(rmins):
        ax = axes[i]
        print(f"Plotting R_min = {rmin}...")
        
        # Load TOUCH (OV00)
        chain_ov00 = load_chain(rmin, 'OV00', field)
        if chain_ov00 is not None:
            contour_2d(ax, chain_ov00[:, 0], chain_ov00[:, 1], C_OV00, 'TOUCH (OV00)')
            
        # Load CENTER (OV05)
        chain_ov05 = load_chain(rmin, 'OV05', field)
        if chain_ov05 is not None:
            contour_2d(ax, chain_ov05[:, 0], chain_ov05[:, 1], C_OV05, 'CENTER (OV05)')
            
        ax.plot(OMEGA_M_TRUE, SIGMA8_TRUE, '*', color='#B58900', ms=15, mec='k', mew=1, zorder=10, label='Truth')
        ax.set_title(f'$R_{{min}} = {rmin:.1f} \\ h^{{-1}}\\mathrm{{Mpc}}$', fontsize=14)
        ax.set_xlabel(r'$\Omega_m$')
        if i == 0:
            ax.set_ylabel(r'$\sigma_8$')
        ax.set_xlim(0.08, 0.48)
        ax.set_ylim(0.55, 0.95)
        ax.grid(True, ls='--', alpha=0.2)
        ax.legend(loc='upper right')

    plt.suptitle(f'Scale-Cut Sensitivity Scan: TOUCH vs CENTER ({field} Joint MCMC)', fontsize=16, y=1.02)
    plt.tight_layout()
    
    outpath = os.path.join(OUT_DIR, 'validation_test2_scale_cut_scan.png')
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    print(f"\nSaved plot to: {outpath}")

if __name__ == '__main__':
    main()
