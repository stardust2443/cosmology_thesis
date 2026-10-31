#!/usr/bin/env python3
"""
plot_informative_results.py

Plots corner plots for the 4D MCMC runs with informative calibration priors,
and creates comparison plots showing the propagation of calibration uncertainty
from fixed bias to informative prior.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import corner

HERE = os.path.dirname(os.path.abspath(__file__))
CHAINS_DIR = os.path.join(HERE, '..', 'data', 'mcmc_chains')
THESIS_FIG = os.path.join(HERE, '..', '..', 'thesis_docs', 'figures')
os.makedirs(THESIS_FIG, exist_ok=True)

SUBSAMPLES = ['20p', '10p', '5p', '2_5p']
OPTIMAL_RMIN = {
    '20p': 7.0,
    '10p': 8.0,
    '5p': 8.0,
    '2_5p': 10.0
}

def plot_subsample_informative_corner(frac):
    rmin = OPTIMAL_RMIN[frac]
    ov_key = 'OV05'
    
    fname = f"cbl_v10_subsample_{frac}_informative_R{int(rmin)}_{ov_key}.npy"
    fpath = os.path.join(CHAINS_DIR, fname)
    if not os.path.exists(fpath):
        print(f"File not found: {fname}")
        return
        
    chain = np.load(fpath)
    flat = chain[int(len(chain)*0.3):].reshape(-1, 4)
    
    labels = [r'$\Omega_m$', r'$\sigma_8$', r'$B_{\mathrm{slope}}$', r'$B_{\mathrm{offset}}$']
    truths = [0.315, 0.811, None, None]
    
    fig = corner.corner(
        flat,
        labels=labels,
        truths=truths,
        truth_color='#B58900',
        color='#1A237E',
        show_titles=True,
        title_kwargs={"fontsize": 12},
        hist_kwargs={"color": '#1A237E', "alpha": 0.6},
        quantiles=[0.16, 0.50, 0.84]
    )
    
    sub_title = frac.replace('_', '.')
    fig.suptitle(f"4D Posteriors (Informative Calibration Priors): Subsample {sub_title} ($R_{{\\mathrm{{min}}}} = {rmin:.1f}\\ h^{{-1}}\\mathrm{{Mpc}}$)", 
                 fontsize=13, fontweight='bold', y=1.02)
    
    out_name = f"corner_subsample_{frac}_informative.png"
    plt.savefig(os.path.join(THESIS_FIG, out_name), dpi=300, bbox_inches='tight')
    print(f"Saved corner plot to {out_name}")
    plt.close()

def plot_comparison_2d(frac):
    rmin = OPTIMAL_RMIN[frac]
    ov_key = 'OV05'
    
    f_fix = os.path.join(CHAINS_DIR, f"cbl_v9_subsample_{frac}_2D_R{int(rmin)}_{ov_key}.npy")
    f_inf = os.path.join(CHAINS_DIR, f"cbl_v10_subsample_{frac}_informative_R{int(rmin)}_{ov_key}.npy")
    
    if not (os.path.exists(f_fix) and os.path.exists(f_inf)):
        return
        
    ch_fix = np.load(f_fix)
    flat_fix = ch_fix[int(len(ch_fix)*0.3):].reshape(-1, 2)
    
    ch_inf = np.load(f_inf)
    flat_inf = ch_inf[int(len(ch_inf)*0.3):].reshape(-1, 4)[:, :2]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    corner.hist2d(flat_fix[:, 0], flat_fix[:, 1], ax=ax, color='#9A031E', levels=[0.68, 0.95], plot_datapoints=False, plot_density=False)
    corner.hist2d(flat_inf[:, 0], flat_inf[:, 1], ax=ax, color='#1A237E', levels=[0.68, 0.95], plot_datapoints=False, plot_density=False)
    
    ax.plot(0.315, 0.811, '*', color='#B58900', ms=14, mec='k', label='True Cosmology')
    ax.plot([], [], color='#9A031E', lw=2, label='Fixed Bias (Zero Calib Uncertainty)')
    ax.plot([], [], color='#1A237E', lw=2, label='Informative Prior (Propagated Uncertainty)')
    
    sub_title = frac.replace('_', '.')
    ax.set_title(f"Uncertainty Propagation: Subsample {sub_title}", fontsize=12, fontweight='bold')
    ax.set_xlabel(r'$\Omega_m$')
    ax.set_ylabel(r'$\sigma_8$')
    ax.legend(loc='upper right', fontsize=9.5)
    ax.grid(True, ls='--', alpha=0.3)
    
    out_name = f"comparison_uncertainty_{frac}.png"
    plt.savefig(os.path.join(THESIS_FIG, out_name), dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {out_name}")
    plt.close()

if __name__ == '__main__':
    for frac in SUBSAMPLES:
        plot_subsample_informative_corner(frac)
        plot_comparison_2d(frac)
