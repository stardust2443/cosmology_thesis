#!/usr/bin/env python3
"""
plot_validation_64jk.py

Plots the validation results from the 64-region Hartlap-corrected full covariance MCMC fits:
1. Comparison plot: Poisson vs 8-Octant Diagonal JK vs 64-Region Full Covariance (Hartlap)
2. Scale-cut scan plot: Rmin = 7.0, 8.0, 9.0 Mpc/h under full covariance.

Saves figures to plots/cbl_mcmc/ and copies them to thesis_docs/figures/.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

HERE = os.path.dirname(os.path.abspath(__file__))
CHAINS_DIR = os.path.join(HERE, '..', 'data', 'mcmc_chains')
OUT_DIR    = os.path.join(HERE, '..', 'plots', 'cbl_mcmc')
THESIS_FIG = os.path.join(HERE, '..', '..', 'thesis_docs', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(THESIS_FIG, exist_ok=True)

OMEGA_M_TRUE, SIGMA8_TRUE = 0.315, 0.811

# Perceptually uniform colorblind-friendly theme
C_POISSON = '#1A237E'  # Indigo
C_JK8     = '#1B5E20'  # Emerald Green
C_JK64    = '#9A031E'  # Cherry Red / Crimson
C_GOLD    = '#B58900'  # Dark Gold for Truth

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 15,
    'font.family': 'sans-serif'
})

def contour_2d(ax, x, y, color, label, levels=[0.6827, 0.9545], linewidth=2.0, alpha_fill=0.15):
    try:
        kernel = stats.gaussian_kde(np.vstack([x, y]))
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
        dx = (xmax - xmin) * 0.1
        dy = (ymax - ymin) * 0.1
        xx, yy = np.mgrid[xmin-dx:xmax+dx:100j, ymin-dy:ymax+dy:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        f = np.reshape(kernel(positions).T, xx.shape)

        t = np.linspace(0, f.max(), 1000)
        integral = np.array([f[f >= val].sum() for val in t])
        integral /= integral[0]

        clevels = []
        for lvl in levels:
            target_val = t[np.argmin(np.abs(integral - lvl))]
            clevels.append(target_val)
        clevels.sort()

        if len(clevels) > 0 and clevels[-1] > 0:
            ax.contour(xx, yy, f, levels=clevels, colors=[color], linewidths=linewidth)
            ax.contourf(xx, yy, f, levels=[clevels[0], f.max()], colors=[color], alpha=alpha_fill)
            ax.plot([], [], color=color, lw=linewidth, label=label)
    except Exception as e:
        print(f"Error drawing contour for {label}: {e}")

def plot_64jk_covariance_impact():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    # 1. TOUCH (OV00)
    ax = axes[0]
    p_pois = os.path.join(CHAINS_DIR, 'cbl_v5_joint_R7_NREC50_OV00_flat.npy')
    p_jk8  = os.path.join(CHAINS_DIR, 'cbl_v5_joint_R7_NREC50_OV00_jk_flat.npy')
    p_jk64 = os.path.join(CHAINS_DIR, 'cbl_v6_joint_64jk_full_R7_OV00.npy')

    if os.path.exists(p_pois):
        c_pois = np.load(p_pois)
        contour_2d(ax, c_pois[:, 0], c_pois[:, 1], C_POISSON, 'Poisson Errors')
    if os.path.exists(p_jk8):
        c_jk8 = np.load(p_jk8)
        contour_2d(ax, c_jk8[:, 0], c_jk8[:, 1], C_JK8, '8-Octant Diagonal JK')
    if os.path.exists(p_jk64):
        c_jk64 = np.load(p_jk64)
        contour_2d(ax, c_jk64[:, 0], c_jk64[:, 1], C_JK64, '64-Region Full Covariance (Hartlap)')

    ax.plot(OMEGA_M_TRUE, SIGMA8_TRUE, '*', color=C_GOLD, ms=15, mec='k', mew=1, zorder=10, label='Truth')
    ax.set_title('TOUCH (OV00, Strict Overlap)', fontsize=14)
    ax.set_xlabel(r'$\Omega_m$')
    ax.set_ylabel(r'$\sigma_8$')
    ax.set_xlim(0.08, 0.48)
    ax.set_ylim(0.52, 1.18)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper right')

    # 2. CENTER (OV05)
    ax = axes[1]
    p_pois = os.path.join(CHAINS_DIR, 'cbl_v5_joint_R7_NREC50_OV05_flat.npy')
    p_jk8  = os.path.join(CHAINS_DIR, 'cbl_v5_joint_R7_NREC50_OV05_jk_flat.npy')
    p_jk64 = os.path.join(CHAINS_DIR, 'cbl_v6_joint_64jk_full_R7_OV05.npy')

    if os.path.exists(p_pois):
        c_pois = np.load(p_pois)
        contour_2d(ax, c_pois[:, 0], c_pois[:, 1], C_POISSON, 'Poisson Errors')
    if os.path.exists(p_jk8):
        c_jk8 = np.load(p_jk8)
        contour_2d(ax, c_jk8[:, 0], c_jk8[:, 1], C_JK8, '8-Octant Diagonal JK')
    if os.path.exists(p_jk64):
        c_jk64 = np.load(p_jk64)
        contour_2d(ax, c_jk64[:, 0], c_jk64[:, 1], C_JK64, '64-Region Full Covariance (Hartlap)')

    ax.plot(OMEGA_M_TRUE, SIGMA8_TRUE, '*', color=C_GOLD, ms=15, mec='k', mew=1, zorder=10, label='Truth')
    ax.set_title('CENTER (OV05, Allowed Overlap)', fontsize=14)
    ax.set_xlabel(r'$\Omega_m$')
    ax.set_xlim(0.08, 0.48)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper right')

    fig.suptitle(r'Impact of Error Model & Full Covariance Matrix on Joint Fits ($R \geq 7.0\ h^{-1}\mathrm{Mpc}$)', fontsize=15, y=1.02)
    plt.tight_layout()
    
    out_path = os.path.join(OUT_DIR, 'validation_64jk_covariance_impact.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(THESIS_FIG, 'validation_64jk_covariance_impact.png'), dpi=300, bbox_inches='tight')
    print(f"Saved plot: {out_path}")
    plt.close()

def plot_64jk_scale_cut_scan():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)

    rmins = [7.0, 8.0, 9.0]
    for i, rmin in enumerate(rmins):
        ax = axes[i]
        p_ov00 = os.path.join(CHAINS_DIR, f'cbl_v6_joint_64jk_full_R7_OV00.npy') if rmin == 7.0 else None
        p_ov05 = os.path.join(CHAINS_DIR, f'cbl_v6_joint_64jk_full_R{int(rmin)}_OV05.npy')

        if p_ov00 and os.path.exists(p_ov00):
            c_ov00 = np.load(p_ov00)
            contour_2d(ax, c_ov00[:, 0], c_ov00[:, 1], '#9A031E', 'TOUCH (OV00)')
        if os.path.exists(p_ov05):
            c_ov05 = np.load(p_ov05)
            contour_2d(ax, c_ov05[:, 0], c_ov05[:, 1], '#1A237E', 'CENTER (OV05)')

        ax.plot(OMEGA_M_TRUE, SIGMA8_TRUE, '*', color=C_GOLD, ms=15, mec='k', mew=1, zorder=10, label='Truth')
        ax.set_title(f'$R_{{min}} = {rmin:.1f} \\ h^{{-1}}\\mathrm{{Mpc}}$ (Full Covariance)', fontsize=13)
        ax.set_xlabel(r'$\Omega_m$')
        if i == 0:
            ax.set_ylabel(r'$\sigma_8$')
        ax.set_xlim(0.08, 0.48)
        ax.set_ylim(0.52, 1.18)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='upper right')

    fig.suptitle(r'Scale-Cut Scan under 64-Region Hartlap Full Covariance', fontsize=15, y=1.02)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, 'validation_64jk_scale_cut_scan.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(THESIS_FIG, 'validation_64jk_scale_cut_scan.png'), dpi=300, bbox_inches='tight')
    print(f"Saved plot: {out_path}")
    plt.close()

if __name__ == '__main__':
    plot_64jk_covariance_impact()
    plot_64jk_scale_cut_scan()
