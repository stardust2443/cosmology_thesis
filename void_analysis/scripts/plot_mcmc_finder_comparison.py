#!/usr/bin/env python3
"""
plot_mcmc_finder_comparison.py

Generates:
1. Joint 2D cosmological contour comparison: BitVF vs REVOLVER in the (Omega_m, sigma8) plane.
2. VSF data points with error bars (no connecting lines) + MCMC best-fit models and residuals.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# CBL bootstrap
CBL_HOME = '/Users/arun/projects/CosmoBolognaLib'
sys.path.insert(0, os.path.join(CBL_HOME, 'Python'))
os.environ.setdefault('DYLD_LIBRARY_PATH', CBL_HOME)
import CosmoBolognaLib as cbl

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, '..', 'data')
PLOT_DIR = os.path.join(HERE, '..', 'plots', 'halos')
os.makedirs(PLOT_DIR, exist_ok=True)

OMEGA_M_TRUE = 0.315
SIGMA8_TRUE = 0.811
OMEGA_B = 0.0493
H0 = 0.6736
N_S = 0.9649
SCALAR_AMP = 2.194e-9
SCALAR_PIVOT = 0.05
TAU = 0.0925

cosmo = cbl.cosmology.LCDM(
    OMEGA_M_TRUE, OMEGA_B, 0.0,
    H0, SCALAR_AMP, SCALAR_PIVOT,
    N_S, TAU, True
)
cosmo.set_parameter("sigma8", SIGMA8_TRUE)
DEL_C = cosmo.deltac(0.0)
sf = cbl.cosmology.SizeFunction(cosmo)

SAMPLES = ['100P', '75P', '50P', '30P']
R_MIN_MAP = {'100P': 18, '75P': 18, '50P': 16, '30P': 16}
B_EFF_MAP = {'100P': 1.0211, '75P': 0.9583, '50P': 0.9108, '30P': 0.8623}

def load_chains(sample):
    rmin = R_MIN_MAP[sample]
    # BitVF chain
    bv_path = os.path.join(DATA_DIR, 'mcmc_chains_halo', f'mcmc_chain_halo_{sample}_touch_informative_Rmin{rmin}.npz')
    # REVOLVER chain
    rv_path = os.path.join(DATA_DIR, 'mcmc_chains_revolver', f'mcmc_chain_revolver_{sample}_touch_Rmin{rmin}.npz')
    
    bv_data = np.load(bv_path) if os.path.exists(bv_path) else None
    rv_data = np.load(rv_path) if os.path.exists(rv_path) else None
    return bv_data, rv_data

def plot_contours_grid():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=200)
    axes = axes.flatten()
    
    for idx, sample in enumerate(SAMPLES):
        ax = axes[idx]
        bv_data, rv_data = load_chains(sample)
        beff = B_EFF_MAP[sample]
        rmin = R_MIN_MAP[sample]
        
        # Plot simulation truth
        ax.axvline(OMEGA_M_TRUE, color='black', ls='--', lw=1.2, alpha=0.7)
        ax.axhline(SIGMA8_TRUE, color='black', ls='--', lw=1.2, alpha=0.7)
        ax.plot(OMEGA_M_TRUE, SIGMA8_TRUE, marker='*', color='#e74c3c', ms=12, label='Simulation Truth')
        
        if bv_data is not None:
            chain_bv = bv_data['chain']
            # Compute 2D histogram / contours
            om_bv = chain_bv[:, 0]
            s8_bv = chain_bv[:, 1]
            H, xedges, yedges = np.histogram2d(om_bv, s8_bv, bins=35)
            xcenters = 0.5 * (xedges[:-1] + xedges[1:])
            ycenters = 0.5 * (yedges[:-1] + yedges[1:])
            H_flat = H.flatten()
            H_sorted = np.sort(H_flat)[::-1]
            cumsum = np.cumsum(H_sorted) / np.sum(H_sorted)
            lvl_68 = H_sorted[np.searchsorted(cumsum, 0.68)]
            lvl_95 = H_sorted[np.searchsorted(cumsum, 0.95)]
            
            cs_bv = ax.contour(xcenters, ycenters, H.T, levels=[lvl_95, lvl_68],
                               colors=['#2980b9', '#1b4f72'], linewidths=[1.5, 2.2])
            ax.contourf(xcenters, ycenters, H.T, levels=[lvl_68, H_sorted[0]],
                        colors=['#3498db'], alpha=0.25)
                        
        if rv_data is not None:
            chain_rv = rv_data['chain']
            om_rv = chain_rv[:, 0]
            s8_rv = chain_rv[:, 1]
            H, xedges, yedges = np.histogram2d(om_rv, s8_rv, bins=35)
            xcenters = 0.5 * (xedges[:-1] + xedges[1:])
            ycenters = 0.5 * (yedges[:-1] + yedges[1:])
            H_flat = H.flatten()
            H_sorted = np.sort(H_flat)[::-1]
            cumsum = np.cumsum(H_sorted) / np.sum(H_sorted)
            lvl_68 = H_sorted[np.searchsorted(cumsum, 0.68)]
            lvl_95 = H_sorted[np.searchsorted(cumsum, 0.95)]
            
            cs_rv = ax.contour(xcenters, ycenters, H.T, levels=[lvl_95, lvl_68],
                               colors=['#d35400', '#7e3200'], linewidths=[1.5, 2.2], linestyles=['--', '-'])
            ax.contourf(xcenters, ycenters, H.T, levels=[lvl_68, H_sorted[0]],
                        colors=['#e67e22'], alpha=0.20)
                        
        ax.set_title(rf'Sample {sample} ($b_{{\rm eff}}={beff:.3f}, R \geq {rmin}\ h^{{-1}}\mathrm{{Mpc}}$)',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel(r'$\Omega_m$', fontsize=12)
        ax.set_ylabel(r'$\sigma_8$', fontsize=12)
        ax.set_xlim(0.24, 0.50)
        ax.set_ylim(0.76, 0.88)
        ax.grid(True, alpha=0.3, ls=':')
        
        # Legend proxies
        p_bv = mpatches.Patch(color='#3498db', alpha=0.5, label='BitVF (calibrated 68% prior)')
        p_rv = mpatches.Patch(color='#e67e22', alpha=0.5, label='REVOLVER (calibrated 68% prior)')
        p_truth = plt.Line2D([0], [0], marker='*', color='#e74c3c', ls='none', ms=10, label='Simulation Truth')
        ax.legend(handles=[p_bv, p_rv, p_truth], loc='upper right', fontsize=9)
        
    plt.tight_layout()
    out_png = os.path.join(PLOT_DIR, 'mcmc_finder_comparison_grid.png')
    plt.savefig(out_png, dpi=200)
    thesis_png = os.path.join(HERE, '..', 'main_thesis_docs', 'figures', 'mcmc_finder_comparison_grid.png')
    if os.path.exists(os.path.dirname(thesis_png)):
        import shutil
        shutil.copy2(out_png, thesis_png)
    plt.close()
    print(f"Saved MCMC finder comparison grid to {out_png}")

if __name__ == '__main__':
    plot_contours_grid()
