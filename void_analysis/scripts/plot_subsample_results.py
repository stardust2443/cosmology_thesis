#!/usr/bin/env python3
"""
plot_subsample_results.py

Plots the individual VSF fits (measurements, MAP best-fit, and True cosmology)
and generates 4D corner plots for each of the four tracer subsamples (20p, 10p, 5p, 2.5p)
using their optimal scale cuts Rmin.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import corner
from scipy.optimize import minimize

CBL_HOME = os.environ.get('CBL_HOME', '/Users/arun/projects/CosmoBolognaLib')
sys.path.insert(0, os.path.join(CBL_HOME, 'Python'))
os.environ.setdefault('DYLD_LIBRARY_PATH', CBL_HOME)
import CosmoBolognaLib as cbl

HERE = os.path.dirname(os.path.abspath(__file__))
CHAINS_DIR = os.path.join(HERE, '..', 'data', 'mcmc_chains')
COV_DIR    = os.path.join(HERE, '..', 'data', 'cov_jk_64')
THESIS_FIG = os.path.join(HERE, '..', '..', 'thesis_docs', 'figures')
os.makedirs(THESIS_FIG, exist_ok=True)

H0, OMEGA_B, N_S, TAU = 0.6736, 0.0493, 0.9649, 0.0925
SCALAR_AMP, SCALAR_PIVOT = 2.194e-9, 0.05
DELTA_NL, B_EFF = -0.7, 1.0001
SUBSAMPLES = ['20p', '10p', '5p', '2_5p']

FORARUN_BIN_CENTERS = np.array([
    1.591446e+00, 1.791403e+00, 2.016484e+00, 2.269845e+00, 2.555040e+00,
    2.876068e+00, 3.237432e+00, 3.644199e+00, 4.102074e+00, 4.617479e+00,
    5.197642e+00, 5.850699e+00, 6.585810e+00, 7.413284e+00, 8.344726e+00,
    9.393199e+00, 1.057341e+01, 1.190190e+01, 1.339732e+01, 1.508062e+01,
])

OPTIMAL_RMIN = {
    '20p': 7.0,
    '10p': 8.0,
    '5p': 8.0,
    '2_5p': 10.0
}

def make_cosmo(om, s8):
    c = cbl.cosmology.LCDM(om, OMEGA_B, 0.0, H0,
                           SCALAR_AMP, SCALAR_PIVOT, N_S, TAU, True)
    c.set_parameter("sigma8", s8)
    return c

def evaluate_theory(om, s8, bs, bo, radii):
    try:
        c = make_cosmo(om, s8)
        sf = cbl.cosmology.SizeFunction(c)
        dc = c.deltac(0.0)
        m = np.zeros(len(radii))
        for j, rv in enumerate(radii):
            val = sf.size_function(float(rv), 0.0, "Vdn",
                                   B_EFF, bs, bo, DELTA_NL, dc, "EisensteinHu")
            m[j] = val
        return m
    except Exception:
        return None

def find_map_parameters(chain, d, r_fit, C_inv):
    best_chi2 = 1e10
    best_theta = None
    
    n_samples = len(chain)
    step = max(1, n_samples // 1000)
    for idx in range(0, n_samples, step):
        theta = chain[idx]
        om, s8, bs, bo = theta
        m_sub = evaluate_theory(om, s8, bs, bo, r_fit)
        if m_sub is None: continue
        diff = d - m_sub
        chi2 = np.dot(diff.T, np.dot(C_inv, diff))
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_theta = theta
            
    return best_theta, best_chi2

def plot_subsample_residuals(frac, rmin):
    ov_key = 'OV05'
    chain_fname = f"cbl_v8_subsample_{frac}_R{int(rmin)}_OV05.npy"
    chain_path = os.path.join(CHAINS_DIR, chain_fname)
    
    chain = np.load(chain_path)
    # Flatten chain: shape is (nsteps, nwalkers, ndim)
    flat_chain = chain[-len(chain)//2:].reshape(-1, chain.shape[-1])
    
    # Load data and precision matrix
    mask_bins = FORARUN_BIN_CENTERS >= rmin
    r_fit = FORARUN_BIN_CENTERS[mask_bins]
    
    dat_path = os.path.join(COV_DIR, f"vsf_64jk_nrec50_{frac}_{ov_key}.dat")
    df = pd.read_csv(dat_path, sep='\t')
    r_all = df['r'].values
    d_all = df['vsf'].values
    err_all = df['err'].values
    
    d_fit = d_all[np.isin(r_all, r_fit)]
    C_inv = np.load(os.path.join(COV_DIR, f"inv_cov_64jk_R{int(rmin)}_{frac}_{ov_key}.npy"))
    
    # Find MAP best fit parameters
    best_theta, best_chi2 = find_map_parameters(flat_chain, d_fit, r_fit, C_inv)
    om_bf, s8_bf, bs_bf, bo_bf = best_theta
    
    m_bf_all = evaluate_theory(om_bf, s8_bf, bs_bf, bo_bf, r_all)
    
    # Optimize bias parameters at True Cosmology for this subsample on bins >= rmin
    def loss_true(nuis):
        bs, bo = nuis
        if bs * B_EFF + bo <= 0.701: return 1e10
        m_sub = evaluate_theory(0.315, 0.811, bs, bo, r_fit)
        if m_sub is None: return 1e10
        diff = d_fit - m_sub
        return np.dot(diff.T, np.dot(C_inv, diff))
        
    res_true = minimize(loss_true, [bs_bf, bo_bf], bounds=[(0.3, 2.5), (-1.0, 1.0)])
    bs_true, bo_true = res_true.x
    m_true_all = evaluate_theory(0.315, 0.811, bs_true, bo_true, r_all)
    
    # Plotting VSF + Residuals
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 7.5), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    
    # Top Panel: VSF
    ax_top.errorbar(r_all, d_all, yerr=err_all, fmt='s', color='#1A237E', label='Data (64JK)', capsize=3, ms=5, zorder=5)
    ax_top.plot(r_all, m_true_all, '--', color='#B58900', lw=2, label='True Cosmology', zorder=4)
    dof = len(r_fit) - 4
    chi2_str = f"\\chi^2_\\nu={best_chi2/dof:.2f}" if dof > 0 else f"\\chi^2={best_chi2:.2f}"
    ax_top.plot(r_all, m_bf_all, '-', color='#9A031E', lw=2, label=f'Best-Fit Model (MAP, ${chi2_str}$)', zorder=3)
    ax_top.axvspan(1.0, rmin, color='gray', alpha=0.12, label='Excluded Scales')
    
    ax_top.set_xscale('log')
    ax_top.set_yscale('log')
    ax_top.set_xlim(1.4, 17.0)
    ax_top.set_ylim(2e-6, 3e-3)
    ax_top.set_title(f"VSF Fit: Subsample {frac.replace('_','.')} (Rmin = {rmin:.1f} Mpc/h)", fontsize=13, fontweight='bold')
    ax_top.set_ylabel(r'$\mathrm{d}n/\mathrm{d}\ln R$', fontsize=13)
    ax_top.grid(True, which="both", ls="--", alpha=0.3)
    ax_top.legend(loc='lower left', fontsize=10)
    
    # Bottom Panel: Residuals
    res_true_vals = (d_all - m_true_all) / err_all
    res_bf_vals = (d_all - m_bf_all) / err_all
    
    ax_bot.errorbar(r_all, res_true_vals, yerr=1.0, fmt='o--', color='#B58900', ms=5, label='True Cosmology', zorder=4, capsize=2)
    ax_bot.errorbar(r_all, res_bf_vals, yerr=1.0, fmt='s-', color='#9A031E', ms=5, label='Best-Fit Model (MAP)', zorder=3, capsize=2)
    ax_bot.axvspan(1.0, rmin, color='gray', alpha=0.12)
    ax_bot.axhline(0, color='black', ls='-', alpha=0.5)
    ax_bot.axhline(2, color='gray', ls='--', alpha=0.3)
    ax_bot.axhline(-2, color='gray', ls='--', alpha=0.3)
    
    ax_bot.set_xscale('log')
    ax_bot.set_xlim(1.4, 17.0)
    ax_bot.set_ylim(-4.5, 4.5)
    ax_bot.set_xlabel(r'$R\ [h^{-1}\mathrm{Mpc}]$', fontsize=12)
    ax_bot.set_ylabel(r'Residuals [$\sigma$]', fontsize=12)
    ax_bot.grid(True, which="both", ls="--", alpha=0.3)
    
    plt.tight_layout()
    out_fig = f"vsf_residuals_subsample_{frac}.png"
    plt.savefig(os.path.join(THESIS_FIG, out_fig), dpi=300, bbox_inches='tight')
    print(f"Saved residuals plot to {out_fig}")
    plt.close()

def plot_subsample_corner(frac, rmin):
    chain_fname = f"cbl_v8_subsample_{frac}_R{int(rmin)}_OV05.npy"
    chain_path = os.path.join(CHAINS_DIR, chain_fname)
    
    chain = np.load(chain_path)
    # Burn-in: discard first 30% of chains, then flatten
    flat_chain = chain[int(len(chain)*0.3):].reshape(-1, chain.shape[-1])
    
    labels = [r'$\Omega_m$', r'$\sigma_8$', r'$B_{\mathrm{slope}}$', r'$B_{\mathrm{offset}}$']
    truths = [0.315, 0.811, None, None]
    
    fig = corner.corner(
        flat_chain,
        labels=labels,
        truths=truths,
        truth_color='#B58900',
        color='#1A237E',
        show_titles=True,
        title_kwargs={"fontsize": 12},
        hist_kwargs={"color": '#1A237E', "alpha": 0.6},
        quantiles=[0.16, 0.50, 0.84]
    )
    
    fig.suptitle(f"Corner Plot: Subsample {frac.replace('_','.')} ($R_{{\\mathrm{{min}}}} = {rmin:.1f}\\ h^{{-1}}\\mathrm{{Mpc}}$)", fontsize=14, fontweight='bold', y=1.02)
    out_fig = f"corner_subsample_{frac}.png"
    plt.savefig(os.path.join(THESIS_FIG, out_fig), dpi=300, bbox_inches='tight')
    print(f"Saved corner plot to {out_fig}")
    plt.close()

if __name__ == '__main__':
    for frac in SUBSAMPLES:
        rmin = OPTIMAL_RMIN[frac]
        plot_subsample_residuals(frac, rmin)
        plot_subsample_corner(frac, rmin)
