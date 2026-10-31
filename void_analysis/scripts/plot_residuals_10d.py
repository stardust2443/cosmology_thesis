#!/usr/bin/env python3
"""
plot_residuals_10d.py

Plots the measurements, the model evaluated at the true cosmology, and the best-fit model
across the full radius range for all 4 subsamples.
Shades the excluded scale cut region (R < Rmin).
Uses the Maximum A Posteriori (MAP) parameters from the 10D MCMC chains (rather than marginalized medians)
to show the true best-fit curves, correcting the layout titles and legends.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
log_c = np.log(FORARUN_BIN_CENTERS)
DLNR  = float(np.median(np.diff(log_c)))
V_FULL = 250.0**3
NORM_FACTOR = V_FULL * DLNR

def make_cosmo(om, s8):
    c = cbl.cosmology.LCDM(om, OMEGA_B, 0.0, H0,
                           SCALAR_AMP, SCALAR_PIVOT, N_S, TAU, True)
    c.set_parameter("sigma8", s8)
    return c

def evaluate_subsample_theory(om, s8, bs, bo, radii):
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

def find_map_parameters(chain, d, r_fit, C_inv, idx_gauss=None, idx_poisson=None, counts=None, is_gp=False):
    # Evaluates MCMC points to find the minimum chi-squared (MAP) point
    best_chi2 = 1e10
    best_theta = None
    
    # We step through the chain (every 10th sample to save time)
    n_samples = len(chain)
    step = max(1, n_samples // 1000)
    for idx in range(0, n_samples, step):
        theta = chain[idx]
        om, s8 = theta[0], theta[1]
        
        m_list = []
        possible = True
        for s_idx in range(4):
            bs = theta[2 + 2*s_idx]
            bo = theta[2 + 2*s_idx + 1]
            m_sub = evaluate_subsample_theory(om, s8, bs, bo, r_fit)
            if m_sub is None:
                possible = False
                break
            m_list.append(m_sub)
        if not possible:
            continue
        m = np.hstack(m_list)
        
        if is_gp:
            # GP likelihood chi2 equivalent
            chi2_g = 0.0
            if len(idx_gauss) > 0:
                diff_g = d[idx_gauss] - m[idx_gauss]
                chi2_g = np.dot(diff_g.T, np.dot(C_inv, diff_g))
            
            chi2_p = 0.0
            mu = m * NORM_FACTOR
            for j in idx_poisson:
                nj = counts[j]
                muj = mu[j]
                if muj <= 0:
                    possible = False
                    break
                if nj > 0:
                    chi2_p += 2.0 * (nj * np.log(nj / muj) - (nj - muj))
                else:
                    chi2_p += 2.0 * muj
            if not possible: continue
            chi2 = chi2_g + chi2_p
        else:
            diff = d - m
            chi2 = np.dot(diff.T, np.dot(C_inv, diff))
            
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_theta = theta
            
    print(f"Found MAP with chi2 = {best_chi2:.4f} | parameters = {best_theta[:2]}")
    return best_theta

def plot_residuals_10d(ov_key, rmin, chain_fname, out_fig_name, is_gp=False):
    # Load 10D chain
    chain_path = os.path.join(CHAINS_DIR, chain_fname)
    if not os.path.exists(chain_path):
        print(f"Chain {chain_fname} not found!")
        return
        
    chain = np.load(chain_path)
    if len(chain.shape) == 3:
        chain = chain.reshape(-1, chain.shape[-1])
        
    # Load full dataset and cov
    mask_bins = FORARUN_BIN_CENTERS >= rmin
    r_fit = FORARUN_BIN_CENTERS[mask_bins]
    
    d_list = []
    err_list = []
    for frac in SUBSAMPLES:
        dat_path = os.path.join(COV_DIR, f"vsf_64jk_nrec50_{frac}_{ov_key}.dat")
        df = pd.read_csv(dat_path, sep='\t')
        df_sub = df[df['r'] >= rmin]
        d_list.append(df_sub['vsf'].values)
        err_list.append(df_sub['err'].values)
        
    d_fit = np.hstack(d_list)
    err_fit = np.hstack(err_list)
    counts_fit = np.round(d_fit * NORM_FACTOR)
    
    cov_path = os.path.join(COV_DIR, f"cov_64jk_R{int(rmin)}_{ov_key}.npy")
    C = np.load(cov_path)
    
    # Hartlap / GP setup
    if is_gp:
        idx_gauss = np.where(counts_fit >= 5.0)[0]
        idx_poisson = np.where(counts_fit < 5.0)[0]
        C_gauss = C[np.ix_(idx_gauss, idx_gauss)]
        p_gauss = len(idx_gauss)
        alpha_gauss = (64 - p_gauss - 2) / 63.0 if p_gauss > 0 else 1.0
        C_inv = np.linalg.inv(C_gauss) * alpha_gauss if p_gauss > 0 else None
        best_theta = find_map_parameters(chain, d_fit, r_fit, C_inv, idx_gauss, idx_poisson, counts_fit, is_gp=True)
    else:
        p = len(d_fit)
        alpha_hartlap = (64 - p - 2) / 63.0
        C_inv = np.linalg.inv(C) * alpha_hartlap
        best_theta = find_map_parameters(chain, d_fit, r_fit, C_inv, is_gp=False)
        
    om_bf, s8_bf = best_theta[0], best_theta[1]
    
    # Setup plotting
    fig, axes = plt.subplots(2, 4, figsize=(18, 9.5), gridspec_kw={'height_ratios': [2, 1]})
    
    for idx, frac in enumerate(SUBSAMPLES):
        ax_top = axes[0, idx]
        ax_bot = axes[1, idx]
        
        # Load complete measured data (all 20 bins)
        dat_path = os.path.join(COV_DIR, f"vsf_64jk_nrec50_{frac}_{ov_key}.dat")
        df = pd.read_csv(dat_path, sep='\t')
        r_all = df['r'].values
        d_all = df['vsf'].values
        err_all = df['err'].values
        
        # MAP parameters for this subsample
        bs_bf = best_theta[2 + 2*idx]
        bo_bf = best_theta[2 + 2*idx + 1]
        
        # Calculate MAP theory curve over full radius range
        m_bf_all = evaluate_subsample_theory(om_bf, s8_bf, bs_bf, bo_bf, r_all)
        
        # Optimize bias parameters at True Cosmology for this subsample on bins >= rmin
        def loss_true(nuis):
            bs, bo = nuis
            if bs * B_EFF + bo <= 0.701: return 1e10
            m_sub = evaluate_subsample_theory(0.315, 0.811, bs, bo, r_fit)
            if m_sub is None: return 1e10
            diff = d_fit[idx*len(r_fit):(idx+1)*len(r_fit)] - m_sub
            e_sub = err_fit[idx*len(r_fit):(idx+1)*len(r_fit)]
            return np.sum((diff / e_sub)**2)
            
        res_true = minimize(loss_true, [bs_bf, bo_bf], bounds=[(0.3, 2.5), (-1.0, 1.0)])
        bs_true, bo_true = res_true.x
        m_true_all = evaluate_subsample_theory(0.315, 0.811, bs_true, bo_true, r_all)
        
        # Top panel: VSF
        # True cosmology as gold dashed, Best-Fit Model as red solid
        ax_top.errorbar(r_all, d_all, yerr=err_all, fmt='s', color='#1A237E', label='Data (64JK)', capsize=3, ms=5, zorder=5)
        ax_top.plot(r_all, m_true_all, '--', color='#B58900', lw=2, label='True Cosmology', zorder=4)
        ax_top.plot(r_all, m_bf_all, '-', color='#9A031E', lw=2, label='Best-Fit Model (MAP)', zorder=3)
        
        # Shading the excluded scales
        ax_top.axvspan(1.0, rmin, color='gray', alpha=0.12, label='Excluded Scales')
        
        ax_top.set_xscale('log')
        ax_top.set_yscale('log')
        ax_top.set_xlim(1.4, 17.0)
        ax_top.set_ylim(2e-6, 3e-3)
        ax_top.set_title(f"Subsample: {frac} ({frac.replace('_','.')} tracers)", fontsize=13, fontweight='bold')
        ax_top.set_ylabel(r'$\mathrm{d}n/\mathrm{d}\ln R$', fontsize=13)
        ax_top.grid(True, which="both", ls="--", alpha=0.3)
        
        if idx == 0:
            ax_top.legend(loc='lower left', fontsize=10)
            
        # Bottom panel: Residuals
        res_true = (d_all - m_true_all) / err_all
        res_bf = (d_all - m_bf_all) / err_all
        
        ax_bot.errorbar(r_all, res_true, yerr=1.0, fmt='o--', color='#B58900', ms=5, label='True Cosmology', zorder=4, capsize=2)
        ax_bot.errorbar(r_all, res_bf, yerr=1.0, fmt='s-', color='#9A031E', ms=5, label='Best-Fit Model (MAP)', zorder=3, capsize=2)
        
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
        
    title_suffix = "Gaussian-Poisson Likelihood" if is_gp else "64JK Full Covariance"
    fig.suptitle(f"VSF Measurement & Model Residuals: {ov_key} ({title_suffix}, Rmin = {rmin:.1f})", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(THESIS_FIG, out_fig_name), dpi=300, bbox_inches='tight')
    print(f"Saved residuals plot to {out_fig_name}")
    plt.close()

if __name__ == '__main__':
    # 1. 10D MCMC fits (v6) residuals
    plot_residuals_10d('OV05', 6.0, 'cbl_v6_joint_64jk_full_R6_OV05.npy', 'vsf_residuals_OV05_R6_64JK_Full.png', is_gp=False)
    plot_residuals_10d('OV05', 7.0, 'cbl_v6_joint_64jk_full_R7_OV05.npy', 'vsf_residuals_OV05_R7_64JK_Full.png', is_gp=False)
    plot_residuals_10d('OV05', 8.0, 'cbl_v6_joint_64jk_full_R8_OV05.npy', 'vsf_residuals_OV05_R8_64JK_Full.png', is_gp=False)
    plot_residuals_10d('OV05', 9.0, 'cbl_v6_joint_64jk_full_R9_OV05.npy', 'vsf_residuals_OV05_R9_64JK_Full.png', is_gp=False)
    plot_residuals_10d('OV05', 10.0, 'cbl_v6_joint_64jk_full_R10_OV05.npy', 'vsf_residuals_OV05_R10_64JK_Full.png', is_gp=False)
    plot_residuals_10d('OV00', 7.0, 'cbl_v6_joint_64jk_full_R7_OV00.npy', 'vsf_residuals_OV00_R7_64JK_Full.png', is_gp=False)
    
    # 2. Gaussian-Poisson (v7) residuals
    plot_residuals_10d('OV05', 7.0, 'cbl_v7_joint_gp_R7_OV05.npy', 'vsf_residuals_OV05_R7_GP.png', is_gp=True)
    plot_residuals_10d('OV05', 9.0, 'cbl_v7_joint_gp_R9_OV05.npy', 'vsf_residuals_OV05_R9_GP.png', is_gp=True)
