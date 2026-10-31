#!/usr/bin/env python3
"""
update_subsample_plots.py

1. Generates zoomed-in VSF fit plots: shows only ~1-2 bins below Rmin in shaded gray,
   and expands the fitted bins up to large radii.
2. Generates 2D corner plots (Omega_m, sigma_8) for the calibrated-bias fixed MCMC.
3. Generates 4D corner plots (Omega_m, sigma_8, bs, bo) with consistent median + 68% CL labels.
4. Outputs all summary statistics to be used in the thesis text and tables.
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

# Zoomed-in x-limits: keep only ~1-2 bins below Rmin
XLIMS_ZOOM = {
    '20p': (5.2, 17.5),
    '10p': (6.0, 17.5),
    '5p': (6.0, 17.5),
    '2_5p': (7.8, 17.5)
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

def calibrate_bias(om, s8, r_fit, d, C_inv):
    best_chi2 = 1e10
    best_nuis = [1.0, 0.0]
    
    bs_vals = np.linspace(0.4, 2.2, 15)
    bo_vals = np.linspace(-0.8, 0.8, 15)
    for bs in bs_vals:
        for bo in bo_vals:
            if bs * B_EFF + bo <= 0.701: continue
            m_sub = evaluate_theory(om, s8, bs, bo, r_fit)
            if m_sub is None: continue
            diff = d - m_sub
            chi2 = np.dot(diff.T, np.dot(C_inv, diff))
            if chi2 < best_chi2:
                best_chi2 = chi2
                best_nuis = [bs, bo]
                
    def loss(nuis):
        bs, bo = nuis
        if bs * B_EFF + bo <= 0.701: return 1e10
        m_sub = evaluate_theory(om, s8, bs, bo, r_fit)
        if m_sub is None: return 1e10
        diff = d - m_sub
        return np.dot(diff.T, np.dot(C_inv, diff))
        
    res = minimize(loss, best_nuis, bounds=[(0.3, 2.5), (-1.0, 1.0)], tol=1e-6)
    return res.x, res.fun

def plot_zoomed_vsf(frac):
    rmin = OPTIMAL_RMIN[frac]
    ov_key = 'OV05'
    
    mask_bins = FORARUN_BIN_CENTERS >= rmin
    r_fit = FORARUN_BIN_CENTERS[mask_bins]
    
    dat_path = os.path.join(COV_DIR, f"vsf_64jk_nrec50_{frac}_{ov_key}.dat")
    df = pd.read_csv(dat_path, sep='\t')
    r_all = df['r'].values
    d_all = df['vsf'].values
    err_all = df['err'].values
    
    d_fit = d_all[np.isin(r_all, r_fit)]
    C_inv = np.load(os.path.join(COV_DIR, f"inv_cov_64jk_R{int(rmin)}_{frac}_{ov_key}.npy"))
    
    # 1. Calibrate bias at True cosmology
    (bs_calib, bo_calib), chi2_true = calibrate_bias(0.315, 0.811, r_fit, d_fit, C_inv)
    nu_true = len(r_fit) - 2
    red_chi2_true = chi2_true / nu_true
    m_true_all = evaluate_theory(0.315, 0.811, bs_calib, bo_calib, r_all)
    
    # 2. Find 2D Best-Fit (fixed calibrated bias)
    def loss_2d(cosmo):
        om, s8 = cosmo
        if not (0.1 <= om <= 0.5 and 0.5 <= s8 <= 1.2): return 1e10
        m_sub = evaluate_theory(om, s8, bs_calib, bo_calib, r_fit)
        if m_sub is None: return 1e10
        diff = d_fit - m_sub
        return np.dot(diff.T, np.dot(C_inv, diff))
        
    res_2d = minimize(loss_2d, [0.315, 0.811], bounds=[(0.1, 0.5), (0.5, 1.2)], tol=1e-6)
    om_bf, s8_bf = res_2d.x
    chi2_2d = res_2d.fun
    nu_2d = len(r_fit) - 2
    red_chi2_2d = chi2_2d / nu_2d
    m_bf_all = evaluate_theory(om_bf, s8_bf, bs_calib, bo_calib, r_all)
    
    # Setup zoomed-in plot
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 7.2), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    xmin, xmax = XLIMS_ZOOM[frac]
    
    # Filter points visible in the zoomed range
    mask_vis = (r_all >= xmin * 0.9) & (r_all <= xmax * 1.1)
    
    # Top Panel: VSF
    ax_top.errorbar(r_all[mask_vis], d_all[mask_vis], yerr=err_all[mask_vis], fmt='s', color='#1A237E', 
                    label='Data (64JK)', capsize=4, ms=6, zorder=5)
    ax_top.plot(r_all[mask_vis], m_true_all[mask_vis], '--', color='#B58900', lw=2.2, 
                label=f'True Cosmo ($\\chi^2_\\nu={red_chi2_true:.2f}$)', zorder=4)
    ax_top.plot(r_all[mask_vis], m_bf_all[mask_vis], '-', color='#9A031E', lw=2.2, 
                label=f'Best-Fit Cosmo ($\\chi^2_\\nu={red_chi2_2d:.2f}$)', zorder=3)
    
    # Shading only the visible excluded region below Rmin
    ax_top.axvspan(xmin, rmin, color='gray', alpha=0.15, label='Excluded ($R < R_{\\mathrm{min}}$)')
    
    ax_top.set_xscale('log')
    ax_top.set_yscale('log')
    ax_top.set_xlim(xmin, xmax)
    
    # Compute appropriate y-limits based on visible points
    y_vis = d_all[mask_vis]
    ax_top.set_ylim(min(y_vis)*0.4, max(y_vis)*2.5)
    
    sub_title = frac.replace('_', '.')
    ax_top.set_title(f"VSF Fit: Subsample {sub_title} ($R_{{\\mathrm{{min}}}} = {rmin:.1f}\\ h^{{-1}}\\mathrm{{Mpc}}$)", fontsize=13, fontweight='bold')
    ax_top.set_ylabel(r'$\mathrm{d}n/\mathrm{d}\ln R\ [h^3\mathrm{Mpc}^{-3}]$', fontsize=12)
    ax_top.grid(True, which="both", ls="--", alpha=0.3)
    ax_top.legend(loc='lower left', fontsize=10.5)
    
    # Bottom Panel: Residuals
    res_true_vals = (d_all - m_true_all) / err_all
    res_bf_vals = (d_all - m_bf_all) / err_all
    
    ax_bot.errorbar(r_all[mask_vis], res_true_vals[mask_vis], yerr=1.0, fmt='o--', color='#B58900', ms=5.5, label='True Cosmo', zorder=4, capsize=3)
    ax_bot.errorbar(r_all[mask_vis], res_bf_vals[mask_vis], yerr=1.0, fmt='s-', color='#9A031E', ms=5.5, label='Best-Fit Cosmo', zorder=3, capsize=3)
    
    ax_bot.axvspan(xmin, rmin, color='gray', alpha=0.15)
    ax_bot.axhline(0, color='black', ls='-', alpha=0.5)
    ax_bot.axhline(1, color='gray', ls=':', alpha=0.4)
    ax_bot.axhline(-1, color='gray', ls=':', alpha=0.4)
    ax_bot.axhline(2, color='gray', ls='--', alpha=0.3)
    ax_bot.axhline(-2, color='gray', ls='--', alpha=0.3)
    
    ax_bot.set_xscale('log')
    ax_bot.set_xlim(xmin, xmax)
    ax_bot.set_ylim(-3.5, 3.5)
    ax_bot.set_xlabel(r'$R\ [h^{-1}\mathrm{Mpc}]$', fontsize=12)
    ax_bot.set_ylabel(r'Residuals [$\sigma$]', fontsize=12)
    ax_bot.grid(True, which="both", ls="--", alpha=0.3)
    
    plt.tight_layout()
    out_name = f"vsf_residuals_subsample_{frac}_zoomed.png"
    plt.savefig(os.path.join(THESIS_FIG, out_name), dpi=300, bbox_inches='tight')
    # Also overwrite the original figure so main.tex picks it up directly!
    plt.savefig(os.path.join(THESIS_FIG, f"vsf_residuals_subsample_{frac}.png"), dpi=300, bbox_inches='tight')
    print(f"Saved zoomed VSF plot to {out_name}")
    plt.close()

def plot_2d_corner(frac):
    rmin = OPTIMAL_RMIN[frac]
    chain_fname = f"cbl_v9_subsample_{frac}_2D_R{int(rmin)}_OV05.npy"
    chain_path = os.path.join(CHAINS_DIR, chain_fname)
    
    chain = np.load(chain_path)
    flat_chain = chain[int(len(chain)*0.3):].reshape(-1, chain.shape[-1])
    
    labels = [r'$\Omega_m$', r'$\sigma_8$']
    truths = [0.315, 0.811]
    
    fig = corner.corner(
        flat_chain,
        labels=labels,
        truths=truths,
        truth_color='#B58900',
        color='#1A237E',
        show_titles=True,
        title_kwargs={"fontsize": 13},
        hist_kwargs={"color": '#1A237E', "alpha": 0.6},
        quantiles=[0.16, 0.50, 0.84]
    )
    
    sub_title = frac.replace('_', '.')
    fig.suptitle(f"Cosmological Constraints: Subsample {sub_title} (Fixed Bias, $R_{{\\mathrm{{min}}}} = {rmin:.1f}\\ h^{{-1}}\\mathrm{{Mpc}}$)", 
                 fontsize=14, fontweight='bold', y=1.03)
    
    out_name = f"corner_subsample_{frac}_2D.png"
    plt.savefig(os.path.join(THESIS_FIG, out_name), dpi=300, bbox_inches='tight')
    print(f"Saved 2D corner plot to {out_name}")
    plt.close()

def plot_4d_corner(frac):
    rmin = OPTIMAL_RMIN[frac]
    chain_fname = f"cbl_v8_subsample_{frac}_R{int(rmin)}_OV05.npy"
    chain_path = os.path.join(CHAINS_DIR, chain_fname)
    
    chain = np.load(chain_path)
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
    
    sub_title = frac.replace('_', '.')
    fig.suptitle(f"4D Joint Posteriors: Subsample {sub_title} (Floating Bias, $R_{{\\mathrm{{min}}}} = {rmin:.1f}\\ h^{{-1}}\\mathrm{{Mpc}}$)", 
                 fontsize=14, fontweight='bold', y=1.02)
    
    out_name = f"corner_subsample_{frac}.png"
    plt.savefig(os.path.join(THESIS_FIG, out_name), dpi=300, bbox_inches='tight')
    print(f"Saved 4D corner plot to {out_name}")
    plt.close()

if __name__ == '__main__':
    for frac in SUBSAMPLES:
        plot_zoomed_vsf(frac)
        plot_2d_corner(frac)
        plot_4d_corner(frac)
