#!/usr/bin/env python3
"""
calibrate_contarini_halo.py

Fits the Contarini et al. void-barrier parameters (B_slope, B_offset)
to the Aletheia halo void size function at the true simulation cosmology,
using the independently measured linear halo bias b_eff.

Addresses the key question from Simone Sartori and the advisor:
"Does calibrating the void barrier (B_slope, B_offset) allow the Vdn model
to absorb model approximations and provide a statistically valid fit (chi^2/nu ~ 1)
for genuine halo tracers?"
"""
import os, sys, time, argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import chi2 as chi2_dist

# CBL bootstrap
CBL_HOME = '/Users/arun/projects/CosmoBolognaLib'
sys.path.insert(0, os.path.join(CBL_HOME, 'Python'))
os.environ.setdefault('DYLD_LIBRARY_PATH', CBL_HOME)
import CosmoBolognaLib as cbl

HERE = os.path.dirname(os.path.abspath(__file__))
COV_DIR = os.path.join(HERE, '..', 'data', 'voids_halo', 'cov')
PLOT_DIR = os.path.join(HERE, '..', 'plots', 'halos')
os.makedirs(PLOT_DIR, exist_ok=True)

# Aletheia true cosmology
OMEGA_M = 0.315
OMEGA_B = 0.0493
H0 = 0.6736
N_S = 0.9649
SIGMA8 = 0.811
SCALAR_AMP = 2.194e-9
SCALAR_PIVOT = 0.05
TAU = 0.0925

cosmo = cbl.cosmology.LCDM(
    OMEGA_M, OMEGA_B, 0.0,
    H0, SCALAR_AMP, SCALAR_PIVOT,
    N_S, TAU, True
)
cosmo.set_parameter("sigma8", SIGMA8)
DEL_C = cosmo.deltac(0.0)
sf = cbl.cosmology.SizeFunction(cosmo)

def run_calibration(sample='100P', criterion='touch', b_eff=1.0211, rmin=18.0, delta_v_NL=-0.79):
    cov_file = os.path.join(COV_DIR, f"cov_jk_halo_{sample}_{criterion}.npz")
    if not os.path.exists(cov_file):
        print(f"Covariance file not found: {cov_file}")
        return
        
    data_dict = np.load(cov_file)
    bin_centers = data_dict['bin_centers']
    dn_dlnR = data_dict['dn_dlnR']
    cov_full = data_dict['cov_full']
    sigma_jk = data_dict['sigma_jk']
    
    # Filter by R_min cut
    mask = (bin_centers >= rmin) & (dn_dlnR > 0.0)
    r_act = bin_centers[mask]
    y_act = dn_dlnR[mask]
    s_act = sigma_jk[mask]
    cov_act = cov_full[np.ix_(mask, mask)]
    p = len(r_act)
    
    # Hartlap correction for p bins and 64 JK samples
    hartlap = (64.0 - p - 2.0) / 63.0
    inv_cov_hartlap = hartlap * np.linalg.pinv(cov_act)
    
    print(f"\n=======================================================")
    print(f" Contarini Barrier Calibration for {sample} ({criterion})")
    print(f" Scale cut: R >= {rmin:.1f} h^-1 Mpc (Active bins: {p})")
    print(f" Fixed Cosmology: Omega_m={OMEGA_M}, sigma8={SIGMA8}")
    print(f" Fixed Linear Halo Bias b_eff: {b_eff:.4f}")
    print(f" Hartlap factor: {hartlap:.4f}")
    print(f"=======================================================")
    
    def model_vsf(params):
        slope, offset = params
        b_void = slope * b_eff + offset
        if b_void <= 0.80 or slope <= 0.0:
            return np.full(p, 1e-30)
        vals = []
        for r in r_act:
            try:
                v = sf.size_function(float(r), 0.0, "Vdn", float(b_eff),
                                     float(slope), float(offset),
                                     float(delta_v_NL), DEL_C, "EisensteinHu")
                vals.append(max(v, 1e-30))
            except:
                vals.append(1e-30)
        return np.array(vals)
        
    def chi2_func(params):
        m = model_vsf(params)
        diff = y_act - m
        chi2 = float(np.dot(diff, np.dot(inv_cov_hartlap, diff)))
        return chi2
        
    # Initial guess from Contarini literature
    init_guess = [0.854, 0.420]
    res = minimize(chi2_func, init_guess, method='Nelder-Mead',
                   options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 400})
                   
    opt_slope, opt_offset = res.x
    opt_b_void = opt_slope * b_eff + opt_offset
    opt_chi2 = res.fun
    ndof = p - 2
    red_chi2 = opt_chi2 / ndof if ndof > 0 else opt_chi2
    p_val = 1.0 - chi2_dist.cdf(opt_chi2, ndof) if ndof > 0 else 0.0
    
    print("\n----------------- Optimization Results -----------------")
    print(f"Optimal B_slope:      {opt_slope:.5f}")
    print(f"Optimal B_offset:     {opt_offset:.5f}")
    print(f"Effective Void Bias:  {opt_b_void:.5f}")
    print(f"Minimum chi^2:        {opt_chi2:.3f}")
    print(f"Degrees of Freedom:   {ndof}")
    print(f"Reduced chi^2/nu:     {red_chi2:.3f}")
    print(f"P-value:              {p_val:.4f}")
    print("--------------------------------------------------------\n")
    
    # Residuals table
    th_act_opt = model_vsf([opt_slope, opt_offset])
    pulls = (y_act - th_act_opt) / s_act
    print(f"{'R [Mpc/h]':<10} | {'Data':<12} | {'Model':<12} | {'Sigma_JK':<12} | {'Pull (d-m)/s':<12}")
    print("-" * 65)
    for ri, yi, mi, si, pi in zip(r_act, y_act, th_act_opt, s_act, pulls):
        print(f"{ri:<10.2f} | {yi:<12.4e} | {mi:<12.4e} | {si:<12.4e} | {pi:<12.2f}")
        
    # Fine curve for plotting
    r_fine = np.linspace(r_act[0] * 0.95, r_act[-1] * 1.05, 80)
    th_fine_opt = np.array([
        sf.size_function(float(r), 0.0, "Vdn", float(b_eff),
                         float(opt_slope), float(opt_offset),
                         float(delta_v_NL), DEL_C, "EisensteinHu")
        for r in r_fine
    ])
    
    # 2D Chi2 contour grid
    s_grid = np.linspace(max(0.1, opt_slope - 0.3), opt_slope + 0.3, 25)
    off_grid = np.linspace(opt_offset - 0.3, opt_offset + 0.3, 25)
    chi2_grid = np.zeros((len(off_grid), len(s_grid)))
    for i, off_val in enumerate(off_grid):
        for j, s_val in enumerate(s_grid):
            chi2_grid[i, j] = chi2_func([s_val, off_val])
            
    # Plotting
    fig = plt.figure(figsize=(15, 4.8), dpi=150)
    
    # Panel 1: VSF fit
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.errorbar(r_act, y_act, yerr=s_act, fmt='o', color='#2c3e50',
                 ecolor='#7f8c8d', elinewidth=1.5, capsize=3, label=f'Aletheia 100P ({criterion})')
    ax1.plot(r_fine, th_fine_opt, color='#e74c3c', lw=2.2,
             label=f'Calibrated Vdn ($\chi^2/\\nu={red_chi2:.2f}$)\n$B_{{slope}}={opt_slope:.3f}, B_{{off}}={opt_offset:.3f}$\n$b_{{void}}={opt_b_void:.3f}$')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$R\ [h^{-1}\mathrm{Mpc}]$', fontsize=12)
    ax1.set_ylabel(r'$\mathrm{d}n/\mathrm{d}\ln R\ [h^3\mathrm{Mpc}^{-3}]$', fontsize=12)
    ax1.set_title(f'Halo Void Size Function (R >= {rmin:.1f})', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='lower left')
    ax1.grid(True, alpha=0.3, ls=':')
    
    # Panel 2: Fractional residuals
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.axhline(0, color='black', lw=1, ls='-')
    ax2.axhspan(-1, 1, color='green', alpha=0.15, label=r'$\pm 1\sigma$')
    ax2.axhspan(-2, 2, color='green', alpha=0.08, label=r'$\pm 2\sigma$')
    ax2.plot(r_act, pulls, 'o-', color='#e74c3c', lw=1.8, label=r'$(d - m)/\sigma_{\mathrm{JK}}$')
    ax2.set_xscale('log')
    ax2.set_xlabel(r'$R\ [h^{-1}\mathrm{Mpc}]$', fontsize=12)
    ax2.set_ylabel(r'Residuals $[\sigma_{\mathrm{JK}}]$', fontsize=12)
    ax2.set_ylim(-3.0, 3.0)
    ax2.set_title('Standardized Fit Residuals', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3, ls=':')
    
    # Panel 3: 2D Chi2 Landscape
    ax3 = fig.add_subplot(1, 3, 3)
    S_mat, OFF_mat = np.meshgrid(s_grid, off_grid)
    dchi2 = chi2_grid - opt_chi2
    cs = ax3.contourf(S_mat, OFF_mat, dchi2, levels=[0, 2.3, 6.18, 11.83, 20.0],
                      colors=['#3498db', '#5dade2', '#aed6f1', '#ebf5fb'])
    cbar = plt.colorbar(cs, ax=ax3)
    cbar.set_label(r'$\Delta \chi^2$', fontsize=11)
    ax3.contour(S_mat, OFF_mat, dchi2, levels=[2.3, 6.18], colors=['#2980b9', '#1f618d'], linewidths=1.5)
    ax3.plot(opt_slope, opt_offset, 'r*', ms=12, label='Best-fit Barrier')
    ax3.set_xlabel(r'$B_{\mathrm{slope}}$', fontsize=12)
    ax3.set_ylabel(r'$B_{\mathrm{offset}}$', fontsize=12)
    ax3.set_title('Contarini Barrier Likelihood Contours', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9, loc='upper right')
    ax3.grid(True, alpha=0.3, ls=':')
    
    plt.tight_layout()
    out_fig = os.path.join(PLOT_DIR, f"contarini_calibration_{sample}_{criterion}_Rmin{int(rmin)}.png")
    plt.savefig(out_fig, dpi=200)
    print(f"\nSaved diagnostic calibration plot to {out_fig}\n")
    plt.close()
    
    return {
        'slope': opt_slope,
        'offset': opt_offset,
        'b_void': opt_b_void,
        'chi2': opt_chi2,
        'ndof': ndof,
        'red_chi2': red_chi2,
        'p_val': p_val
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=str, default='100P')
    parser.add_argument('--criterion', type=str, default='touch', choices=['center', 'touch'])
    parser.add_argument('--b-eff', type=float, default=1.0211)
    parser.add_argument('--rmin', type=float, default=18.0)
    args = parser.parse_args()
    
    run_calibration(args.sample, args.criterion, args.b_eff, args.rmin)
