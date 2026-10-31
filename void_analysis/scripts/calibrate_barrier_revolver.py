#!/usr/bin/env python3
"""
calibrate_barrier_revolver.py

Fits the effective void barrier parameter b_void to the measured REVOLVER
halo void size function at the true Aletheia simulation cosmology (Omega_m=0.315, sigma8=0.811),
using CosmoBolognaLib's Vdn model and the 64-region spatial Jackknife covariance.

Extracts:
1. Optimal b_void
2. Minimum chi^2, degrees of freedom, and reduced chi^2/nu
3. Exact 68% calibration uncertainty sigma(b_void) from Delta chi^2 = 1.0 profile
"""
import os, sys, time, argparse
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2 as chi2_dist
import matplotlib.pyplot as plt

# CBL bootstrap
CBL_HOME = '/Users/arun/projects/CosmoBolognaLib'
sys.path.insert(0, os.path.join(CBL_HOME, 'Python'))
os.environ.setdefault('DYLD_LIBRARY_PATH', CBL_HOME)
import CosmoBolognaLib as cbl

HERE = os.path.dirname(os.path.abspath(__file__))
COV_DIR = os.path.join(HERE, '..', 'data', 'voids_halo_revolver', 'cov')
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

def calibrate_sample(sample='100P', criterion='touch', b_eff=1.0211, rmin=18.0, delta_v_NL=-0.79):
    cov_file = os.path.join(COV_DIR, f"cov_jk_halo_{sample}_{criterion}.npz")
    if not os.path.exists(cov_file):
        print(f"Covariance file not found: {cov_file}")
        return None
        
    data = np.load(cov_file)
    bin_centers = data['bin_centers']
    dn_dlnR = data['dn_dlnR']
    cov_full = data['cov_full']
    sigma_jk = data['sigma_jk']
    
    # Filter by R_min scale cut
    mask = (bin_centers >= rmin) & (dn_dlnR > 0.0)
    r_act = bin_centers[mask]
    y_act = dn_dlnR[mask]
    s_act = sigma_jk[mask]
    cov_act = cov_full[np.ix_(mask, mask)]
    p = len(r_act)
    
    # Hartlap correction for 64 JK samples and p active bins
    hartlap = (64.0 - p - 2.0) / 63.0
    inv_cov_hartlap = hartlap * np.linalg.pinv(cov_act)
    
    print("=" * 65)
    print(f" REVOLVER Barrier Calibration: {sample} ({criterion})")
    print(f" Scale cut: R >= {rmin:.1f} h^-1 Mpc ({p} active bins)")
    print(f" Linear Halo Bias b_eff: {b_eff:.4f}")
    print(f" Hartlap Factor: {hartlap:.4f}")
    print("=" * 65)
    
    def model_vsf(b_void):
        if b_void <= 0.5:
            return np.full(p, 1e-30)
        # Use slope = 1.0, offset = b_void - b_eff
        slope = 1.0
        offset = b_void - b_eff
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
        
    def chi2_func(b_void):
        m = model_vsf(b_void)
        diff = y_act - m
        chi2 = float(np.dot(diff, np.dot(inv_cov_hartlap, diff)))
        return chi2
        
    # Minimize 1D chi2 profile
    res = minimize_scalar(chi2_func, bounds=(0.80, 2.50), method='bounded')
    opt_b_void = res.x
    opt_chi2 = res.fun
    ndof = p - 1  # 1 fitted parameter (b_void)
    red_chi2 = opt_chi2 / ndof if ndof > 0 else opt_chi2
    p_val = 1.0 - chi2_dist.cdf(opt_chi2, ndof) if ndof > 0 else 0.0
    
    # Compute 68% calibration uncertainty: Delta chi^2 = 1.0
    b_grid = np.linspace(opt_b_void - 0.05, opt_b_void + 0.05, 100)
    chi2_grid = np.array([chi2_func(b) for b in b_grid])
    dchi2 = chi2_grid - opt_chi2
    
    # 2nd derivative curvature: sigma = 1 / sqrt(0.5 * d2chi2/db2)
    poly = np.polyfit(b_grid - opt_b_void, dchi2, 2)
    # poly[0] * delta_b^2 = 1 => delta_b = 1 / sqrt(poly[0])
    if poly[0] > 0:
        sigma_b_void = 1.0 / np.sqrt(poly[0])
    else:
        sigma_b_void = 0.01
        
    print("\n----------------- Optimization Results -----------------")
    print(f"Optimal b_void:        {opt_b_void:.5f} +/- {sigma_b_void:.5f}")
    print(f"Minimum chi^2:         {opt_chi2:.3f}")
    print(f"Degrees of Freedom:    {ndof}")
    print(f"Reduced chi^2/nu:      {red_chi2:.3f}")
    print(f"P-value:               {p_val:.4f}")
    print("--------------------------------------------------------\n")
    
    # Residuals table
    th_act_opt = model_vsf(opt_b_void)
    pulls = (y_act - th_act_opt) / s_act
    print(f"{'R [Mpc/h]':<10} | {'Data':<12} | {'Model':<12} | {'Sigma_JK':<12} | {'Pull (d-m)/s':<12}")
    print("-" * 65)
    for ri, yi, mi, si, pi in zip(r_act, y_act, th_act_opt, s_act, pulls):
        print(f"{ri:<10.2f} | {yi:<12.4e} | {mi:<12.4e} | {si:<12.4e} | {pi:<12.2f}")
        
    # Plotting: Measured VSF as points with error bars (no connecting line) + Model Curve
    r_fine = np.linspace(r_act[0] * 0.95, r_act[-1] * 1.05, 100)
    th_fine = np.array([
        sf.size_function(float(r), 0.0, "Vdn", float(b_eff),
                         1.0, float(opt_b_void - b_eff),
                         float(delta_v_NL), DEL_C, "EisensteinHu")
        for r in r_fine
    ])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08}, dpi=150)
    
    # Top panel: VSF
    ax1.errorbar(r_act, y_act, yerr=s_act, fmt='o', color='#2c3e50',
                 ecolor='#7f8c8d', elinewidth=1.5, capsize=3, label=f'REVOLVER {sample} ({criterion})')
    ax1.plot(r_fine, th_fine, color='#e74c3c', lw=2.2,
             label=f'Calibrated Vdn ($\chi^2/\\nu={red_chi2:.2f}$)\n$b_{{void}}={opt_b_void:.3f} \pm {sigma_b_void:.4f}$')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylabel(r'$\mathrm{d}n/\mathrm{d}\ln R\ [h^3\mathrm{Mpc}^{-3}]$', fontsize=12)
    ax1.set_title(f'REVOLVER Halo Void Size Function ({sample} {criterion})\n'
                  rf'$R \geq {rmin:.0f}\ h^{{-1}}\mathrm{{Mpc}}$ | $b_{{\rm eff}}={b_eff:.3f}$',
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='lower left')
    ax1.grid(True, alpha=0.3, ls=':')
    
    # Bottom panel: Pulls
    ax2.axhline(0, color='black', lw=1, ls='-')
    ax2.axhspan(-1, 1, color='green', alpha=0.15, label=r'$\pm 1\sigma$')
    ax2.axhspan(-2, 2, color='green', alpha=0.08, label=r'$\pm 2\sigma$')
    ax2.plot(r_act, pulls, 'o', color='#e74c3c', ms=6, label=r'$(d - m)/\sigma_{\mathrm{JK}}$')
    ax2.set_xscale('log')
    ax2.set_xlabel(r'$R\ [h^{-1}\mathrm{Mpc}]$', fontsize=12)
    ax2.set_ylabel(r'Pull $[\sigma_{\mathrm{JK}}]$', fontsize=12)
    ax2.set_ylim(-3.0, 3.0)
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3, ls=':')
    
    out_fig = os.path.join(PLOT_DIR, f"revolver_calibration_{sample}_{criterion}_Rmin{int(rmin)}.png")
    plt.savefig(out_fig, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved calibration plot to {out_fig}\n")
    
    return {
        'sample': sample,
        'criterion': criterion,
        'b_eff': b_eff,
        'b_void': opt_b_void,
        'sigma_b_void': sigma_b_void,
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
    
    calibrate_sample(args.sample, args.criterion, args.b_eff, args.rmin)
