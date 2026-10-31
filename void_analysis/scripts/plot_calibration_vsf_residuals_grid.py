#!/usr/bin/env python3
"""
plot_calibration_vsf_residuals_grid.py

Generates a publication-grade 4-sample comparative grid for Prof. Marulli:
For all 4 halo mass cuts (100P, 75P, 50P, 30P):
- Top subpanel: Measured VSF data points (points only, error bars) vs recalibrated Vdn model curves at fiducial cosmology (Omega_m=0.315, sigma8=0.811) for BOTH BitVF and REVOLVER.
- Bottom subpanel: Standardized pulls (d - m) / sigma_JK with +/-1sigma and +/-2sigma reference bands.
- Accompanying table with b_eff, b_void, chi^2, ndof, and reduced chi^2/nu.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt

CBL_HOME = '/Users/arun/projects/CosmoBolognaLib'
sys.path.insert(0, os.path.join(CBL_HOME, 'Python'))
os.environ.setdefault('DYLD_LIBRARY_PATH', CBL_HOME)
import CosmoBolognaLib as cbl

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, '..', 'data')
PLOT_DIR = os.path.join(HERE, '..', 'plots', 'halos')
THESIS_FIG_DIR = os.path.join(HERE, '..', 'main_thesis_docs', 'figures')
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(THESIS_FIG_DIR, exist_ok=True)

# Fiducial Aletheia cosmology
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
del_c = cosmo.deltac(0.0)
sf = cbl.cosmology.SizeFunction(cosmo)

SAMPLES = ['100P', '75P', '50P', '30P']
B_EFF_MAP = {'100P': 1.0211, '75P': 0.9583, '50P': 0.9108, '30P': 0.8623}
R_MIN_MAP = {'100P': 18.0, '75P': 18.0, '50P': 16.0, '30P': 16.0}

# Optimal calibrated b_void from grid search at true cosmology
CALIB_PARAMS = {
    'BitVF': {
        '100P': 1.5130,
        '75P':  1.4780,
        '50P':  1.4240,
        '30P':  1.3730
    },
    'REVOLVER': {
        '100P': 1.5030,
        '75P':  1.4690,
        '50P':  1.4050,
        '30P':  1.3650
    }
}

def generate_grid_plot():
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=200,
                             gridspec_kw={'height_ratios': [3, 1.2], 'hspace': 0.08, 'wspace': 0.22})
    
    stats = []

    for col, s in enumerate(SAMPLES):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]
        
        beff = B_EFF_MAP[s]
        rmin = R_MIN_MAP[s]
        
        # Colors
        c_bv = '#1f77b4'  # Blue
        c_rv = '#d95f02'  # Orange
        
        for finder, cov_subdir, color, marker, label_name in [
            ('BitVF', 'voids_halo', c_bv, 'o', 'BitVF'),
            ('REVOLVER', 'voids_halo_revolver', c_rv, 's', 'REVOLVER')
        ]:
            cov_file = os.path.join(DATA_DIR, cov_subdir, 'cov', f'cov_jk_halo_{s}_touch.npz')
            d = np.load(cov_file)
            r = d['bin_centers']
            y = d['dn_dlnR']
            cov = d['cov_full']
            sig = d['sigma_jk']
            
            mask = (r >= rmin) & (y > 0.0)
            r_act = r[mask]
            y_act = y[mask]
            s_act = sig[mask]
            cov_act = cov[np.ix_(mask, mask)]
            p = len(r_act)
            hartlap = (64.0 - p - 2.0) / 63.0
            inv_cov = hartlap * np.linalg.pinv(cov_act)
            
            opt_b = CALIB_PARAMS[finder][s]
            
            # Theoretical model on active bins
            m_act = np.zeros(p)
            for i, ri in enumerate(r_act):
                try:
                    m_act[i] = sf.size_function(float(ri), 0.0, 'Vdn', float(beff), 1.0, float(opt_b - beff), -0.79, del_c, 'EisensteinHu')
                except:
                    m_act[i] = 1e-30
            
            diff = y_act - m_act
            chi2_val = float(np.dot(diff, np.dot(inv_cov, diff)))
            ndof = p - 1
            red_chi2 = chi2_val / ndof
            
            # Standardized pulls
            pulls = (y_act - m_act) / s_act
            
            stats.append({
                'sample': s,
                'beff': beff,
                'finder': finder,
                'rmin': rmin,
                'p_bins': p,
                'b_void': opt_b,
                'chi2': chi2_val,
                'ndof': ndof,
                'red_chi2': red_chi2,
                'pulls': pulls,
                'r_act': r_act
            })
            
            # Plot data points with error bars (no connecting line!)
            # Prevent lower error bar clipping on log scale
            yerr_low = np.minimum(s_act, y_act * 0.95)
            # Slight offset on x for visual clarity between finders
            x_shift = 1.015 if finder == 'REVOLVER' else 0.985
            ax_top.errorbar(r_act * x_shift, y_act, yerr=[yerr_low, s_act], fmt=marker,
                            color=color, ms=6, capsize=3, lw=1.2,
                            label=rf'{label_name}: $b_{{\rm void}}={opt_b:.3f}$ ($\chi^2/\nu={red_chi2:.2f}$)')
            
            # Smooth model curve
            r_dense = np.linspace(r_act[0] * 0.92, r_act[-1] * 1.08, 80)
            th_dense = np.array([
                sf.size_function(float(ri), 0.0, "Vdn", float(beff),
                                 1.0, float(opt_b - beff),
                                 -0.79, del_c, "EisensteinHu")
                for ri in r_dense
            ])
            ax_top.plot(r_dense, th_dense, color=color, lw=1.8, ls='-')
            
            # Plot pulls on bottom subpanel
            ax_bot.plot(r_act * x_shift, pulls, marker=marker, ls='none', color=color, ms=6, label=label_name)

        # Style top panel
        ax_top.set_xscale('log')
        ax_top.set_yscale('log')
        ax_top.set_ylim(2e-9, 2e-5)
        ax_top.set_title(rf'Sample {s} ($b_{{\rm eff}}={beff:.3f}$)', fontsize=13, fontweight='bold')
        if col == 0:
            ax_top.set_ylabel(r'$\mathrm{d}n/\mathrm{d}\ln R\ [h^3\mathrm{Mpc}^{-3}]$', fontsize=12)
        ax_top.legend(fontsize=8.5, loc='lower left')
        ax_top.grid(True, alpha=0.3, ls=':')
        ax_top.set_xticklabels([])
        
        # Style bottom panel (pulls)
        ax_bot.axhline(0, color='black', lw=1, ls='-')
        ax_bot.axhspan(-1, 1, color='green', alpha=0.15)
        ax_bot.axhspan(-2, 2, color='green', alpha=0.08)
        ax_bot.set_xscale('log')
        ax_bot.set_xlim(ax_top.get_xlim())
        ax_bot.set_ylim(-4.5, 4.5)
        ax_bot.set_xlabel(r'Void Radius $R\ [h^{-1}\mathrm{Mpc}]$', fontsize=12)
        if col == 0:
            ax_bot.set_ylabel(r'Pull $[\sigma_{\rm JK}]$', fontsize=11)
        ax_bot.grid(True, alpha=0.3, ls=':')
        
    out_png = os.path.join(PLOT_DIR, 'calibration_vsf_residuals_grid.png')
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(THESIS_FIG_DIR, 'calibration_vsf_residuals_grid.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved 4-sample calibration grid to {out_png}")
    return stats

if __name__ == '__main__':
    generate_grid_plot()
