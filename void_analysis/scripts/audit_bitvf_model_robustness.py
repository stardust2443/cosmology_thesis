#!/usr/bin/env python3
"""
audit_bitvf_model_robustness.py

Executes a comprehensive 5-part robustness audit of the Vdn theoretical model
on BitVF halo void catalogues across all four Aletheia mass thresholds (100P, 75P, 50P, 30P):

1. Tail-Truncation Audit (R_max sensitivity, filtering N ~ 1 Poisson tail bins)
2. Scale-Validity Plateau (R_min sweep from 14 to 20 Mpc/h)
3. Global Multi-Tracer Joint Barrier Fit (joint B_slope, B_offset across all 4 samples)
4. Cleaning Overlap Sensitivity (TOUCH vs CENTER / OV00 vs OV05)
5. Poisson Likelihood vs Gaussian Chi^2 (Cash / C-statistic in sparse tail)

Generates publication-quality diagnostic figure:
plots/halos/bitvf_model_robustness_audit.png
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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

# Aletheia true cosmology
OMEGA_M = 0.315
OMEGA_B = 0.0493
H0 = 0.6736
N_S = 0.9649
SIGMA8 = 0.811
SCALAR_AMP = 2.194e-9
SCALAR_PIVOT = 0.05
TAU = 0.0925
V_BOX = 1e9  # (Mpc/h)^3

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
COLORS = {'100P': '#e74c3c', '75P': '#8e44ad', '50P': '#16a085', '30P': '#d35400'}

def fit_1d_bvoid(r_act, y_act, cov_act, beff):
    p = len(r_act)
    hartlap = (64.0 - p - 2.0) / 63.0 if p < 60 else 1.0
    inv_cov = hartlap * np.linalg.pinv(cov_act)
    
    b_grid = np.linspace(1.20, 1.70, 501)
    chi2_vals = []
    for b_cand in b_grid:
        m = np.zeros(p)
        for i, ri in enumerate(r_act):
            try:
                m[i] = sf.size_function(float(ri), 0.0, 'Vdn', float(beff), 1.0, float(b_cand - beff), -0.79, del_c, 'EisensteinHu')
            except:
                m[i] = 1e-30
        diff = y_act - m
        chi2_vals.append(float(np.dot(diff, np.dot(inv_cov, diff))))
    opt_idx = np.argmin(chi2_vals)
    opt_b = b_grid[opt_idx]
    opt_chi2 = chi2_vals[opt_idx]
    ndof = max(1, p - 1)
    return opt_b, opt_chi2, ndof, opt_chi2 / ndof

def run_all_audits():
    print("=" * 75)
    print("  COMPREHENSIVE ROBUSTNESS AUDIT OF BitVF + Vdn THEORETICAL MODEL")
    print("=" * 75)
    
    # =========================================================================
    # TEST 1: TAIL TRUNCATION AUDIT (R_max sensitivity)
    # =========================================================================
    print("\n[TEST 1] Tail Truncation Audit (Sensitivity to Poisson N ~ 1 Bins)")
    print("-" * 75)
    print(f"{'Sample':<8} | {'R_max':<8} | {'p bins':<6} | {'b_void':<8} | {'Delta b [%]':<12} | {'chi^2':<8} | {'chi^2/nu':<8}")
    print("-" * 75)
    
    t1_results = {s: [] for s in SAMPLES}
    for s in SAMPLES:
        beff = B_EFF_MAP[s]
        rmin = 18.0 if s in ['100P', '75P'] else 16.0
        cov_file = os.path.join(DATA_DIR, 'voids_halo', 'cov', f'cov_jk_halo_{s}_touch.npz')
        d = np.load(cov_file)
        r, y, cov = d['bin_centers'], d['dn_dlnR'], d['cov_full']
        
        # Base calibration (all bins)
        mask_base = (r >= rmin) & (y > 0.0)
        b_base, _, _, _ = fit_1d_bvoid(r[mask_base], y[mask_base], cov[np.ix_(mask_base, mask_base)], beff)
        
        for rmax in [40.0, 32.0, 28.0]:
            mask = (r >= rmin) & (r <= rmax) & (y > 0.0)
            if np.sum(mask) < 3: continue
            b_val, chi2, ndof, red_chi2 = fit_1d_bvoid(r[mask], y[mask], cov[np.ix_(mask, mask)], beff)
            pct_shift = ((b_val - b_base) / b_base) * 100.0
            print(f"{s:<8} | {rmax:<8.1f} | {len(r[mask]):<6} | {b_val:<8.4f} | {pct_shift:>+10.2f}% | {chi2:<8.2f} | {red_chi2:<8.2f}")
            t1_results[s].append({'rmax': rmax, 'b_void': b_val, 'chi2_nu': red_chi2, 'p': len(r[mask])})

    # =========================================================================
    # TEST 2: SCALE-VALIDITY AUDIT (R_min sweep)
    # =========================================================================
    print("\n[TEST 2] Scale-Validity Plateau Audit (R_min Sweep: 14 to 20 Mpc/h)")
    print("-" * 75)
    t2_results = {s: [] for s in SAMPLES}
    for s in SAMPLES:
        beff = B_EFF_MAP[s]
        cov_file = os.path.join(DATA_DIR, 'voids_halo', 'cov', f'cov_jk_halo_{s}_touch.npz')
        d = np.load(cov_file)
        r, y, cov = d['bin_centers'], d['dn_dlnR'], d['cov_full']
        
        for rmin_cand in [14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]:
            mask = (r >= rmin_cand) & (y > 0.0)
            if np.sum(mask) < 3: continue
            b_val, chi2, ndof, red_chi2 = fit_1d_bvoid(r[mask], y[mask], cov[np.ix_(mask, mask)], beff)
            t2_results[s].append({'rmin': rmin_cand, 'b_void': b_val, 'chi2_nu': red_chi2, 'p': len(r[mask])})
        
        b_vals = [res['b_void'] for res in t2_results[s]]
        max_spread = (max(b_vals) - min(b_vals)) / np.mean(b_vals) * 100.0
        print(f"Sample {s:<5} (b_eff={beff:.3f}) | b_void range: [{min(b_vals):.4f}, {max(b_vals):.4f}] | Max Variation: {max_spread:.2f}% across R_min in [14, 20]")

    # =========================================================================
    # TEST 3: GLOBAL MULTI-TRACER JOINT FIT
    # =========================================================================
    print("\n[TEST 3] Global Multi-Tracer Joint Barrier Fit Across All 4 Catalogues")
    print("-" * 75)
    def joint_fit(rmax_filter=40.0):
        def obj(params):
            slope, offset = params
            tot = 0.0
            for s in SAMPLES:
                beff = B_EFF_MAP[s]
                rmin = 18.0 if s in ['100P', '75P'] else 16.0
                cov_file = os.path.join(DATA_DIR, 'voids_halo', 'cov', f'cov_jk_halo_{s}_touch.npz')
                d = np.load(cov_file)
                r, y, cov = d['bin_centers'], d['dn_dlnR'], d['cov_full']
                mask = (r >= rmin) & (r <= rmax_filter) & (y > 0.0)
                r_act, y_act = r[mask], y[mask]
                cov_act = cov[np.ix_(mask, mask)]
                p = len(r_act)
                hartlap = (64.0 - p - 2.0) / 63.0
                inv_cov = hartlap * np.linalg.pinv(cov_act)
                b_cand = slope * beff + offset
                m = np.array([sf.size_function(float(ri), 0.0, 'Vdn', float(beff), 1.0, float(b_cand - beff), -0.79, del_c, 'EisensteinHu') for ri in r_act])
                diff = y_act - m
                tot += float(np.dot(diff, np.dot(inv_cov, diff)))
            return tot
        
        res = minimize(obj, [0.91, 0.58], method='Nelder-Mead')
        tot_bins = sum([len(np.load(os.path.join(DATA_DIR, 'voids_halo', 'cov', f'cov_jk_halo_{s}_touch.npz'))['bin_centers'][(np.load(os.path.join(DATA_DIR, 'voids_halo', 'cov', f'cov_jk_halo_{s}_touch.npz'))['bin_centers'] >= (18.0 if s in ['100P', '75P'] else 16.0)) & (np.load(os.path.join(DATA_DIR, 'voids_halo', 'cov', f'cov_jk_halo_{s}_touch.npz'))['bin_centers'] <= rmax_filter) & (np.load(os.path.join(DATA_DIR, 'voids_halo', 'cov', f'cov_jk_halo_{s}_touch.npz'))['dn_dlnR'] > 0)]) for s in SAMPLES])
        ndof = tot_bins - 2
        return res.x[0], res.x[1], res.fun, tot_bins, ndof, res.fun / ndof

    s_all, off_all, chi2_all, n_all, nu_all, red_all = joint_fit(40.0)
    print(f"Full Vector (All 26 bins):   B_slope = {s_all:.4f}, B_offset = {off_all:.4f} | chi^2 = {chi2_all:.2f}/{nu_all} | chi^2/nu = {red_all:.2f}")
    s_cut, off_cut, chi2_cut, n_cut, nu_cut, red_cut = joint_fit(32.0)
    print(f"Robust Tail (R <= 32 Mpc/h): B_slope = {s_cut:.4f}, B_offset = {off_cut:.4f} | chi^2 = {chi2_cut:.2f}/{nu_cut} | chi^2/nu = {red_cut:.2f}")

    # =========================================================================
    # TEST 4: CLEANING OVERLAP SENSITIVITY (TOUCH vs CENTER)
    # =========================================================================
    print("\n[TEST 4] Cleaning Overlap Sensitivity: TOUCH (0% overlap) vs CENTER (<=50% overlap)")
    print("-" * 75)
    print(f"{'Sample':<8} | {'TOUCH b_void':<14} | {'TOUCH chi^2/nu':<14} | {'CENTER b_void':<14} | {'CENTER chi^2/nu':<14} | {'Shift':<8}")
    print("-" * 75)
    
    t4_touch_b, t4_center_b = [], []
    t4_beff = []
    for s in SAMPLES:
        beff = B_EFF_MAP[s]
        rmin = 18.0 if s in ['100P', '75P'] else 16.0
        t4_beff.append(beff)
        
        # TOUCH
        cov_t = os.path.join(DATA_DIR, 'voids_halo', 'cov', f'cov_jk_halo_{s}_touch.npz')
        dt = np.load(cov_t)
        mask_t = (dt['bin_centers'] >= rmin) & (dt['dn_dlnR'] > 0.0)
        bt, chi2_t, _, red_t = fit_1d_bvoid(dt['bin_centers'][mask_t], dt['dn_dlnR'][mask_t], dt['cov_full'][np.ix_(mask_t, mask_t)], beff)
        t4_touch_b.append(bt)
        
        # CENTER
        cov_c = os.path.join(DATA_DIR, 'voids_halo', 'cov', f'cov_jk_halo_{s}_center.npz')
        dc = np.load(cov_c)
        mask_c = (dc['bin_centers'] >= rmin) & (dc['dn_dlnR'] > 0.0)
        bc, chi2_c, _, red_c = fit_1d_bvoid(dc['bin_centers'][mask_c], dc['dn_dlnR'][mask_c], dc['cov_full'][np.ix_(mask_c, mask_c)], beff)
        t4_center_b.append(bc)
        
        shift_pct = ((bc - bt) / bt) * 100.0
        print(f"{s:<8} | {bt:<14.4f} | {red_t:<14.2f} | {bc:<14.4f} | {red_c:<14.2f} | {shift_pct:>+6.2f}%")

    poly_touch = np.polyfit(t4_beff, t4_touch_b, 1)
    poly_center = np.polyfit(t4_beff, t4_center_b, 1)
    print(f"Barrier Scaling (TOUCH):  b_void = {poly_touch[0]:.3f} * b_eff + {poly_touch[1]:.3f}")
    print(f"Barrier Scaling (CENTER): b_void = {poly_center[0]:.3f} * b_eff + {poly_center[1]:.3f}")

    # =========================================================================
    # TEST 5: POISSON LIKELIHOOD (CASH / C-STATISTIC)
    # =========================================================================
    print("\n[TEST 5] Poisson Cash Statistic vs Gaussian Chi^2 in Discrete Tail")
    print("-" * 75)
    print(f"{'Sample':<8} | {'p bins':<6} | {'Gaussian chi^2/nu':<18} | {'Poisson C / nu':<18} | {'Improvement':<12}")
    print("-" * 75)
    
    for s in SAMPLES:
        beff = B_EFF_MAP[s]
        rmin = 18.0 if s in ['100P', '75P'] else 16.0
        cov_file = os.path.join(DATA_DIR, 'voids_halo', 'cov', f'cov_jk_halo_{s}_touch.npz')
        d = np.load(cov_file)
        r, y, cov = d['bin_centers'], d['dn_dlnR'], d['cov_full']
        mask = (r >= rmin) & (y > 0.0)
        r_act, y_act = r[mask], y[mask]
        p = len(r_act)
        ndof = p - 1
        
        # Best b_void
        b_val, chi2, _, red_chi2 = fit_1d_bvoid(r_act, y_act, cov[np.ix_(mask, mask)], beff)
        
        # Compute theoretical counts and observed counts
        dlnR = np.log(r[1] / r[0])
        bin_vol_eff = V_BOX * dlnR
        N_obs = y_act * bin_vol_eff
        
        m_act = np.array([sf.size_function(float(ri), 0.0, 'Vdn', float(beff), 1.0, float(b_val - beff), -0.79, del_c, 'EisensteinHu') for ri in r_act])
        mu_th = m_act * bin_vol_eff
        
        # Cash C-statistic: 2 * sum(mu - N + N * ln(N / mu))
        # Where N=0, N * ln(N/mu) = 0
        c_stat = 0.0
        for n_i, mu_i in zip(N_obs, mu_th):
            if n_i > 0:
                c_stat += 2.0 * (mu_i - n_i + n_i * np.log(n_i / mu_i))
            else:
                c_stat += 2.0 * mu_i
        red_c = c_stat / ndof
        diff = red_chi2 - red_c
        print(f"{s:<8} | {p:<6} | {red_chi2:<18.2f} | {red_c:<18.2f} | {diff:>+10.2f}")

    # =========================================================================
    # GENERATE 4-PANEL PUBLICATION SUMMARY FIGURE
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=200)
    
    # Subplot (1,1): Scale-Validity Plateau b_void(R_min)
    ax = axes[0, 0]
    for s in SAMPLES:
        rmins = [res['rmin'] for res in t2_results[s]]
        bvoids = [res['b_void'] for res in t2_results[s]]
        ax.plot(rmins, bvoids, 'o-', color=COLORS[s], lw=2, ms=6, label=f'Sample {s}')
    ax.set_xlabel(r'Scale Cut $R_{\rm min}\ [h^{-1}\mathrm{Mpc}]$', fontsize=12)
    ax.set_ylabel(r'Calibrated $b_{\rm void}$', fontsize=12)
    ax.set_title(r'Audit 1: Scale-Validity Plateau ($b_{\rm void}$ Invariance)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, ls=':')
    ax.legend(fontsize=9, loc='upper right')
    
    # Subplot (1,2): Tail Truncation Chi^2/nu vs R_max
    ax = axes[0, 1]
    rmax_labels = ['R <= 28', 'R <= 32', 'All Bins']
    x_pos = np.arange(len(rmax_labels))
    width = 0.18
    for idx, s in enumerate(SAMPLES):
        chi2_nu_vals = [res['chi2_nu'] for res in reversed(t1_results[s])]
        ax.bar(x_pos + idx * width, chi2_nu_vals, width, label=s, color=COLORS[s], alpha=0.85)
    ax.axhline(1.0, color='black', ls='--', lw=1.2, label='Ideal Fit')
    ax.set_xticks(x_pos + 1.5 * width)
    ax.set_xticklabels(rmax_labels, fontsize=11)
    ax.set_ylabel(r'Reduced $\chi^2/\nu$', fontsize=12)
    ax.set_title(r'Audit 2: Tail Poisson Noise Removal ($R_{\rm max}$)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, ls=':')
    ax.legend(fontsize=9, loc='upper right')
    
    # Subplot (2,1): TOUCH vs CENTER Cleaning Comparison
    ax = axes[1, 0]
    b_plot = np.linspace(0.82, 1.05, 50)
    ax.plot(t4_beff, t4_touch_b, 'o', color='#2980b9', ms=9, label=rf'TOUCH: $B_{{\rm slope}}={poly_touch[0]:.2f}$')
    ax.plot(b_plot, poly_touch[0] * b_plot + poly_touch[1], color='#2980b9', ls='--', lw=2)
    ax.plot(t4_beff, t4_center_b, 's', color='#e67e22', ms=9, label=rf'CENTER: $B_{{\rm slope}}={poly_center[0]:.2f}$')
    ax.plot(b_plot, poly_center[0] * b_plot + poly_center[1], color='#e67e22', ls='-', lw=2)
    ax.set_xlabel(r'Linear Halo Bias $b_{\rm eff}$', fontsize=12)
    ax.set_ylabel(r'Effective Void Barrier $b_{\rm void}$', fontsize=12)
    ax.set_title(r'Audit 3: Cleaning Overlap Sensitivity (TOUCH vs CENTER)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, ls=':')
    ax.legend(fontsize=10, loc='upper left')
    
    # Subplot (2,2): Gaussian vs Poisson deviance comparison
    ax = axes[1, 1]
    samples_pos = np.arange(len(SAMPLES))
    gauss_vals = []
    poisson_vals = []
    for s in SAMPLES:
        beff = B_EFF_MAP[s]
        rmin = 18.0 if s in ['100P', '75P'] else 16.0
        cov_file = os.path.join(DATA_DIR, 'voids_halo', 'cov', f'cov_jk_halo_{s}_touch.npz')
        d = np.load(cov_file)
        r, y, cov = d['bin_centers'], d['dn_dlnR'], d['cov_full']
        mask = (r >= rmin) & (y > 0.0)
        r_act, y_act = r[mask], y[mask]
        p = len(r_act)
        ndof = p - 1
        b_val, chi2, _, red_chi2 = fit_1d_bvoid(r_act, y_act, cov[np.ix_(mask, mask)], beff)
        dlnR = np.log(r[1] / r[0])
        bin_vol_eff = V_BOX * dlnR
        N_obs = y_act * bin_vol_eff
        m_act = np.array([sf.size_function(float(ri), 0.0, 'Vdn', float(beff), 1.0, float(b_val - beff), -0.79, del_c, 'EisensteinHu') for ri in r_act])
        mu_th = m_act * bin_vol_eff
        c_stat = sum([2.0 * (mu_i - n_i + n_i * np.log(n_i / mu_i)) if n_i > 0 else 2.0 * mu_i for n_i, mu_i in zip(N_obs, mu_th)])
        gauss_vals.append(red_chi2)
        poisson_vals.append(c_stat / ndof)
        
    width = 0.35
    ax.bar(samples_pos - width/2, gauss_vals, width, label=r'Gaussian $\chi^2/\nu$ (Overweights $N=1$)', color='#e74c3c', alpha=0.85)
    ax.bar(samples_pos + width/2, poisson_vals, width, label=r'Poisson $C/\nu$ (Exact for Low Counts)', color='#27ae60', alpha=0.85)
    ax.axhline(1.0, color='black', ls='--', lw=1.2, label='Ideal Model')
    ax.set_xticks(samples_pos)
    ax.set_xticklabels(SAMPLES, fontsize=11)
    ax.set_ylabel('Goodness of Fit per Degree of Freedom', fontsize=12)
    ax.set_title(r'Audit 4: Gaussian $\chi^2$ vs Exact Poisson Statistic', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, ls=':')
    ax.legend(fontsize=9, loc='upper right')
    
    plt.tight_layout()
    out_png = os.path.join(PLOT_DIR, 'bitvf_model_robustness_audit.png')
    plt.savefig(out_png, dpi=200)
    plt.savefig(os.path.join(THESIS_FIG_DIR, 'bitvf_model_robustness_audit.png'), dpi=200)
    plt.close()
    print(f"\n[DONE] Saved comprehensive robustness audit figure to {out_png}")

if __name__ == '__main__':
    run_all_audits()
