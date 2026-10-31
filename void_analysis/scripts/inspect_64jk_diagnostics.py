#!/usr/bin/env python3
"""
inspect_64jk_diagnostics.py

Computes chi-square and reduced chi-square values at both the True cosmology
and the Best-Fit cosmology for all configurations:
- Poisson (R7)
- 8-Octant Diagonal JK (R7)
- 64-Region Full Covariance (R7, R8, R9)

Generates diagnostic plots showing VSF measurements, True model, and Best-fit model.
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
OUT_DIR    = os.path.join(HERE, '..', 'plots', 'cbl_mcmc')
THESIS_FIG = os.path.join(HERE, '..', '..', 'thesis_docs', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(THESIS_FIG, exist_ok=True)

# Constants
H0, OMEGA_B, N_S, TAU = 0.6736, 0.0493, 0.9649, 0.0925
SCALAR_AMP, SCALAR_PIVOT = 2.194e-9, 0.05
DELTA_NL, B_EFF = -0.7, 1.0
SUBSAMPLES = ['20p', '10p', '5p', '2_5p']
FA_MAP = {'2_5p':'025','5p':'05','10p':'10','20p':'20'}

FORARUN_BIN_CENTERS = np.array([
    1.591446e+00, 1.791403e+00, 2.016484e+00, 2.269845e+00, 2.555040e+00,
    2.876068e+00, 3.237432e+00, 3.644199e+00, 4.102074e+00, 4.617479e+00,
    5.197642e+00, 5.850699e+00, 6.585810e+00, 7.413284e+00, 8.344726e+00,
    9.393199e+00, 1.057341e+01, 1.190190e+01, 1.339732e+01, 1.508062e+01,
])

def make_cosmo(om, s8):
    c = cbl.cosmology.LCDM(om, OMEGA_B, 0.0, H0,
                           SCALAR_AMP, SCALAR_PIVOT, N_S, TAU, True)
    c.set_parameter("sigma8", s8)
    return c

def get_theory(om, s8, bs, bo, r_fit):
    try:
        c = make_cosmo(om, s8)
        sf = cbl.cosmology.SizeFunction(c)
        dc = c.deltac(0.0)
        m_list = []
        for _ in SUBSAMPLES:
            m_sub = np.zeros(len(r_fit))
            for j, rv in enumerate(r_fit):
                val = sf.size_function(float(rv), 0.0, "Vdn",
                                       B_EFF, bs, bo, DELTA_NL, dc, "EisensteinHu")
                m_sub[j] = val
            m_list.append(m_sub)
        return np.hstack(m_list)
    except Exception as e:
        print(f"Theory evaluation failed: {e}")
        return None

def compute_chi2_diagonal(d, m, err):
    return np.sum(((d - m) / err)**2)

def compute_chi2_full(d, m, C_inv):
    diff = d - m
    return np.dot(diff.T, np.dot(C_inv, diff))

def evaluate_case(ov_key, rmin, error_type):
    """
    error_type: 'Poisson', '8JK_Diag', '64JK_Full'
    """
    mask_bins = FORARUN_BIN_CENTERS >= rmin
    r_fit = FORARUN_BIN_CENTERS[mask_bins]
    
    d_list = []
    err_list = []
    
    # Load VSF data and errors
    for frac in SUBSAMPLES:
        if error_type == '64JK_Full':
            dat_path = os.path.join(COV_DIR, f"vsf_64jk_nrec50_{frac}_{ov_key}.dat")
            df = pd.read_csv(dat_path, sep='\t')
        else:
            # Poisson or 8JK reference files
            base = os.path.join(HERE, '..', 'data', 'ForArun')
            if error_type == '8JK_Diag':
                dat_path = os.path.join(HERE, '..', 'data', 'vsf_jk_simple0', 
                                        f"vsf_dm_250cube_ss0p{FA_MAP[frac]}_NREC50_SM0p00_CS0p79_TH-0p80_{ov_key}.dat")
            else:
                dat_path = os.path.join(base, f"vsf_dm_250cube_ss0p{FA_MAP[frac]}_NREC50_SM0p00_CS0p79_TH-0p80_{ov_key}.dat")
            df = pd.read_csv(dat_path, sep=r'\s+', comment='#', names=['r', 'vsf', 'err'])
            
        df_sub = df[df['r'] >= rmin]
        d_list.append(df_sub['vsf'].values)
        err_list.append(df_sub['err'].values)
        
    d = np.hstack(d_list)
    err = np.hstack(err_list)
    
    # Load full covariance if needed
    C_inv = None
    if error_type == '64JK_Full':
        inv_cov_path = os.path.join(COV_DIR, f"inv_cov_64jk_R{int(rmin)}_{ov_key}.npy")
        C_inv = np.load(inv_cov_path)
        
    # Get Best-Fit parameters from chain
    if error_type == '64JK_Full':
        chain_path = os.path.join(CHAINS_DIR, f"cbl_v5_joint_64jk_full_R{int(rmin)}_{ov_key}.npy")
    elif error_type == '8JK_Diag':
        chain_path = os.path.join(CHAINS_DIR, f"cbl_v5_joint_R{int(rmin)}_NREC50_{ov_key}_jk_flat.npy")
    else:
        chain_path = os.path.join(CHAINS_DIR, f"cbl_v5_joint_R{int(rmin)}_NREC50_{ov_key}_flat.npy")
        
    if os.path.exists(chain_path):
        chain = np.load(chain_path)
        if len(chain.shape) == 3:
            chain = chain.reshape(-1, chain.shape[-1])
        # Find MAP (best fit) parameters
        # For simplicity, take the median or average of the last 20% of the chain
        best_theta = np.median(chain[-len(chain)//5:], axis=0)
    else:
        print(f"Chain file not found: {chain_path}, using defaults.")
        best_theta = [0.315, 0.811, 0.5, 0.5]
        
    # True cosmology: (0.315, 0.811)
    # To find the best nuisance parameters at true cosmology, optimize bs, bo
    def loss_true(nuis):
        bs, bo = nuis
        m = get_theory(0.315, 0.811, bs, bo, r_fit)
        if m is None: return 1e10
        if error_type == '64JK_Full':
            return compute_chi2_full(d, m, C_inv)
        else:
            return compute_chi2_diagonal(d, m, err)
            
    res_true = minimize(loss_true, [best_theta[2], best_theta[3]], bounds=[(-2.0, 3.0), (-2.0, 3.0)])
    bs_true, bo_true = res_true.x
    m_true = get_theory(0.315, 0.811, bs_true, bo_true, r_fit)
    
    # Best-Fit model
    om_bf, s8_bf, bs_bf, bo_bf = best_theta
    m_bf = get_theory(om_bf, s8_bf, bs_bf, bo_bf, r_fit)
    
    # Calculate Chi-squares
    if error_type == '64JK_Full':
        chi2_true = compute_chi2_full(d, m_true, C_inv)
        chi2_bf   = compute_chi2_full(d, m_bf, C_inv)
    else:
        chi2_true = compute_chi2_diagonal(d, m_true, err)
        chi2_bf   = compute_chi2_diagonal(d, m_bf, err)
        
    p = len(d)
    nu = p - 4
    
    print(f"\n=======================================================")
    print(f"Case: {ov_key} | Rmin={rmin:.1f} | Error: {error_type}")
    print(f"Data Bins (DoF): {p} ({nu})")
    print(f"True Cosmology (om=0.315, s8=0.811, bs={bs_true:.3f}, bo={bo_true:.3f}):")
    print(f"  Chi2: {chi2_true:.2f} | Red Chi2: {chi2_true/nu:.3f}")
    print(f"Best-Fit Cosmology (om={om_bf:.3f}, s8={s8_bf:.3f}, bs={bs_bf:.3f}, bo={bo_bf:.3f}):")
    print(f"  Chi2: {chi2_bf:.2f} | Red Chi2: {chi2_bf/nu:.3f}")
    
    # Plotting VSF comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    n_bins = len(r_fit)
    for idx, frac in enumerate(SUBSAMPLES):
        ax = axes[idx]
        d_sub = d[idx*n_bins:(idx+1)*n_bins]
        err_sub = err[idx*n_bins:(idx+1)*n_bins]
        m_true_sub = m_true[idx*n_bins:(idx+1)*n_bins]
        m_bf_sub = m_bf[idx*n_bins:(idx+1)*n_bins]
        
        ax.errorbar(r_fit, d_sub, yerr=err_sub, fmt='s', color='#1A237E', label='VSF Data', capsize=4, ms=6)
        ax.plot(r_fit, m_true_sub, '--', color='#B58900', lw=2, label='True Cosmology')
        ax.plot(r_fit, m_bf_sub, '-', color='#9A031E', lw=2, label='Best-Fit Model')
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f"Subsample: {frac}")
        ax.set_xlabel(r'$R\ [h^{-1}\mathrm{Mpc}]$')
        ax.set_ylabel(r'$\mathrm{d}n/\mathrm{d}\ln R$')
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend()
        
    fig.suptitle(f"VSF Fit: {ov_key} | $R_{{min}}={rmin:.1f}$ | {error_type}\n"
                 f"True Red $\chi^2 = {chi2_true/nu:.2f}$ | Best-Fit Red $\chi^2 = {chi2_bf/nu:.2f}$", fontsize=14)
    plt.tight_layout()
    
    out_name = f"vsf_fit_{ov_key}_R{int(rmin)}_{error_type}.png"
    plt.savefig(os.path.join(OUT_DIR, out_name), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(THESIS_FIG, out_name), dpi=300, bbox_inches='tight')
    print(f"Saved diagnostic VSF plot to {out_name}")
    plt.close()
    
    return {
        'case': f"{ov_key}_R{rmin}_{error_type}",
        'p': p,
        'nu': nu,
        'chi2_true': chi2_true,
        'red_chi2_true': chi2_true/nu,
        'chi2_bf': chi2_bf,
        'red_chi2_bf': chi2_bf/nu,
        'om_bf': om_bf,
        's8_bf': s8_bf
    }

def main():
    results = []
    # 1. Poisson R7
    results.append(evaluate_case('OV00', 7.0, 'Poisson'))
    results.append(evaluate_case('OV05', 7.0, 'Poisson'))
    
    # 2. 8JK Diag R7
    results.append(evaluate_case('8JK_Diag', 7.0, '8JK_Diag'))  # wait, evaluate_case needs ov_key = 'OV00' or 'OV05'
    # Actually evaluate_case takes ov_key as argument for file finding, so let's adjust:
    
def run_all():
    cases = [
        ('OV00', 7.0, 'Poisson'),
        ('OV05', 7.0, 'Poisson'),
        ('OV00', 7.0, '8JK_Diag'),
        ('OV05', 7.0, '8JK_Diag'),
        ('OV00', 7.0, '64JK_Full'),
        ('OV05', 7.0, '64JK_Full'),
        ('OV05', 8.0, '64JK_Full'),
        ('OV05', 9.0, '64JK_Full'),
    ]
    results = []
    for ov, rmin, err_type in cases:
        try:
            res = evaluate_case(ov, rmin, err_type)
            results.append(res)
        except Exception as e:
            print(f"Failed to evaluate case {ov} R={rmin} {err_type}: {e}")
            
    # Save summary table
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(COV_DIR, 'chi2_summary_table.csv'), index=False)
    print("\nSummary Table:")
    print(df_res.to_string())

if __name__ == '__main__':
    run_all()
