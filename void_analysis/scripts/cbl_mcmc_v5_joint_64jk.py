#!/usr/bin/env python3
"""
cbl_mcmc_v5_joint_64jk.py

Runs joint MCMC fits across all 4 subsamples using the 64-region
Hartlap-corrected full precision matrix:
    ln L = -0.5 * (d - m)^T C_inv_hartlap (d - m)

Supports scale cuts Rmin in [7.0, 8.0, 9.0] and overlap protocols OV00 (touch) / OV05 (center).
"""
import os, sys, time, argparse
import numpy as np
import pandas as pd

CBL_HOME = os.environ.get('CBL_HOME', '/Users/arun/projects/CosmoBolognaLib')
sys.path.insert(0, os.path.join(CBL_HOME, 'Python'))
os.environ.setdefault('DYLD_LIBRARY_PATH', CBL_HOME)
import CosmoBolognaLib as cbl
import emcee

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CHAINS = os.path.join(HERE, '..', 'data', 'mcmc_chains')
OUT_PLOTS  = os.path.join(HERE, '..', 'plots', 'cbl_mcmc')
COV_DIR    = os.path.join(HERE, '..', 'data', 'cov_jk_64')
os.makedirs(OUT_CHAINS, exist_ok=True)
os.makedirs(OUT_PLOTS, exist_ok=True)

OMEGA_M_TRUE, SIGMA8_TRUE = 0.315, 0.811
H0, OMEGA_B, N_S, TAU = 0.6736, 0.0493, 0.9649, 0.0925
SCALAR_AMP, SCALAR_PIVOT = 2.194e-9, 0.05
DELTA_NL, B_EFF = -0.7, 1.0

SUBSAMPLES = ['20p', '10p', '5p', '2_5p']
FORARUN_BIN_CENTERS = np.array([
    1.591446e+00, 1.791403e+00, 2.016484e+00, 2.269845e+00, 2.555040e+00,
    2.876068e+00, 3.237432e+00, 3.644199e+00, 4.102074e+00, 4.617479e+00,
    5.197642e+00, 5.850699e+00, 6.585810e+00, 7.413284e+00, 8.344726e+00,
    9.393199e+00, 1.057341e+01, 1.190190e+01, 1.339732e+01, 1.508062e+01,
])

PRIORS = {
    'omega_m':  (0.10, 0.50),
    'sigma8':   (0.50, 1.20),
    'b_slope':  (-2.0, 3.0),
    'b_offset': (-2.0, 3.0),
}
NDIM = 4

def make_cosmo(om, s8):
    c = cbl.cosmology.LCDM(om, OMEGA_B, 0.0, H0,
                           SCALAR_AMP, SCALAR_PIVOT, N_S, TAU, True)
    c.set_parameter("sigma8", s8)
    return c

def load_data_and_cov(ov_key, rmin):
    """
    Loads joint data vector d and Hartlap-corrected precision matrix C_inv.
    """
    mask_bins = FORARUN_BIN_CENTERS >= rmin
    r_fit = FORARUN_BIN_CENTERS[mask_bins]
    
    d_list = []
    for frac in SUBSAMPLES:
        dat_path = os.path.join(COV_DIR, f"vsf_64jk_nrec50_{frac}_{ov_key}.dat")
        df = pd.read_csv(dat_path, sep='\t')
        df['r'] = df['r'].astype(float)
        df['vsf'] = df['vsf'].astype(float)
        # Match bins >= rmin
        df_sub = df[df['r'] >= rmin]
        d_list.append(df_sub['vsf'].values)
        
    d = np.hstack(d_list)
    inv_cov_path = os.path.join(COV_DIR, f"inv_cov_64jk_R{int(rmin)}_{ov_key}.npy")
    C_inv = np.load(inv_cov_path)
    
    return r_fit, d, C_inv

def log_prior(theta):
    om, s8, bs, bo = theta
    if not (PRIORS['omega_m'][0]  <= om <= PRIORS['omega_m'][1]):  return -np.inf
    if not (PRIORS['sigma8'][0]   <= s8 <= PRIORS['sigma8'][1]):   return -np.inf
    if not (PRIORS['b_slope'][0]  <= bs <= PRIORS['b_slope'][1]):  return -np.inf
    if not (PRIORS['b_offset'][0] <= bo <= PRIORS['b_offset'][1]): return -np.inf
    return 0.0

def log_likelihood(theta, data_dict):
    om, s8, bs, bo = theta
    r_fit = data_dict['r_fit']
    d     = data_dict['d']
    C_inv = data_dict['C_inv']
    
    try:
        c = make_cosmo(om, s8)
        sf = cbl.cosmology.SizeFunction(c)
        dc = c.deltac(0.0)
    except Exception:
        return -np.inf

    m_list = []
    for _ in SUBSAMPLES:
        m_sub = np.zeros(len(r_fit))
        for j, rv in enumerate(r_fit):
            try:
                # size_function signature: (R, z, model, beff, slope, offset, delta_nl, deltac, transfer)
                val = sf.size_function(float(rv), 0.0, "Vdn",
                                       B_EFF, bs, bo, DELTA_NL, dc, "EisensteinHu")
                if not np.isfinite(val) or val <= 0: return -np.inf
                m_sub[j] = val
            except Exception:
                return -np.inf
        m_list.append(m_sub)
        
    m = np.hstack(m_list)
    diff = d - m
    chi2 = float(np.dot(diff.T, np.dot(C_inv, diff)))
    return -0.5 * chi2

def log_posterior(theta, data_dict):
    lp = log_prior(theta)
    if not np.isfinite(lp): return -np.inf
    ll = log_likelihood(theta, data_dict)
    return lp + ll

def run_mcmc(ov_key, rmin, nwalkers=32, nsteps=1500):
    r_fit, d, C_inv = load_data_and_cov(ov_key, rmin)
    data_dict = {'r_fit': r_fit, 'd': d, 'C_inv': C_inv}
    
    p0 = np.array([0.315, 0.811, 0.5, 0.5])
    pos = p0 + 1e-2 * np.random.randn(nwalkers, NDIM)
    
    # Clip pos to prior bounds
    pos[:, 0] = np.clip(pos[:, 0], 0.12, 0.48)
    pos[:, 1] = np.clip(pos[:, 1], 0.55, 1.15)
    pos[:, 2] = np.clip(pos[:, 2], -1.5, 2.5)
    pos[:, 3] = np.clip(pos[:, 3], -1.5, 2.5)
    
    sampler = emcee.EnsembleSampler(nwalkers, NDIM, log_posterior, args=(data_dict,))
    sampler.run_mcmc(pos, nsteps, progress=True)
    
    chain = sampler.get_chain(discard=300, thin=5, flat=True)
    out_chain = os.path.join(OUT_CHAINS, f"cbl_v5_joint_64jk_full_R{int(rmin)}_{ov_key}.npy")
    np.save(out_chain, chain)
    print(f"[{ov_key}] Rmin={rmin:.1f} Saved chain to {out_chain} | shape: {chain.shape}")
    return chain

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ov', type=str, default='OV05', choices=['OV00', 'OV05'])
    parser.add_argument('--rmin', type=float, default=7.0)
    parser.add_argument('--nwalkers', type=int, default=32)
    parser.add_argument('--nsteps', type=int, default=1500)
    args = parser.parse_args()
    
    run_mcmc(args.ov, args.rmin, args.nwalkers, args.nsteps)
