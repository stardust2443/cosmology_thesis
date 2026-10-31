#!/usr/bin/env python3
"""
cbl_mcmc_v6_joint_10d.py

Performs joint cosmological fits on 4 binned VSF subsamples (20p, 10p, 5p, 2.5p)
using a 10D parameter space:
- Cosmology: Omega_m, sigma_8
- Nuisance parameters: B_slope, B_offset separately for each subsample
Uses the 64-region Jackknife precision matrix and sets B_EFF = 1.0001 to activate bias parameters.
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import emcee

# Add CosmoBolognaLib to path
CBL_HOME = os.environ.get('CBL_HOME', '/Users/arun/projects/CosmoBolognaLib')
sys.path.insert(0, os.path.join(CBL_HOME, 'Python'))
os.environ.setdefault('DYLD_LIBRARY_PATH', CBL_HOME)
import CosmoBolognaLib as cbl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CHAINS = os.path.join(HERE, '..', 'data', 'mcmc_chains')
OUT_PLOTS  = os.path.join(HERE, '..', 'plots', 'cbl_mcmc')
COV_DIR    = os.path.join(HERE, '..', 'data', 'cov_jk_64')
os.makedirs(OUT_CHAINS, exist_ok=True)
os.makedirs(OUT_PLOTS, exist_ok=True)

# Constants
OMEGA_M_TRUE, SIGMA8_TRUE = 0.315, 0.811
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

NDIM = 10

def make_cosmo(om, s8):
    c = cbl.cosmology.LCDM(om, OMEGA_B, 0.0, H0,
                           SCALAR_AMP, SCALAR_PIVOT, N_S, TAU, True)
    c.set_parameter("sigma8", s8)
    return c

def load_data_and_cov(ov_key, rmin):
    mask_bins = FORARUN_BIN_CENTERS >= rmin
    r_fit = FORARUN_BIN_CENTERS[mask_bins]
    
    d_list = []
    for frac in SUBSAMPLES:
        dat_path = os.path.join(COV_DIR, f"vsf_64jk_nrec50_{frac}_{ov_key}.dat")
        df = pd.read_csv(dat_path, sep='\t')
        df['r'] = df['r'].astype(float)
        df['vsf'] = df['vsf'].astype(float)
        df_sub = df[df['r'] >= rmin]
        d_list.append(df_sub['vsf'].values)
        
    d = np.hstack(d_list)
    inv_cov_path = os.path.join(COV_DIR, f"inv_cov_64jk_R{int(rmin)}_{ov_key}.npy")
    C_inv = np.load(inv_cov_path)
    
    return r_fit, d, C_inv

def log_prior(theta):
    om, s8 = theta[0], theta[1]
    if not (0.10 <= om <= 0.50): return -np.inf
    if not (0.50 <= s8 <= 1.20): return -np.inf
    for idx in range(4):
        bs = theta[2 + 2*idx]
        bo = theta[2 + 2*idx + 1]
        if not (0.3 <= bs <= 2.5): return -np.inf
        if not (-1.0 <= bo <= 1.0): return -np.inf
        if bs * B_EFF + bo <= 0.701: return -np.inf
    return 0.0

def log_likelihood(theta, data_dict):
    om, s8 = theta[0], theta[1]
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
    for idx in range(4):
        bs = theta[2 + 2*idx]
        bo = theta[2 + 2*idx + 1]
        m_sub = np.zeros(len(r_fit))
        for j, rv in enumerate(r_fit):
            try:
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

def run_mcmc(ov_key, rmin, nwalkers=40, nsteps=3000):
    r_fit, d, C_inv = load_data_and_cov(ov_key, rmin)
    data_dict = {'r_fit': r_fit, 'd': d, 'C_inv': C_inv}
    
    p0 = np.array([0.315, 0.811] + [1.0, 0.0] * 4)
    pos = p0 + 1e-2 * np.random.randn(nwalkers, NDIM)
    
    # Clip pos to prior bounds
    pos[:, 0] = np.clip(pos[:, 0], 0.12, 0.48)
    pos[:, 1] = np.clip(pos[:, 1], 0.55, 1.15)
    for idx in range(4):
        pos[:, 2 + 2*idx] = np.clip(pos[:, 2 + 2*idx], 0.4, 2.0)
        pos[:, 2 + 2*idx + 1] = np.clip(pos[:, 2 + 2*idx + 1], -0.2, 0.2)
    
    sampler = emcee.EnsembleSampler(nwalkers, NDIM, log_posterior, args=(data_dict,))
    sampler.run_mcmc(pos, nsteps, progress=True)
    
    chain = sampler.get_chain(discard=500, thin=5, flat=True)
    out_chain = os.path.join(OUT_CHAINS, f"cbl_v6_joint_64jk_full_R{int(rmin)}_{ov_key}.npy")
    np.save(out_chain, chain)
    print(f"[{ov_key}] Rmin={rmin:.1f} Saved 10D chain to {out_chain} | shape: {chain.shape}")
    return chain

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ov', type=str, default='OV05', choices=['OV00', 'OV05'])
    parser.add_argument('--rmin', type=float, default=7.0)
    parser.add_argument('--nwalkers', type=int, default=40)
    parser.add_argument('--nsteps', type=int, default=3000)
    args = parser.parse_args()
    
    run_mcmc(args.ov, args.rmin, args.nwalkers, args.nsteps)
