#!/usr/bin/env python3
"""
cbl_mcmc_v8_subsample.py

Performs cosmological fits on a SINGLE VSF subsample (20p, 10p, 5p, or 2.5p)
using a 4D parameter space:
- Cosmology: Omega_m, sigma_8
- Nuisance parameters: B_slope, B_offset
Uses the 64-region Jackknife precision matrix for that specific subsample
and sets B_EFF = 1.0001 to activate bias parameters.
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
COV_DIR    = os.path.join(HERE, '..', 'data', 'cov_jk_64')
os.makedirs(OUT_CHAINS, exist_ok=True)

# Constants
H0, OMEGA_B, N_S, TAU = 0.6736, 0.0493, 0.9649, 0.0925
SCALAR_AMP, SCALAR_PIVOT = 2.194e-9, 0.05
DELTA_NL, B_EFF = -0.7, 1.0001

FORARUN_BIN_CENTERS = np.array([
    1.591446e+00, 1.791403e+00, 2.016484e+00, 2.269845e+00, 2.555040e+00,
    2.876068e+00, 3.237432e+00, 3.644199e+00, 4.102074e+00, 4.617479e+00,
    5.197642e+00, 5.850699e+00, 6.585810e+00, 7.413284e+00, 8.344726e+00,
    9.393199e+00, 1.057341e+01, 1.190190e+01, 1.339732e+01, 1.508062e+01,
])

NDIM = 4

def make_cosmo(om, s8):
    c = cbl.cosmology.LCDM(om, OMEGA_B, 0.0, H0,
                           SCALAR_AMP, SCALAR_PIVOT, N_S, TAU, True)
    c.set_parameter("sigma8", s8)
    return c

def load_data_and_cov(ov_key, frac, rmin):
    mask_bins = FORARUN_BIN_CENTERS >= rmin
    r_fit = FORARUN_BIN_CENTERS[mask_bins]
    
    dat_path = os.path.join(COV_DIR, f"vsf_64jk_nrec50_{frac}_{ov_key}.dat")
    df = pd.read_csv(dat_path, sep='\t')
    df['r'] = df['r'].astype(float)
    df['vsf'] = df['vsf'].astype(float)
    df_sub = df[df['r'] >= rmin]
    d = df_sub['vsf'].values
    
    inv_cov_path = os.path.join(COV_DIR, f"inv_cov_64jk_R{int(rmin)}_{frac}_{ov_key}.npy")
    C_inv = np.load(inv_cov_path)
    
    return r_fit, d, C_inv

def log_prior(theta):
    om, s8, bs, bo = theta[0], theta[1], theta[2], theta[3]
    if not (0.10 <= om <= 0.50): return -np.inf
    if not (0.50 <= s8 <= 1.20): return -np.inf
    if not (0.3 <= bs <= 2.5): return -np.inf
    if not (-1.0 <= bo <= 1.0): return -np.inf
    if bs * B_EFF + bo <= 0.701: return -np.inf
    return 0.0

def log_likelihood(theta, data_dict):
    om, s8, bs, bo = theta[0], theta[1], theta[2], theta[3]
    r_fit = data_dict['r_fit']
    d     = data_dict['d']
    C_inv = data_dict['C_inv']
    
    try:
        c = make_cosmo(om, s8)
        sf = cbl.cosmology.SizeFunction(c)
        dc = c.deltac(0.0)
    except Exception:
        return -np.inf

    m_sub = np.zeros(len(r_fit))
    for j, rv in enumerate(r_fit):
        try:
            val = sf.size_function(float(rv), 0.0, "Vdn",
                                   B_EFF, bs, bo, DELTA_NL, dc, "EisensteinHu")
            m_sub[j] = val
        except Exception:
            return -np.inf
            
    diff = d - m_sub
    chi2 = np.dot(diff.T, np.dot(C_inv, diff))
    return -0.5 * chi2

def log_probability(theta, data_dict):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, data_dict)

def run_mcmc():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frac', type=str, required=True, help="Subsample: 20p, 10p, 5p, 2_5p")
    parser.add_argument('--ov', type=str, default='OV05', help="OV00 or OV05")
    parser.add_argument('--rmin', type=float, default=9.0, help="Scale cut Rmin")
    parser.add_argument('--nsteps', type=int, default=4000)
    parser.add_argument('--nwalkers', type=int, default=40)
    args = parser.parse_args()
    
    print(f"[{args.ov}][{args.frac}] Loading data for Rmin = {args.rmin:.1f}...")
    r_fit, d, C_inv = load_data_and_cov(args.ov, args.frac, args.rmin)
    
    data_dict = {
        'r_fit': r_fit,
        'd': d,
        'C_inv': C_inv
    }
    
    # Starting guess (centered near True cosmology and physical bias)
    p0 = []
    for _ in range(args.nwalkers):
        guess = [
            0.315 + 0.01 * np.random.randn(),
            0.811 + 0.02 * np.random.randn(),
            1.200 + 0.05 * np.random.randn(),
            0.000 + 0.05 * np.random.randn()
        ]
        p0.append(guess)
        
    sampler = emcee.EnsembleSampler(args.nwalkers, NDIM, log_probability, args=(data_dict,))
    
    print(f"[{args.ov}][{args.frac}] Running {args.nsteps} steps of MCMC with {args.nwalkers} walkers...")
    sampler.run_mcmc(p0, args.nsteps, progress=True)
    
    # Save the chain
    chain = sampler.get_chain()
    out_name = f"cbl_v8_subsample_{args.frac}_R{int(args.rmin)}_{args.ov}.npy"
    np.save(os.path.join(OUT_CHAINS, out_name), chain)
    print(f"[{args.ov}][{args.frac}] Saved 4D chain to {out_name} | shape: {chain.shape}")

if __name__ == '__main__':
    run_mcmc()
