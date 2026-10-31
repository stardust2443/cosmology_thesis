#!/usr/bin/env python3
"""
calibrate_and_run_informative_mcmc.py

1. Calibrates (B_slope, B_offset) on each subsample at True Cosmology (Omega_m=0.315, sigma_8=0.811)
   at its optimal Rmin, computing both central values and covariance matrix (uncertainties).
2. Runs 4-parameter MCMC (Omega_m, sigma_8, B_slope, B_offset) with informative Gaussian priors
   derived from the calibration step.
3. Propagates calibration uncertainty into cosmological constraints.
"""
import os, sys
import numpy as np
import pandas as pd
import emcee
from scipy.optimize import minimize

CBL_HOME = os.environ.get('CBL_HOME', '/Users/arun/projects/CosmoBolognaLib')
sys.path.insert(0, os.path.join(CBL_HOME, 'Python'))
os.environ.setdefault('DYLD_LIBRARY_PATH', CBL_HOME)
import CosmoBolognaLib as cbl

HERE = os.path.dirname(os.path.abspath(__file__))
CHAINS_DIR = os.path.join(HERE, '..', 'data', 'mcmc_chains')
COV_DIR    = os.path.join(HERE, '..', 'data', 'cov_jk_64')
os.makedirs(CHAINS_DIR, exist_ok=True)

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

def calibrate_nuisance_with_uncertainty(frac):
    rmin = OPTIMAL_RMIN[frac]
    ov_key = 'OV05'
    
    mask_bins = FORARUN_BIN_CENTERS >= rmin
    r_fit = FORARUN_BIN_CENTERS[mask_bins]
    
    dat_path = os.path.join(COV_DIR, f"vsf_64jk_nrec50_{frac}_{ov_key}.dat")
    df = pd.read_csv(dat_path, sep='\t')
    df_sub = df[df['r'] >= rmin]
    d = df_sub['vsf'].values
    
    inv_cov_path = os.path.join(COV_DIR, f"inv_cov_64jk_R{int(rmin)}_{frac}_{ov_key}.npy")
    C_inv = np.load(inv_cov_path)
    
    # Grid search for central values
    best_chi2 = 1e10
    best_nuis = [1.0, 0.0]
    for bs in np.linspace(0.4, 2.2, 15):
        for bo in np.linspace(-0.8, 0.8, 15):
            if bs * B_EFF + bo <= 0.701: continue
            m_sub = evaluate_theory(0.315, 0.811, bs, bo, r_fit)
            if m_sub is None: continue
            diff = d - m_sub
            chi2 = np.dot(diff.T, np.dot(C_inv, diff))
            if chi2 < best_chi2:
                best_chi2 = chi2
                best_nuis = [bs, bo]
                
    def loss(nuis):
        bs, bo = nuis
        if bs * B_EFF + bo <= 0.701: return 1e10
        m_sub = evaluate_theory(0.315, 0.811, bs, bo, r_fit)
        if m_sub is None: return 1e10
        diff = d - m_sub
        return np.dot(diff.T, np.dot(C_inv, diff))
        
    res = minimize(loss, best_nuis, bounds=[(0.3, 2.5), (-1.0, 1.0)], tol=1e-6)
    mu_calib = res.x
    
    # Sample calibration posterior via MCMC at fixed True cosmology to obtain exact covariance
    def log_calib_prob(nuis):
        bs, bo = nuis
        if not (0.3 <= bs <= 2.5 and -1.0 <= bo <= 1.0): return -np.inf
        if bs * B_EFF + bo <= 0.701: return -np.inf
        m_sub = evaluate_theory(0.315, 0.811, bs, bo, r_fit)
        if m_sub is None: return -np.inf
        diff = d - m_sub
        return -0.5 * np.dot(diff.T, np.dot(C_inv, diff))
        
    nw, nd, ns = 24, 2, 1500
    p0 = [mu_calib + 0.02 * np.random.randn(nd) for _ in range(nw)]
    calib_sampler = emcee.EnsembleSampler(nw, nd, log_calib_prob)
    calib_sampler.run_mcmc(p0, ns, progress=False)
    
    flat_calib = calib_sampler.get_chain()[int(ns*0.3):].reshape(-1, nd)
    mu_chain = np.mean(flat_calib, axis=0)
    cov_calib = np.cov(flat_calib, rowvar=False)
    sig_bs = np.sqrt(cov_calib[0, 0])
    sig_bo = np.sqrt(cov_calib[1, 1])
    rho = cov_calib[0, 1] / (sig_bs * sig_bo)
    
    print(f"[{frac}] Calibration: bs={mu_chain[0]:.4f} +/- {sig_bs:.4f}, bo={mu_chain[1]:.4f} +/- {sig_bo:.4f}, corr={rho:.3f}")
    return r_fit, d, C_inv, mu_chain, cov_calib

def run_informative_prior_mcmc(frac):
    r_fit, d, C_inv, mu_calib, cov_calib = calibrate_nuisance_with_uncertainty(frac)
    inv_cov_prior = np.linalg.inv(cov_calib)
    rmin = OPTIMAL_RMIN[frac]
    ov_key = 'OV05'
    
    def log_prior(theta):
        om, s8, bs, bo = theta
        if not (0.10 <= om <= 0.50): return -np.inf
        if not (0.50 <= s8 <= 1.20): return -np.inf
        if not (0.3 <= bs <= 2.5 and -1.0 <= bo <= 1.0): return -np.inf
        if bs * B_EFF + bo <= 0.701: return -np.inf
        
        # Informative Gaussian prior on (bs, bo)
        delta_b = np.array([bs - mu_calib[0], bo - mu_calib[1]])
        lp_bias = -0.5 * np.dot(delta_b.T, np.dot(inv_cov_prior, delta_b))
        return lp_bias

    def log_likelihood(theta):
        om, s8, bs, bo = theta
        m_sub = evaluate_theory(om, s8, bs, bo, r_fit)
        if m_sub is None: return -np.inf
        diff = d - m_sub
        return -0.5 * np.dot(diff.T, np.dot(C_inv, diff))

    def log_prob(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp): return -np.inf
        return lp + log_likelihood(theta)

    nwalkers, ndim, nsteps = 32, 4, 3000
    p0 = []
    for _ in range(nwalkers):
        guess = [
            0.315 + 0.01 * np.random.randn(),
            0.811 + 0.02 * np.random.randn(),
            mu_calib[0] + 0.01 * np.random.randn(),
            mu_calib[1] + 0.01 * np.random.randn()
        ]
        p0.append(guess)
        
    print(f"[{frac}] Running {nsteps} steps of 4D MCMC with informative calibration priors...")
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(p0, nsteps, progress=False)
    
    chain = sampler.get_chain()
    out_name = f"cbl_v10_subsample_{frac}_informative_R{int(rmin)}_{ov_key}.npy"
    np.save(os.path.join(CHAINS_DIR, out_name), chain)
    
    flat = chain[int(nsteps * 0.3):].reshape(-1, ndim)
    p16, p50, p84 = np.percentile(flat, [16, 50, 84], axis=0)
    print(f"[{frac}] Final Posteriors (Informative Prior):")
    print(f"  Omega_m: {p50[0]:.3f} +{p84[0]-p50[0]:.3f}/-{p50[0]-p16[0]:.3f}")
    print(f"  sigma_8: {p50[1]:.3f} +{p84[1]-p50[1]:.3f}/-{p50[1]-p16[1]:.3f}")
    print(f"  B_slope: {p50[2]:.3f} +{p84[2]-p50[2]:.3f}/-{p50[2]-p16[2]:.3f}")
    print(f"  B_offset: {p50[3]:.3f} +{p84[3]-p50[3]:.3f}/-{p50[3]-p16[3]:.3f}\n")

if __name__ == '__main__':
    for frac in SUBSAMPLES:
        run_informative_prior_mcmc(frac)
