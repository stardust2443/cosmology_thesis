#!/usr/bin/env python3
"""
07_run_mcmc_bitvf.py
Runs an MCMC parameter estimation (Omega_m, sigma_8) using the BitVF catalogs.
Since we may not have 50 BitVF mock realizations to build an empirical covariance matrix natively,
this script leverages the REVOLVER covariance matrix as a stand-in, which is a standard
approximation when validating a new geometrical void finder against a trusted baseline.
"""
import os
import sys
import numpy as np
import pandas as pd
import emcee
import corner
import matplotlib.pyplot as plt
from scipy.linalg import inv
import time

# Theoretical model
try:
    from vsf_model import get_theoretical_vsf
except ImportError:
    print("Warning: vsf_model.py not found.")
    sys.exit(1)

def build_covariance_revolver(frac="20p", num_mocks=50, num_bins=20, r_bins=None):
    """
    Builds the covariance matrix from REVOLVER mocks as a proxy for BitVF covariance.
    """
    base_dir = "/Users/arun/projects/my_cosmo_analysis/void_analysis/data/cleaned/"
    vsf_matrix = []
    
    for i in range(num_mocks):
        file_path = os.path.join(base_dir, f"cleaned_revolver_cat_{frac}_R{i}.dat")
        if not os.path.exists(file_path):
            file_path = file_path.replace(".dat", ".csv")
            
        if os.path.exists(file_path):
            df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_csv(file_path, sep=r'\s+', header=None)
            r_col = 'radius' if 'radius' in df.columns else 3
            
            counts, _ = np.histogram(df[r_col].values, bins=r_bins)
            vol = 250**3
            dn_dr = counts / vol / np.diff(r_bins)
            vsf_matrix.append(dn_dr)
            
    if len(vsf_matrix) < 2:
        print("Warning: Could not load enough REVOLVER mocks to build covariance matrix.")
        print("Falling back to a diagonal mock precision matrix.")
        return np.eye(num_bins) * 1e8 # Highly precise dummy matrix
        
    vsf_matrix = np.array(vsf_matrix)
    cov_matrix = np.cov(vsf_matrix, rowvar=False)
    # Add small ridge penalty for numerical stability
    inv_cov = inv(cov_matrix + np.eye(num_bins) * 1e-12) 
    return inv_cov

def load_bitvf_observation(frac="20p", r_bins=None):
    """Loads the target BitVF catalog we are trying to fit."""
    base_dir = "/Users/arun/projects/my_cosmo_analysis/void_analysis/data/cleaned/"
    
    # Try to find the specific BitVF catalog
    import glob
    files = glob.glob(os.path.join(base_dir, f"*bitvf*{frac}*.csv"))
    if not files:
        raise ValueError(f"No cleaned BitVF catalog found for fraction {frac}.")
        
    file_path = files[0]
    print(f"Fitting target BitVF catalog: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path)
    
    counts, _ = np.histogram(df['radius'].values, bins=r_bins)
    vol = 250**3
    obs_vsf = counts / vol / np.diff(r_bins)
    return obs_vsf

def log_prior(theta):
    """Flat priors for LCDM parameters."""
    omega_m, sigma_8 = theta
    if 0.1 < omega_m < 0.6 and 0.5 < sigma_8 < 1.2:
        return 0.0
    return -np.inf

def log_likelihood(theta, r_centers, obs_vsf, inv_cov):
    """Gaussian log-likelihood."""
    omega_m, sigma_8 = theta
    
    # Theoretical prediction
    model_vsf = get_theoretical_vsf(r_centers, omega_m, sigma_8)
    
    # Residuals
    diff = obs_vsf - model_vsf
    
    # chi^2
    chi2 = np.dot(diff.T, np.dot(inv_cov, diff))
    
    return -0.5 * chi2

def log_probability(theta, r_centers, obs_vsf, inv_cov):
    """Combines prior and likelihood."""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, r_centers, obs_vsf, inv_cov)

def main():
    frac = "20p" # Default sparsity to fit
    num_bins = 15
    r_bins = np.linspace(5, 45, num_bins + 1)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    
    print("Step 1: Building Covariance Matrix...")
    inv_cov = build_covariance_revolver(frac=frac, num_bins=num_bins, r_bins=r_bins)
    
    print("\nStep 2: Loading BitVF Target...")
    try:
        obs_vsf = load_bitvf_observation(frac=frac, r_bins=r_bins)
    except ValueError as e:
        print(e)
        return
    
    # Mask out bins with 0 observations to prevent singular likelihoods
    valid_mask = obs_vsf > 0
    if not np.all(valid_mask):
        print(f"Masking out {np.sum(~valid_mask)} empty bins.")
        obs_vsf = obs_vsf[valid_mask]
        r_centers = r_centers[valid_mask]
        # Re-slice covariance matrix to match valid bins
        inv_cov = inv_cov[valid_mask][:, valid_mask]

    print("\nStep 3: Initializing MCMC...")
    nwalkers = 32
    ndim = 2
    nsteps = 100 # Adjust this for full runs
    
    # Initialize near Planck values
    initial_guess = np.array([0.31, 0.81])
    pos = initial_guess + 1e-4 * np.random.randn(nwalkers, ndim)
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(r_centers, obs_vsf, inv_cov))
    
    print("Running MCMC...")
    t0 = time.time()
    # For a quick test we can run fewer steps if we want, but we'll do the full run here
    sampler.run_mcmc(pos, nsteps, progress=True)
    t1 = time.time()
    print(f"MCMC completed in {t1-t0:.2f} seconds.")
    
    print("\nStep 4: Post-Processing & Corner Plots...")
    burn_in = int(0.2 * nsteps)
    flat_samples = sampler.get_chain(discard=burn_in, thin=15, flat=True)
    
    plot_dir = "/Users/arun/projects/my_cosmo_analysis/void_analysis/plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    np.save(os.path.join(plot_dir, "mcmc_chain_bitvf.npy"), flat_samples)
    
    labels = [r"$\Omega_m$", r"$\sigma_8$"]
    fig = corner.corner(
        flat_samples, 
        labels=labels, 
        truths=[0.31, 0.81], 
        quantiles=[0.16, 0.5, 0.84], 
        show_titles=True, 
        title_kwargs={"fontsize": 12},
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2))
    )
    
    plot_path = os.path.join(plot_dir, "mcmc_corner_bitvf.png")
    fig.savefig(plot_path, bbox_inches='tight')
    print(f"Success! Corner plot saved to: {plot_path}")

if __name__ == "__main__":
    main()
