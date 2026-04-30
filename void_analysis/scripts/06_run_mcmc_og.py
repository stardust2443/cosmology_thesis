import os
import numpy as np
import pandas as pd
import emcee
import corner
import matplotlib.pyplot as plt
from scipy.linalg import inv
import time

# Attempt to import your theoretical model
try:
    from vsf_model import get_theoretical_vsf
except ImportError:
    print("Warning: vsf_model.py not found or missing get_theoretical_vsf.")
    print("Please ensure your theoretical model script is in the same directory.")
    exit(1)

def load_data_and_build_covariance(frac="20p", num_mocks=50, num_bins=20):
    """
    Reads the 50 REVOLVER mocks, computes the mean VSF, 
    and builds the inverse covariance matrix natively.
    """
    base_dir = "../data/cleaned/"
    vsf_matrix = []
    r_bins = np.linspace(10, 50, num_bins + 1)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

    print(f"Loading {num_mocks} REVOLVER realizations to build Covariance Matrix...")
    for i in range(num_mocks):
        file_path = os.path.join(base_dir, f"cleaned_revolver_cat_{frac}_R{i}.dat")
        if not os.path.exists(file_path):
            file_path = file_path.replace(".dat", ".csv") # Fallback
            
        # Assuming column 3 is radius based on standard ZOBOV outputs
        try:
            df = pd.read_csv(file_path, delim_whitespace=True, header=None)
            radii = df.iloc[:, 3].values
            
            # Compute Void Size Function (number density per bin)
            counts, _ = np.histogram(radii, bins=r_bins)
            vol = 250**3  # Box size
            dn_dr = counts / vol / np.diff(r_bins)
            vsf_matrix.append(dn_dr)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
            
    vsf_matrix = np.array(vsf_matrix)
    if len(vsf_matrix) == 0:
        raise ValueError("No mock files were loaded. Check your file paths and naming format.")

    # Calculate mean observation and precision (inverse covariance) matrix
    mean_vsf = np.mean(vsf_matrix, axis=0)
    cov_matrix = np.cov(vsf_matrix, rowvar=False)
    
    # Add small ridge penalty to diagonal to ensure inversion stability if highly correlated
    inv_cov = inv(cov_matrix + np.eye(num_bins) * 1e-10) 
    
    return r_centers, mean_vsf, inv_cov

def log_prior(theta):
    """
    Flat priors for LCDM parameters.
    theta[0] = Omega_m
    theta[1] = sigma_8
    """
    omega_m, sigma_8 = theta
    if 0.1 < omega_m < 0.6 and 0.5 < sigma_8 < 1.2:
        return 0.0
    return -np.inf

def log_likelihood(theta, r_centers, mean_vsf, inv_cov):
    """
    Vectorized Gaussian log-likelihood. O(1) in pure Python loops.
    """
    omega_m, sigma_8 = theta
    
    # Call your theoretical model (Ensure it maps Eulerian R to Lagrangian R_L internally)
    model_vsf = get_theoretical_vsf(r_centers, omega_m, sigma_8)
    
    # Residuals
    diff = mean_vsf - model_vsf
    
    # chi^2 matrix multiplication natively
    chi2 = np.dot(diff.T, np.dot(inv_cov, diff))
    
    return -0.5 * chi2

def log_probability(theta, r_centers, mean_vsf, inv_cov):
    """ Combines prior and likelihood. """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, r_centers, mean_vsf, inv_cov)

def main():
    # 1. Setup Data
    r_centers, mean_vsf, inv_cov = load_data_and_build_covariance(frac="20p")
    
    # 2. Setup MCMC
    nwalkers = 32
    ndim = 2
    nsteps = 1500
    
    # Initialize walkers tightly around Planck LCDM truth
    initial_guess = np.array([0.31, 0.81])
    pos = initial_guess + 1e-4 * np.random.randn(nwalkers, ndim)
    
    # 3. Run emcee
    print("Initializing MCMC Walkers...")
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(r_centers, mean_vsf, inv_cov))
    
    t0 = time.time()
    sampler.run_mcmc(pos, nsteps, progress=True)
    t1 = time.time()
    print(f"MCMC completed in {t1-t0:.2f} seconds.")
    
    # 4. Post-Processing & Corner Plots
    burn_in = int(0.2 * nsteps) # 20% burn-in removal
    flat_samples = sampler.get_chain(discard=burn_in, thin=15, flat=True)
    
    # Save raw chain for safety
    np.save("../data/mcmc_chain.npy", flat_samples)
    
    print("Generating Corner Plot...")
    labels = [r"$\Omega_m$", r"$\sigma_8$"]
    fig = corner.corner(
        flat_samples, 
        labels=labels, 
        truths=[0.31, 0.81], # Overlay true LCDM parameters
        quantiles=[0.16, 0.5, 0.84], 
        show_titles=True, 
        title_kwargs={"fontsize": 12},
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2)) # 1-sigma and 2-sigma contours
    )
    
    os.makedirs("../plots", exist_ok=True)
    plot_path = "../plots/mcmc_corner_vsf.pdf"
    fig.savefig(plot_path, bbox_inches='tight')
    print(f"Success! Corner plot saved to: {plot_path}")

if __name__ == "__main__":
    main()