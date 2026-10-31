#!/usr/bin/env python3
"""
run_mcmc_halo.py

MCMC cosmological parameter inference from the Aletheia halo void size function.
Infers (Omega_m, sigma8, b_void) using emcee and CosmoBolognaLib.

- Tracer: Aletheia halo catalogues (e.g. 100P, b_eff = 1.0211)
- Cleaning: TOUCH overlap removal (isolated spheres)
- Cutoff: R >= 18.0 h^-1 Mpc (completeness scale ~ 2.3 * MPS)
- Covariance: 64-region spatial Jackknife with Hartlap correction
- Theory: Vdn model via CosmoBolognaLib C++ bindings
"""
import os, sys, time, argparse
import numpy as np
import pandas as pd
import emcee
import corner
import matplotlib.pyplot as plt

# CBL bootstrap
CBL_HOME = '/Users/arun/projects/CosmoBolognaLib'
sys.path.insert(0, os.path.join(CBL_HOME, 'Python'))
os.environ.setdefault('DYLD_LIBRARY_PATH', CBL_HOME)
import CosmoBolognaLib as cbl

HERE = os.path.dirname(os.path.abspath(__file__))
COV_DIR = os.path.join(HERE, '..', 'data', 'voids_halo', 'cov')
CHAINS_DIR = os.path.join(HERE, '..', 'data', 'mcmc_chains_halo')
PLOT_DIR = os.path.join(HERE, '..', 'plots', 'halos')
os.makedirs(CHAINS_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# True Aletheia Cosmology
OMEGA_M_TRUE = 0.315
SIGMA8_TRUE = 0.811
OMEGA_B = 0.0493
H0 = 0.6736
N_S = 0.9649
SCALAR_AMP = 2.194e-9
SCALAR_PIVOT = 0.05
TAU = 0.0925
DELTA_V_NL = -0.79

def make_cosmo(om, s8):
    c = cbl.cosmology.LCDM(om, OMEGA_B, 0.0, H0,
                           SCALAR_AMP, SCALAR_PIVOT, N_S, TAU, True)
    c.set_parameter("sigma8", s8)
    return c

def run_mcmc(sample='100P', criterion='touch', rmin=18.0, b_eff=1.0211,
             nwalkers=24, nsteps=1000, burnin=300, prior_type='informative', b_void_calib=None, b_void_sigma=None):
    
    cov_file = os.path.join(COV_DIR, f"cov_jk_halo_{sample}_{criterion}.npz")
    if not os.path.exists(cov_file):
        raise FileNotFoundError(f"Covariance file not found: {cov_file}")
        
    data = np.load(cov_file)
    bin_centers = data['bin_centers']
    dn_dlnR = data['dn_dlnR']
    cov_full = data['cov_full']
    sigma_jk = data['sigma_jk']
    
    # Apply scale cut
    mask = (bin_centers >= rmin) & (dn_dlnR > 0.0)
    r_act = bin_centers[mask]
    y_act = dn_dlnR[mask]
    s_act = sigma_jk[mask]
    cov_act = cov_full[np.ix_(mask, mask)]
    p = len(r_act)
    
    hartlap = (64.0 - p - 2.0) / 63.0
    inv_cov = hartlap * np.linalg.pinv(cov_act)
    
    print("=" * 65)
    print(f" MCMC Cosmological Inference for Aletheia {sample} ({criterion})")
    print(f" Scale cut: R >= {rmin:.1f} h^-1 Mpc ({p} bins)")
    print(f" Physical Halo Bias: b_eff = {b_eff:.4f}")
    print(f" Prior Type: {prior_type}")
    print(f" Walkers: {nwalkers}, Steps: {nsteps}, Burn-in: {burnin}")
    print("=" * 65)
    
    CALIB_MAP = {
        '100P': {'b_void': 1.513, 'sigma': 0.0053},
        '75P':  {'b_void': 1.478, 'sigma': 0.0051},
        '50P':  {'b_void': 1.424, 'sigma': 0.0050},
        '30P':  {'b_void': 1.371, 'sigma': 0.0048}
    }
    cfg = CALIB_MAP.get(sample, {'b_void': 1.50, 'sigma': 0.0050})
    if b_void_calib is None:
        b_void_calib = cfg['b_void']
    if b_void_sigma is None:
        b_void_sigma = cfg['sigma']
    print(f" Prior on b_void: N({b_void_calib:.4f}, {b_void_sigma:.4f}^2) [68% calibration uncertainty]")
    
    # Log-prior
    def log_prior(theta):
        om, s8, b_void = theta
        if not (0.10 <= om <= 0.55):
            return -np.inf
        if not (0.50 <= s8 <= 1.20):
            return -np.inf
        if not (1.00 <= b_void <= 2.20):
            return -np.inf
            
        if prior_type == 'informative':
            # Gaussian prior on b_void calibrated from simulation
            return -0.5 * ((b_void - b_void_calib) / b_void_sigma) ** 2
        else:
            return 0.0
            
    # Log-likelihood
    def log_likelihood(theta):
        om, s8, b_void = theta
        try:
            c = make_cosmo(om, s8)
            sf = cbl.cosmology.SizeFunction(c)
            dc = c.deltac(0.0)
            # Use slope=1.0, offset = b_void - b_eff
            slope = 1.0
            offset = b_void - b_eff
            
            m = np.zeros(p)
            for i, r in enumerate(r_act):
                val = sf.size_function(float(r), 0.0, "Vdn", float(b_eff),
                                       float(slope), float(offset),
                                       float(DELTA_V_NL), dc, "EisensteinHu")
                if val <= 0.0 or np.isnan(val):
                    return -np.inf
                m[i] = val
                
            diff = y_act - m
            chi2 = float(np.dot(diff, np.dot(inv_cov, diff)))
            return -0.5 * chi2
        except Exception:
            return -np.inf
            
    def log_posterior(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = log_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll
        
    # Initial walker positions tightly around truth + calibration
    init_pos = []
    for _ in range(nwalkers):
        init_pos.append([
            np.random.normal(OMEGA_M_TRUE, 0.015),
            np.random.normal(SIGMA8_TRUE, 0.025),
            np.random.normal(b_void_calib, 0.02)
        ])
    init_pos = np.array(init_pos)
    
    print("\nStarting MCMC sampling...")
    t0 = time.time()
    sampler = emcee.EnsembleSampler(nwalkers, 3, log_posterior)
    sampler.run_mcmc(init_pos, nsteps, progress=True)
    dt = time.time() - t0
    print(f"MCMC complete in {dt:.1f} s ({dt/nsteps*1000:.1f} ms/step)\n")
    
    # Flatten chain after burnin
    chain = sampler.get_chain(discard=burnin, flat=True)
    log_prob = sampler.get_log_prob(discard=burnin, flat=True)
    
    # MAP estimate
    map_idx = np.argmax(log_prob)
    map_theta = chain[map_idx]
    
    # Percentiles
    pcts = np.percentile(chain, [16, 50, 84], axis=0)
    labels = [r'\Omega_m', r'\sigma_8', r'b_{\mathrm{void}}']
    param_names = ['Omega_m', 'sigma8', 'b_void']
    
    print("------------------- MCMC Constraints -------------------")
    for idx, name in enumerate(param_names):
        med = pcts[1, idx]
        err_low = med - pcts[0, idx]
        err_high = pcts[2, idx] - med
        print(f"{name:<10}: {med:.4f} +{err_high:.4f} -{err_low:.4f} (MAP: {map_theta[idx]:.4f})")
    print(f"True Omega_m: {OMEGA_M_TRUE:.4f}, True sigma8: {SIGMA8_TRUE:.4f}")
    print("--------------------------------------------------------\n")
    
    # Save chain
    chain_file = os.path.join(CHAINS_DIR, f"mcmc_chain_halo_{sample}_{criterion}_{prior_type}_Rmin{int(rmin)}.npz")
    np.savez(
        chain_file,
        chain=chain,
        log_prob=log_prob,
        map_theta=map_theta,
        percentiles=pcts,
        r_act=r_act,
        y_act=y_act,
        s_act=s_act
    )
    print(f"Saved chain to {chain_file}")
    
    # Corner plot
    fig = corner.corner(
        chain,
        labels=[r'$\Omega_m$', r'$\sigma_8$', r'$b_{\rm void}$'],
        truths=[OMEGA_M_TRUE, SIGMA8_TRUE, b_void_calib],
        truth_color='#e74c3c',
        quantiles=[0.16, 0.50, 0.84],
        show_titles=True,
        title_fmt='.3f',
        title_kwargs={'fontsize': 12},
        color='#2980b9',
        smooth=1.0
    )
    fig.subplots_adjust(top=0.88)
    fig.suptitle(f'Aletheia Halo VSF Cosmological Inference ({sample} {criterion})\n'
                 rf'$R \geq {rmin:.0f}\ h^{{-1}}\mathrm{{Mpc}}$ | $b_{{\rm eff}}={b_eff:.3f}$',
                 fontsize=13, fontweight='bold')
                 
    out_corner = os.path.join(PLOT_DIR, f"corner_mcmc_halo_{sample}_{criterion}_{prior_type}_Rmin{int(rmin)}.png")
    fig.savefig(out_corner, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved corner plot to {out_corner}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=str, default='100P')
    parser.add_argument('--criterion', type=str, default='touch', choices=['center', 'touch'])
    parser.add_argument('--rmin', type=float, default=18.0)
    parser.add_argument('--b-eff', type=float, default=1.0211)
    parser.add_argument('--b-void-calib', type=float, default=None)
    parser.add_argument('--b-void-sigma', type=float, default=None)
    parser.add_argument('--walkers', type=int, default=24)
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--burnin', type=int, default=250)
    parser.add_argument('--prior', type=str, default='informative', choices=['informative', 'flat'])
    args = parser.parse_args()
    
    run_mcmc(args.sample, args.criterion, args.rmin, args.b_eff,
             args.walkers, args.steps, args.burnin, args.prior, args.b_void_calib, args.b_void_sigma)
