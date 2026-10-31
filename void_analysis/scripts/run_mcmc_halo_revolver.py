#!/usr/bin/env python3
"""
run_mcmc_halo_revolver.py

MCMC cosmological parameter inference from the REVOLVER halo void size function.
Infers (Omega_m, sigma8, b_void) using emcee and CosmoBolognaLib.
Uses the 68% calibration uncertainty as the Gaussian prior on b_void (advisor recommendation).
"""
import os, sys, time, argparse
import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt

# CBL bootstrap
CBL_HOME = '/Users/arun/projects/CosmoBolognaLib'
sys.path.insert(0, os.path.join(CBL_HOME, 'Python'))
os.environ.setdefault('DYLD_LIBRARY_PATH', CBL_HOME)
import CosmoBolognaLib as cbl

HERE = os.path.dirname(os.path.abspath(__file__))
COV_DIR = os.path.join(HERE, '..', 'data', 'voids_halo_revolver', 'cov')
CHAINS_DIR = os.path.join(HERE, '..', 'data', 'mcmc_chains_revolver')
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

CALIB_MAP = {
    '100P': {'b_void': 1.503, 'sigma': 0.0062, 'b_eff': 1.0211, 'rmin': 18.0},
    '75P':  {'b_void': 1.469, 'sigma': 0.0062, 'b_eff': 0.9583, 'rmin': 18.0},
    '50P':  {'b_void': 1.405, 'sigma': 0.0046, 'b_eff': 0.9108, 'rmin': 16.0},
    '30P':  {'b_void': 1.365, 'sigma': 0.0045, 'b_eff': 0.8623, 'rmin': 16.0}
}

def make_cosmo(om, s8):
    c = cbl.cosmology.LCDM(om, OMEGA_B, 0.0, H0,
                           SCALAR_AMP, SCALAR_PIVOT, N_S, TAU, True)
    c.set_parameter("sigma8", s8)
    return c

def run_mcmc(sample='100P', criterion='touch', rmin=None, b_eff=None,
             nwalkers=24, nsteps=1000, burnin=300, b_void_calib=None, b_void_sigma=None):
    
    cfg = CALIB_MAP.get(sample, CALIB_MAP['100P'])
    if rmin is None:
        rmin = cfg['rmin']
    if b_eff is None:
        b_eff = cfg['b_eff']
    if b_void_calib is None:
        b_void_calib = cfg['b_void']
    if b_void_sigma is None:
        b_void_sigma = cfg['sigma']
        
    cov_file = os.path.join(COV_DIR, f"cov_jk_halo_{sample}_{criterion}.npz")
    if not os.path.exists(cov_file):
        raise FileNotFoundError(f"Covariance file not found: {cov_file}")
        
    data = np.load(cov_file)
    bin_centers = data['bin_centers']
    dn_dlnR = data['dn_dlnR']
    cov_full = data['cov_full']
    sigma_jk = data['sigma_jk']
    
    # Scale cut
    mask = (bin_centers >= rmin) & (dn_dlnR > 0.0)
    r_act = bin_centers[mask]
    y_act = dn_dlnR[mask]
    s_act = sigma_jk[mask]
    cov_act = cov_full[np.ix_(mask, mask)]
    p = len(r_act)
    
    hartlap = (64.0 - p - 2.0) / 63.0
    inv_cov = hartlap * np.linalg.pinv(cov_act)
    
    print("=" * 65)
    print(f" MCMC Inference for REVOLVER {sample} ({criterion})")
    print(f" Scale cut: R >= {rmin:.1f} h^-1 Mpc ({p} bins)")
    print(f" Linear Halo Bias: b_eff = {b_eff:.4f}")
    print(f" Calibrated Prior on b_void: N({b_void_calib:.4f}, {b_void_sigma:.4f}^2) [68% uncertainty]")
    print(f" Walkers: {nwalkers}, Steps: {nsteps}, Burn-in: {burnin}")
    print("=" * 65)
    
    def log_prior(theta):
        om, s8, b_void = theta
        if not (0.10 <= om <= 0.55):
            return -np.inf
        if not (0.50 <= s8 <= 1.20):
            return -np.inf
        if not (1.00 <= b_void <= 2.20):
            return -np.inf
        return -0.5 * ((b_void - b_void_calib) / b_void_sigma) ** 2
        
    def log_likelihood(theta):
        om, s8, b_void = theta
        try:
            c = make_cosmo(om, s8)
            sf = cbl.cosmology.SizeFunction(c)
            dc = c.deltac(0.0)
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
        
    # Walker initialization
    init_pos = []
    for _ in range(nwalkers):
        init_pos.append([
            np.random.normal(OMEGA_M_TRUE, 0.015),
            np.random.normal(SIGMA8_TRUE, 0.025),
            np.random.normal(b_void_calib, b_void_sigma)
        ])
    init_pos = np.array(init_pos)
    
    print("\nStarting MCMC sampling...")
    t0 = time.time()
    sampler = emcee.EnsembleSampler(nwalkers, 3, log_posterior)
    sampler.run_mcmc(init_pos, nsteps, progress=False)
    dt = time.time() - t0
    print(f"MCMC complete in {dt:.1f} s ({dt/nsteps*1000:.1f} ms/step)\n")
    
    chain = sampler.get_chain(discard=burnin, flat=True)
    log_prob = sampler.get_log_prob(discard=burnin, flat=True)
    
    map_idx = np.argmax(log_prob)
    map_theta = chain[map_idx]
    pcts = np.percentile(chain, [16, 50, 84], axis=0)
    param_names = ['Omega_m', 'sigma8', 'b_void']
    
    print("------------------- MCMC Constraints -------------------")
    for idx, name in enumerate(param_names):
        med = pcts[1, idx]
        err_low = med - pcts[0, idx]
        err_high = pcts[2, idx] - med
        print(f"{name:<10}: {med:.4f} +{err_high:.4f} -{err_low:.4f} (MAP: {map_theta[idx]:.4f})")
    print(f"True Omega_m: {OMEGA_M_TRUE:.4f}, True sigma8: {SIGMA8_TRUE:.4f}")
    print("--------------------------------------------------------\n")
    
    chain_file = os.path.join(CHAINS_DIR, f"mcmc_chain_revolver_{sample}_{criterion}_Rmin{int(rmin)}.npz")
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
    
    fig = corner.corner(
        chain,
        labels=[r'$\Omega_m$', r'$\sigma_8$', r'$b_{\rm void}$'],
        truths=[OMEGA_M_TRUE, SIGMA8_TRUE, b_void_calib],
        truth_color='#e74c3c',
        quantiles=[0.16, 0.50, 0.84],
        show_titles=True,
        title_fmt='.3f',
        title_kwargs={'fontsize': 12},
        color='#e67e22',
        smooth=1.0
    )
    fig.subplots_adjust(top=0.88)
    fig.suptitle(f'REVOLVER Halo VSF Cosmological Inference ({sample} {criterion})\n'
                 rf'$R \geq {rmin:.0f}\ h^{{-1}}\mathrm{{Mpc}}$ | Calibrated Prior $\sigma(b_{{\rm void}})={b_void_sigma:.4f}$',
                 fontsize=13, fontweight='bold')
                 
    out_corner = os.path.join(PLOT_DIR, f"corner_mcmc_revolver_{sample}_{criterion}_Rmin{int(rmin)}.png")
    fig.savefig(out_corner, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved corner plot to {out_corner}\n")
    
    return {
        'om_med': pcts[1, 0], 'om_err_p': pcts[2, 0] - pcts[1, 0], 'om_err_m': pcts[1, 0] - pcts[0, 0],
        's8_med': pcts[1, 1], 's8_err_p': pcts[2, 1] - pcts[1, 1], 's8_err_m': pcts[1, 1] - pcts[0, 1],
        'bv_med': pcts[1, 2], 'bv_err_p': pcts[2, 2] - pcts[1, 2], 'bv_err_m': pcts[1, 2] - pcts[0, 2]
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=str, default='100P')
    parser.add_argument('--criterion', type=str, default='touch', choices=['center', 'touch'])
    parser.add_argument('--rmin', type=float, default=None)
    parser.add_argument('--walkers', type=int, default=24)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--burnin', type=int, default=300)
    args = parser.parse_args()
    
    run_mcmc(args.sample, args.criterion, args.rmin,
             nwalkers=args.walkers, nsteps=args.steps, burnin=args.burnin)
