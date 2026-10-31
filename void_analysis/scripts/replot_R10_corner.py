#!/usr/bin/env python3
import os
import numpy as np
import corner
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CHAIN_PATH = os.path.join(HERE, '..', 'data', 'mcmc_chains', 'cbl_v5_joint_R10_NREC50_rmin10_test_flat.npy')
OUT_PATH = os.path.join(HERE, '..', '..', 'thesis_docs', 'figures', 'corner_joint_R10_NREC50_rmin10_test.png')

OMEGA_M_TRUE, SIGMA8_TRUE = 0.315, 0.811

print(f"Loading chain: {CHAIN_PATH}")
flat = np.load(CHAIN_PATH)

print("Plotting corner...")
fig = corner.corner(
    flat,
    labels=[r'$\Omega_m$', r'$\sigma_8$', r'$B_{\rm slope}$', r'$B_{\rm offset}$'],
    truths=[OMEGA_M_TRUE, SIGMA8_TRUE, None, None],
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_fmt='.3f',  # Fixes the \pm 0.00 issue by showing 3 decimal places
    title_kwargs={"fontsize": 13},
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2)),
    color='#1A237E',
    truth_color='#9A031E'
)
fig.suptitle(r'Joint MCMC ($R \geq 10\ h^{-1}\mathrm{Mpc}$ Validity Audit)', fontsize=14, y=1.02)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.savefig(OUT_PATH, dpi=180, bbox_inches='tight')
print(f"Saved: {OUT_PATH}")
