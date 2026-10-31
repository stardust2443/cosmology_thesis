#!/usr/bin/env python3
import os, argparse
import numpy as np
import corner
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--sample', type=str, default='75P')
parser.add_argument('--criterion', type=str, default='touch')
parser.add_argument('--rmin', type=float, default=18.0)
args = parser.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
CHAIN_PATH = os.path.join(HERE, '..', 'data', 'mcmc_chains_halo', f'mcmc_chain_halo_{args.sample}_{args.criterion}_informative_Rmin{int(args.rmin)}.npz')
OUT_PATH = os.path.join(HERE, '..', 'plots', 'halos', f'corner_mcmc_halo_{args.sample}_{args.criterion}_informative_Rmin{int(args.rmin)}.png')

b_eff_map = {'100P': 1.021, '75P': 0.958, '50P': 0.911, '30P': 0.862}
b_void_map = {'100P': 1.513, '75P': 1.478, '50P': 1.425, '30P': 1.373}
b_eff = b_eff_map.get(args.sample, 1.0)
b_void_truth = b_void_map.get(args.sample, 1.50)

data = np.load(CHAIN_PATH)
chain = data['chain']

fig = corner.corner(
    chain,
    labels=[r'$\Omega_m$', r'$\sigma_8$', r'$b_{\rm void}$'],
    truths=[0.315, 0.811, b_void_truth],
    truth_color='#e74c3c',
    quantiles=[0.16, 0.50, 0.84],
    show_titles=True,
    title_fmt='.3f',
    title_kwargs={'fontsize': 13},
    color='#2980b9',
    smooth=1.0
)
fig.subplots_adjust(top=0.88)
fig.suptitle(rf'Aletheia Halo VSF Cosmological Inference ({args.sample} {args.criterion})' + '\n' +
             rf'$R \geq {int(args.rmin)}\ h^{{-1}}\mathrm{{Mpc}}$ | $b_{{\mathrm{{eff}}}} = {b_eff:.3f}$',
             fontsize=14, fontweight='bold')
fig.savefig(OUT_PATH, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"Saved clean corner plot to: {OUT_PATH}")
