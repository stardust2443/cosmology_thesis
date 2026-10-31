#!/usr/bin/env python3
"""
plot_multi_tracer_scaling.py

Generates a publication-grade 3-panel figure summarizing multi-tracer void cosmology:
Panel 1: Binned Void Size Functions across halo mass cuts (100P, 75P, 30P)
Panel 2: Physical Barrier Scaling: b_void vs b_eff (Demonstrating linear barrier calibration)
Panel 3: Cosmological constraints: sigma8 vs Omega_m across mass cuts enclosing truth.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
VSF_DIR = os.path.join(HERE, '..', 'data', 'voids_halo', 'vsf')
CHAINS_DIR = os.path.join(HERE, '..', 'data', 'mcmc_chains_halo')
OUT_DIR = os.path.join(HERE, '..', 'plots', 'halos')
os.makedirs(OUT_DIR, exist_ok=True)

# Colors and styling
COLORS = {
    '100P': '#2980b9', # Blue
    '75P':  '#27ae60', # Green
    '50P':  '#e67e22', # Orange
    '30P':  '#8e44ad'  # Purple
}
LABELS = {
    '100P': r'100P ($M \geq 2.6 \times 10^{12} M_\odot,\ b_{\rm eff}=1.021$)',
    '75P':  r'75P ($M \geq 2.0 \times 10^{12} M_\odot,\ b_{\rm eff}=0.958$)',
    '50P':  r'50P ($M \geq 1.3 \times 10^{12} M_\odot,\ b_{\rm eff}=0.911$)',
    '30P':  r'30P ($M \geq 7.8 \times 10^{11} M_\odot,\ b_{\rm eff}=0.862$)'
}

# 1. Load VSF data
vsf_data = {}
for s in ['100P', '75P', '50P', '30P']:
    f = os.path.join(VSF_DIR, f"vsf_halo_{s}_touch.dat")
    if os.path.exists(f):
        # bin_idx, R_center, count, dn_dlnR, sigma_poisson, sigma_jk
        d = np.loadtxt(f)
        vsf_data[s] = {
            'r': d[:, 1],
            'dn': d[:, 3],
            'err': d[:, 5]
        }

# 2. Scaling parameters (b_eff vs b_void)
scaling_data = [
    ('100P', 1.0211, 1.513, 0.048),
    ('75P',  0.9583, 1.478, 0.051),
    ('50P',  0.9108, 1.424, 0.050),
    ('30P',  0.8623, 1.371, 0.048)
]

# 3. Load MCMC chains
chains = {}
chain_files = {
    '100P': os.path.join(CHAINS_DIR, "mcmc_chain_halo_100P_touch_informative_Rmin18.npz"),
    '75P':  os.path.join(CHAINS_DIR, "mcmc_chain_halo_75P_touch_informative_Rmin18.npz"),
    '50P':  os.path.join(CHAINS_DIR, "mcmc_chain_halo_50P_touch_informative_Rmin16.npz"),
    '30P':  os.path.join(CHAINS_DIR, "mcmc_chain_halo_30P_touch_informative_Rmin16.npz")
}
for s, path in chain_files.items():
    if os.path.exists(path):
        chains[s] = np.load(path)['chain']

# Set up 3-panel figure
fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=200)

# ----------------- Panel 1: Multi-Tracer VSF -----------------
ax1 = axes[0]
for s in ['100P', '75P', '50P', '30P']:
    if s in vsf_data:
        mask = vsf_data[s]['dn'] > 0
        r = vsf_data[s]['r'][mask]
        dn = vsf_data[s]['dn'][mask]
        err = vsf_data[s]['err'][mask]
        ax1.errorbar(r, dn, yerr=err, fmt='o-', color=COLORS[s],
                     label=LABELS[s], capsize=3, lw=1.8, ms=5)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel(r'$R\ [h^{-1}\mathrm{Mpc}]$', fontsize=12)
ax1.set_ylabel(r'$\mathrm{d}n/\mathrm{d}\ln R\ [h^3\mathrm{Mpc}^{-3}]$', fontsize=12)
ax1.set_title('Void Size Functions Across Mass Cuts', fontsize=12, fontweight='bold')
ax1.legend(fontsize=8.5, loc='lower left')
ax1.grid(True, alpha=0.3, ls=':')

# ----------------- Panel 2: Barrier Scaling (b_void vs b_eff) -----------------
ax2 = axes[1]
b_effs = np.array([x[1] for x in scaling_data])
b_voids = np.array([x[2] for x in scaling_data])
b_errs = np.array([x[3] for x in scaling_data])

# Linear fit
fit_coeff = np.polyfit(b_effs, b_voids, 1)
fit_line = np.poly1d(fit_coeff)
b_grid = np.linspace(0.80, 1.10, 50)

ax2.plot(b_grid, fit_line(b_grid), color='#e74c3c', lw=2.0, ls='--',
         label=rf'Linear Barrier: $b_{{\rm void}} = {fit_coeff[0]:.2f} b_{{\rm eff}} + {fit_coeff[1]:.2f}$')

for s, beff, bvoid, berr in scaling_data:
    ax2.errorbar(beff, bvoid, yerr=berr, fmt='s', color=COLORS[s],
                 ms=8, capsize=4, elinewidth=1.8, label=f'{s} ($b_{{\\rm eff}}={beff:.3f}$)')

ax2.set_xlabel(r'Linear Halo Bias $b_{\mathrm{eff}}$', fontsize=12)
ax2.set_ylabel(r'Effective Void Bias $b_{\mathrm{void}}$', fontsize=12)
ax2.set_title('Tracer Bias vs. Void Barrier Scaling', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9, loc='upper left')
ax2.grid(True, alpha=0.3, ls=':')

# ----------------- Panel 3: Cosmological Inference -----------------
ax3 = axes[2]
import scipy.stats as stats

def plot_2d_contour(ax, x, y, color, label, alpha=0.25):
    try:
        kernel = stats.gaussian_kde(np.vstack([x, y]))
        xmin, xmax = 0.25, 0.45
        ymin, ymax = 0.60, 1.02
        xx, yy = np.mgrid[xmin:xmax:80j, ymin:ymax:80j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        f = np.reshape(kernel(positions).T, xx.shape)
        
        # 1-sigma and 2-sigma thresholds
        t = np.linspace(0, f.max(), 1000)
        integral = np.array([f[f >= val].sum() for val in t])
        integral /= integral[0]
        l1 = t[np.argmin(np.abs(integral - 0.6827))]
        l2 = t[np.argmin(np.abs(integral - 0.9545))]
        
        ax.contour(xx, yy, f, levels=[l2, l1], colors=[color], linewidths=[1.2, 2.0])
        ax.contourf(xx, yy, f, levels=[l1, f.max()], colors=[color], alpha=alpha)
        ax.plot([], [], color=color, lw=2.0, label=label)
    except:
        pass

for s in ['100P', '75P', '50P', '30P']:
    if s in chains:
        plot_2d_contour(ax3, chains[s][:, 0], chains[s][:, 1], COLORS[s], rf'{s} Posterior ($1\sigma$)')

# Truth
ax3.plot(0.315, 0.811, '*', color='#f39c12', mec='black', ms=14, zorder=10,
         label=r'Simulation Truth ($\Omega_m=0.315, \sigma_8=0.811$)')

ax3.set_xlim(0.27, 0.43)
ax3.set_ylim(0.68, 0.96)
ax3.set_xlabel(r'$\Omega_m$', fontsize=12)
ax3.set_ylabel(r'$\sigma_8$', fontsize=12)
ax3.set_title(r'Cosmological Posteriors ($\Omega_m - \sigma_8$)', fontsize=12, fontweight='bold')
ax3.legend(fontsize=8.5, loc='upper right')
ax3.grid(True, alpha=0.3, ls=':')

plt.tight_layout()
out_file = os.path.join(OUT_DIR, "multi_tracer_cosmology_summary.png")
fig.savefig(out_file, dpi=250, bbox_inches='tight')
print(f"Saved multi-tracer summary figure to: {out_file}")
plt.close(fig)
