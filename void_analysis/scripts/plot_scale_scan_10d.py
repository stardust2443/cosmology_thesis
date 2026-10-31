import os, sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

CHAINS_DIR = '/Users/arun/projects/my_cosmo_analysis/void_analysis/data/mcmc_chains'
THESIS_FIG = '/Users/arun/projects/my_cosmo_analysis/thesis_docs/figures'
os.makedirs(THESIS_FIG, exist_ok=True)

C_GOLD  = '#B58900'  # True cosmology star

# Perceptually uniform color palette for scale cuts
COLORS = {
    'R6': '#1A237E',  # Indigo
    'R7': '#0D47A1',  # Deep Blue
    'R8': '#1B5E20',  # Emerald Green
    'R9': '#9A031E',  # Crimson
    'R10': '#E65100', # Orange
}

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'font.family': 'sans-serif'
})

def contour_2d(ax, x, y, color, label, levels=[0.6827, 0.9545], linewidth=2.0, alpha_fill=0.10):
    try:
        kernel = stats.gaussian_kde(np.vstack([x, y]))
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
        dx = (xmax - xmin) * 0.1
        dy = (ymax - ymin) * 0.1
        xx, yy = np.mgrid[xmin-dx:xmax+dx:100j, ymin-dy:ymax+dy:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        f = np.reshape(kernel(positions).T, xx.shape)

        t = np.linspace(0, f.max(), 1000)
        integral = np.array([f[f >= val].sum() for val in t])
        integral /= integral[0]

        clevels = []
        for lvl in levels:
            target_val = t[np.argmin(np.abs(integral - lvl))]
            clevels.append(target_val)
        clevels.sort()

        if len(clevels) > 0 and clevels[-1] > 0:
            ax.contour(xx, yy, f, levels=clevels, colors=[color], linewidths=linewidth)
            ax.contourf(xx, yy, f, levels=[clevels[0], f.max()], colors=[color], alpha=alpha_fill)
            ax.plot([], [], color=color, lw=linewidth, label=label)
    except Exception as e:
        print(f"Error for {label}: {e}")

def plot_scale_scan():
    fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=True)
    
    rmins = [6.0, 7.0, 8.0, 9.0, 10.0]
    
    for idx, rmin in enumerate(rmins):
        ax = axes[idx]
        fname = f"cbl_v6_joint_64jk_full_R{int(rmin)}_OV05.npy"
        fpath = os.path.join(CHAINS_DIR, fname)
        
        if os.path.exists(fpath):
            chain = np.load(fpath)
            if len(chain.shape) == 3:
                chain = chain.reshape(-1, chain.shape[-1])
            
            contour_2d(ax, chain[:, 0], chain[:, 1], COLORS[f'R{int(rmin)}'], f'$R_{{\\mathrm{{min}}}} \\geq {rmin:.1f}$')
            
        ax.plot(0.315, 0.811, '*', color=C_GOLD, ms=15, mec='k', mew=1, zorder=10, label='Truth')
        ax.set_title(f'$R_{{\\mathrm{{min}}}} \\geq {rmin:.1f}\\ h^{{-1}}\\mathrm{{Mpc}}$', fontsize=13, fontweight='bold')
        ax.set_xlabel(r'$\Omega_m$')
        if idx == 0:
            ax.set_ylabel(r'$\sigma_8$')
        ax.set_xlim(0.08, 0.48)
        ax.set_ylim(0.52, 1.18)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='upper right')
        
    fig.suptitle('10D Joint MCMC Posteriors: Scale-Cut scan ($R_{\\mathrm{min}}$ from 6.0 to 10.0)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(THESIS_FIG, 'validation_64jk_scale_cut_scan_full.png'), dpi=300, bbox_inches='tight')
    print("Saved extended scale cut scan to validation_64jk_scale_cut_scan_full.png")
    plt.close()

plot_scale_scan()
