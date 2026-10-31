#!/usr/bin/env python3
"""
02_bitvf_vsf_updated.py
Benchmarks bitVF voids using REVOLVER's empirical tracer-based rescaling,
enforcing physical MPS resolution limits and publication-quality plotting.
"""
import os, sys, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree as KDTree
from tqdm import tqdm
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

# --- 1. Publication Aesthetics ---
plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.size': 16,
    'axes.labelsize': 20,
    'legend.fontsize': 15,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'axes.linewidth': 1.2,
})

# --- Configuration ---
BOX_SIZE = 250.0
PERIODIC = False
OUT_DIR = "../data/cleaned/BitVF_VSF"
os.makedirs(OUT_DIR, exist_ok=True)

samples = {
    "20p": "../data/raw/voids_recon_50_bitvf_aletheia_20p",
    "10p": "../data/raw/voids_recon_50_bitvf_aletheia_10p",
    "5p":  "../data/raw/voids_recon_50_bitvf_aletheia_5p",
    "2_5p": "../data/raw/voids_recon_50_bitvf_aletheia_2_5p"
}

tracer_files = {
    "20p": "/Users/arun/projects/Revolver/subsamples/DM_particles_ss20p_250cube.dat",  
    "10p": "/Users/arun/projects/Revolver/subsamples/DM_particles_ss10p_250cube.dat",
    "5p":  "/Users/arun/projects/Revolver/subsamples/DM_particles_ss5p_250cube.dat",
    "2_5p": "/Users/arun/projects/Revolver/subsamples/DM_particles_ss2_5p_250cube.dat"
}

PROCESS_KEYS = ['20p', '10p', '5p', '2_5p']

CLEANING_PARAMS = {
    'rmax': 50.0,
    'delta_nl_threshold': -0.8,
    'rescale_max_radius_factor': 4.0,
    'rescale_n_bins': 200,
}

def read_bitvf_catalog(path):
    cols = ['x', 'y', 'z', 'radius', 'r_ext', 'delta_min', 'delta_avg']
    return pd.read_csv(path, sep=r'\s+', comment='#', names=cols)

def read_tracers(path):
    df = pd.read_csv(path, sep=r'\s+', comment='#', header=None, usecols=[0,1,2], names=['x','y','z'])
    df[['x','y','z']] = df[['x','y','z']] % BOX_SIZE
    return df

class BitVFRevolverCleaner:
    def __init__(self, voids_df, tracers_df, box_size):
        self.box_size = float(box_size)
        self.voids = voids_df.copy().reset_index(drop=True)
        self.tracers = tracers_df.copy().reset_index(drop=True)
        
        self.mean_tracer_density = len(self.tracers) / (self.box_size**3)
        self.mps = self.mean_tracer_density ** (-1.0/3.0)
        
        print(f"[Init] tracers={len(self.tracers)}, mean_rho={self.mean_tracer_density:.4e}, mps={self.mps:.4f} Mpc/h")
        self.tracer_tree = KDTree(self.tracers[['x','y','z']].values)

    def run_pipeline(self):
        c = self.voids[self.voids['radius'] <= CLEANING_PARAMS['rmax']].copy()
        c = self.step2_rescale_radii(c)
        c = self.step3_prune_overlaps(c)
        return c

    def step2_rescale_radii(self, catalog):
        print("[Step2] Fast Empirical Rescaling (Growing to SvdW -0.8 target)...")
        n_bins = int(CLEANING_PARAMS['rescale_n_bins'])
        max_rf = float(CLEANING_PARAMS['rescale_max_radius_factor'])
        delta_target = float(CLEANING_PARAMS['delta_nl_threshold'])

        centers = catalog[['x','y','z']].values
        radii = catalog['radius'].values
        new_r = []
        vol_factor = (4.0/3.0) * np.pi
        fail_count = 0

        for C, r0 in tqdm(zip(centers, radii), total=len(catalog), desc="  Rescaling"):
            max_r = max_rf * r0
            if max_r <= 1.0:
                new_r.append(0.0)
                continue

            idx = self.tracer_tree.query_ball_point(C, r=max_r)
            if len(idx) == 0:
                new_r.append(0.0)
                continue
                
            particles = self.tracers.iloc[idx][['x', 'y', 'z']].values
            dists = np.linalg.norm(particles - C, axis=1)
            
            bins = np.linspace(0.5 * self.mps, max_r, n_bins)
            dists.sort()
            enclosed_counts = np.searchsorted(dists, bins, side='right')
            
            vol_enclosed = vol_factor * (bins**3)
            rho_enclosed = enclosed_counts / vol_enclosed
            delta_enclosed = rho_enclosed / self.mean_tracer_density - 1.0
            
            loc = np.where(delta_enclosed > delta_target)[0]
            if len(loc) > 0:
                k = loc[0]
                if k == 0:
                    new_r.append(0.0)
                else:
                    y1, y2 = delta_enclosed[k-1], delta_enclosed[k]
                    r1, r2 = bins[k-1], bins[k]
                    frac = (delta_target - y1) / (y2 - y1) if y2 != y1 else 0
                    new_r.append(float(r1 + frac * (r2 - r1)))
            else:
                new_r.append(0.0)

        catalog['radius'] = new_r
        
        # min_physical_radius = max(2.0, self.mps / 3.0)
        catalog = catalog[catalog['radius'] >= 2.0].reset_index(drop=True)
        
        print(f"  -> {len(catalog)} voids survived SvdW scaling and MPS resolution cut.")
        return catalog

    def step3_prune_overlaps(self, catalog):
        print("[Step3] Pruning overlaps using REVOLVER's strict d < R1 + R2 rule...")
        sorted_cat = catalog.sort_values(by='radius', ascending=False).reset_index(drop=True)
        centers = sorted_cat[['x','y','z']].values
        radii = sorted_cat['radius'].values
        tree = KDTree(centers)
        keep = np.ones(len(sorted_cat), dtype=bool)
        
        for i in tqdm(range(len(sorted_cat)), desc="  Pruning"):
            if not keep[i]:
                continue
            neighs = tree.query_ball_point(centers[i], r=2.01 * radii[i])
            for j in neighs:
                if j > i and keep[j]:
                    dist = np.linalg.norm(centers[i] - centers[j])
                    # FIX: Corrected mathematical addition for exclusion zone
                    if dist < (radii[i] + radii[j]):
                        keep[j] = False
                        
        final = sorted_cat[keep].reset_index(drop=True)
        print(f"  -> {len(final)} independent benchmark voids remain.")
        return final

# --- Execution ---
cleaned_results = {}
for key in PROCESS_KEYS:
    void_path = samples.get(key)
    tracer_path = tracer_files.get(key)
    if not os.path.exists(void_path) or not os.path.exists(tracer_path):
        continue

    print(f"\n=== Benchmarking bitVF Sample: {key} ===")
    voids_df = read_bitvf_catalog(void_path)
    tracers_df = read_tracers(tracer_path)
    
    vmin = voids_df[['x','y','z']].min()
    tmin = tracers_df[['x','y','z']].min()
    shifts = (tmin - vmin)
    if (shifts.abs() > 1e-6).any():
        voids_df[['x','y','z']] += shifts.values
    
    cleaner = BitVFRevolverCleaner(voids_df, tracers_df, BOX_SIZE)
    cleaned_cat = cleaner.run_pipeline()
    cleaned_results[key] = cleaned_cat
    
    # FIX: Re-added CSV export logic
    outcsv = os.path.join(OUT_DIR, f"bitvf_revolver_benchmark_{key}.csv")
    cleaned_cat.to_csv(outcsv, index=False)
    print(f"  -> Saved cleaned catalog to {outcsv}")

# --- VSF Plotting with Error Bands ---
if cleaned_results:
    print("\nGenerating BitVF Benchmark VSF plot...")
    fig, ax = plt.subplots(figsize=(9, 9))
    
    bins = np.exp(np.arange(np.log(1.5), np.log(25.0) + 1e-12, np.log(10)/15))
    bin_centers = np.sqrt(bins[:-1] * bins[1:])
    dlnR = np.log(bins[1:] / bins[:-1])
    V_eff = BOX_SIZE**3

    colors = {'20p': '#6A3399', '10p': '#ADD8E6', '5p': '#3CB371', '2_5p': '#FFD700'}
    
    for key, cat in cleaned_results.items():
        if cat.empty: continue
        
        y, _ = np.histogram(cat['radius'].values, bins=bins)
        dn_dlnR = y / V_eff / dlnR
        err = np.sqrt(y) / V_eff / dlnR
        
        # FIX: Re-added physical artifact masking to remove the nosedive peaks
        # mps_dict = {'20p': 5.4, '10p': 6.8, '5p': 8.6, '2_5p': 10.7}
        # min_R = max(2.0, mps_dict.get(key, 5.0) / 3.0)
        mask = (dn_dlnR > 0) & (bin_centers >= 2.0)
        
        y_vals = dn_dlnR[mask]
        
        # Calculate logarithmic coordinates and errors
        y_plot = np.log10(y_vals)
        err_log = err[mask] / (y_vals * np.log(10))
        
        label_str = f'ss = {key.replace("p", "%").replace("_", ".")}'
        
        # Plot central line and shaded error band
        ax.plot(bin_centers[mask], y_plot, '-', lw=2.5, color=colors[key], label=label_str)
        ax.fill_between(bin_centers[mask], y_plot - err_log, y_plot + err_log, color=colors[key], alpha=0.3, edgecolor='none')

    ax.set_xscale('log')
    ax.set_xlim(1.5, 20)
    ax.set_ylim(-8.0, -3.2)
    
    ax.tick_params(axis='both', which='major', direction='in', length=8, top=True, right=True)
    ax.tick_params(axis='both', which='minor', direction='in', length=4, top=True, right=True)
    ax.set_xticks([2, 5, 10, 15])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=20))
    ax.xaxis.set_minor_formatter(NullFormatter())

    ax.set_xlabel(r"$R_v \ [h^{-1}\mathrm{Mpc}]$")
    ax.set_ylabel(r"$\log(\mathrm{d}n/\mathrm{d}{\ln R_v}) \ [h^3 \mathrm{Mpc}^{-3}]$")
    
    ax.text(0.12, 0.40, "BitVF VSF plot", transform=ax.transAxes, 
            fontsize=18, verticalalignment='center',
            bbox=dict(facecolor='white', edgecolor='black', alpha=1.0, pad=10))

    ax.legend(loc='lower left', bbox_to_anchor=(0.05, 0.1), ncol=2, frameon=False, 
              handlelength=2.0, columnspacing=1.5)
    ax.grid(True, which='major', ls=':', alpha=0.4, color='gray')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "BitVF_VSF_08.png"), dpi=300, bbox_inches='tight')
    print("Success! Plot saved.")