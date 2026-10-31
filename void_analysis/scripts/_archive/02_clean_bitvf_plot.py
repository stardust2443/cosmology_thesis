#!/usr/bin/env python3
"""
02_clean_bitvf_revolver_style.py
Benchmarks bitVF voids using REVOLVER's empirical tracer-based rescaling.
"""
import os, sys, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree as KDTree
from tqdm import tqdm

# --- Configuration ---
BOX_SIZE = 250.0
PERIODIC = False
OUT_DIR = "../data/cleaned/benchmark_revolver_style"
os.makedirs(OUT_DIR, exist_ok=True)

# Update these paths to your bitVF output and the original tracer particle files
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
    'rmin': 2.0,
    'rmax': 50.0,
    'delta_nl_threshold': -0.8,
    'rescale_max_radius_factor': 4.0,
    'rescale_n_bins': 200,
    'border_cut': 0.0,
}

# --- Data Loaders ---
def read_bitvf_catalog(path):
    cols = ['x', 'y', 'z', 'radius', 'r_ext', 'delta_min', 'delta_avg']
    df = pd.read_csv(path, sep=r'\s+', comment='#', names=cols)
    return df

def read_tracers(path):
    df = pd.read_csv(path, sep=r'\s+', comment='#', header=None, usecols=[0,1,2], names=['x','y','z'])
    df[['x','y','z']] = df[['x','y','z']] % BOX_SIZE
    return df

# --- The Benchmark Cleaner ---
class BitVFRevolverCleaner:
    def __init__(self, voids_df, tracers_df, box_size, params, periodic=False):
        self.box_size = float(box_size)
        self.periodic = bool(periodic)
        self.voids = voids_df.copy().reset_index(drop=True)
        self.tracers = tracers_df.copy().reset_index(drop=True)
        
        self.mean_tracer_density = len(self.tracers) / (self.box_size**3)
        self.mps = self.mean_tracer_density ** (-1.0/3.0)
        
        print(f"[Init] tracers={len(self.tracers)}, mean_rho={self.mean_tracer_density:.4e}, mps={self.mps:.4f} Mpc/h")

    def run_pipeline(self):
        t0 = time.time()
        c = self.step1_border_cut()
        c = self.step2_rescale_radii(c)
        c = self.step3_prune_overlaps(c)
        dt = time.time() - t0
        print(f"[Done] Pipeline finished in {dt:.1f}s. Final void count: {len(c)}")
        return c

    def step1_border_cut(self):
        p = CLEANING_PARAMS
        rmin, rmax = p['rmin'], p['rmax']
        cat = self.voids[(self.voids['radius'] >= rmin) & (self.voids['radius'] <= rmax)].copy()
        
        print(f"  -> {len(cat)} voids after initial radius cut")
        
        # Build tracer tree once
        self.tracer_tree = KDTree(self.tracers[['x','y','z']].values)
        
        bc = p.get('border_cut', 0.0)
        if bc > 0.0 and not self.periodic:
            mask = ((cat['x'] > bc) & (cat['x'] < (self.box_size - bc)) &
                    (cat['y'] > bc) & (cat['y'] < (self.box_size - bc)) &
                    (cat['z'] > bc) & (cat['z'] < (self.box_size - bc)))
            cat = cat[mask].copy()
            print(f"  -> {len(cat)} voids after border cut {bc}")
            
        return cat.reset_index(drop=True)

    def step2_rescale_radii(self, catalog):
        print("[Step2] Fast Empirical Rescaling (Growing to SvdW -0.8 target)...")
        p = CLEANING_PARAMS
        n_bins = int(p['rescale_n_bins'])
        max_rf = float(p['rescale_max_radius_factor'])
        delta_target = float(p['delta_nl_threshold']) # -0.8

        centers = catalog[['x','y','z']].values
        radii = catalog['radius'].values
        new_r = []
        vol_factor = (4.0/3.0) * np.pi
        fail_count = 0

        for C, r0 in tqdm(zip(centers, radii), total=len(catalog), desc="  Rescaling"):
            max_r = max_rf * r0
            
            # Singularity Guard: Ensure max_r is reasonably large
            if max_r <= 1.0:
                new_r.append(0.0)
                fail_count += 1
                continue

            # Query all particles in the max volume ONCE
            idx = self.tracer_tree.query_ball_point(C, r=max_r)
            
            if len(idx) == 0:
                # Void is completely empty up to its expanded boundary
                new_r.append(0.0)
                fail_count += 1
                continue
                
            particles = self.tracers.iloc[idx][['x', 'y', 'z']].values
            dists = np.linalg.norm(particles - C, axis=1)
            
            # Start bins at 1.0 to avoid single-particle density singularities at the dead center
            bins = np.linspace(1.0, max_r, n_bins)
            
            dists.sort()
            enclosed_counts = np.searchsorted(dists, bins, side='right')
            
            vol_enclosed = vol_factor * (bins**3)
            rho_enclosed = enclosed_counts / vol_enclosed
            delta_enclosed = rho_enclosed / self.mean_tracer_density - 1.0
            
            # Find the exact crossing point of the galaxy wall
            loc = np.where(delta_enclosed > delta_target)[0]
            
            if len(loc) > 0:
                k = loc[0]
                if k == 0:
                    # The very first safe bin is already dense. This is a fake void!
                    new_r.append(0.0)
                    fail_count += 1
                else:
                    y1, y2 = delta_enclosed[k-1], delta_enclosed[k]
                    r1, r2 = bins[k-1], bins[k]
                    frac = (delta_target - y1) / (y2 - y1) if y2 != y1 else 0
                    r_interp = r1 + frac * (r2 - r1)
                    new_r.append(float(r_interp))
            else:
                new_r.append(0.0)
                fail_count += 1

        catalog['radius'] = new_r
        catalog = catalog[catalog['radius'] > 0].reset_index(drop=True)
        print(f"  -> {len(catalog)} voids successfully rescaled. (Shrunk/Failed: {fail_count})")
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
        print(f"[Warning] Skipping {key}: Missing void or tracer file.")
        continue

    print(f"\n=== Benchmarking bitVF Sample: {key} ===")
    voids_df = read_bitvf_catalog(void_path)
    tracers_df = read_tracers(tracer_path)
    
    # THE CRITICAL FIX: Align coordinates
    vmin = voids_df[['x','y','z']].min()
    tmin = tracers_df[['x','y','z']].min()
    shifts = (tmin - vmin)
    if (shifts.abs() > 1e-6).any():
        print(f"  [Align] Shifting bitVF coordinates by: {shifts.to_dict()}")
        voids_df[['x','y','z']] += shifts.values
    
    cleaner = BitVFRevolverCleaner(voids_df, tracers_df, BOX_SIZE, CLEANING_PARAMS, periodic=PERIODIC)
    cleaned_cat = cleaner.run_pipeline()
    
    outcsv = os.path.join(OUT_DIR, f"bitvf_revolver_benchmark_{key}.csv")
    cleaned_cat.to_csv(outcsv, index=False)
    cleaned_results[key] = cleaned_cat

# --- VSF Plotting ---
if cleaned_results:
    print("\nGenerating Benchmark VSF plot...")
    plt.figure(figsize=(9, 7))
    
    bins = np.geomspace(2.0, 25.0, 15)
    bin_centers = np.sqrt(bins[:-1] * bins[1:])
    dlnR = np.log(bins[1:] / bins[:-1])
    V_eff = BOX_SIZE**3

    colors = {'20p': '#6A3399', '10p': '#ADD8E6', '5p': '#3CB371', '2_5p': '#FFD700'}
    
    for key, cat in cleaned_results.items():
        if cat.empty:
            continue
        y, _ = np.histogram(cat['radius'].values, bins=bins)
        dn_dlnR = y / V_eff / dlnR
        mask = dn_dlnR > 0
        
        plt.plot(bin_centers[mask], np.log10(dn_dlnR[mask]), '-', lw=2.5, 
                 color=colors.get(key, '#000000'), label=f'bitVF Benchmark = {key}')

    plt.xscale('log')
    plt.xlim(1.5, 20)
    plt.ylim(-8.0, -2.8)
    plt.xlabel(r"$R_v \ [h^{-1}\mathrm{Mpc}]$", fontsize=16)
    plt.ylabel(r"$\log(\mathrm{d}n/\mathrm{d}{\ln R_v}) \ [h^3 \mathrm{Mpc}^{-3}]$", fontsize=16)
    plt.title("bitVF Catalog with REVOLVER Empirical Scaling", fontsize=16)
    plt.legend(loc='lower left', fontsize=12, frameon=False)
    plt.grid(True, which='major', ls=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "BitVF_Revolver_Benchmark.pdf"), dpi=300)
    plt.show()