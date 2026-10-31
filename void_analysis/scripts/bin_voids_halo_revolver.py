#!/usr/bin/env python3
"""
bin_voids_halo_revolver.py

Bins cleaned REVOLVER halo void catalogs and computes 64-region spatial Jackknife
covariance matrices, diagonal uncertainties, and Hartlap-corrected precision matrices.

Spatial sub-cubes: 4x4x4 grid across [0, 1000]^3, each of width 250 h^-1 Mpc.
Survey volume: V_eff = (1000 - 2 * 20)^3 = 960^3 = 8.84736e8 h^-3 Mpc^3.
"""
import os, sys, argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN_DIR = os.path.join(HERE, '..', 'data', 'voids_halo_revolver', 'cleaned')
VSF_DIR = os.path.join(HERE, '..', 'data', 'voids_halo_revolver', 'vsf')
COV_DIR = os.path.join(HERE, '..', 'data', 'voids_halo_revolver', 'cov')
os.makedirs(VSF_DIR, exist_ok=True)
os.makedirs(COV_DIR, exist_ok=True)

BOX = 1000.0
BORDER_PAD = 20.0
V_EFF = (BOX - 2.0 * BORDER_PAD) ** 3  # 960^3 = 8.84736e8 h^-3 Mpc^3

# Define logarithmic bins matching Contarini / standard convention
R_BASE = 1.591446
DLNR = 0.11867127167
ALL_CENTERS = np.array([R_BASE * np.exp(i * DLNR) for i in range(35)])
MASK_BINS = (ALL_CENTERS >= 8.0) & (ALL_CENTERS <= 36.0)
BIN_CENTERS = ALL_CENTERS[MASK_BINS]
LOG_C = np.log(BIN_CENTERS)
EDGES = np.exp(np.append(LOG_C - DLNR / 2.0, LOG_C[-1] + DLNR / 2.0))
N_BINS = len(BIN_CENTERS)

# 64 Jackknife sub-cubes
N_SUB = 4
W_SUB = BOX / N_SUB  # 250.0 h^-1 Mpc
V_LOO = (63.0 / 64.0) * V_EFF

def assign_subcube(coords):
    ix = np.clip((coords[:, 0] / W_SUB).astype(int), 0, N_SUB - 1)
    iy = np.clip((coords[:, 1] / W_SUB).astype(int), 0, N_SUB - 1)
    iz = np.clip((coords[:, 2] / W_SUB).astype(int), 0, N_SUB - 1)
    return ix * (N_SUB * N_SUB) + iy * N_SUB + iz

def compute_revolver_vsf_and_cov(sample='100P', criterion='touch', radius_col='R_resc'):
    in_file = os.path.join(CLEAN_DIR, f"cleaned_voids_halo_{sample}_{criterion}.csv")
    if not os.path.exists(in_file):
        print(f"Error: Catalog not found: {in_file}")
        return
        
    df = pd.read_csv(in_file)
    coords = df[['x', 'y', 'z']].values
    radii = df[radius_col].values
    n_voids = len(radii)
    print(f"=== Binned REVOLVER VSF & 64-JK Covariance for {sample} ({criterion}, {radius_col}) ===")
    print(f"Total voids: {n_voids}, V_eff = {V_EFF:.4e} h^-3 Mpc^3")
    
    # Full VSF
    counts_full, _ = np.histogram(radii, bins=EDGES)
    dn_full = counts_full / (V_EFF * DLNR)
    poisson_err = np.sqrt(np.maximum(counts_full, 1.0)) / (V_EFF * DLNR)
    
    # 64 Jackknife Leave-One-Out
    sub_ids = assign_subcube(coords)
    n_jk = N_SUB ** 3  # 64
    loo_matrix = np.zeros((n_jk, N_BINS))
    
    for k in range(n_jk):
        mask_k = (sub_ids != k)
        c_k, _ = np.histogram(radii[mask_k], bins=EDGES)
        loo_matrix[k] = c_k / (V_LOO * DLNR)
        
    mean_loo = loo_matrix.mean(axis=0)
    cov_jk = (float(n_jk - 1) / float(n_jk)) * np.dot((loo_matrix - mean_loo).T, (loo_matrix - mean_loo))
    sigma_jk = np.sqrt(np.diag(cov_jk))
    
    # Select active bins (bins with count >= 5)
    valid_bins = counts_full >= 5
    active_indices = np.where(valid_bins)[0]
    p = len(active_indices)
    
    hartlap_factor = (n_jk - p - 2.0) / (n_jk - 1.0)
    cov_active = cov_jk[np.ix_(active_indices, active_indices)]
    inv_cov_raw = np.linalg.pinv(cov_active)
    inv_cov_hartlap = hartlap_factor * inv_cov_raw
    
    print(f"Active bins (count >= 5): {p}/{N_BINS}")
    print(f"Hartlap factor: {hartlap_factor:.4f}")
    print(f"{'Bin':<4} | {'R_center':<10} | {'Count':<8} | {'dn/dlnR':<14} | {'Sigma_Poisson':<14} | {'Sigma_JK':<14} | {'JK/Poiss':<10}")
    print("-" * 82)
    for i in range(N_BINS):
        ratio = sigma_jk[i] / poisson_err[i] if poisson_err[i] > 0 else 0.0
        print(f"{i:<4} | {BIN_CENTERS[i]:<10.3f} | {counts_full[i]:<8} | {dn_full[i]:<14.4e} | {poisson_err[i]:<14.4e} | {sigma_jk[i]:<14.4e} | {ratio:<10.2f}")
        
    # Save results
    cov_out = os.path.join(COV_DIR, f"cov_jk_halo_{sample}_{criterion}.npz")
    np.savez(
        cov_out,
        bin_centers=BIN_CENTERS,
        bin_edges=EDGES,
        counts=counts_full,
        dn_dlnR=dn_full,
        sigma_poisson=poisson_err,
        sigma_jk=sigma_jk,
        cov_full=cov_jk,
        active_indices=active_indices,
        hartlap_factor=hartlap_factor,
        inv_cov_hartlap=inv_cov_hartlap,
        v_eff=V_EFF,
        dlnr=DLNR
    )
    print(f"Saved covariance and precision matrix to {cov_out}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=str, default='100P')
    parser.add_argument('--criterion', type=str, default='touch', choices=['center', 'touch'])
    parser.add_argument('--radius-col', type=str, default='R_resc')
    args = parser.parse_args()
    
    compute_revolver_vsf_and_cov(args.sample, args.criterion, args.radius_col)
