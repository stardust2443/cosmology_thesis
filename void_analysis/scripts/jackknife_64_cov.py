#!/usr/bin/env python3
"""
jackknife_64_cov.py

Computes 64-region (4x4x4 spatial grid) Jackknife covariance matrices,
diagonal error vectors, and Hartlap-corrected precision matrices
for both TOUCH (OV00) and CENTER (OV05) void cleaning protocols across all
4 tracer subsamples (20p, 10p, 5p, 2_5p).

Outputs stored in data/cov_jk_64/.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CLN  = os.path.join(HERE, '..', 'data', 'cleaned_simple0')
OUT_DIR = os.path.join(HERE, '..', 'data', 'cov_jk_64')
os.makedirs(OUT_DIR, exist_ok=True)

FORARUN_BIN_CENTERS = np.array([
    1.591446e+00, 1.791403e+00, 2.016484e+00, 2.269845e+00, 2.555040e+00,
    2.876068e+00, 3.237432e+00, 3.644199e+00, 4.102074e+00, 4.617479e+00,
    5.197642e+00, 5.850699e+00, 6.585810e+00, 7.413284e+00, 8.344726e+00,
    9.393199e+00, 1.057341e+01, 1.190190e+01, 1.339732e+01, 1.508062e+01,
])
log_c = np.log(FORARUN_BIN_CENTERS)
DLNR  = float(np.median(np.diff(log_c)))
EDGES = np.exp(np.append(log_c - DLNR/2, log_c[-1] + DLNR/2))

V_FULL = 250.0**3
V_SUB  = 62.5**3
V_LOO  = 63.0 * V_SUB  # Leave-one-out volume (63 sub-cubes)

FA_MAP = {'2_5p':'025','5p':'05','10p':'10','20p':'20'}
SUBSAMPLES = ['20p','10p','5p','2_5p']
OVERLAPS = {'touch': 'OV00', 'center': 'OV05'}

def counts_in_radii(radii):
    c, _ = np.histogram(radii, bins=EDGES)
    return c

def assign_64_regions(df):
    """Return integer region id 0..63 for each void center."""
    x0 = 375.0
    w  = 62.5
    ix = np.clip(((df['x'].values - x0) / w).astype(int), 0, 3)
    iy = np.clip(((df['y'].values - x0) / w).astype(int), 0, 3)
    iz = np.clip(((df['z'].values - x0) / w).astype(int), 0, 3)
    return ix * 16 + iy * 4 + iz

def compute_64_jackknife(df):
    """
    Returns (R, vsf_full, sigma_jk_64, loo_matrix [shape 64 x 20]).
    """
    radii = df['R_resc'].values
    reg_id = assign_64_regions(df)

    counts_full = counts_in_radii(radii)
    vsf_full    = counts_full / (V_FULL * DLNR)

    loo = np.zeros((64, len(FORARUN_BIN_CENTERS)))
    for k in range(64):
        mask = reg_id != k
        c_k = counts_in_radii(radii[mask])
        loo[k] = c_k / (V_LOO * DLNR)

    mean_loo = loo.mean(axis=0)
    var_jk   = (63.0 / 64.0) * np.sum((loo - mean_loo)**2, axis=0)
    sigma_jk = np.sqrt(var_jk)

    return FORARUN_BIN_CENTERS, vsf_full, sigma_jk, loo

def process_all():
    print("Computing 64-region Jackknife covariance matrices & Hartlap factors...")
    
    for overlap, ov_key in OVERLAPS.items():
        loo_joint_dict = {}
        vsf_full_dict = {}
        
        for frac in SUBSAMPLES:
            fpath = os.path.join(CLN, f'cleaned_simple0_nrec50_cat_{frac}_{overlap}.csv')
            if not os.path.exists(fpath):
                print(f"File not found: {fpath}")
                continue
            df = pd.read_csv(fpath)
            R, vsf, sig_jk, loo = compute_64_jackknife(df)
            
            loo_joint_dict[frac] = loo
            vsf_full_dict[frac]  = vsf

            # Save per-subsample dat file with 64-region JK error
            out_df = pd.DataFrame({'r': R, 'vsf': vsf, 'err': sig_jk})
            out_df = out_df[out_df['vsf'] > 0]
            out_name = f"vsf_64jk_nrec50_{frac}_{ov_key}.dat"
            out_df.to_csv(os.path.join(OUT_DIR, out_name), sep='\t', index=False, float_format='%.6e')

        # Construct joint covariance for scale cuts Rmin in [6.0, 7.0, 8.0, 9.0, 10.0]
        for rmin in [6.0, 7.0, 8.0, 9.0, 10.0]:
            # Bins matching R >= rmin
            mask_bins = FORARUN_BIN_CENTERS >= rmin
            n_bins_sub = int(np.sum(mask_bins))
            
            # Save single-subsample covariances first
            for frac in SUBSAMPLES:
                loo_sub = loo_joint_dict[frac][:, mask_bins]
                mean_loo_sub = loo_sub.mean(axis=0)
                devs_sub = loo_sub - mean_loo_sub
                C_sub = (63.0 / 64.0) * np.dot(devs_sub.T, devs_sub)
                
                cond_sub = np.linalg.cond(C_sub)
                C_sub_inv = np.linalg.inv(C_sub)
                
                N = 64
                p_sub = n_bins_sub
                alpha_sub = (N - p_sub - 2.0) / (N - 1.0)
                C_sub_inv_hartlap = alpha_sub * C_sub_inv
                
                np.save(os.path.join(OUT_DIR, f'cov_64jk_R{int(rmin)}_{frac}_{ov_key}.npy'), C_sub)
                np.save(os.path.join(OUT_DIR, f'inv_cov_64jk_R{int(rmin)}_{frac}_{ov_key}.npy'), C_sub_inv_hartlap)
                
            n_total_bins = n_bins_sub * len(SUBSAMPLES)

            # Build concatenated LOO matrix of shape (64, n_total_bins)
            loo_concat = []
            full_vsf_concat = []
            for frac in SUBSAMPLES:
                loo_concat.append(loo_joint_dict[frac][:, mask_bins])
                full_vsf_concat.append(vsf_full_dict[frac][mask_bins])

            loo_matrix = np.hstack(loo_concat)  # shape (64, n_total_bins)
            mean_loo = loo_matrix.mean(axis=0)

            # Covariance matrix C (shape n_total_bins x n_total_bins)
            devs = loo_matrix - mean_loo
            C = (63.0 / 64.0) * np.dot(devs.T, devs)

            # Invert covariance matrix
            cond = np.linalg.cond(C)
            C_inv = np.linalg.inv(C)

            # Hartlap correction factor: alpha = (N - p - 2) / (N - 1)
            # N = 64, p = n_total_bins
            N = 64
            p = n_total_bins
            alpha_hartlap = (N - p - 2.0) / (N - 1.0)
            C_inv_hartlap = alpha_hartlap * C_inv

            # Save covariance matrices and precision matrices
            np.save(os.path.join(OUT_DIR, f'cov_64jk_R{int(rmin)}_{ov_key}.npy'), C)
            np.save(os.path.join(OUT_DIR, f'inv_cov_64jk_R{int(rmin)}_{ov_key}.npy'), C_inv_hartlap)
            
            print(f"[{ov_key}] Rmin={rmin:.1f} | n_bins={n_total_bins} | Cond={cond:.2e} | Hartlap alpha={alpha_hartlap:.4f}")

if __name__ == '__main__':
    process_all()
