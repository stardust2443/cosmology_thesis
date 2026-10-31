#!/usr/bin/env python3
"""
jackknife_vsf_overlap.py
Compute jackknife (leave-one-octant-out) errors for the cleaned_simple0 void catalogs
for both 'touch' (OV00) and 'center' (OV05) overlap criteria.
Saves the binned VSF dat files with jackknife errors into data/vsf_jk_simple0.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CLN  = os.path.join(HERE, '..', 'data', 'cleaned_simple0')
OUT_DIR = os.path.join(HERE, '..', 'data', 'vsf_jk_simple0')
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
V_OCT  = 125.0**3
V_LOO  = 7.0 * V_OCT  # leave-one-out volume

FA_MAP = {'2_5p':'025','5p':'05','10p':'10','20p':'20'}
FRACTIONS = ['20p','10p','5p','2_5p']
OVERLAPS = {'touch': 'OV00', 'center': 'OV05'}

def counts_in_radii(radii):
    c, _ = np.histogram(radii, bins=EDGES)
    return c

def assign_octants(df):
    """Return integer octant id 0..7 for each void center."""
    # Box range is [375, 625] for Aletheia raws, so x0 = 375.0, half = 125.0
    x0 = 375.0
    half = 125.0
    ix = ((df['x'].values - x0) >= half).astype(int)
    iy = ((df['y'].values - x0) >= half).astype(int)
    iz = ((df['z'].values - x0) >= half).astype(int)
    return ix * 4 + iy * 2 + iz

def jackknife_vsf(df):
    """
    Returns (R, vsf_full, sigma_JK, counts_full).
    vsf_full uses V_eff = 250^3 (the natural single-realization estimate).
    sigma_JK is computed by leave-one-octant-out on V_eff_LOO = 7 * 125^3.
    """
    radii = df['R_resc'].values
    oct_id = assign_octants(df)

    # central value: full-box VSF
    counts_full = counts_in_radii(radii)
    vsf_full    = counts_full / (V_FULL * DLNR)

    # leave-one-out VSFs
    loo = np.zeros((8, len(FORARUN_BIN_CENTERS)))
    for k in range(8):
        mask = oct_id != k
        c_k = counts_in_radii(radii[mask])
        loo[k] = c_k / (V_LOO * DLNR)

    # standard jackknife variance (N-1)/N * Σ (θ_(i) - <θ_(i)>)²
    mean_loo = loo.mean(axis=0)
    var_jk   = (7.0 / 8.0) * np.sum((loo - mean_loo)**2, axis=0)
    sigma_jk = np.sqrt(var_jk)
    return FORARUN_BIN_CENTERS, vsf_full, sigma_jk, counts_full

def main():
    print(f"Calculating Jackknife errors for all subsamples and overlap criteria...")
    print(f"{'overlap':<10}{'frac':<6}{'N_voids':>8}")
    print("-" * 30)
    
    for overlap, ov_key in OVERLAPS.items():
        for frac in FRACTIONS:
            fpath = os.path.join(CLN, f'cleaned_simple0_nrec50_cat_{frac}_{overlap}.csv')
            if not os.path.exists(fpath):
                print(f"Skipping {frac} {overlap} (not found)")
                continue
            df = pd.read_csv(fpath)
            R, vsf, sig_jk, counts = jackknife_vsf(df)
            
            # Save dat file
            out_df = pd.DataFrame({'r': R, 'vsf': vsf, 'err': sig_jk})
            # Keep only bins with positive VSF
            out_df = out_df[out_df['vsf'] > 0]
            
            out_name = f"vsf_dm_250cube_ss0p{FA_MAP[frac]}_NREC50_SM0p00_CS0p79_TH-0p80_{ov_key}.dat"
            out_path = os.path.join(OUT_DIR, out_name)
            
            with open(out_path, 'w') as f:
                f.write('# r_v[h^-1 Mpc]    dn/dlnR[h^3 Mpc^-3]    sigma_JK\n')
                out_df.to_csv(f, sep='\t', index=False, header=False, float_format='%.6e')
                
            print(f"{overlap:<10}{frac:<6}{len(df):>8}  -> Saved {out_name}")

if __name__ == '__main__':
    main()
