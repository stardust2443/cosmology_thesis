#!/usr/bin/env python3
"""
bin_simple0_to_vsf.py

Bins our newly validated simple-cleaned void catalogs (from data/cleaned_simple0/)
into binned VSF .dat files in:
  - data/cleaned_simple0_VSF_touch/  (for touch overlap)
  - data/cleaned_simple0_VSF_center/ (for center overlap)
so that the joint MCMC script can load them directly via the --data-dir flag.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(HERE, '..', 'data', 'cleaned_simple0')
OUT_TOUCH = os.path.join(HERE, '..', 'data', 'cleaned_simple0_VSF_touch')
OUT_CENTER = os.path.join(HERE, '..', 'data', 'cleaned_simple0_VSF_center')

# Survey parameters
V_EFF = (250.0 - 2 * 20.0) ** 3  # 210^3 = 9,261,000 h^-3 Mpc^3

# Standard binning
FORARUN_BIN_CENTERS = np.array([
    1.591446e+00, 1.791403e+00, 2.016484e+00, 2.269845e+00, 2.555040e+00,
    2.876068e+00, 3.237432e+00, 3.644199e+00, 4.102074e+00, 4.617479e+00,
    5.197642e+00, 5.850699e+00, 6.585810e+00, 7.413284e+00, 8.344726e+00,
    9.393199e+00, 1.057341e+01, 1.190190e+01, 1.339732e+01, 1.508062e+01,
])
LOG_C = np.log(FORARUN_BIN_CENTERS)
DLNR = float(np.median(np.diff(LOG_C)))
EDGES = np.exp(np.append(LOG_C - DLNR / 2, LOG_C[-1] + DLNR / 2))

FRAC_MAP = {'2_5p': '025', '5p': '05', '10p': '10', '20p': '20'}
SUBSAMPLES = ['20p', '10p', '5p', '2_5p']
FIELDS = ['NREC50', 'BIT']
OVERLAPS = ['touch', 'center']

def bin_radii(radii):
    counts, _ = np.histogram(radii, bins=EDGES)
    dn = counts / (V_EFF * DLNR)
    err = np.sqrt(counts) / (V_EFF * DLNR)
    return counts, dn, err

def main():
    print("Binning simple-cleaned catalogs into VSF .dat format...")
    print(f"  Effective volume: {V_EFF:.0f} h^-3 Mpc^3")
    print(f"  DLNR:             {DLNR:.6f}")

    for overlap in OVERLAPS:
        out_dir = OUT_TOUCH if overlap == 'touch' else OUT_CENTER
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nProcessing {overlap.upper()} overlap -> {out_dir}")

        for field in FIELDS:
            for frac in SUBSAMPLES:
                filename = f"cleaned_simple0_{field.lower()}_cat_{frac}_{overlap}.csv"
                filepath = os.path.join(CLEAN, filename)
                
                if not os.path.exists(filepath):
                    continue
                
                df = pd.read_csv(filepath)
                radii = df['R_resc'].values
                counts, dn, err = bin_radii(radii)

                # Keep filenames as _OV00.dat so cbl_mcmc_v5_joint.py can load them natively
                out_filename = (
                    f"vsf_dm_250cube_ss0p{FRAC_MAP[frac]}_{field}_"
                    "SM0p00_CS0p79_TH-0p80_OV00.dat"
                )
                out_path = os.path.join(out_dir, out_filename)

                with open(out_path, 'w') as f:
                    f.write('# ## bin_center vsf_value poisson_error\n')
                    for r, v, e in zip(FORARUN_BIN_CENTERS, dn, err):
                        f.write(f'{r:.6e} {v:.6e} {e:.6e}\n')

                n_R7 = int((radii >= 7.0).sum())
                print(f"  {field} {frac:<6}: {len(radii):>5d} voids ({n_R7:>3d} at R>=7) -> {out_filename}")

if __name__ == '__main__':
    main()
