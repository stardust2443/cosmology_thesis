#!/usr/bin/env python3
"""
bin_v4_cleaned_to_vsf.py

Bin our locally cleaned v4 BitVF catalogs into VSF files matching Simone's
reference format, so the v5 joint MCMC can consume them directly.

Input :  data/cleaned/cleaned_v4_recon_cat_{frac}.csv  (per-void: x,y,z,radius,r_ext,R_resc,...)
Output:  data/ForArun_v4cleaned/vsf_dm_250cube_ss0p{XX}_NREC50_SM0p00_CS0p79_TH-0p80_OV00.dat
  (or data/ForArun_v4cleaned_noMULT/ when --no-mult is passed)

Format mimics the ForArun reference:
  # ## bin_center vsf_value poisson_error
  <r>  <dn/dlnR>  <Poisson_err>

CLI:
  python scripts/bin_v4_cleaned_to_vsf.py            # default (with MULT peak-alignment)
  python scripts/bin_v4_cleaned_to_vsf.py --no-mult  # set MULT[*]=1.0 across subsamples
"""
import os, argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(HERE, '..', 'data', 'cleaned')
DEFAULT_OUT_MULT   = os.path.join(HERE, '..', 'data', 'ForArun_v4cleaned')
DEFAULT_OUT_NOMULT = os.path.join(HERE, '..', 'data', 'ForArun_v4cleaned_noMULT')

# Aletheia 250 h^-1 Mpc cube.
# Per Simone (2026-05): voids are cut to those whose centers are ≥ 20 h⁻¹Mpc
# from every face, so the effective survey volume is (250-2·20)³ = 210³,
# NOT 250³. Keep these two consistent at all times.
BORDER_PAD = 20.0
V_EFF = (250.0 - 2 * BORDER_PAD) ** 3

# ForArun binning (matches Simone exactly)
FORARUN_BIN_CENTERS = np.array([
    1.591446e+00, 1.791403e+00, 2.016484e+00, 2.269845e+00, 2.555040e+00,
    2.876068e+00, 3.237432e+00, 3.644199e+00, 4.102074e+00, 4.617479e+00,
    5.197642e+00, 5.850699e+00, 6.585810e+00, 7.413284e+00, 8.344726e+00,
    9.393199e+00, 1.057341e+01, 1.190190e+01, 1.339732e+01, 1.508062e+01,
])
LOG_C = np.log(FORARUN_BIN_CENTERS)
DLNR = float(np.median(np.diff(LOG_C)))
EDGES = np.exp(np.append(LOG_C - DLNR / 2, LOG_C[-1] + DLNR / 2))

# Phase-1 calibration: scales R_resc to match Simone's NREC50 peak per subsample.
# Advisor (May 2026) flagged this as empirical and "potentially arbitrary"; use the
# --no-mult flag to disable it.
MULT = {'20p': 1.13, '10p': 1.27, '5p': 1.43, '2_5p': 1.27}
FRAC_MAP = {'2_5p': '025', '5p': '05', '10p': '10', '20p': '20'}


def bin_radii(radii, v_eff=V_EFF):
    counts, _ = np.histogram(radii, bins=EDGES)
    dn = counts / (v_eff * DLNR)
    err = np.sqrt(counts) / (v_eff * DLNR)
    return counts, dn, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-mult', action='store_true',
                    help='Disable the per-subsample MULT factor (set to 1.0 for all '
                         'subsamples). Output goes to data/ForArun_v4cleaned_noMULT/.')
    ap.add_argument('--out-dir', default=None,
                    help='Override the default output dir.')
    args = ap.parse_args()

    if args.no_mult:
        active_mult = {k: 1.0 for k in MULT}
        out_root = args.out_dir or DEFAULT_OUT_NOMULT
        print(f"[MULT MODE] DISABLED (all subsamples get MULT = 1.0)")
    else:
        active_mult = dict(MULT)
        out_root = args.out_dir or DEFAULT_OUT_MULT
        print(f"[MULT MODE] PEAK-ALIGN  (MULT = {MULT})")
    os.makedirs(out_root, exist_ok=True)

    print("Binning cleaned v4 BitVF catalogs into Simone-format VSFs...")
    print(f"  Volume:    {V_EFF:.0f} h^-3 Mpc^3 (250^3)")
    print(f"  Δln R:     {DLNR:.6f}")
    print(f"  N bins:    {len(FORARUN_BIN_CENTERS)}")
    print(f"  Output:    {out_root}\n")

    summary = []
    for frac in ['20p', '10p', '5p', '2_5p']:
        cat_path = os.path.join(CLEAN, f'cleaned_v4_recon_cat_{frac}.csv')
        if not os.path.exists(cat_path):
            print(f"  {frac}: missing → skip")
            continue
        df = pd.read_csv(cat_path)
        radii = df['R_resc'].values * active_mult[frac]
        counts, dn, err = bin_radii(radii)

        out_path = os.path.join(
            out_root,
            f'vsf_dm_250cube_ss0p{FRAC_MAP[frac]}_NREC50_'
            'SM0p00_CS0p79_TH-0p80_OV00.dat')

        # Match Simone's header and column order
        with open(out_path, 'w') as f:
            f.write('# ## bin_center vsf_value poisson_error\n')
            for r, v, e in zip(FORARUN_BIN_CENTERS, dn, err):
                f.write(f'{r:.6e} {v:.6e} {e:.6e}\n')

        n_R7 = int(((radii >= 7.0)).sum())
        bins_with_data = int((counts > 0).sum())
        summary.append((frac, len(radii), n_R7, bins_with_data, out_path))
        print(f"  {frac}: {len(radii)} voids → {bins_with_data} occupied bins, "
              f"{n_R7} voids at R≥7 → {out_path}")

    print("\nSummary:")
    print(f"  {'frac':<6} {'N_voids':>8} {'N_R≥7':>7} {'occ.bins':>9}")
    for frac, n, n7, occ, _ in summary:
        print(f"  {frac:<6} {n:>8d} {n7:>7d} {occ:>9d}")


if __name__ == '__main__':
    main()
