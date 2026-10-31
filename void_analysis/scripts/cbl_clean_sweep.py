#!/usr/bin/env python3
"""
cbl_clean_sweep.py

Phase-1 driver: scan the CBL `clean_void_catalogue` outer-sphere `ratio`
parameter on the 5% NREC50 subsample, looking for the value that brings the
CBL-cleaned VSF into amplitude agreement with Simone's reference in the
R ≥ 7 h⁻¹Mpc fit range.

Sweep grid (default): ratio ∈ {0.5, 0.79, 1.0, 1.5}.
All other parameters held at Simone-conventional values (threshold=0.20,
ol_crit=central, radius_col=r_ext).

For each sweep point:
  1. Run cbl_clean_voids.clean_one(...) with the sweep ratio.
  2. Bin the cleaned R distribution onto the ForArun grid.
  3. Compare to Simone's 5% NREC50 reference VSF: report per-bin counts,
     amplitude ratio v_cbl/v_simone, and the median ratio in R ≥ 7.

Outputs:
  data/cleaned_cbl/sweep/cleaned_cbl_nrec50_cat_5p_r{ratio}.csv     (per ratio)
  data/cleaned_cbl/sweep/sweep_summary.csv                          (table)

CLI:
  python scripts/cbl_clean_sweep.py
  python scripts/cbl_clean_sweep.py --ratios 0.5 0.79 1.0 1.5
"""
import os, sys, argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Reuse the cleaner and binning utilities already in the codebase
from cbl_clean_voids import clean_one
from bin_v4_cleaned_to_vsf import (
    FORARUN_BIN_CENTERS, EDGES, DLNR, V_EFF
)

# Where to dump per-ratio outputs
SWEEP_DIR = os.path.join(HERE, '..', 'data', 'cleaned_cbl', 'sweep')
os.makedirs(SWEEP_DIR, exist_ok=True)

SIMONE_REF = os.path.join(
    HERE, '..', 'data', 'ForArun',
    'vsf_dm_250cube_ss0p05_NREC50_SM0p00_CS0p79_TH-0p80_OV00.dat')

R_MIN = 7.0


def bin_vsf(radii):
    """Bin a void-radius array onto the ForArun grid. Returns (counts, vsf, err)."""
    counts, _ = np.histogram(radii, bins=EDGES)
    vsf = counts / (V_EFF * DLNR)
    err = np.sqrt(counts) / (V_EFF * DLNR)
    return counts, vsf, err


def load_simone():
    df = pd.read_csv(SIMONE_REF, sep=r'\s+', comment='#',
                     names=['r', 'vsf', 'err'])
    return df[df['vsf'] > 0].reset_index(drop=True)


def ratio_tag(r):
    """Format ratio for filenames: 0.79 → '_r0p79', 1.5 → '_r1p5'."""
    return f"_r{str(r).replace('.', 'p')}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ratios', type=float, nargs='+',
                    default=[0.5, 0.79, 1.0, 1.5],
                    help='Ratios to sweep.')
    ap.add_argument('--frac', default='5p', choices=['20p', '10p', '5p', '2_5p'])
    ap.add_argument('--field', default='NREC50', choices=['NREC50', 'BIT'])
    args = ap.parse_args()

    print(f"\n{'#'*64}")
    print(f"#  CBL cleaning sweep: ratio ∈ {args.ratios}")
    print(f"#  field = {args.field}   frac = {args.frac}")
    print(f"{'#'*64}\n")

    simone = load_simone()
    simone_by_r = dict(zip(simone['r'].round(3), simone['vsf']))

    rows = []
    binned = {}  # ratio → (counts, vsf, err)

    for r in args.ratios:
        tag = ratio_tag(r)
        print(f"\n→ Sweep point: ratio = {r}  (tag = {tag})")
        out_csv = clean_one(
            args.frac, field=args.field,
            ratio=r, threshold=0.20, ol_crit='central',
            radius_col='r_ext', out_dir=SWEEP_DIR, tag=tag)

        df = pd.read_csv(out_csv)
        counts, vsf, err = bin_vsf(df['R'].values)
        binned[r] = (counts, vsf, err)

        # Amplitude ratio in fit range
        fit_ratios = []
        for rbin, c, v in zip(FORARUN_BIN_CENTERS, counts, vsf):
            if rbin < R_MIN:
                continue
            v_sim = simone_by_r.get(round(rbin, 3))
            if v_sim and v_sim > 0 and v > 0:
                fit_ratios.append(v / v_sim)
        med_ratio = float(np.median(fit_ratios)) if fit_ratios else float('nan')

        n_total = len(df)
        n_R7 = int((df['R'] >= R_MIN).sum())
        n_empty_fitbins = int(sum(
            1 for rbin, c in zip(FORARUN_BIN_CENTERS, counts)
            if rbin >= R_MIN and c == 0))

        rows.append({
            'ratio': r,
            'N_voids': n_total,
            'N_R>=7': n_R7,
            'R_median': float(np.median(df['R'])),
            'median_ratio_R>=7': med_ratio,
            'empty_fit_bins': n_empty_fitbins,
        })
        print(f"  → N={n_total}, N(R≥7)={n_R7}, median v_cbl/v_simone (R≥7) "
              f"= {med_ratio:.3f}")

    # Summary table
    print(f"\n{'='*64}")
    print("SWEEP SUMMARY")
    print('='*64)
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.3f}'))

    # Save summary
    summary_path = os.path.join(SWEEP_DIR, 'sweep_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")

    # Save per-bin VSFs for plotting (one column per ratio)
    binned_df = pd.DataFrame({'r': FORARUN_BIN_CENTERS})
    for r in args.ratios:
        c, v, e = binned[r]
        tag = f'r{str(r).replace(".", "p")}'
        binned_df[f'count_{tag}'] = c
        binned_df[f'vsf_{tag}'] = v
        binned_df[f'err_{tag}'] = e
    binned_path = os.path.join(SWEEP_DIR, 'sweep_binned_vsfs.csv')
    binned_df.to_csv(binned_path, index=False)
    print(f"Saved binned VSFs: {binned_path}")

    # Pick recommended best ratio: closest median to 1.0, must have ≥ 4 occupied fit bins
    candidates = [row for row in rows
                  if row['empty_fit_bins'] <= 3
                  and not np.isnan(row['median_ratio_R>=7'])]
    if candidates:
        best = min(candidates, key=lambda r: abs(r['median_ratio_R>=7'] - 1.0))
        print(f"\nRECOMMENDED BEST RATIO: {best['ratio']}  "
              f"(median ratio = {best['median_ratio_R>=7']:.3f}, "
              f"N(R≥7) = {best['N_R>=7']})")
    else:
        print("\nNo ratio gave usable fit-range statistics. Reconsider parameters.")


if __name__ == '__main__':
    main()
