#!/usr/bin/env python3
"""
sweep_veff_per_subsample.py

For each NREC50 subsample, find the V_eff that minimizes SSR against
the ForArun Simone data VSF, using the *current* cleaned catalogs.
Tests a wide range of effective box volumes parametrized as L_eff^3.

Outputs: per-subsample best L_eff, SSR(best), and SSR at L=250 for comparison.
"""
import os, numpy as np, pandas as pd

HERE = os.path.dirname(__file__)
CLN  = os.path.join(HERE, '..', 'data', 'cleaned')
FA   = os.path.join(HERE, '..', 'data', 'ForArun')

FORARUN_BIN_CENTERS = np.array([
    1.591446e+00, 1.791403e+00, 2.016484e+00, 2.269845e+00, 2.555040e+00,
    2.876068e+00, 3.237432e+00, 3.644199e+00, 4.102074e+00, 4.617479e+00,
    5.197642e+00, 5.850699e+00, 6.585810e+00, 7.413284e+00, 8.344726e+00,
    9.393199e+00, 1.057341e+01, 1.190190e+01, 1.339732e+01, 1.508062e+01,
])
log_c = np.log(FORARUN_BIN_CENTERS)
DLNR  = np.median(np.diff(log_c))
EDGES = np.exp(np.append(log_c - DLNR/2, log_c[-1] + DLNR/2))

FA_MAP = {'2_5p': '025', '5p': '05', '10p': '10', '20p': '20'}

def cleaned_vsf(frac, v_eff):
    df = pd.read_csv(os.path.join(CLN, f'cleaned_recon_cat_{frac}.csv'))
    counts, _ = np.histogram(df['r_ext'].values, bins=EDGES)
    dn = counts / (v_eff * DLNR)
    return FORARUN_BIN_CENTERS, dn, counts

def forarun(frac):
    f = os.path.join(FA, f'vsf_dm_250cube_ss0p{FA_MAP[frac]}_NREC50_SM0p00_CS0p79_TH-0p80_OV00.dat')
    df = pd.read_csv(f, sep=r'\s+', comment='#', names=['r','vsf','err'])
    return df

def ssr(frac, L_eff):
    v_eff = L_eff ** 3
    centers, dn, counts = cleaned_vsf(frac, v_eff)
    fa = forarun(frac)
    # Match on common bins (within 1e-3 tolerance)
    fa_dict = dict(zip(np.round(fa['r'].values, 4), fa['vsf'].values))
    res = []
    for c, d, n in zip(centers, dn, counts):
        if n == 0: continue
        v_fa = fa_dict.get(round(c, 4))
        if v_fa is None or v_fa <= 0: continue
        # log-space residual normalized
        r = (np.log10(d) - np.log10(v_fa)) ** 2
        res.append(r)
    return np.sum(res) / max(len(res), 1)

# Sweep
L_grid = np.arange(150.0, 401.0, 5.0)
fractions = ['20p', '10p', '5p', '2_5p']

print(f"{'frac':<6}{'L*':>8}{'V*^(1/3)':>11}{'SSR*':>10}{'SSR(L=250)':>14}{'ratio':>9}")
print("-"*60)

results = {}
for frac in fractions:
    ssrs = np.array([ssr(frac, L) for L in L_grid])
    i_best = int(np.argmin(ssrs))
    L_star = L_grid[i_best]
    ssr_star = ssrs[i_best]
    ssr_250  = ssr(frac, 250.0)
    results[frac] = (L_star, ssr_star, ssr_250)
    print(f"{frac:<6}{L_star:>8.0f}{L_star:>11.1f}{ssr_star:>10.4f}{ssr_250:>14.4f}{ssr_250/max(ssr_star,1e-9):>9.2f}")

print("\nNote: SSR is mean squared log10 residual per bin.")
print("If L* is far from 250, the cleaned catalog amplitude is mismatched")
print("with ForArun -- a constant rescaling can fix it.")
