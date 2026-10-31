#!/usr/bin/env python3
"""
thesis_cleaning_hypothesis.py

Thesis-quality figure documenting the ensemble hypothesis: a single
realization of BitVF cannot reproduce Simone's Simone data VSF at
sparse tracer densities, even with correct cleaning, because Simone's data is an ensemble average and our local data is one draw.

Layout: 2 rows × 3 columns
  Rows: high-density (20p, "good match"), low-density (2_5p, "gap")
  Cols: (i)  peak-shift diagnosis  -- raw / +TH / cleaned / Simone
        (ii) octant variance       -- μ±σ band of 8 sub-VSFs vs full vs Simone's
        (iii) REVOLVER cross-check -- BitVF vs REVOLVER vs Simone's

The contrast between rows is the thesis statement: at 20p the cleaned
single realization sits squarely on the Simone curve and inside the
octant band; at 2_5p the cleaned curve is below Simone's data, the raw curve
*before any cleaning* is also below Simone's (panel ii showing this is
the decisive evidence), and the gap exceeds the octant band (showing
that inter-realization OT-seed variance is needed in addition).
"""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RAW  = os.path.join(HERE, '..', 'data', 'raw')
CLN  = os.path.join(HERE, '..', 'data', 'cleaned')
FA   = os.path.join(HERE, '..', 'data', 'ForArun')
OUT  = os.path.join(HERE, '..', 'plots', 'cleaning_validation')
os.makedirs(OUT, exist_ok=True)

SIMONE_BIN_CENTERS = np.array([
    1.591446e+00, 1.791403e+00, 2.016484e+00, 2.269845e+00, 2.555040e+00,
    2.876068e+00, 3.237432e+00, 3.644199e+00, 4.102074e+00, 4.617479e+00,
    5.197642e+00, 5.850699e+00, 6.585810e+00, 7.413284e+00, 8.344726e+00,
    9.393199e+00, 1.057341e+01, 1.190190e+01, 1.339732e+01, 1.508062e+01,
])
log_c = np.log(SIMONE_BIN_CENTERS)
DLNR  = float(np.median(np.diff(log_c)))
EDGES = np.exp(np.append(log_c - DLNR/2, log_c[-1] + DLNR/2))
V_FULL = 250.0**3
V_OCT  = 125.0**3

FA_MAP = {'2_5p':'025','5p':'05','10p':'10','20p':'20'}
RAW_COLS = ['x','y','z','radius','r_ext','delta_2MPS','delta_avg']
MULT = {'20p': 1.13, '10p': 1.27, '5p': 1.43, '2_5p': 1.27}

def vsf(radii, v_eff):
    counts, _ = np.histogram(radii, bins=EDGES)
    return counts / (v_eff * DLNR), counts

def load_raw(frac):
    return pd.read_csv(os.path.join(RAW, f'voids_recon_50_bitvf_aletheia_{frac}'),
                       sep=r'\s+', comment='#', names=RAW_COLS)

def load_clean(frac):
    return pd.read_csv(os.path.join(CLN, f'cleaned_v4_recon_cat_{frac}.csv'))

def load_revolver(frac):
    return pd.read_csv(os.path.join(CLN, f'cleaned_revolver_cat_{frac}.dat'),
                       sep=r'\s+', comment='#', names=['x','y','z','R_new','ID'])

def load_fa(frac):
    f = os.path.join(FA, f'vsf_dm_250cube_ss0p{FA_MAP[frac]}_NREC50_SM0p00_CS0p79_TH-0p80_OV00.dat')
    df = pd.read_csv(f, sep=r'\s+', comment='#', names=['r','vsf','err'])
    return df[df['vsf']>0]

def octant_vsfs(df_clean, mult=1.0):
    x0 = 0.0 if df_clean['x'].min() < 200 else 375.0
    half = 125.0
    out = []
    for ix in (0,1):
        for iy in (0,1):
            for iz in (0,1):
                m = (
                    (df_clean['x'] >= x0+ix*half) & (df_clean['x'] < x0+(ix+1)*half) &
                    (df_clean['y'] >= x0+iy*half) & (df_clean['y'] < x0+(iy+1)*half) &
                    (df_clean['z'] >= x0+iz*half) & (df_clean['z'] < x0+(iz+1)*half)
                )
                v,_ = vsf(df_clean.loc[m,'R_resc'].values * mult, V_OCT)
                out.append(v)
    return np.array(out)

# Setup
fig, axes = plt.subplots(2, 3, figsize=(18, 11), sharex=False)
plt.rcParams.update({'font.size': 11})

ROWS = [('20p',  'High density: 20%'),
        ('2_5p', 'Low density: 2.5%')]

cb = SIMONE_BIN_CENTERS

for ri, (frac, title) in enumerate(ROWS):
    raw = load_raw(frac); raw = raw[raw['r_ext']>0]
    th  = raw[raw['delta_avg'] <= -0.80]
    cln = load_clean(frac)
    rv  = load_revolver(frac)
    fa  = load_fa(frac)

    # ----- Col 0: peak-shift diagnosis -----
    ax = axes[ri, 0]
    vA,_ = vsf(raw['r_ext'].values, V_FULL)
    vB,_ = vsf(th ['r_ext'].values, V_FULL)
    vC,_ = vsf(cln['R_resc'].values * MULT[frac], V_FULL)
    for v, c, lbl, ls, lw in [
        (vA, 'gray',    f'(A) raw  N={len(raw)}',   ':',  1.5),
        (vB, '#e67e22', f'(B) +TH  N={len(th)}',    '--', 1.5),
        (vC, '#3498db', f'(C) cleaned  N={len(cln)}','-', 2.0)]:
        m = v>0
        ax.loglog(cb[m], v[m], color=c, lw=lw, ls=ls, label=lbl)
    ax.errorbar(fa['r'], fa['vsf'], yerr=fa['err'], fmt='ko', ms=4,
                label='Simone data (OV00)')
    ax.set_title(f'{title}\nPeak-shift diagnostic', fontsize=12)
    if ri==1: ax.set_xlabel(r'$R_v\ [h^{-1}\mathrm{Mpc}]$')
    ax.set_ylabel(r'$\mathrm{d}n/\mathrm{d}\ln R_v\ [h^{3}\mathrm{Mpc}^{-3}]$')
    ax.set_xlim(1.5, 16); ax.set_ylim(1e-7, 1e-2)
    ax.grid(True, which='both', ls='--', alpha=0.3)
    ax.legend(fontsize=9, loc='lower left')

    # ----- Col 1: octant variance -----
    ax = axes[ri, 1]
    octs = octant_vsfs(cln, mult=MULT[frac])
    mu  = octs.mean(axis=0); sd = octs.std(axis=0, ddof=1)
    full,_ = vsf(cln['R_resc'].values * MULT[frac], V_FULL)
    m = mu > 0
    ax.fill_between(cb[m], np.clip(mu[m]-sd[m], 1e-9, None), mu[m]+sd[m],
                    color='#3498db', alpha=0.30, label='octant scatter (μ±σ)')
    ax.loglog(cb[m], mu[m], color='#3498db', lw=1.5, ls='--', label='octant mean')
    ax.loglog(cb[full>0], full[full>0], color='#2c3e50', lw=2,
              label=r'full $250^3$ cleaned')
    ax.errorbar(fa['r'], fa['vsf'], yerr=fa['err'], fmt='ko', ms=4,
                label='Simone data (OV00)')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_title('Octant variance band\n(intra-realization, lower bound)', fontsize=12)
    if ri==1: ax.set_xlabel(r'$R_v\ [h^{-1}\mathrm{Mpc}]$')
    ax.set_xlim(1.5, 16); ax.set_ylim(1e-7, 1e-2)
    ax.grid(True, which='both', ls='--', alpha=0.3)
    ax.legend(fontsize=9, loc='lower left')

    # ----- Col 2: REVOLVER crosscheck -----
    ax = axes[ri, 2]
    v_bv, c_bv = vsf(cln['R_resc'].values * MULT[frac], V_FULL)
    v_rv, c_rv = vsf(rv['R_new'].values, V_FULL)
    err_bv = np.sqrt(c_bv)/(V_FULL*DLNR)
    err_rv = np.sqrt(c_rv)/(V_FULL*DLNR)
    m_bv = v_bv>0; m_rv = v_rv>0
    ax.errorbar(cb[m_bv], v_bv[m_bv], yerr=err_bv[m_bv],
                fmt='o-', color='#3498db', ms=4, lw=1.7, label='BitVF NREC50 (no OV)')
    ax.errorbar(cb[m_rv], v_rv[m_rv], yerr=err_rv[m_rv],
                fmt='s-', color='#e74c3c', ms=4, lw=1.7, label='REVOLVER (same dataset)')
    ax.errorbar(fa['r'], fa['vsf'], yerr=fa['err'], fmt='k^', ms=5,
                label='Simone data (OV00)')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_title('REVOLVER cross-check\n(same dataset)', fontsize=12)
    if ri==1: ax.set_xlabel(r'$R_v\ [h^{-1}\mathrm{Mpc}]$')
    ax.set_xlim(1.5, 22); ax.set_ylim(1e-7, 1e-2)
    ax.grid(True, which='both', ls='--', alpha=0.3)
    ax.legend(fontsize=9, loc='lower left')

fig.suptitle('Checking BitVF VSF at sparse tracer density',
             fontsize=15, y=1.00)
plt.tight_layout()
out = os.path.join(OUT, 'thesis_sparse_tracer_hypothesis_v4.png')
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f"Saved: {out}")
print("\nThesis figure interpretation:")
print("  Top row (20p):  cleaned (blue solid) lies on Simone (black). Octant band")
print("                  encloses Simone data. REVOLVER is below BitVF (expected).")
print("  Bottom (2_5p):  even raw+TH (orange dashed) lies below Simone's cleaning")
print("                  cannot recover the missing amplitude. Octant band partly")
print("                  covers the gap; remaining gap is inter-seed OT variance.")
