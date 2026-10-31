#!/usr/bin/env python3
"""
diag_cbl_followup.py

Follow-up to diag_cbl_vs_handrolled.py. The first diagnostic showed Vdn+bias
still overpredicts at small R, with χ²/ν of O(100-500). Two hypotheses:

  (1) Disagreement is concentrated at R < MPS — the resolution-limited regime
      where Vdn is known to break. Test: recompute χ²/ν on R > MPS only.

  (2) The DM-tracer effective bias differs from CBL's BOSS-calibrated value
      (b_eff=1.122). Test: sweep b_eff ∈ [0.3, 2.0] at the truth cosmology
      and find the value that minimizes χ²/ν on the resolved range.

Both run on 20p NREC50 only (the cleanest case) for speed.

Outputs:
  plots/diagnostics/cbl_resolved_range_audit.png
  plots/diagnostics/cbl_beff_sweep.png
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CBL_HOME = os.environ.get('CBL_HOME', '/Users/arun/projects/CosmoBolognaLib')
sys.path.insert(0, os.path.join(CBL_HOME, 'Python'))
os.environ.setdefault('DYLD_LIBRARY_PATH', CBL_HOME)
import CosmoBolognaLib as cbl

HERE = os.path.dirname(os.path.abspath(__file__))
CLN  = os.path.join(HERE, '..', 'data', 'cleaned')
FA   = os.path.join(HERE, '..', 'data', 'ForArun')
OUT  = os.path.join(HERE, '..', 'plots', 'diagnostics')
os.makedirs(OUT, exist_ok=True)

# Aletheia cosmology
cosmo = cbl.cosmology.LCDM(0.315, 0.0493, 0.0, 0.6736,
                           2.194e-9, 0.05, 0.9649, 0.0925, True)
cosmo.set_parameter("sigma8", 0.811)
sf = cbl.cosmology.SizeFunction(cosmo)
DEL_C = cosmo.deltac(0.0)

MPS_OF = {'20p': 1.16, '10p': 1.47, '5p': 1.85, '2_5p': 2.33}

def theory(R_arr, model, b_eff, slope=0.85676, offset=0.42153, dvNL=-0.7):
    """Evaluate CBL size_function over R_arr. Returns NaN where GSL fails."""
    out = np.full(len(R_arr), np.nan)
    for i, R in enumerate(R_arr):
        try:
            out[i] = sf.size_function(float(R), 0.0, model, b_eff, slope, offset,
                                       dvNL, DEL_C, "EisensteinHu")
        except Exception as e:
            # GSL qag fails at some b_eff. Leave NaN and continue.
            pass
    return out

def chi2_per_dof_at_data(R_data, vsf_data, err_data, R_fine, theory_fine):
    """log-space chi^2 / dof at data points within R range. NaN-safe."""
    if len(R_data) == 0: return np.nan
    th = np.interp(R_data, R_fine, theory_fine)
    valid = np.isfinite(th) & (th > 0)
    if valid.sum() == 0: return np.nan
    err_log = err_data[valid] / (vsf_data[valid] * np.log(10))
    chi2 = np.sum(((np.log10(vsf_data[valid]) - np.log10(th[valid])) / err_log)**2)
    return chi2 / valid.sum()

def load_simone(frac):
    fa_map = {'2_5p':'025','5p':'05','10p':'10','20p':'20'}
    f = os.path.join(FA, f'vsf_dm_250cube_ss0p{fa_map[frac]}_NREC50_SM0p00_CS0p79_TH-0p80_OV00.dat')
    df = pd.read_csv(f, sep=r'\s+', comment='#', names=['r','vsf','err'])
    return df[df['vsf'] > 0].reset_index(drop=True)

# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: restricted range, all 4 subsamples
# ═══════════════════════════════════════════════════════════════════════

R_fine = np.logspace(np.log10(1.5), np.log10(20.), 100)
vdn_bias   = theory(R_fine, "Vdn",  1.1218389)
vdn_nobias = theory(R_fine, "Vdn",  1.0, slope=0.0, offset=0.0)
linear     = theory(R_fine, "linear", 1.1218389)

print("\n=== Experiment 1: Restricted-range χ²/ν on Simone data ===")
print(f"{'subsample':<8} {'MPS':>6} {'cut R>':>8} {'Nbins':>7} "
      f"{'Vdn+b':>8} {'Vdn nob':>9} {'linear':>8}")
print('-' * 60)

fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), sharey=True)
COLORS = {'20p': '#9b59b6', '10p': '#3498db', '5p': '#2ecc71', '2_5p': '#f1c40f'}

for ax, frac in zip(axes, ['20p','10p','5p','2_5p']):
    fa = load_simone(frac)
    mps = MPS_OF[frac]

    # Choose three cuts: R > 0, R > MPS, R > 2*MPS
    for cut_label, R_cut in [('all', 0.0), (f'R>{mps:.1f}', mps), (f'R>{2*mps:.1f}', 2*mps)]:
        m = fa['r'].values >= R_cut
        Rd, vd, ed = fa['r'].values[m], fa['vsf'].values[m], fa['err'].values[m]
        c_b = chi2_per_dof_at_data(Rd, vd, ed, R_fine, vdn_bias)
        c_n = chi2_per_dof_at_data(Rd, vd, ed, R_fine, vdn_nobias)
        c_l = chi2_per_dof_at_data(Rd, vd, ed, R_fine, linear)
        print(f"{frac:<8} {mps:>6.2f} {cut_label:>8} {len(Rd):>7} "
              f"{c_b:>8.2f} {c_n:>9.2f} {c_l:>8.2f}")

    # Plot full range, mark MPS and 2*MPS
    ax.loglog(R_fine, vdn_bias,   '-',  color='#c0392b', lw=2.2, label='Vdn + bias')
    ax.loglog(R_fine, vdn_nobias, '--', color='#7f8c8d', lw=1.8, label='Vdn (no bias)')
    ax.loglog(R_fine, linear,     ':',  color='#2c3e50', lw=1.5, label='linear')
    ax.errorbar(fa['r'], fa['vsf'], yerr=fa['err'], fmt='o',
                color=COLORS[frac], ms=4, label=f'Simone {frac}')
    ax.axvline(mps,    color='#bdc3c7', ls='-',  lw=1, alpha=0.7)
    ax.axvline(2*mps,  color='#bdc3c7', ls='--', lw=1, alpha=0.7)
    ax.set_xlim(1.5, 20); ax.set_ylim(1e-7, 1e-2)
    ax.grid(True, which='both', ls='--', alpha=0.3)
    ax.set_xlabel(r'$R_v\ [h^{-1}\mathrm{Mpc}]$')
    ax.set_title(f'{frac} (MPS={mps:.2f})')
    if ax is axes[0]:
        ax.set_ylabel(r'$\mathrm{d}n/\mathrm{d}\ln R_v$')
        ax.legend(fontsize=9, loc='lower left')

fig.suptitle('Experiment 1: Theory vs Simone data, restricted-range χ²/ν '
             '(grey vertical: MPS, dashed: 2×MPS)', fontsize=13)
plt.tight_layout()
out = os.path.join(OUT, 'cbl_resolved_range_audit.png')
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f"\nSaved: {out}")

# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: b_eff sweep on 20p NREC50, restricted to R > MPS
# ═══════════════════════════════════════════════════════════════════════
print("\n=== Experiment 2: b_eff sweep on 20p, R > MPS ===")

frac = '20p'
fa = load_simone(frac)
mps = MPS_OF[frac]
m = fa['r'].values >= mps
Rd, vd, ed = fa['r'].values[m], fa['vsf'].values[m], fa['err'].values[m]

b_grid = np.linspace(0.3, 2.0, 18)
chi2_b = np.zeros_like(b_grid)
for i, b in enumerate(b_grid):
    th = theory(R_fine, "Vdn", b)
    chi2_b[i] = chi2_per_dof_at_data(Rd, vd, ed, R_fine, th)

i_best = int(np.argmin(chi2_b))
print(f"\n{'b_eff':>8} {'χ²/ν':>10}")
for b, c in zip(b_grid, chi2_b):
    marker = ' <-- min' if abs(b - b_grid[i_best]) < 1e-6 else ''
    print(f"{b:>8.3f} {c:>10.3f}{marker}")

# Plot the sweep
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].plot(b_grid, chi2_b, 'o-', color='#c0392b')
axes[0].axvline(b_grid[i_best], color='#27ae60', ls='--',
                label=f'best b_eff = {b_grid[i_best]:.2f}')
axes[0].axvline(1.122, color='#7f8c8d', ls=':', label='CBL example default (1.122)')
axes[0].axvline(1.0,   color='#bdc3c7', ls=':', label='no-bias (1.0)')
axes[0].set_xlabel(r'$b_{\rm eff}$', fontsize=13)
axes[0].set_ylabel(r'$\chi^2/\nu$ (R > MPS)', fontsize=13)
axes[0].set_title(f'b_eff sweep on {frac} Simone, R > {mps:.2f}')
axes[0].legend(fontsize=10); axes[0].grid(True, ls='--', alpha=0.3)

# Plot best-fit theory vs data
th_best = theory(R_fine, "Vdn", b_grid[i_best])
axes[1].loglog(R_fine, th_best, '-', color='#27ae60', lw=2.2,
               label=f'Vdn at b_eff={b_grid[i_best]:.2f} (best)')
axes[1].loglog(R_fine, vdn_bias, '--', color='#c0392b', lw=1.5,
               label='Vdn at b_eff=1.122 (CBL default)')
axes[1].errorbar(fa['r'], fa['vsf'], yerr=fa['err'], fmt='o',
                 color=COLORS[frac], ms=4, label='Simone 20p')
axes[1].axvline(mps, color='#bdc3c7', ls='-', lw=1, label='MPS')
axes[1].set_xlim(1.5, 20); axes[1].set_ylim(1e-7, 1e-2)
axes[1].set_xlabel(r'$R_v$'); axes[1].set_ylabel(r'$\mathrm{d}n/\mathrm{d}\ln R_v$')
axes[1].set_title('Best-fit overlay')
axes[1].legend(fontsize=9); axes[1].grid(True, which='both', ls='--', alpha=0.3)

plt.tight_layout()
out2 = os.path.join(OUT, 'cbl_beff_sweep.png')
fig.savefig(out2, dpi=200, bbox_inches='tight')
print(f"\nSaved: {out2}")

print("\n--- Interpretation ---")
print("Exp 1: if χ²/ν drops sharply (e.g. <10) when R > MPS, the small-R")
print("       disagreement was resolution-limited and Vdn is fine in the")
print("       resolved regime. Phase 3 (MCMC) should restrict to R > MPS.")
print()
print("Exp 2: if best b_eff is very different from CBL default (1.122),")
print("       the DM-tracer bias needs calibration. If it's near 1.0,")
print("       no bias correction is needed (the DM tracer field is unbiased).")
