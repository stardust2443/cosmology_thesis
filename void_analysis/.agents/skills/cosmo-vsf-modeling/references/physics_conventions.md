# Physics Conventions & Simulation Parameters Reference

This document summarizes the simulation parameters, physical models, and sample conventions used across the void analysis and thesis project.

---

## 1. Aletheia N-body Simulation Benchmark

| Parameter | Symbol | Fiducial Value | Description |
| :--- | :--- | :--- | :--- |
| Matter Density | $\Omega_m$ | `0.315` | Total matter density parameter |
| Amplitude of Fluctuations | $\sigma_8$ | `0.811` | Linear amplitude on $8\,h^{-1}\mathrm{Mpc}$ scale |
| Box Length | $L$ | $1000\,h^{-1}\mathrm{Mpc}$ | Comoving box dimension |
| Redshift | $z$ | `0.0` | Present-day simulation snapshot |
| Hubble Parameter | $h$ | `0.67` | $H_0 / (100\,\mathrm{km\,s^{-1}\,Mpc^{-1}})$ |
| Spectral Index | $n_s$ | `0.96` | Primordial scalar spectral index |
| Dark Energy EOS | $w$ | `-1.0` | Cosmological constant equation of state |

---

## 2. Tracer Samples & Subsample Definitions

### Halo Catalogs (Phase 3)
Four mass/number-density thresholded halo catalogs extracted from Aletheia:
- **100P**: 100% of detected halos ($b_{\mathrm{eff}} \approx 0.9583$)
- **75P**: 75% densest halo subset ($b_{\mathrm{eff}} \approx 1.0211$)
- **50P**: 50% densest halo subset ($b_{\mathrm{eff}} \approx 1.1542$)
- **30P**: 30% densest halo subset ($b_{\mathrm{eff}} \approx 1.3418$)

### Dark Matter Subsamples (Phase 1 & 2)
Random particle downsamplings with mean particle separation (MPS):
- **20p**: $20\%$ downsample, $\text{MPS} = 1.16\,h^{-1}\mathrm{Mpc}$
- **10p**: $10\%$ downsample, $\text{MPS} = 1.47\,h^{-1}\mathrm{Mpc}$
- **5p**: $5\%$ downsample, $\text{MPS} = 1.85\,h^{-1}\mathrm{Mpc}$
- **2_5p**: $2.5\%$ downsample, $\text{MPS} = 2.33\,h^{-1}\mathrm{Mpc}$

---

## 3. Void Finders & Model Compatibility Summary

| Property | BitVF (Sartori 2025) | REVOLVER (Nadathur 2019 / ZOBOV) |
| :--- | :--- | :--- |
| **Detection Method** | Back-in-time dynamical displacement field | Voronoi tessellation + watershed algorithm |
| **Lagrangian Geometry** | Smooth quasi-spherical underdense basins | Irregular, hierarchical watershed topology |
| **Vdn Compatibility** | $\chi^2/\nu \approx 0.90\text{--}1.67$ (Compatible) | $\chi^2/\nu \approx 3.14\text{--}8.12$ (Incompatible) |
| **Volume Cleaning** | `TOUCH` (OV00) preserves volume conservation | Watershed basins merge or remain unmerged |
| **MCMC Posteriors** | Recovers true $(\Omega_m, \sigma_8)$ within $1\sigma$ | Biased posteriors ($\Omega_m \sim 0.40\text{--}0.45$) |
| **Barrier Slope** | $B_{\mathrm{slope}} = 0.914 \pm 0.043$ | $B_{\mathrm{slope}} = 0.910 \pm 0.046$ |

---

## 4. Covariance & Degrees of Freedom Rules

1. **Covariance Matrix**:
   64-region spatial Jackknife ($N_{\mathrm{sub}} = 64$ sub-volumes).
   Hartlap factor:
   $$\alpha_{\mathrm{Hartlap}} = \frac{N_{\mathrm{sub}} - p - 2}{N_{\mathrm{sub}} - 1}$$
   where $p$ is the number of radial bins.
2. **Degrees of Freedom**:
   - 1-parameter barrier fit ($b_{\mathrm{void}}$ at fixed cosmology): $\nu = p - 1$
   - 2-parameter barrier regression ($B_{\mathrm{slope}}, B_{\mathrm{offset}}$): $\nu = p - 2$
   - 3-parameter MCMC ($\Omega_m, \sigma_8, b_{\mathrm{void}}$): $\nu = p - 3$
