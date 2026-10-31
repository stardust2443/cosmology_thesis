---
name: cosmo-vsf-modeling
description: >-
  Use this skill when calculating Void Size Functions (VSF), fitting Jennings et al. (Vdn)
  excursion-set models, running CosmoBolognaLib (CBL) void analysis, evaluating chi2/nu goodness-of-fit,
  or configuring void cleaning algorithms (TOUCH vs CENTER).
---

# Void Size Function (VSF) & Theoretical Modeling Runbook

This skill details the execution environment, theoretical modeling procedures, and strict physical constraints for cosmic void statistics on the Aletheia simulation suite using CosmoBolognaLib (CBL), BitVF, and REVOLVER.

---

## 1. Execution Environment & Dependencies

Always execute cosmology analysis scripts with Homebrew Python 3.12 and explicit CBL paths:

```bash
export CBL_HOME=/Users/arun/projects/CosmoBolognaLib
export PYTHONPATH=$CBL_HOME/Python:$PYTHONPATH
export DYLD_LIBRARY_PATH=$CBL_HOME:$DYLD_LIBRARY_PATH
export MPLCONFIGDIR=/tmp/matplotlib_cache

/opt/homebrew/bin/python3.12 <script_name>.py
```

### Critical C++ Bias Bug in CosmoBolognaLib
> [!CAUTION]
> If effective bias $b_{\mathrm{eff}} == 1.0$, CosmoBolognaLib's internal C++ implementation silently overrides the user's bias parameters with hardcoded defaults.
> **Fix**: Never pass exactly `1.0`. Always pass `b_eff = 1.0001` for unbiased tracers (or the measured halo bias, e.g., $1.0211, 0.9583, \dots$).

---

## 2. Core Physical Constraints & Model Compatibility

### Rule 1: Strict Volume Conservation (`TOUCH` vs `CENTER`)
- The **Jennings et al. (2013) Vdn model** is derived from spherical excursion-set theory with an isolated volume-conserving barrier.
- **`TOUCH` (or `OV00`)**: Voids are cleaned as strictly non-overlapping spheres. This preserves volume conservation and yields $\chi^2/\nu \approx 0.90\text{--}1.67$.
- **`CENTER` (or `OV05`)**: Voids allow mutual overlap up to the center of adjacent voids. This **double-counts volume**, violating Vdn theory and blowing up $\chi^2/\nu \sim 25$.
- **Protocol**: When fitting Vdn models, **only** use `TOUCH` cleaned catalogs. Never evaluate Vdn fits on `CENTER` catalogs without explicitly documenting the volume double-counting breakdown.

### Rule 2: Model-Finder Compatibility (BitVF vs REVOLVER)
- **BitVF** (dynamical displacement divergence): Voids expand back in time into spherical Lagrangian regions. Structurally compatible with Vdn excursion-set physics ($\chi^2/\nu \approx 0.90\text{--}1.67$).
- **REVOLVER** (ZOBOV Voronoi watershed basins): Unmerged watershed basins have complex, non-spherical, hierarchical topologies.
  - Imposes an intrinsic physical shape mismatch with spherical Vdn theory ($\chi^2/\nu \approx 3.14\text{--}8.12$).
  - **MCMC Artifact**: The apparent shift in recovered $\Omega_m \sim 0.40\text{--}0.45$ for REVOLVER is **not** a physical cosmological detection; it is a mathematical artifact of forcing an incompatible spherical model onto watershed basins.

### Rule 3: Linear Barrier Universality
- Linear barrier relation:
  $$b_{\mathrm{void}} = B_{\mathrm{slope}} b_{\mathrm{eff}} + B_{\mathrm{offset}}$$
- Measured values across Aletheia halo catalogs:
  - BitVF: $B_{\mathrm{slope}} = 0.914 \pm 0.043$
  - REVOLVER: $B_{\mathrm{slope}} = 0.910 \pm 0.046$
  - Difference: $\Delta B_{\mathrm{slope}} = 0.004 \pm 0.063$ ($0.07\sigma$ tension $\implies$ fully universal).
- **Scale Invariance**: Stable above $R_{\min} \ge 16\,h^{-1}\mathrm{Mpc}$ ($< 1\%$ drift).

---

## 3. Statistical Goodness of Fit & Degrees of Freedom ($\nu$)

When computing $\chi^2/\nu$:
$$\chi^2 = \sum_{i=1}^{p} \sum_{j=1}^{p} (d_i - m_i) \left(C_{\mathrm{JK}}^{-1}\right)_{ij} (d_j - m_j)$$
where $C_{\mathrm{JK}}$ is the 64-region spatial Jackknife covariance matrix (with Hartlap correction factor if applicable: $(N_{\mathrm{sub}} - p - 2)/(N_{\mathrm{sub}} - 1)$):

1. **1D Calibration** ($b_{\mathrm{void}}$ fitted at fiducial cosmology):
   $$\nu = p - 1$$
2. **2D Barrier Calibration** ($B_{\mathrm{slope}}, B_{\mathrm{offset}}$):
   $$\nu = p - 2$$
3. **MCMC Parameter Inference** ($\Omega_m, \sigma_8, b_{\mathrm{void}}$):
   $$\nu = p - k \quad (k = 3)$$

See [physics_conventions.md](./references/physics_conventions.md) for fiducial simulation parameters and sample definitions.
