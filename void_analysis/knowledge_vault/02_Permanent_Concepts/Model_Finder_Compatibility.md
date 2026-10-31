---
tags: [concept, void-cosmology, epistemology, bitvf, revolver, vdn]
aliases: [Model-Finder Compatibility, Epistemology of Voids]
---

# 🧠 First-Principles Concept: Model-Finder Compatibility

> **Core Thesis**: Void finders do not find "pre-existing objects"; they algorithmically define them. Therefore, an apparent cosmological parameter bias (e.g. $\Omega_m \sim 0.45$ instead of $0.315$) is frequently an artifact of **theoretical model mismatch**, not bad data or a physical discovery.

---

## 1. First-Principles Mathematical Foundation

The theoretical Void Size Function (VSF) of Jennings et al. (2013) is formulated via excursion-set theory:
$$\frac{\mathrm{d}n}{\mathrm{d}\ln R} = \frac{f(\ln \sigma)}{V(R)} \left| \frac{\mathrm{d}\ln \sigma}{\mathrm{d}\ln R} \right|$$
where:
1. $V(R) = \frac{4\pi}{3} R^3$ is the **volume of an isolated sphere**.
2. $f(\ln \sigma)$ is the first-crossing distribution of an underdensity barrier $\delta_v$ with a collapse barrier $\delta_c = 1.686$.
3. The mapping between Lagrangian radius $r_L$ and Eulerian radius $R$ relies on **spherical mass conservation**:
   $$\rho_v R^3 = \bar{\rho} r_L^3 \implies R = r_L (1 + \Delta_v)^{-1/3}$$

---

## 2. The Algorithmic Divergence

```
           [Excursion Set Theory]
             (Spherical, isolated)
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
   [BitVF Finder]          [REVOLVER / ZOBOV]
• Inverts displacement    • Voronoi tessellation
• Spherical underdense    • Watershed basins
  Lagrangian basins       • Arbitrary irregular shapes
• Non-overlapping TOUCH   • Unmerged watershed valleys
         │                         │
         ▼                         ▼
  STRUCTURALLY MATCHED      INTRINSIC SHAPE MISMATCH
   chi2/nu ~ 0.9 - 1.7        chi2/nu ~ 3.1 - 8.1
   Unbiased (Omega_m, s8)     Biased Omega_m ~ 0.40 - 0.45
```

### Why BitVF Matches Vdn:
BitVF tracks particle trajectories back in time via the displacement field divergence:
$$\theta(\mathbf{x}) = \nabla \cdot \mathbf{\Psi}(\mathbf{x})$$
Underdense regions in Lagrangian space are naturally quasi-spherical perturbations that expand outward. When cleaned with `TOUCH` (zero sphere overlap), each void represents a distinct, non-overlapping Lagrangian volume that directly obeys Jennings' spherical mass conservation.

### Why REVOLVER Mismatches Vdn:
REVOLVER identifies voids using Voronoi tessellation and watershed basin identification. Unmerged watershed basins:
1. Have arbitrary, fractal, polyhedral boundaries with large filamentary indentations.
2. An effective spherical radius $R_{\mathrm{eff}} \equiv (3V_{\mathrm{voronoi}} / 4\pi)^{1/3}$ forces an irregular geometry into an equivalent sphere, skewing the derivative $|\mathrm{d}\ln\sigma / \mathrm{d}\ln R|$.
3. Produce a characteristic **S-curve residual tilt** against Vdn: overpredicting at $R \approx 20\,h^{-1}\mathrm{Mpc}$ ($+4\sigma$) and underpredicting at $R \approx 32\,h^{-1}\mathrm{Mpc}$ ($-3.5\sigma$).

---

## 3. The MCMC Mechanism: How Model Incompatibility Mimics Cosmology

When an MCMC sampler is asked to fit REVOLVER data using the Vdn model:
1. The sampler cannot alter the geometric shape of the Vdn curve without changing cosmological parameters.
2. To flatten the $+4\sigma$ excess at $R \sim 20$ and lift the $-3.5\sigma$ deficit at $R \sim 32$, the likelihood forces $\Omega_m$ upward ($\Omega_m \to 0.42\text{--}0.45$) and $\sigma_8$ downward.
3. **Conclusion for Committee & Papers**: The parameter bias in watershed voids is an artifact of forcing an incompatible spherical model onto watershed basins.

---

## 🔗 Related Notes
- [[02_Permanent_Concepts/Volume_Conservation_TOUCH_vs_CENTER|Volume Conservation (TOUCH vs CENTER)]]
- [[02_Permanent_Concepts/MCMC_Prior_Wall_Diagnostics|MCMC Prior-Wall Diagnostics]]
- [[01_Literature_Notes/Sartori2025_BitVF|Sartori et al. (2025) Notes]]
- [[01_Literature_Notes/Nadathur2019_REVOLVER|Nadathur et al. (2019) Notes]]
