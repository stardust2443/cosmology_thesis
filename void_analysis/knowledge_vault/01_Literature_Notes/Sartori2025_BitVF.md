---
title: "BitVF: Back-in-time Void Finder via Displacement Divergence"
authors: [Simone Sartori, et al.]
year: 2025
tags: [paper, bitvf, void-finder, lagrangian, displacement]
---

# 📖 Literature Note: Sartori et al. (2025) — BitVF

## 1. Executive Summary
Sartori et al. introduce **BitVF**, a cosmic void finder that uses the divergence of the displacement field $\mathbf{\Psi}(\mathbf{x})$ obtained via dynamical reconstruction to track particles back to their Lagrangian initial positions, identifying underdense basins that expand into Eulerian voids.

## 2. Core Mathematical Formulation
1. **Displacement Field**:
   $$\mathbf{x} = \mathbf{q} + \mathbf{\Psi}(\mathbf{q})$$
   where $\mathbf{q}$ is the Lagrangian coordinate and $\mathbf{x}$ is the Eulerian position.
2. **Divergence Field**:
   $$\theta(\mathbf{x}) = \nabla \cdot \mathbf{\Psi}(\mathbf{x})$$
   In linear perturbation theory, $\theta(\mathbf{x}) \approx -\delta(\mathbf{x}) / f(\Omega_m)$.
3. **Void Identification**:
   Regions where $\theta(\mathbf{x}) > \theta_{\mathrm{threshold}}$ correspond to diverging displacement fields (matter rushing outward), defining proto-voids.
4. **Volume Filtering**:
   - `TOUCH`: Non-overlapping spheres.
   - `CENTER`: Concentric sphere pruning.

## 3. Assumptions (Stated & Unstated)
- *Stated*: Lagrangian proto-voids are approximately spherical and evolve into isolated Eulerian voids.
- *Unstated*: When mapped to Eulerian space, if voids are allowed to overlap (e.g. `CENTER`), the total volume exceeds the box volume, conflicting with excursion-set normalization.

## 4. Points of Contact with Our Thesis
- **Phase 1 Validation**: We successfully replicated Simone Sartori's ForArun reference VSF to within $1\%\text{--}7\%$ precision across dark matter downsamplings.
- **Phase 3 Discovery**: We proved that BitVF is structurally compatible with the Jennings et al. Vdn model ($\chi^2/\nu = 0.90\text{--}1.67$), confirming the physical validity of Sartori's displacement-based approach over watershed basin methods.

## 🔗 Related Notes
- [[02_Permanent_Concepts/Model_Finder_Compatibility|Model-Finder Compatibility]]
- [[02_Permanent_Concepts/Volume_Conservation_TOUCH_vs_CENTER|Volume Conservation (TOUCH vs CENTER)]]
- [[03_Defense_Dossiers/Committee_Profile_Sartori|Committee Profile: Sartori]]
