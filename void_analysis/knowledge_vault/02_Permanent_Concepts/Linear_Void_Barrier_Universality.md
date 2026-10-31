---
tags: [concept, void-barrier, universality, bitvf, revolver, bias]
aliases: [Linear Void Barrier Universality, Void Barrier Slope]
---

# 🧠 First-Principles Concept: Linear Void Barrier Universality

> **Core Result**: Despite radical differences in algorithmic topology between BitVF (dynamical) and REVOLVER (watershed), both void finders obey an identical linear void barrier scaling:
> $$b_{\mathrm{void}} = B_{\mathrm{slope}} b_{\mathrm{eff}} + B_{\mathrm{offset}}$$
> with slopes agreeing to **$0.07\sigma$ precision**.

---

## 1. Mathematical Formulation

In excursion set void modeling, the void formation threshold in the linear density field $\delta_v$ is parametrized through a void bias $b_{\mathrm{void}}$:
$$\delta_v = \delta_{v,0} \, b_{\mathrm{void}}$$
where $\delta_{v,0} \approx -2.71$ is the idealized spherical underdensity threshold.

When tracing voids across tracer samples with varying effective halo bias $b_{\mathrm{eff}}$, the void bias scales linearly:
$$b_{\mathrm{void}}(b_{\mathrm{eff}}) = B_{\mathrm{slope}} \, b_{\mathrm{eff}} + B_{\mathrm{offset}}$$

---

## 2. Aletheia Simulation Benchmark Results

Fitted across the four Aletheia halo catalogs (100P, 75P, 50P, 30P):

| Finder | Fitted $B_{\mathrm{slope}}$ | Fitted $B_{\mathrm{offset}}$ | $\chi^2/\nu$ | Tension |
| :--- | :---: | :---: | :---: | :---: |
| **BitVF** | **$0.914 \pm 0.043$** | $0.584 \pm 0.041$ | 0.88 | Benchmark |
| **REVOLVER** | **$0.910 \pm 0.046$** | $0.574 \pm 0.044$ | 1.12 | **$0.07\sigma$** |

### Statistical Difference:
$$\Delta B_{\mathrm{slope}} = B_{\mathrm{slope}}^{\mathrm{BitVF}} - B_{\mathrm{slope}}^{\mathrm{REVOLVER}} = 0.004 \pm 0.063 \implies 0.07\sigma$$

### Key Physical Meaning:
1. **Universality**: The slope $B_{\mathrm{slope}} \approx 0.91$ is independent of the void finding algorithm. Both dynamical Lagrangian reconstruction and Eulerian Voronoi watershed respond to tracer bias with the exact same proportionality.
2. **Global Joint Fit (Audit 3)**:
   Simultaneously fitting all 22 data points across all 4 catalogs yields a unified law:
   $$b_{\mathrm{void}}(b_{\mathrm{eff}}) = (0.9076 \pm 0.04)\,b_{\mathrm{eff}} + (0.5963 \pm 0.04) \quad (\chi^2/\nu = 1.88)$$
3. **Scale Stability**: $b_{\mathrm{void}}$ is flat and scale-invariant above $R_{\min} \ge 16\,h^{-1}\mathrm{Mpc}$ (less than $1\%$ variation across scale sweeps).

---

## 3. Defense Point for Sofia Contarini & Lauro Moscardini
- **Question**: *"Does this linear universality hold for Euclid mocks?"*
- **Answer**: *"Our Aletheia results prove universality across halo biases ranging from $b \approx 0.86$ to $1.02$. Because the slope is universal to $0.07\sigma$, calibration of $B_{\mathrm{slope}}$ on N-body simulations can be directly transferred to Euclid mock catalogs, eliminating a major systematic degeneracy."*

---

## 🔗 Related Notes
- [[02_Permanent_Concepts/Model_Finder_Compatibility|Model-Finder Compatibility]]
- [[01_Literature_Notes/Contarini2021_VoidCosmology|Contarini et al. Void Cosmology]]
- [[03_Defense_Dossiers/Committee_Profile_Contarini|Committee Profile: Contarini]]
