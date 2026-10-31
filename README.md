# Cosmological Parameter Estimation and Universality from the Void Size Function
### Benchmarking Dynamical Reconstruction (BitVF) against Voronoi Watershed (REVOLVER) on Aletheia Simulations

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-green.svg)](https://isocpp.org/)
[![CosmoBolognaLib](https://img.shields.io/badge/CosmoBolognaLib-V5-orange.svg)](https://github.com/federicomarulli/CosmoBolognaLib)
[![Status](https://img.shields.io/badge/Status-Thesis%20Complete%20%7C%20Validated-brightgreen.svg)]()

**Author:** [Arun Mitra](https://github.com/stardust2443) (BITS Pilani)  
**Collaborators & Mentors:** Prof. Federico Marulli (Univ. of Bologna / INAF), Dr. Simone Sartori, Dr. Sofia Contarini, Dr. Tommaso Ronconi  
**Target Inquiries:** Prospective PhD Advisors & Committee Referees  

---

## 1. Executive Summary & Research Scope

Cosmic voids—underdense regions spanning tens of megaparsecs across the cosmic web—offer an exceptionally clean cosmological probe. Due to their low-density interiors, voids preserve linear and mildly non-linear gravitational physics and are minimally contaminated by complex baryonic processes, making them sensitive indicators of dark energy, neutrino masses, and modified gravity models.

This repository encapsulates the full end-to-end research pipeline and LaTeX thesis documentation for confronting theoretical **Void Size Function (VSF)** excursion-set models with high-resolution cosmological simulations. Specifically, this work benchmarks:
1. **The Back-In-Time Void Finder (BitVF)**: A state-of-the-art dynamical void finder utilizing Optimal Transport (OT) displacement field reconstruction to identify Lagrangian underdensities in the initial conditions.
2. **REVOLVER (ZOBOV Voronoi Watershed)**: The established geometric Voronoi watershed tessellation algorithm widely applied to galaxy redshift surveys (BOSS, DESI, eBOSS).
3. **Jennings et al. (2013) Theoretical Model ($V_{\mathrm{dn}}$)**: A volume-conserving excursion-set formulation implemented within **CosmoBolognaLib (CBL)**.
4. **Contarini et al. (2019, 2021) Linear Void Barrier Framework**: Parameterizing the response of cosmic voids to biased matter tracers:
   $$\delta_{v,\mathrm{eff}} = B_{\mathrm{slope}}\,b_{\mathrm{eff}} + B_{\mathrm{offset}}$$

---

## 2. Key Scientific Discoveries

### A. Universality of the Linear Void Barrier Slope ($B_{\mathrm{slope}} \sim 0.91$)
Across four halo mass threshold cuts ($100P, 75P, 50P, 30P$) on the $1000\,h^{-1}\mathrm{Mpc}$ Aletheia simulation box under identical 64-subvolume spatial Jackknife covariances with Hartlap corrections:
* **BitVF Linear Fit:** $b_{\mathrm{void}} = (0.914 \pm 0.043)\,b_{\mathrm{eff}} + (0.589 \pm 0.040)$ [$R^2 = 0.995$]
* **REVOLVER Linear Fit:** $b_{\mathrm{void}} = (0.910 \pm 0.046)\,b_{\mathrm{eff}} + (0.582 \pm 0.042)$ [$R^2 = 0.995$]
* **Slope Concordance:** $\Delta B_{\mathrm{slope}} = -0.005 \pm 0.063$ (**$0.07\sigma$** difference).
* **Significance:** Demonstrates for the first time that the linear response of cosmic voids to tracer bias is a universal feature of the excursion-set barrier mechanism, holding invariant regardless of whether voids are detected via dynamical Lagrangian reconstruction or Voronoi watershed boundaries.

### B. Precision Cosmological Parameter Recovery
* Marginalized posterior constraints on the amplitude of matter fluctuations recovered $\sigma_8$ to **$1.2\%$ precision**:
  $$\sigma_8 = 0.8208 \pm 0.009 \quad (\text{within } 1.09\sigma \text{ of true } \sigma_8 = 0.8110)$$
* Solved the multi-tracer parameter degeneracy using calibrated informative priors, validating the statistical reproducibility across multiple tracer density regimes.

### C. Theoretical Model-Finder Compatibility Diagnosis
Diagnosing theoretical compatibility between the Jennings $V_{\mathrm{dn}}$ model and observational void finders resolved an ongoing literature anomaly:
* **BitVF yields excellent physical fits** ($\chi^2/\nu \approx 0.90 - 1.67$) with Gaussian, uncorrelated residual pulls across all scale regimes.
* **REVOLVER exhibits a systematic S-curve residual tilt** ($\chi^2/\nu \approx 3.14 - 8.12$), driven by unmerged Voronoi watershed basins over-partitioning large voids. In unconstrained MCMC, this tilts parameter posteriors along the $\Omega_m - \sigma_8$ degeneracy ridge toward unphysical values.

---

## 3. Repository Architecture

```
.
├── void_analysis/
│   ├── main_thesis_docs/     # Complete PhD/Master's thesis LaTeX source, chapters, bibtex
│   │   ├── chapters/         # Modular chapters (Introduction, VSF Theory, Simulations, MCMC)
│   │   ├── figures/          # High-resolution publication-quality PNG figures
│   │   ├── main.tex          # Master LaTeX document
│   │   └── references.bib    # Curated cosmological literature database
│   │
│   ├── scripts/              # Production cosmological analysis pipeline
│   │   ├── clean_voids*.py   # Sphere cleaning algorithms (TOUCH vs CENTER geometry)
│   │   ├── jackknife_64_cov.py # 64-region spatial Jackknife covariance engine
│   │   ├── cbl_mcmc_*.py     # CosmoBolognaLib MCMC cosmological inference drivers
│   │   ├── cbl_sampler_shim.cpp # Native C++ sampler interface for CBL
│   │   ├── calibrate_barrier_*.py # Linear barrier calibration routines
│   │   └── thesis_figure_*.py # Publication figure generation scripts
│   │
│   ├── configs/              # Cosmological parameters & run configurations
│   ├── plots/                # Generated corner plots, residual pull panels & MCMC chains
│   ├── academic_cv/          # Academic Curriculum Vitae (LaTeX source)
│   ├── handoff.md            # Computational session logs & milestone tracking
│   └── MASTER_PROJECT_CONTEXT.md # Comprehensive living source of truth
│
├── aletheia_bitvf.ini        # BitVF configuration for Aletheia simulation boxes
├── first_cosmo.cpp           # CBL cosmology initialization & growth factor diagnostics
└── .gitignore                # Optimized ignore rules for large N-body simulation data
```

---

## 4. Methodological Highlights & Code Standards

- **Strict Spatial Independence**: Uses 64-region subvolume Jackknife resampling with unbiased Hartlap covariance scaling factors.
- **Physical Boundary Verification**: Implements isolated sphere overlap filtering (`TOUCH` criteria) to ensure volume-conserving correspondence with the Jennings $V_{\mathrm{dn}}$ spherical excursion-set barrier.
- **Publication-Ready Visualization**: Follows scientific styling standards with residual pull sub-panels ($[y_{\mathrm{obs}} - y_{\mathrm{model}}] / \sigma$), parameter contours, and LaTeX typography.
- **LaTeX Source Integrity**: Complete thesis LaTeX suite compiles with 0 errors, full cross-referencing, and standard astronomical citation keys (`\citep`, `\citet`).

---

## 5. Branch Structure

This repository maintains a clean branching model tailored for academic review and ongoing research:
* `main`: Stable, peer-review ready branch containing verified pipelines, final thesis documents, and calibrated results.
* `develop`: Active integration branch for computational experiments, new void catalog tests, and pipeline extensions.
* `thesis`: Dedicated branch for thesis LaTeX source updates, chapter writing, and presentation materials.

---

## 6. Contact & Academic Profile

* **Researcher:** Arun Mitra
* **Institutional Email:** [f20212443@hyderabad.bits-pilani.ac.in](mailto:f20212443@hyderabad.bits-pilani.ac.in)
* **Personal Email:** [arunmitra.k@gmail.com](mailto:arunmitra.k@gmail.com)
* **GitHub:** [@stardust2443](https://github.com/stardust2443)
