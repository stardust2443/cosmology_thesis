# 🌌 Master PhD Research Statement

**Applicant**: Arun Mitra  
**Institution**: University of Bologna (Alma Mater Studiorum), Department of Physics and Astronomy (DIFA)  
**Supervisor**: Prof. Federico Marulli  
**Field**: Large-Scale Structure Cosmology, Cosmic Voids, Galaxy Surveys (Euclid, DESI), and Computational Astrophysics

---

## 1. Executive Summary & Research Trajectory

My research centers on extracting precision cosmological information from the underdense Large-Scale Structure of the Universe—**cosmic voids**. Under the supervision of Prof. Federico Marulli at the University of Bologna, my Master's thesis has tackled one of the foundational challenges in void statistics: **the theoretical compatibility between algorithmic void finders and excursion-set analytical models**.

By systematically benchmarking the **BitVF** dynamical back-in-time void finder (Sartori 2025) against the standard **REVOLVER** Voronoi watershed algorithm (Nadathur 2019) across the $(1\,h^{-1}\mathrm{Gpc})^3$ **Aletheia** $N$-body simulation suite, my work has delivered three key results:
1. **Model-Finder Compatibility Established**: We proved that the excursion-set Void Size Function (VSF) of Jennings et al. (2013) is structurally compatible with dynamical Lagrangian reconstruction (BitVF, $\chi^2/\nu \approx 0.90\text{--}1.67$), but fundamentally incompatible with unmerged Voronoi watershed basins (REVOLVER, $\chi^2/\nu \approx 3.14\text{--}8.12$). We demonstrated that REVOLVER's apparent MCMC parameter bias ($\Omega_m \sim 0.44$) is not a physical detection, but a mathematical artifact of the watershed boundary mismatch driving the likelihood against prior boundaries.
2. **Universality of the Linear Void Barrier**: We discovered that despite their radical topological divergence, both finders obey an identical linear void barrier scaling:
   $$b_{\mathrm{void}} = B_{\mathrm{slope}} b_{\mathrm{eff}} + B_{\mathrm{offset}}$$
   with slopes agreeing to **$0.07\sigma$ precision** ($B_{\mathrm{slope}}^{\mathrm{BitVF}} = 0.914 \pm 0.043$ vs. $B_{\mathrm{slope}}^{\mathrm{REVOLVER}} = 0.910 \pm 0.046$).
3. **Volume-Conservation Prerequisite**: Through a 5-part robustness audit, we demonstrated that Jennings et al. Vdn modeling mathematically requires non-overlapping spherical voids (`TOUCH`), where overlapping spheres (`CENTER`) double-count volume and explode $\chi^2/\nu$ to $\sim 25$.

---

## 2. Technical & Computational Toolkit

- **Cosmological Statistics & Software**: Extensive development with **CosmoBolognaLib (CBL)** (C++ and Python bindings), Jackknife spatial covariances, and Hartlap precision unbiasing.
- **Inference & Sampling**: Advanced MCMC parameter recovery using **`emcee`**, **`getdist`**, and **`corner`**, diagnosing autocorrelation times ($\tau$), chain mixing ($\hat{R}$), and prior-wall boundary mechanics.
- **Simulations & Big Data**: Handling gigaparsec $N$-body simulation fields ($1024^3$ particles), dark matter downsamplings, and mass-thresholded halo catalogs (100P, 75P, 50P, 30P).
- **Agentic & AI-Augmented Research**: Pioneer in leveraging agentic development environments (Google Antigravity, Consensus, NotebookLM) to accelerate computational physics, pipeline verification, and literature synthesis.

---

## 3. Prospective PhD Research Vision

During my doctoral studies, I seek to bridge the gap between simulation benchmarks and the first wave of **Euclid** and **DESI** data releases:

1. **Scalable Dynamical Void Finding on Survey Lightcones**:
   - Extend the BitVF dynamical displacement reconstruction from periodic simulation boxes to realistic survey geometries, incorporating spectroscopic selection functions, redshift-space distortions (RSD), and angular masks.
   - Benchmark against standard Voronoi finders on the Euclid Flagship mock catalogs.
2. **Multi-Tracer Barrier Priors**:
   - Utilize our discovered linear barrier universality ($B_{\mathrm{slope}} \approx 0.91$) to formulate tight multi-tracer priors across galaxy color and stellar mass bins, breaking the classic degeneracy between $\Omega_m$, $\sigma_8$, and void bias.
3. **Joint VSF + Void-Galaxy Clustering Likelihoods**:
   - Construct a unified Bayesian inference framework combining the 1-point Void Size Function with the 2-point void-galaxy cross-correlation function $\xi_{\mathrm{vg}}(r)$ to maximize constraints on the dark energy equation of state $w(z)$ and test modified gravity theories ($f(R)$, DGP) in cosmic underdensities.

I am eager to contribute this combination of computational rigor, simulation experience, and theoretical modeling to a dynamic European research group.
