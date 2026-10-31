# Arun Mitra Kommireddy
**Large-Scale Structure Cosmology · Cosmic Voids · Computational Astrophysics**

- **Email:** [arunmitra.k@gmail.com](mailto:arunmitra.k@gmail.com) | [f20212443@hyderabad.bits-pilani.ac.in](mailto:f20212443@hyderabad.bits-pilani.ac.in)
- **Location:** Hyderabad, India
- **GitHub:** [github.com/stardust2443](https://github.com/stardust2443)
- **LinkedIn:** [linkedin.com/in/arun-mitra](https://linkedin.com/in/arun-mitra)
- **Thesis Codebase:** [github.com/stardust2443/cosmology_thesis](https://github.com/stardust2443/cosmology_thesis)

---

## 🎯 Research Interests
- **Large-Scale Structure Cosmology:** Cosmic void statistics, Void Size Functions (VSF), and excursion-set theoretical modeling.
- **Algorithmic Void Identification:** Dynamical displacement field reconstruction (`BitVF`) vs. Voronoi watershed algorithms (`REVOLVER` / `ZOBOV`).
- **Early Universe & Relics:** Primordial Black Holes (PBHs), critical collapse overdensity thresholds, and cosmological perturbation growth.
- **Statistical Inference:** Bayesian parameter recovery ($\Omega_m, \sigma_8, b_{\mathrm{void}}$), Markov Chain Monte Carlo (MCMC), and spatial Jackknife covariance matrices with Anderson-Hartlap unbiasing.
- **Survey Science:** Preparing cosmic void observables for Stage-IV dark energy probes (Euclid, DESI).

---

## 🎓 Education

### Integrated Dual Degree: M.Sc. (Hons.) Physics & B.E. (Hons.) Mechanical Engineering
**Birla Institute of Technology and Science (BITS), Pilani**, Hyderabad Campus, India  
*August 2021 – July 2026*
- **Admission Selectivity:** Admitted through nationwide BITSAT examination (<2% acceptance rate among ~200,000 candidates).
- **Curriculum:** 5-year rigorous dual-degree curriculum combining theoretical/computational physics with advanced applied mathematics and continuum mechanics.
- **Relevant Coursework:** General Relativity & Cosmology, Classical Field Theory, Quantum Mechanics I & II, Statistical Mechanics, Mathematical Methods of Physics, Computational Physics, Fluid Dynamics, Numerical Optimization, Probability & Applied Statistics.

---

## 🔬 Research Experience

### Off-Campus Master's Research Thesis — Cosmic Void Statistics & Parameter Inference
*September 2025 – June 2026*  
- **External Thesis Advisors:** Prof. Federico Marulli (University of Bologna / INAF), Dr. Simone Sartori (INAF-IASF Milano).  
- **BITS Pilani Co-Supervisors:** Prof. Sashideep Gutti, Prof. Ram Chandra Murthy Kalluri.
- **Excursion-Set Void Modeling on $N$-body Simulations:** Evaluated the Jennings et al. (2013) Volume-Conserving (Vdn) theoretical model on the $(1\,h^{-1}\mathrm{Gpc})^3$ Aletheia simulation suite across four mass-thresholded halo catalogues (100P, 75P, 50P, 30P; effective linear bias $b_{\mathrm{eff}} \in [0.86, 1.02]$).
- **Discovery of Linear Void Barrier Universality ($B_{\mathrm{slope}} \sim 0.9$):** Discovered that linear barrier scaling ($b_{\mathrm{void}} = B_{\mathrm{slope}} b_{\mathrm{eff}} + B_{\mathrm{offset}}$) matches between dynamical Optimal Transport reconstruction (`BitVF`) and Voronoi watershed (`REVOLVER`) to an unprecedented **$0.07\sigma$** ($B_{\mathrm{slope}}^{\mathrm{BitVF}} = 0.914 \pm 0.043$ vs. $B_{\mathrm{slope}}^{\mathrm{REVOLVER}} = 0.910 \pm 0.046$). Proved that linear barrier response to tracer bias is a universal property of excursion-set void cosmology rather than an algorithm-specific artifact.
- **Model-Finder Compatibility & MCMC Bias Resolution:** Proved that the idealized spherical Vdn excursion-set model is structurally compatible with dynamical Lagrangian reconstruction (`BitVF`, $\chi^2/\nu \approx 0.90\text{--}1.67$), but fails for Voronoi watershed finders (`REVOLVER`, $\chi^2/\nu \approx 3.14\text{--}8.12$). Demonstrated that unmerged watershed basins over-partition large voids, creating a systematic residual slope tilt that drives Bayesian MCMC samplers toward unphysical cosmological parameters ($\Omega_m \sim 0.44$).
- **Bayesian Parameter Inference Pipeline:** Developed complete Python/C++ MCMC sampling framework using `CosmoBolognaLib` (CBL) and `emcee`. Calibrated informative Gaussian priors from simulation mocks, successfully recovering primordial amplitude $\sigma_8$ to **$1.2\%$ precision** ($\bar{\sigma}_8 = 0.8208 \pm 0.009$ for BitVF vs. fiducial $0.8110$).
- **Robustness Audits & Volume Conservation Prerequisite:** Constructed 64-region spatial Jackknife covariance matrices debiased via the Anderson-Hartlap factor. Demonstrated that Vdn modeling mathematically requires non-overlapping spherical voids (`TOUCH`), where concentric center overlaps (`CENTER`) double-count volume and artificially inflate $\chi^2/\nu$ to $\sim 25$.

### Research Projects — Primordial Black Holes (PBHs) & Early Universe Cosmology
**Department of Physics, BITS Pilani** | *2024 – 2025*
- **PBH Formation Mechanisms & Horizon Re-entry:** Investigated the formation of Primordial Black Holes from large-amplitude cosmological density perturbations re-entering the Hubble horizon during radiation domination. Studied the critical collapse overdensity threshold $\delta_c$ and mass fraction $\beta(M)$.
- **Dark Matter Fraction & Observational Windows:** Explored current observational limits on $f_{\mathrm{PBH}} = \Omega_{\mathrm{PBH}}/\Omega_{\mathrm{DM}}$ across different mass regimes, including Hawking evaporation bounds (low mass), gravitational microlensing (intermediate mass), and accretion limits from CMB anisotropies.
- **Linear Perturbation Theory & Growth Factors:** Analyzed the evolution of scalar cosmological perturbations, Jeans mass scales, and power spectrum calculations using numerical Boltzmann solvers (CAMB).

---

## 📄 Publications & Scientific Manuscripts

1. **A. Mitra**, S. Sartori, F. Marulli, S. Contarini, T. Ronconi, and L. Moscardini (2026).  
   *“Universality of the Linear Void Barrier and Model-Finder Compatibility in Excursion-Set Void Cosmology: Benchmarking Dynamical Reconstruction against Voronoi Watershed on Aletheia Simulations”*.  
   *Manuscript in preparation for preprint / journal submission based on Master's thesis discoveries.*
2. **A. Mitra Kommireddy** (2026).  
   *“Study of Cosmic Void Size Function using Computational Methods: Validation with Simulated Catalogs, Bayesian Parameter Estimation, and an Audit of Overlap-Based Degeneracy Breaking”*.  
   Master's Research Thesis (BITS F422T, 16 units), BITS Pilani. Supervised by Prof. F. Marulli & Dr. S. Sartori.

---

## 🛠️ Technical & Computational Skills

| Category | Competencies |
| :--- | :--- |
| **Cosmology Software** | **CosmoBolognaLib (CBL)** (C++ routines and Python numerical bindings), **BitVF** (Optimal Transport displacement reconstruction), **REVOLVER** / **ZOBOV** (Voronoi watershed), **CAMB**. |
| **Statistical Inference** | **emcee**, **GetDist**, **corner**, Markov Chain Monte Carlo (MCMC), spatial Jackknife covariance matrices, Anderson-Hartlap precision unbiasing, informative Gaussian priors, $\chi^2/\nu$ goodness-of-fit. |
| **Programming Languages** | **Python** (NumPy, SciPy, Matplotlib, Astropy, Pandas, multiprocessing), **C/C++** (scientific compilation, shared libraries, OpenMP), **Shell** (Bash/Zsh). |
| **Data & HPC Workflows** | Gigaparsec $N$-body simulations ($1024^3$ particles), halo and void catalog processing, geometric sphere cleaning, FITS, HDF5, Git/GitHub, Linux/macOS Unix environments. |
| **Scientific Typesetting** | LaTeX, BibTeX, Overleaf, publication-grade vector graphics (Matplotlib). |

---

## 🏆 Scholarships & Academic Recognition

- **Merit-cum-Need Scholarship:** Birla Institute of Technology and Science, Pilani. Awarded for **5 academic semesters** across the dual degree program.
- **National Admission Selectivity:** Top percentile rank in nationwide BITSAT examination (<2% acceptance rate among ~200,000 candidates).

---

## 👥 Academic References

1. **Prof. Federico Marulli** *(Primary Thesis Advisor)*  
   Associate Professor of Astrophysics & Cosmology  
   Department of Physics and Astronomy (DIFA), Alma Mater Studiorum – Università di Bologna; INAF - OAS Bologna, Italy  
   **Email:** [federico.marulli3@unibo.it](mailto:federico.marulli3@unibo.it)  
   *Focus: Large-Scale Structure, Galaxy Clustering, Cosmic Voids, CosmoBolognaLib*

2. **Dr. Simone Sartori** *(Thesis Co-Advisor & BitVF Author)*  
   Researcher / Staff Astrophysicist  
   INAF – Istituto di Astrofisica Spaziale e Fisica Cosmica (IASF), Milano, Italy  
   **Email:** [simone.sartori@inaf.it](mailto:simone.sartori@inaf.it)  
   *Focus: BitVF Algorithm, Dynamical Void Reconstruction, CMB Lensing Cross-Correlations*

3. **Prof. Sashideep Gutti** *(BITS Pilani Co-Supervisor)*  
   Professor of Physics  
   Department of Physics, BITS Pilani, Hyderabad Campus, India  
   **Email:** [asashideep@hyderabad.bits-pilani.ac.in](mailto:asashideep@hyderabad.bits-pilani.ac.in)  
   *Focus: Gravitational Physics, Black Hole Physics, Theoretical Physics*
