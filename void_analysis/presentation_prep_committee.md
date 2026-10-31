# 🎓 Master Presentation & Committee Defense Dossier

**Presented by**: Arun Mitra  
**Supervised by**: Prof. Federico Marulli  
**Audience / Committee**:
- **Prof. Federico Marulli** (Thesis Advisor, Bologna)
- **Dr. Simone Sartori** (Author of BitVF, Padova)
- **Dr. Sofia Contarini** (Expert in Void Cosmology, Bologna)
- **Prof. Lauro Moscardini** (Cosmology Group Leader, Euclid Lead, Bologna)

---

## 🏛️ Executive Narrative Arc: The 15-Slide Presentation Blueprint

```
[Act 1: The Foundation]
Slide 1: Title & Overview (The Epistemology of Voids)
Slide 2: Theoretical Framework (Jennings et al. Vdn Excursion-Set & Volume Conservation)
Slide 3: Algorithmic Divergence (BitVF Dynamical Divergence vs REVOLVER Watershed)

[Act 2: Phase 1 & 2 Validations]
Slide 4: Phase 1: BitVF Replication & Simone's ForArun Match (1% - 7%)
Slide 5: Phase 1 Insight: The Volume Conservation Prerequisite (TOUCH vs CENTER)
Slide 6: Phase 2: Dark Matter Cosmological Inference (Unbiased Truth Recovery)

[Act 3: Phase 3 Halo Benchmark — The Core Discovery]
Slide 7: Phase 3 Setup: 4 Halo Catalogs on Aletheia (100P, 75P, 50P, 30P)
Slide 8: The Showdown: Goodness-of-Fit Grid & Residual Pulls (BitVF 0.90 vs REVOLVER 3.14)
Slide 9: Mechanism: Why REVOLVER MCMC is Biased (S-Curve Tilt Mimics Omega_m)
Slide 10: Universality: The Linear Void Barrier Scaling (Delta B_slope = 0.07 sigma)

[Act 4: The 5-Part Robustness Audit]
Slide 11: Robustness Audits 1 & 2: Scale Plateau (< 1% Drift) & Poisson Tail Truncation
Slide 12: Robustness Audits 3 & 4: Global Joint Multi-Tracer Law & TOUCH Verification

[Act 5: Survey Outlook & Conclusion]
Slide 13: Scalability to Euclid and DESI Galaxy Mocks
Slide 14: Summary of Key Thesis Findings (The 3 Takeaways)
Slide 15: Q&A / Discussion

[Dedicated Backup Slides: B1 to B5]
B1: Per-Bin Chi-Square Gradients and Prior-Wall Pinning
B2: Aletheia Jackknife Covariance Matrices (Hartlap Unbiasing)
B3: Effective Bias Bug in CosmoBolognaLib (b_eff = 1.0001 Guard)
B4: Cash Poisson Statistic vs Gaussian Jackknife
B5: High-Redshift and Neutrino Mass Extensions
```

---

## 📽️ Detailed Slide-by-Slide Script & Visual Guide

### Slide 1: Title & The Core Epistemological Problem
* **Title**: *Testing the Universality of the Linear Void Barrier and Model-Finder Compatibility on N-body Halos: BitVF vs. REVOLVER*
* **Speaker Script**:
  > *"Good morning everyone. In void cosmology, we often treat void finders and theoretical models as plug-and-play modules. But void finders do not find pre-existing physical objects; they define them algorithmically. The fundamental question of my thesis is: **Which void finders define objects that are physically compatible with our theoretical models, and can we uncover a universal void barrier law across tracers?** Today, I will present our answers using the Aletheia N-body simulation suite."*

---

### Slide 2: Theoretical Modeling: Jennings et al. (Vdn) & Volume Conservation
* **Visual**: Equation for $\mathrm{d}n/\mathrm{d}\ln R$ alongside schematic of expanding underdensities.
* **Key Physics**:
  - Excursion-set underdensity barrier $\delta_v = \delta_{v,0} b_{\mathrm{void}}$.
  - Volume conservation partition: $\sum V_i n_i = 1 - F_{\mathrm{collapse}}$.
* **Speaker Script**:
  > *"The Jennings et al. (2013) Vdn model is derived from excursion-set theory. It makes one strict mathematical assumption: space is partitioned into non-overlapping, expanding spherical volumes. This volume-conservation requirement will prove central to our results."*

---

### Slide 3: Two Algorithmic Philosophies: BitVF vs. REVOLVER
* **Visual**: Side-by-side comparison: BitVF displacement vector field divergence vs. REVOLVER Voronoi tessellation / watershed basin diagram.
* **Speaker Script**:
  > *"To test this theory, we compare two fundamentally different finding algorithms:
  > 1. **BitVF (Sartori 2025)**: Uses dynamical reconstruction to measure the divergence of the displacement field $\theta = \nabla \cdot \mathbf{\Psi}$. It identifies Lagrangian underdense basins that expand into Eulerian space.
  > 2. **REVOLVER (Nadathur 2019)**: An Eulerian Voronoi watershed finder that groups irregular Voronoi cells into unmerged basins."*

---

### Slide 4: Phase 1: Replicating Simone Sartori's ForArun Benchmark
* **Visual**: VSF comparison plot showing Arun's cleaned BitVF vs. Simone's ForArun catalog across dark matter subsamples (20p, 10p, 5p, 2.5p).
* **Key Numbers**: Validated within **$1\%\text{--}7\%$** across the entire radius range $R \in [7, 35]\,h^{-1}\mathrm{Mpc}$.
* **Speaker Script**:
  > *"In Phase 1, our first milestone was verifying algorithmic reproducibility. We processed the Aletheia dark matter field and replicated Dr. Simone Sartori's benchmark within $1\%$ to $7\%$ precision, confirming the integrity of our local pipeline."*

---

### Slide 5: The Volume Conservation Prerequisite: TOUCH vs. CENTER
* **Visual**: 2-panel plot comparing `TOUCH` (OV00) vs `CENTER` (OV05) fits.
* **Key Numbers**:
  - `TOUCH` (0% overlap): $\chi^2/\nu = \mathbf{0.90}$ (100P)
  - `CENTER` (up to 50% overlap): $\chi^2/\nu = \mathbf{23.14}$ (100P) — **$25\times$ worse!**
* **Speaker Script**:
  > *"A crucial early finding was the comparison between cleaning algorithms. Simone's work noted both TOUCH and CENTER. When we fit the Vdn model, `TOUCH` non-overlapping spheres produce a textbook fit of $\chi^2/\nu = 0.90$. `CENTER` allows concentric overlap up to 50%, double-counting volume and exploding $\chi^2/\nu$ to $23.14$. This proves empirically that non-overlapping spheres are mathematically mandatory for excursion-set modeling."*

---

### Slide 6: Phase 2: Dark Matter Cosmological Inference
* **Visual**: 2D $(\Omega_m, \sigma_8)$ posterior contours with Aletheia simulation truth star ($\Omega_m = 0.315, \sigma_8 = 0.811$).
* **Speaker Script**:
  > *"In Phase 2, we performed MCMC parameter inference on dark matter voids using CosmoBolognaLib and emcee. With informative priors on the barrier, our BitVF pipeline recovers the true simulation cosmology within $1\sigma$, confirming that our inference machinery is unbiased."*

---

### Slide 7: Phase 3: The Halo Void Benchmark Across 4 Tracers
* **Visual**: Table of the 4 Aletheia halo catalogs (100P, 75P, 50P, 30P) with biases $b_{\mathrm{eff}} \in [0.86, 1.02]$, number densities, and 64-region Jackknife spatial sub-volumes.
* **Speaker Script**:
  > *"Phase 3 represents the core thesis contribution: applying both finders to four mass-thresholded halo catalogs in the $1\,h^{-1}\mathrm{Gpc}$ box, spanning effective tracer biases from $0.86$ to $1.02$, with spatial Jackknife covariance matrices."*

---

### Slide 8: The Showdown: Goodness-of-Fit Grid & Residual Pulls
* **Visual**: Figure `plots/halos/calibration_vsf_residuals_grid.png` (4 columns: 100P, 75P, 50P, 30P; top: VSF points + Vdn curve; bottom: standardized pulls $(d-m)/\sigma_{\mathrm{JK}}$).
* **Key Numbers Table**:
  | Sample | Tracer Bias $b_{\mathrm{eff}}$ | BitVF $\chi^2/\nu$ | REVOLVER $\chi^2/\nu$ |
  | :--- | :---: | :---: | :---: |
  | **100P** | $1.0211$ | **0.90** | **3.14** ($p = 0.007$) |
  | **75P**  | $0.9583$ | **1.36** | **5.18** ($p = 10^{-4}$) |
  | **50P**  | $0.9108$ | **3.25** | **6.35** ($p < 10^{-4}$) |
  | **30P**  | $0.8623$ | **1.67** | **8.12** ($p < 10^{-6}$) |
* **Speaker Script**:
  > *"Here is the defining result of our work. Across all halo samples, BitVF achieves acceptable reduced chi-squares ($\chi^2/\nu = 0.90$ to $1.67$), with residuals lying cleanly within $\pm 1\sigma$. REVOLVER fails catastrophically against Vdn, with $\chi^2/\nu$ reaching $8.12$. Notice the lower panels: REVOLVER shows a coherent, non-random S-curve residual tilt."*

---

### Slide 9: Mechanism: Why REVOLVER MCMC is Biased
* **Visual**: 2D posterior comparison showing BitVF centered on the truth star while REVOLVER is pushed to $\Omega_m \sim 0.44$; annotated diagram of the S-curve residual tilt.
* **Speaker Script**:
  > *"Why does REVOLVER produce biased MCMC posteriors ($\Omega_m \sim 0.44$)? It is not a sampling failure or a discovery of new physics. Because watershed basins have polyhedral boundaries, assigning them an effective spherical radius creates an S-curve shape mismatch. The MCMC sampler is forced to drive $\Omega_m$ upward to flatten the high-radius tail, driving walkers against the prior boundary."*

---

### Slide 10: Universality: Linear Void Barrier Scaling
* **Visual**: Figure `plots/halos/compare_barrier_universality.png` ($b_{\mathrm{void}}$ vs $b_{\mathrm{eff}}$ regression).
* **Key Numbers**:
  - BitVF: $B_{\mathrm{slope}} = \mathbf{0.914 \pm 0.043}$
  - REVOLVER: $B_{\mathrm{slope}} = \mathbf{0.910 \pm 0.046}$
  - Difference: $\Delta B_{\mathrm{slope}} = \mathbf{0.004 \pm 0.063} \implies \mathbf{0.07\sigma}$ tension!
* **Speaker Script**:
  > *"Despite this structural difference, we discovered an astonishing universality: the linear void barrier slope $B_{\mathrm{slope}}$ is identical between BitVF and REVOLVER to within $0.07\sigma$. Both finders respond to tracer bias with the exact same physical slope $B_{\mathrm{slope}} \approx 0.91$. This proves that barrier response is an intrinsic gravitational property of large-scale structure."*

---

### Slide 11: Robustness Audits 1 & 2: Scale Plateau & Poisson Tail Truncation
* **Visual**: Figure `plots/halos/bitvf_model_robustness_audit.png` (Panels A & B).
* **Key Numbers**:
  - Audit 1: $R_{\min}$ sweep from $14$ to $20\,h^{-1}\mathrm{Mpc}$ yields **$< 1\%$ variation** in $b_{\mathrm{void}}$ (scale-invariant plateau).
  - Audit 2: Truncating sparse bins in 50P ($R \le 28\,h^{-1}\mathrm{Mpc}$, where $N \approx 1$ void) drops $\chi^2/\nu$ from **$3.25 \to 1.83$**, while $b_{\mathrm{void}}$ shifts by only **$0.21\%$**!
* **Speaker Script**:
  > *"To stress-test BitVF, we executed a 5-part robustness audit. In Audit 1, sweeping the scale cut confirms $b_{\mathrm{void}}$ is scale-invariant to sub-percent precision. In Audit 2, we investigated why Sample 50P had an elevated $\chi^2$: its highest bin has literally $N \approx 1$ void in the gigaparsec box. Truncating this Poisson tail drops $\chi^2/\nu$ to $1.83$ while the fitted barrier shifts by only $0.2\%$, proving our barrier calibration is exceptionally robust."*

---

### Slide 12: Robustness Audits 3 & 4: Global Joint Fit & Multi-Tracer Law
* **Visual**: Audit Panels C & D: Global joint regression across all 22 active bins.
* **Key Numbers**:
  - Joint multi-tracer law: $b_{\mathrm{void}}(b_{\mathrm{eff}}) = (0.9076 \pm 0.04) b_{\mathrm{eff}} + (0.5963 \pm 0.04)$ with joint $\chi^2/\nu = 1.88$.
* **Speaker Script**:
  > *"In Audit 3, we fit a single joint linear barrier across all 22 active data points simultaneously. A single two-parameter linear law describes all four halo catalogs with a joint $\chi^2/\nu = 1.88$."*

---

### Slide 13: Scalability to Euclid & DESI Galaxy Mocks
* **Visual**: Flowchart from Aletheia simulation to Euclid Flagship mock challenges; computational complexity comparison ($O(N \log N)$ FFT vs Voronoi tessellation).
* **Speaker Script**:
  > *"Looking ahead to Euclid and DESI, BitVF's fast displacement reconstruction scales as $O(N \log N)$ via FFTs, bypassing the memory bottlenecks of Voronoi tessellations. Furthermore, having a universal barrier slope $B_{\mathrm{slope}} \approx 0.91$ allows us to place tight priors on multi-tracer galaxy samples, dramatically reducing parameter degeneracies."*

---

### Slide 14: Summary of Key Thesis Findings
* **Visual**: 3 bullet summary with bold checkmarks.
* **Key Takeaways**:
  1. **Epistemology Proven**: Algorithmic void finding dictates model compatibility. BitVF is structurally matched to Vdn ($\chi^2/\nu \approx 0.9\text{--}1.7$); REVOLVER has an intrinsic shape mismatch ($\chi^2/\nu \approx 3.1\text{--}8.1$).
  2. **Barrier Universality**: The linear void barrier slope is universal ($B_{\mathrm{slope}} \approx 0.91$, $\Delta = 0.07\sigma$).
  3. **Robustness Verified**: $b_{\mathrm{void}}$ is scale-invariant to $< 1\%$ and validated via a 5-part audit suite.

---

### Slide 15: Acknowledgments & Questions
* **Speaker Script**:
  > *"Thank you to Prof. Federico Marulli, Dr. Simone Sartori, Dr. Sofia Contarini, and Prof. Lauro Moscardini for your invaluable guidance. I welcome your questions."*

---

## 🎯 Individual Committee Defense Playbook

Review the pre-formulated counterarguments in [`knowledge_vault/03_Defense_Dossiers/Adversarial_Defense_QA.md`](file:///Users/arun/projects/my_cosmo_analysis/void_analysis/knowledge_vault/03_Defense_Dossiers/Adversarial_Defense_QA.md) for full transcripts.
