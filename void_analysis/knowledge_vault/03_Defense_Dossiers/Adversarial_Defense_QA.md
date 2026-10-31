# ⚔️ Adversarial Defense Q&A: BitVF, REVOLVER, and Vdn Modeling

Prepare for tough, technical, and adversarial questions during the committee presentation.

---

### Q1 (Federico Marulli): "In Sample 50P, your reduced chi-square is $\chi^2/\nu = 3.25$. That indicates a poor fit ($p < 10^{-4}$). Doesn't this invalidate your claim that BitVF is compatible with the Jennings Vdn model?"

> **Your Answer**:
> *"Thank you, Professor. We specifically audited Sample 50P in Audit 2 of our robustness suite. In Sample 50P, the highest radial bin ($R \approx 35\,h^{-1}\mathrm{Mpc}$) contains an average count of literally $N = 1.1$ void in the entire $(1\,h^{-1}\mathrm{Gpc})^3$ volume. In this ultra-sparse Poisson tail, small integer fluctuations dominate the Gaussian Jackknife covariance.
> 
> When we truncate this sparse tail ($R \le 28\,h^{-1}\mathrm{Mpc}$), $\chi^2/\nu$ immediately drops from $3.25$ down to **$1.83$**. Most importantly, the calibrated void barrier $b_{\mathrm{void}}$ shifts by a mere **$0.21\%$** ($1.424 \to 1.427$). This proves that the elevated $\chi^2$ is not a structural failure of BitVF or Vdn, but rather a Poisson tail artifact at $N \approx 1$."*

---

### Q2 (Simone Sartori): "In your analysis, you discarded the CENTER cleaning algorithm and exclusively used TOUCH. Why did you discard CENTER, given that it retains more void candidates?"

> **Your Answer**:
> *"Simone, while CENTER retains more candidates by allowing concentric overlap up to $50\%$, Jennings et al.'s excursion-set model strictly requires non-overlapping spherical volumes to satisfy the volume-conservation partition condition $\sum V_i n_i = 1 - F_{\mathrm{collapse}}$.
> 
> In Audit 4, we directly tested CENTER against TOUCH across all four halo catalogs. For 100P, CENTER caused $\chi^2/\nu$ to explode from **$0.90$ to $23.14$**; for 75P, it exploded from **$1.36$ to $28.62$**. This demonstrated that allowing volume double-counting breaks the theoretical foundation of excursion-set modeling. TOUCH non-overlapping spheres are a physical prerequisite for Vdn modeling."*

---

### Q3 (Sofia Contarini): "REVOLVER has been widely used in BOSS and eBOSS analyses. Why are you confident that REVOLVER's biased $\Omega_m \approx 0.44$ is a model mismatch artifact rather than an MCMC sampling error or a physical parameter degeneracy?"

> **Your Answer**:
> *"Dr. Contarini, we evaluated the residual pull profiles $(d - m)/\sigma_{\mathrm{JK}}$ at the true cosmology for both finders. REVOLVER displays a severe, systematic S-curve tilt across every single halo catalog: it overpredicts at $R \approx 20\,h^{-1}\mathrm{Mpc}$ by $+3.6\sigma$ to $+4.1\sigma$, and underpredicts at $R \approx 32\,h^{-1}\mathrm{Mpc}$ by $-3.2\sigma$ to $-3.5\sigma$.
> 
> In contrast, BitVF's residuals are flat and within $\pm 1\sigma$. The MCMC sampler cannot change the geometric shape of the Vdn curve without changing $\Omega_m$. To flatten REVOLVER's S-curve tilt, the sampler is forced to drive $\Omega_m$ upward toward $0.44$. When we inspect the per-bin $\chi^2$ gradients, we find that widening the prior only pushes $\Omega_m$ further into unphysical territory ($\sim 0.60$), confirming that the issue is a geometrical mismatch between watershed polygons and spherical excursion-set models."*

---

### Q4 (Lauro Moscardini): "You found that the linear void barrier slope $B_{\mathrm{slope}} \approx 0.91$ is universal to $0.07\sigma$ between BitVF and REVOLVER. What is the physical reason for this universality, and can it be used for Euclid cosmological forecasts?"

> **Your Answer**:
> *"Prof. Moscardini, the universality of $B_{\mathrm{slope}} = 0.914 \pm 0.043$ (BitVF) and $0.910 \pm 0.046$ (REVOLVER) shows that the response of void formation thresholds to tracer bias $b_{\mathrm{eff}}$ is governed by the underlying gravitational dynamics of the matter field, rather than the specific edge-finding algorithm.
> 
> For Euclid, this is a major practical advantage: in multi-tracer mock challenges, instead of treating $b_{\mathrm{void}}$ as an unconstrained nuisance parameter for every galaxy sub-sample, we can impose a tight prior on $B_{\mathrm{slope}} \approx 0.91$, substantially reducing the parameter volume and breaking degeneracies between $\Omega_m$, $\sigma_8$, and void bias."*

---

### Q5 (General): "How does your pipeline handle the CosmoBolognaLib effective bias bug when $b_{\mathrm{eff}} = 1.0$?"

> **Your Answer**:
> *"In CosmoBolognaLib's internal C++ implementation, passing $b_{\mathrm{eff}} = 1.0$ triggers a default branch that overrides the user's bias parameters with hardcoded defaults. In our automated pipeline, we strictly guard against this by setting $b_{\mathrm{eff}} = 1.0001$ for unbiased dark matter samples or passing the exact measured halo bias (e.g. $1.0211, 0.9583$), preventing any silent override."*
