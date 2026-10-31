# MCMC Convergence & Quality Assurance Checklist

Use this checklist before and after running MCMC parameter estimation chains.

---

## 1. Pre-Flight Configuration Checklist

- [ ] **Python Interpreter**: Confirmed `/opt/homebrew/bin/python3.12`.
- [ ] **Covariance Matrix**: Using the correct 64-region Jackknife matrix ($C_{\mathrm{JK}}$) matching the tracer subsample.
- [ ] **Hartlap Factor**: Applied $\alpha = (N_{\mathrm{sub}} - p - 2) / (N_{\mathrm{sub}} - 1)$ to scale $C^{-1}$.
- [ ] **Data Vector Radius Cut**: Excluded small non-linear scales ($R < R_{\min}$, typically $7$ or $16\,h^{-1}\mathrm{Mpc}$).
- [ ] **Priors Defined**:
  - Uninformative: $\Omega_m \in [0.10, 0.60]$, $\sigma_8 \in [0.40, 1.40]$.
  - Informative (Barrier/Bias): Gaussian prior $\mathcal{N}(\mu_{b}, \sigma_{b}^2)$ calibrated from fiducial fits.
- [ ] **Walker Initialization**:
  - $N_{\mathrm{walkers}} \ge 32$ (for 2D or 3D parameter spaces).
  - Small Gaussian ball centered around maximum likelihood estimate (MLE).

---

## 2. Post-Run Diagnostics Checklist

- [ ] **Autocorrelation Time**:
  - Compute `tau = sampler.get_autocorr_time(quiet=True)`.
  - Verify $N_{\mathrm{steps}} \ge 50 \times \max(\tau)$.
- [ ] **Acceptance Fraction**:
  - Average acceptance fraction across walkers: $0.20 \le \bar{a}_f \le 0.50$.
  - If $\bar{a}_f < 0.15$, proposal step size is too large.
  - If $\bar{a}_f > 0.60$, walkers are taking tiny diffusive steps.
- [ ] **Burn-in Discarded**:
  - Stripped at least $3 \times \tau$ initial steps from the chain.
- [ ] **Prior Wall Check**:
  - Is the median or posterior mode within $2\sigma$ of the hard boundary?
  - If YES $\implies$ Check for theoretical model mismatch (e.g. REVOLVER with Vdn).
- [ ] **Truth Recovery**:
  - Does the 68% credible interval enclose the Aletheia truth ($\Omega_m = 0.315$, $\sigma_8 = 0.811$)?
  - BitVF: Recovers truth within $1\sigma$.
  - Document tension in units of $\sigma$: $|\theta_{\mathrm{median}} - \theta_{\mathrm{true}}| / \sigma_{\theta}$.
