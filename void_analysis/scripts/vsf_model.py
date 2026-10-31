#!/usr/bin/env python3
"""
vsf_model.py
Theoretical Void Size Function (VSF) using the Volume-Conserving (Vdn) model.

Implements the framework from:
  - Sheth & van de Weygaert (2004): SvdW two-barrier multiplicity function
  - Jennings et al. (2013): Vdn volume-conserving correction

The model predicts dn/dlnR_E [h^3 Mpc^-3] as a function of Eulerian void
radius R_E, given cosmological parameters (Omega_m, sigma_8).

References:
  Thesis Eq. 2.5  — SvdW multiplicity f(sigma)
  Thesis Eq. 2.12 — Lagrangian-to-Eulerian mapping
  Thesis Eq. 2.18 — Vdn model
"""
import numpy as np
from colossus.cosmology import cosmology
from colossus.lss import peaks

# ─── Physical constants ───────────────────────────────────────────────
DELTA_V_LINEAR = -2.71      # Linear void formation threshold (shell-crossing)
DELTA_C = 1.686             # Linear halo collapse threshold
DELTA_NL_V = -0.80          # Nonlinear density contrast inside the void (TH-0p80)

# Lagrangian-to-Eulerian expansion factor: R_E = R_L * (1 + delta_NL)^(-1/3)
EXPANSION_FACTOR = (1.0 + DELTA_NL_V) ** (-1.0 / 3.0)  # ≈ 1.71


def _sigma_from_R(R_lagrangian):
    """
    Compute σ(R_L) using colossus peakHeight.
    
    peakHeight returns ν = δ_c / σ(M), so σ = δ_c / ν.
    We convert R_L [Mpc/h] → M [Msun/h] → ν → σ.
    """
    M = peaks.lagrangianM(R_lagrangian)
    nu = peaks.peakHeight(M, z=0.0)
    sigma = DELTA_C / nu
    return sigma


def svdw_multiplicity(sigma, delta_v=DELTA_V_LINEAR):
    """
    Sheth-van de Weygaert (2004) two-barrier multiplicity function.
    
    Thesis Eq. 2.5:
        f(σ) = 2 Σ_{j=1}^{∞} exp[-(jπx)²/2] · jπx² · sin(jπD)
    where:
        D = |δ_v| / (δ_c + |δ_v|)
        x = D·σ / |δ_v|
    
    We sum 50 terms for convergence (the series converges rapidly).
    """
    abs_dv = np.abs(delta_v)
    D = abs_dv / (DELTA_C + abs_dv)
    x = D * sigma / abs_dv
    
    f = np.zeros_like(sigma, dtype=float)
    for j in range(1, 51):
        term = j * np.pi * x**2 * np.sin(j * np.pi * D) * np.exp(-0.5 * (j * np.pi * x)**2)
        f += term
    
    return 2.0 * f


def get_theoretical_vsf(R_eulerian, omega_m, sigma8, delta_v=DELTA_V_LINEAR):
    """
    Compute the Vdn (Volume-Conserving) theoretical VSF: dn/dlnR_E.
    
    Thesis Eq. 2.18:
        dn/dlnR_E = [f(σ) / V(R_E)] · |dlnσ⁻¹/dlnR_L| · dlnR_L/dlnR_E
    
    With volume conservation (Vdn correction):
        dn/dlnR_E = [f(σ) / V(R_E)] · |dlnσ⁻¹/dlnR_L|
        
    Since R_E = R_L · C  (C = expansion factor),  dlnR_L/dlnR_E = 1.
    And V(R_E) = (4π/3) R_E³.
    
    Args:
        R_eulerian: Array of Eulerian void radii [Mpc/h].
        omega_m:    Matter density parameter Ω_m.
        sigma8:     Amplitude of matter fluctuations σ_8.
        delta_v:    Linear void formation threshold (default: -2.71).
    
    Returns:
        dn/dlnR_E in [h³ Mpc⁻³].
    """
    R_eulerian = np.atleast_1d(np.asarray(R_eulerian, dtype=float))
    
    # 1. Set cosmology for this MCMC step
    params = {
        'flat': True,
        'H0': 67.36,
        'Om0': omega_m,
        'Ob0': 0.0493,
        'sigma8': sigma8,
        'ns': 0.9649,
    }
    cosmology.setCosmology('mcmc_step', params)
    
    # 2. Map Eulerian R_E → Lagrangian R_L
    R_L = R_eulerian / EXPANSION_FACTOR
    
    # 3. Compute σ(R_L)
    sigma = _sigma_from_R(R_L)
    
    # 4. Compute |dlnσ⁻¹/dlnR_L| via central finite difference
    eps = 1e-4
    sigma_plus = _sigma_from_R(R_L * (1.0 + eps))
    sigma_minus = _sigma_from_R(R_L * (1.0 - eps))
    # dlnσ/dlnR = (ln σ_+ - ln σ_-) / (2 eps)
    # dlnσ⁻¹/dlnR = -dlnσ/dlnR
    dln_sigma_inv_dlnR = -np.log(sigma_plus / sigma_minus) / (2.0 * eps)
    dln_sigma_inv_dlnR = np.abs(dln_sigma_inv_dlnR)
    
    # 5. Multiplicity function
    f_nu = svdw_multiplicity(sigma, delta_v=delta_v)
    
    # 6. Eulerian volume
    V_E = (4.0 * np.pi / 3.0) * R_eulerian**3
    
    # 7. Assemble Vdn model (Eq. 2.18)
    dn_dlnR_E = (f_nu / V_E) * dln_sigma_inv_dlnR
    
    return dn_dlnR_E


# ─── Quick self-test ──────────────────────────────────────────────────
if __name__ == "__main__":
    # Test at Aletheia true cosmology
    R_test = np.linspace(3.0, 16.0, 20)
    vsf = get_theoretical_vsf(R_test, omega_m=0.315, sigma8=0.811)
    
    print("Vdn Model Test (Aletheia cosmology: Ωm=0.315, σ8=0.811)")
    print(f"{'R_E [Mpc/h]':>12s}  {'dn/dlnR_E':>12s}")
    print("-" * 28)
    for r, n in zip(R_test, vsf):
        print(f"{r:12.2f}  {n:12.5e}")