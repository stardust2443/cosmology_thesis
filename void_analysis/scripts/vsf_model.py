import numpy as np
from colossus.cosmology import cosmology
from colossus.lss import variance

def svdw_multiplicity(sigma: np.ndarray, delta_v: float = -2.71, delta_c: float = 1.686) -> np.ndarray:
    """
    Computes the Sheth-van de Weygaert (2004) two-barrier multiplicity function.
    
    Args:
        sigma: Variance of the linear density field.
        delta_v: Linear void formation threshold (default: -2.71).
        delta_c: Linear halo collapse threshold (default: 1.686).
        
    Returns:
        The void multiplicity function f(nu).
    """
    nu = np.abs(delta_v) / sigma
    D = np.abs(delta_v) / (delta_c + np.abs(delta_v))
    x = D / nu
    
    # 4-term approximation of the infinite series expansion
    f = np.zeros_like(nu)
    for j in range(1, 5):
        term = j * np.pi * x**2 * np.sin(j * np.pi * D) * np.exp(-0.5 * (j * np.pi * x)**2)
        f += term
        
    return 2.0 * f

def get_theoretical_vsf(r_bins: np.ndarray, omega_m: float, sigma8: float) -> np.ndarray:
    """
    Calculates the Volume-Conserving (Vdn) theoretical Void Size Function.
    
    Args:
        r_bins: Array of void radii in Mpc/h.
        omega_m: Matter density parameter.
        sigma8: Amplitude of matter fluctuations.
        
    Returns:
        Theoretical abundance dn/dlnR in (h/Mpc)^3.
    """
    # 1. Initialize custom cosmology for the MCMC step
    params = {'flat': True, 'H0': 67.74, 'Om0': omega_m, 'Ob0': 0.0486, 'sigma8': sigma8, 'ns': 0.966}
    cosmology.setCosmology('mcmc_step', params)
    
    # 2. Convert Eulerian radius (R) to Lagrangian radius (R_L)
    # Vdn model requires mapping the observed non-linear radius to the linear initial state.
    delta_nl = -0.8
    r_L = r_bins * (1.0 + delta_nl)**(-1.0 / 3.0)
    
    # 3. Calculate Linear Variance sigma(R_L)
    sig = variance.sigmaR(r_L)
    
    # 4. Calculate the logarithmic derivative: |dln(sigma) / dlnR|
    # Using a standard central finite difference
    eps = 1e-4
    sig_plus = variance.sigmaR(r_L * (1.0 + eps))
    sig_minus = variance.sigmaR(r_L * (1.0 - eps))
    dln_sig_dlnR = np.abs(np.log(sig_plus / sig_minus) / (2.0 * eps))
    
    # 5. Compute the multiplicity function f(nu)
    f_nu = svdw_multiplicity(sig)
    
    # 6. Assemble the final VSF equation
    V_L = (4.0 * np.pi / 3.0) * r_L**3
    dn_dlnR = (f_nu / V_L) * dln_sig_dlnR
    
    return dn_dlnR

if __name__ == "__main__":
    # Minimal Test
    r_test = np.linspace(2.0, 20.0, 10)
    abundance = get_theoretical_vsf(r_test, omega_m=0.31, sigma8=0.81)
    
    print("R [Mpc/h]  |  dn/dlnR")
    print("-" * 25)
    for r, n in zip(r_test, abundance):
        print(f"{r:8.2f}   |  {n:.5e}")