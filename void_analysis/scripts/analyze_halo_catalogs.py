#!/usr/bin/env python3
"""
analyze_halo_catalogs.py

Robust analysis of Aletheia halo catalogues at z=0.0 (30P, 50P, 75P, 100P):
- Accurate parsing of fixed-width columns
- Counts, number densities, mean particle separations (MPS)
- Mass distributions
- Theoretical effective linear bias (Tinker10 and Sheth01)
"""
import os, sys
import numpy as np
from colossus.cosmology import cosmology
from colossus.lss import bias

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, '..', 'AletheiaForArun', 'z0p00')

BOX_SIZE = 1000.0 # h^-1 Mpc
VOLUME = BOX_SIZE**3 # (h^-1 Mpc)^3

# Initialize Aletheia cosmology in Colossus
params = {'flat': True, 'H0': 67.36, 'Om0': 0.315, 'Ob0': 0.0493, 'sigma8': 0.811, 'ns': 0.9649}
cosmo = cosmology.setCosmology('aletheia', params)
h = 0.6736

FILES = [
    ('30P', 'halos_30P_78e10.dat'),
    ('50P', 'halos_50P_131e10.dat'),
    ('75P', 'halos_75P_196e10.dat'),
    ('100P', 'halos_100P_262e10.dat')
]

def parse_line_fixed(line):
    if "M200b" in line:
        line = line.split("M200b")[-1]
    tokens = line.strip().split()
    if len(tokens) == 7:
        return float(tokens[6])
    elif len(tokens) == 6:
        s = tokens[5]
        dot_idx = s.find('.')
        if dot_idx != -1:
            m_str = s[dot_idx + 3:]
            return float(m_str)
    return None

print("="*82)
print(f"{'Sample':<6} | {'N_halos':<9} | {'n_bar [h^3 Mpc^-3]':<18} | {'MPS [h^-1 Mpc]':<14} | {'<b_Tinker>':<10} | {'<b_Sheth>':<10}")
print("="*82)

for label, fname in FILES:
    fpath = os.path.join(DATA_DIR, fname)
    if not os.path.exists(fpath):
        print(f"File not found: {fpath}")
        continue
        
    masses = []
    # Read file line by line to parse masses
    with open(fpath, 'r') as f:
        for line in f:
            m = parse_line_fixed(line)
            if m is not None:
                masses.append(m)
                
    masses = np.array(masses, dtype=np.float64)
    n_halos = len(masses)
    n_bar = n_halos / VOLUME
    mps = n_bar**(-1.0/3.0)
    
    # Convert masses to Msun/h for Colossus: M_h = M * h
    masses_h = masses * h
    
    # Sample 100,000 halos to compute mean bias
    sub_sample = np.random.choice(masses_h, size=min(100000, n_halos), replace=False)
    b_tinker = np.mean(bias.haloBias(sub_sample, model='tinker10', z=0.0, mdef='200m'))
    b_sheth = np.mean(bias.haloBias(sub_sample, model='sheth01', z=0.0, mdef='200m'))
    
    print(f"{label:<6} | {n_halos:<9} | {n_bar:<18.5e} | {mps:<14.3f} | {b_tinker:<10.4f} | {b_sheth:<10.4f}")

print("="*82)
