#!/usr/bin/env python3
"""
prepare_clean_halo_catalogs.py

Reads the raw Aletheia halo files from AletheiaForArun/z0p00, parses the
concatenated columns cleanly, and writes standard space-separated files:
- X Y Z (for void finding)
- X Y Z Vx Vy Vz Mass (full catalog for clustering / bias calculations)
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, '..', 'AletheiaForArun', 'z0p00')
OUT_DIR = os.path.join(HERE, '..', 'data', 'halos', 'z0p00')
os.makedirs(OUT_DIR, exist_ok=True)

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
        return [float(x) for x in tokens]
    elif len(tokens) == 6:
        s = tokens[5]
        dot_idx = s.find('.')
        if dot_idx != -1:
            vz_str = s[:dot_idx + 3]
            m_str = s[dot_idx + 3:]
            return [float(tokens[0]), float(tokens[1]), float(tokens[2]),
                    float(tokens[3]), float(tokens[4]), float(vz_str), float(m_str)]
    return None

for label, fname in FILES:
    in_path = os.path.join(RAW_DIR, fname)
    out_xyz_path = os.path.join(OUT_DIR, f"halos_{label}_xyz.dat")
    out_full_path = os.path.join(OUT_DIR, f"halos_{label}_full.dat")
    
    print(f"Processing {label} ({fname})...")
    count = 0
    with open(in_path, 'r') as fin, open(out_xyz_path, 'w') as f_xyz, open(out_full_path, 'w') as f_full:
        for line in fin:
            p = parse_line_fixed(line)
            if p is not None:
                # X Y Z
                f_xyz.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")
                # X Y Z Vx Vy Vz Mass
                f_full.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {p[3]:.2f} {p[4]:.2f} {p[5]:.2f} {p[6]:.2e}\n")
                count += 1
                
    print(f"  Done {label}: {count} halos written to {out_xyz_path}")

print("All halo catalogues parsed and standardized successfully!")
