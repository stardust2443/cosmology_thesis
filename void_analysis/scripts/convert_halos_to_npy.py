#!/usr/bin/env python3
"""
convert_halos_to_npy.py

Converts text-based halo catalogs (X Y Z) in data/halos/z0p00/ to binary .npy
files for fast loading in REVOLVER and analysis scripts.
"""
import os, time
import numpy as np

DATA_DIR = "/Users/arun/projects/my_cosmo_analysis/void_analysis/data/halos/z0p00"
SAMPLES = ['100P', '75P', '50P', '30P']

for s in SAMPLES:
    txt_path = os.path.join(DATA_DIR, f"halos_{s}_xyz.dat")
    npy_path = os.path.join(DATA_DIR, f"halos_{s}_xyz.npy")
    if os.path.exists(npy_path):
        print(f"[{s}] .npy file already exists: {npy_path}")
        continue
    if not os.path.exists(txt_path):
        print(f"[{s}] Warning: text file not found: {txt_path}")
        continue
    print(f"[{s}] Loading {txt_path}...")
    t0 = time.time()
    data = np.loadtxt(txt_path, dtype=np.float32)
    dt = time.time() - t0
    print(f"[{s}] Loaded {len(data)} haloes in {dt:.1f}s. Saving to {npy_path}...")
    np.save(npy_path, data)
    print(f"[{s}] Saved successfully.")

print("All available halo catalogues converted to .npy!")
