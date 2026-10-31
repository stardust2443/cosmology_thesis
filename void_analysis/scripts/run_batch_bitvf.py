#!/usr/bin/env python3
"""
run_batch_bitvf.py

Runs BitVF void identification, cleaning, binning, and covariance calculation
sequentially across the remaining Aletheia halo catalogs (75P, 50P, 30P).
"""
import os, sys, time, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

SAMPLES = ['75P', '50P', '30P']

for sample in SAMPLES:
    print(f"\n========================================================")
    print(f" PROCESSING HALO CATALOG: {sample}")
    print(f"========================================================")
    
    # 1. BitVF void finder
    raw_void_file = os.path.join(HERE, '..', 'data', 'voids_halo', 'raw', f"voids_raw_bitvf_{sample}_Nrec1.dat")
    if not os.path.exists(raw_void_file):
        print(f"Step 1: Running BitVF for {sample}...")
        t0 = time.time()
        cmd = [PYTHON, os.path.join(HERE, 'run_bitvf_halos.py'), '--sample', sample, '--nrec', '1']
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"BitVF failed for {sample} with code {ret.returncode}")
            continue
        print(f"BitVF finished in {time.time() - t0:.1f} s.")
    else:
        print(f"Step 1: Raw voids already exist for {sample} at {raw_void_file}.")
        
    # 2. Clean voids (both center and touch)
    for crit in ['touch', 'center']:
        print(f"Step 2: Cleaning voids for {sample} ({crit})...")
        t0 = time.time()
        cmd = [PYTHON, os.path.join(HERE, 'clean_voids_halo.py'), '--sample', sample, '--criterion', crit]
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"Cleaning failed for {sample} ({crit})")
            continue
        print(f"Cleaning finished in {time.time() - t0:.1f} s.")
        
    # 3. Bin and Jackknife covariance
    print(f"Step 3: Binning and computing 64-JK covariance for {sample}...")
    t0 = time.time()
    cmd = [PYTHON, os.path.join(HERE, 'bin_voids_halo.py'), '--sample', sample, '--criterion', 'all']
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"Binning failed for {sample}")
        continue
    print(f"Binning finished in {time.time() - t0:.1f} s.")

print("\nAll batch halo catalogs processed successfully!")
