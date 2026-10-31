#!/usr/bin/env python3
"""
run_revolver_halos.py

Orchestrates running REVOLVER (ZOBOV Voronoi watershed void finder) on the
Aletheia halo catalogs (z=0.0, L=1000.0 h^-1 Mpc periodic box).
"""
import os, sys, time, argparse, subprocess

REVOLVER_DIR = "/Users/arun/projects/Revolver"
PYTHON_BIN = os.path.join(REVOLVER_DIR, ".venv", "bin", "python")
DATA_DIR = "/Users/arun/projects/my_cosmo_analysis/void_analysis/data"
HALO_DIR = os.path.join(DATA_DIR, "halos", "z0p00")
OUT_BASE = os.path.join(DATA_DIR, "voids_halo_revolver", "raw")

PARAM_TEMPLATE = """# AUTO-GENERATED PARAMS FOR {sample}
from parameters.default_params import *

handle = 'halo_{sample}'
output_folder = '{out_dir}/'

verbose = True
nthreads = 8

is_box = True
box_length = 1000.0
tracer_file_type = 2
tracer_posn_cols = [0, 1, 2]
tracer_file = '{halo_file}'

run_zobov = True
run_voxelvoids = False
do_recon = False
do_tessellation = True

zobov_box_div = {box_div}
zobov_buffer = 0.08
min_dens_cut = 1.0
void_min_num = 5
use_barycentres = False
find_clusters = False
"""

def run_sample(sample='100P', box_div=4):
    halo_npy = os.path.join(HALO_DIR, f"halos_{sample}_xyz.npy")
    if not os.path.exists(halo_npy):
        print(f"Error: {halo_npy} does not exist!")
        return False
        
    out_dir = os.path.join(OUT_BASE, sample)
    os.makedirs(out_dir, exist_ok=True)
    
    param_file = os.path.join(REVOLVER_DIR, "parameters", f"params_halo_{sample}.py")
    content = PARAM_TEMPLATE.format(
        sample=sample,
        out_dir=out_dir,
        halo_file=halo_npy,
        box_div=box_div
    )
    with open(param_file, 'w') as f:
        f.write(content)
    print(f"[{sample}] Wrote param file: {param_file}")
    
    cmd = [PYTHON_BIN, "revolver.py", "--par", f"parameters/params_halo_{sample}.py"]
    print(f"[{sample}] Launching REVOLVER in {REVOLVER_DIR}...")
    t0 = time.time()
    res = subprocess.run(cmd, cwd=REVOLVER_DIR)
    dt = time.time() - t0
    
    if res.returncode == 0:
        print(f"[{sample}] REVOLVER completed successfully in {dt:.1f}s ({dt/60:.2f} min).")
        # Check generated output
        cat_file = os.path.join(out_dir, "zobov-Voids_cat.txt")
        if os.path.exists(cat_file):
            size = os.path.getsize(cat_file)
            print(f"[{sample}] Void catalog created: {cat_file} ({size/1024/1024:.2f} MB)")
            return True
        else:
            print(f"[{sample}] Warning: {cat_file} not found despite returncode 0.")
            return False
    else:
        print(f"[{sample}] REVOLVER failed with return code {res.returncode}!")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=str, default='100P', choices=['100P', '75P', '50P', '30P', 'all'])
    parser.add_argument('--box-div', type=int, default=4)
    args = parser.parse_args()
    
    if args.sample == 'all':
        for s in ['100P', '75P', '50P', '30P']:
            ok = run_sample(s, args.box_div)
            if not ok:
                print(f"Stopping at {s} due to error.")
                break
    else:
        run_sample(args.sample, args.box_div)
