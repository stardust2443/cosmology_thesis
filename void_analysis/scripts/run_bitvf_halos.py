#!/usr/bin/env python3
"""
run_bitvf_halos.py

Runs BitVFbox void identification on the Aletheia halo catalogues (z=0.0).
Parameters aligned with Sartori et al. 2026 / CosmoBolognaLib:
- cell_size: 1.0 (in units of MPS)
- threshold: 1e-2
- Nrec: 1 or 8
"""
import os, sys, time
CBL_HOME = '/Users/arun/projects/CosmoBolognaLib'
sys.path.insert(0, os.path.join(CBL_HOME, 'Python'))
os.environ.setdefault('DYLD_LIBRARY_PATH', CBL_HOME)
import CosmoBolognaLib as cbl

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, '..', 'data', 'halos', 'z0p00')
OUT_DIR = os.path.join(HERE, '..', 'data', 'voids_halo', 'raw') + '/'
os.makedirs(OUT_DIR, exist_ok=True)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--sample', type=str, default='100P', choices=['30P', '50P', '75P', '100P'])
parser.add_argument('--nrec', type=int, default=1)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

sample = args.sample
tracer_file = os.path.join(DATA_DIR, f"halos_{sample}_xyz.dat")
output_file = f"raw_bitvf_{sample}_Nrec{args.nrec}.dat"

col_trac = [0, 1, 2]
Nrec = args.nrec
threshold = 1e-2
sigma = 0.0
cell_size = 1.0
printDispl = False
printDiv = False
printGal = False
delimiter = ' '
comment = '#'
seed = args.seed

print(f"=== Running BitVF on {sample} with Nrec={Nrec} ===")
t0 = time.time()
bv = cbl.bitvf.BitVFbox(
    tracer_file,
    col_trac,
    Nrec,
    threshold,
    sigma,
    cell_size,
    printDispl,
    printDiv,
    printGal,
    OUT_DIR,
    output_file,
    delimiter,
    comment,
    seed
)
t1 = time.time()
print(f"=== Successfully completed {sample} in {t1 - t0:.2f} seconds ===")
