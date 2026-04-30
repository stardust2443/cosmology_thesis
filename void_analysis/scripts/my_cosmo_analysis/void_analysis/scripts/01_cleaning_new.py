#!/usr/bin/env python3
"""
01_cleaning_new.py
"""
import os
import yaml
import pandas as pd
import numpy as np

def load_config(config_path="../configs/pipeline_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def clean_catalog(df, config):
    """Applies standard physical cuts to the raw void catalog."""
    c = config['cleaning']
    
    # Cut 1: Central density threshold
    mask = df['delta_min'] < c['delta_nl_threshold']
    
    # Cut 2: Radius bounds
    mask &= (df['radius'] > c['rmin']) & (df['radius'] < c['rmax'])
    
    # TODO: Insert KDTree overlap removal logic here if REVOLVER didn't pre-filter it.
    
    return df[mask].copy()

def main():
    config = load_config()
    raw_dir = config['paths']['raw_dir']
    clean_dir = config['paths']['clean_dir']
    
    for frac in config['fractions']:

        raw_file = os.path.join(raw_dir, f"voids_bitvf_aletheia_{frac}.txt")
        
        if not os.path.exists(raw_file):
            print(f"Skipping {frac}: Raw file not found at {raw_file}")
            continue
            
        print(f"Processing fraction: {frac}")
        
        # Output headers
        cols = ['VoidID', 'x', 'y', 'z', 'radius', 'delta_min', 'delta_avg', 'lambda_v', 'DensRatio']
        
        # Read the raw file (skipping the two header rows starting with '#')
        df_raw = pd.read_csv(raw_file, sep='\s+', names=cols, comment='#')
        
        # Clean the catalog
        df_clean = clean_catalog(df_raw, config)
        
        # Save out to CSV
        out_path = os.path.join(clean_dir, f"cleaned_bitvf_aletheia_cat_{frac}.csv")
        df_clean.to_csv(out_path, index=False)
        
        print(f"  -> Saved {len(df_clean)} actual voids (out of {len(df_raw)} raw).")

if __name__ == "__main__":
    main()