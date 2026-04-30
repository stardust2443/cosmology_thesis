#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    frac = '5p', '2_5p' 
    
    Z_MIN = 492.5
    Z_MAX = 507.5
    
    print(f"Plotting 2D slice for Z between {Z_MIN} and {Z_MAX}...")

    # 1. Load the Cleaned Voids (The Red Circles)
    void_file = f"../data/cleaned/cleaned_bitvf_aletheia_cat_0_{frac}.csv"
    if not os.path.exists(void_file):
        print(f"Error: Could not find {void_file}")
        return
        
    df_voids = pd.read_csv(void_file)
    voids_in_slice = df_voids[(df_voids['z'] >= Z_MIN) & (df_voids['z'] <= Z_MAX)]
    print(f"  -> Found {len(voids_in_slice)} voids in this slice.")

    # 2. Load the Dark Matter Particles (The Black Cosmic Web)
    part_file = f"../data/subsamples/DM_particles_ss{frac}_250cube.dat"
    if not os.path.exists(part_file):
        print(f"\nCRITICAL ERROR: Missing Dark Matter File!")
        print(f"You must download {part_file} from the cluster.")
        print("Run this in your terminal: scp arun.mitra@ophfe1:~/workspace/subsamples/DM_particles_ss5p_250cube.dat ../data/raw/")
        return
        
    print("Loading DM particles (this may take a moment)...")
    chunk_list = []
    # Note: DM particles don't have a '#' header, so we just read the first 3 columns
    for chunk in pd.read_csv(part_file, sep=r'\s+', names=['x', 'y', 'z'], usecols=[0, 1, 2], chunksize=500000):
        filtered_chunk = chunk[(chunk['z'] >= Z_MIN) & (chunk['z'] <= Z_MAX)]
        chunk_list.append(filtered_chunk)
    
    df_parts = pd.concat(chunk_list)
    print(f"  -> Found {len(df_parts)} particles in this slice.")

    # 3. Generate the Plot
    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)
    
    # Background: The Cosmic Web
    ax.scatter(df_parts['x'], df_parts['y'], s=0.05, color='black', alpha=0.3, label='DM Particles')
    
    # Foreground: The Voids
    for idx, row in voids_in_slice.iterrows():
        circ = plt.Circle(
            (row['x'], row['y']), 
            row['radius'], 
            color='red', 
            fill=False, 
            linewidth=1.5, 
            alpha=0.8
        )
        ax.add_patch(circ)
        
    ax.set_xlim(375, 625)
    ax.set_ylim(375, 625)
    ax.set_aspect('equal') 
    
    ax.set_xlabel(r"$X \ [h^{-1}\mathrm{Mpc}]$", fontsize=14)
    ax.set_ylabel(r"$Y \ [h^{-1}\mathrm{Mpc}]$", fontsize=14)
    # Fixed the \in syntax warning using rf"..."
    ax.set_title(rf"Aletheia {frac} Real Space of 2D Cross Section", fontsize=16)
    
    ax.plot([], [], color='red', label='BitVF Voids')
    ax.legend(loc='upper right', framealpha=0.9)
    
    out_path = f"../plots/new_2D_Slice_a_{frac}.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Success! Map saved to {out_path}")

if __name__ == "__main__":
    main()