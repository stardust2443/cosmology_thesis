#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    frac = '5p'  # Let's start with the 5% catalog for good resolution
    
    # 1. Define the Slice Parameters
    # Your box runs from 375 to 625. We will take a 15 Mpc slice right through the middle.
    Z_MIN = 492.5
    Z_MAX = 507.5
    
    print(f"Plotting 2D slice for Z between {Z_MIN} and {Z_MAX}...")

    # 2. Load the Cleaned Voids
    void_file = f"../data/cleaned/cleaned_bitvf_aletheia_cat_0_{frac}.csv"
    if not os.path.exists(void_file):
        print(f"Error: Could not find {void_file}")
        return
        
    df_voids = pd.read_csv(void_file)
    
    # Filter voids: Only keep voids whose centers fall inside our Z-slice
    voids_in_slice = df_voids[(df_voids['z'] >= Z_MIN) & (df_voids['z'] <= Z_MAX)]
    print(f"  -> Found {len(voids_in_slice)} voids in this slice.")

    # 3. Load the Dark Matter Particles
    # Assuming the particle file is a space-separated text file with at least X, Y, Z
    part_file = f"../data/subsamples/DM_particles_ss{frac}_250cube.dat"
    if not os.path.exists(part_file):
        print(f"Error: Could not find particle file at {part_file}")
        return
        
    print("Loading DM particles (this may take a moment)...")
    # Using chunking to save memory if the file is massive
    chunk_list = []
    for chunk in pd.read_csv(part_file, sep=r'\s+', names=['x', 'y', 'z'], usecols=[0, 1, 2], chunksize=500000):
        # Only keep particles inside the Z-slice
        filtered_chunk = chunk[(chunk['z'] >= Z_MIN) & (chunk['z'] <= Z_MAX)]
        chunk_list.append(filtered_chunk)
    
    df_parts = pd.concat(chunk_list)
    print(f"  -> Found {len(df_parts)} particles in this slice.")

    # 4. Generate the Plot
    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)
    
    # Background: The Cosmic Web (Dark Matter Particles)
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
        
    # Formatting
    ax.set_xlim(375, 625)
    ax.set_ylim(375, 625)
    ax.set_aspect('equal') # CRITICAL: Ensures circles don't stretch into ovals
    
    ax.set_xlabel(r"$X \ [h^{-1}\mathrm{Mpc}]$", fontsize=14)
    ax.set_ylabel(r"$Y \ [h^{-1}\mathrm{Mpc}]$", fontsize=14)
    ax.set_title(f"Aletheia {frac} Real Space 2D Cross Section", fontsize=16)
    
    # Add a dummy line for the void legend since patches don't auto-add
    ax.plot([], [], color='red', label='BitVF Voids')
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Save
    out_path = f"../plots/2D_Slice_2704_{frac}.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Success! Map saved to {out_path}")

if __name__ == "__main__":
    main()