#!/usr/bin/env python3
"""
plot_void_cross_sections.py
Generates 2D real-space cross-sections (15 h^-1 Mpc thick slices) to map the
identified voids against the dark matter halos, proving that BitVF geometrically
respects the cosmic web topology.
"""
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_dm_particles(filepath: str) -> pd.DataFrame:
    """Loads DM particles from .dat file (X, Y, Z, vx, vy, vz)."""
    df = pd.read_csv(filepath, sep=r'\s+', header=None, usecols=[0, 1, 2], names=['x', 'y', 'z'])
    return df

def plot_slice(dm_df: pd.DataFrame, void_df: pd.DataFrame, z_center: float, slice_thickness: float, output_path: str):
    """Plots a 2D slice of DM particles and overlaid voids."""
    z_min = z_center - slice_thickness / 2.0
    z_max = z_center + slice_thickness / 2.0
    
    # Filter DM particles in slice
    mask_dm = (dm_df['z'] >= z_min) & (dm_df['z'] <= z_max)
    dm_slice = dm_df[mask_dm]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('black')
    
    # Plot DM particles
    ax.scatter(dm_slice['x'], dm_slice['y'], s=0.1, c='white', alpha=0.5, label='DM Particles')
    
    # Filter voids that intersect the slice
    # Distance from void center to slice center must be less than void radius
    mask_voids = np.abs(void_df['z'] - z_center) < void_df['radius']
    void_slice = void_df[mask_voids]
    
    # Plot voids
    for _, row in void_slice.iterrows():
        # Calculate the 2D projected radius at the z_center slice
        dz = np.abs(row['z'] - z_center)
        # projected_radius = sqrt(R^2 - dz^2)
        proj_r = np.sqrt(max(0, row['radius']**2 - dz**2))
        if proj_r > 0:
            circle = patches.Circle((row['x'], row['y']), proj_r, 
                                    linewidth=1.5, edgecolor='cyan', facecolor='none', alpha=0.8)
            ax.add_patch(circle)
            
    # Add a single dummy circle for legend
    ax.scatter([], [], edgecolor='cyan', facecolor='none', label='BitVF Voids', s=100)
    
    ax.set_xlabel(r'$X \ [h^{-1}\rm{Mpc}]$', fontsize=14)
    ax.set_ylabel(r'$Y \ [h^{-1}\rm{Mpc}]$', fontsize=14)
    ax.set_title(f'Cosmic Web and Voids Cross-section\nSlice: $Z \in [{z_min:.1f}, {z_max:.1f}]$', fontsize=16)
    
    # Set limits based on DM particles
    if not dm_slice.empty:
        ax.set_xlim(dm_slice['x'].min(), dm_slice['x'].max())
        ax.set_ylim(dm_slice['y'].min(), dm_slice['y'].max())
    else:
        ax.set_xlim(375, 625)
        ax.set_ylim(375, 625)
        
    ax.legend(loc='upper right', fontsize=12, facecolor='black', edgecolor='white', labelcolor='white')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  -> Saved cross-section plot to {output_path}")

def main():
    data_dir = '/Users/arun/projects/my_cosmo_analysis/void_analysis/data'
    plot_dir = '/Users/arun/projects/my_cosmo_analysis/void_analysis/plots'
    os.makedirs(plot_dir, exist_ok=True)
    
    # Subsamples mapping
    sparsities = ['2_5p', '5p']
    
    for sparsity in sparsities:
        dm_file = os.path.join(data_dir, 'subsamples', f'DM_particles_ss{sparsity}_250cube.dat')
        if not os.path.exists(dm_file):
            print(f"DM particle file not found: {dm_file}")
            continue
            
        print(f"\nProcessing sparsity {sparsity}...")
        dm_df = load_dm_particles(dm_file)
        print(f"  -> Loaded {len(dm_df)} DM particles.")
        
        # Find corresponding cleaned void catalog
        clean_dir = os.path.join(data_dir, 'cleaned')
        # match something like cleaned_0_voids_bitvf_aletheia_2_5p.txt.csv
        void_files = glob.glob(os.path.join(clean_dir, f"*_{sparsity}*"))
        if not void_files:
            print(f"No cleaned void catalog found for {sparsity}.")
            continue
            
        void_file = void_files[0] # Pick the first match
        print(f"  -> Using void catalog: {os.path.basename(void_file)}")
        void_df = pd.read_csv(void_file)
        
        # Plot at a slice roughly in the middle, Z ~ 500
        # The box is 375 to 625
        output_path = os.path.join(plot_dir, f'void_cross_section_ss{sparsity}.png')
        plot_slice(dm_df, void_df, z_center=500.0, slice_thickness=15.0, output_path=output_path)

if __name__ == "__main__":
    main()
