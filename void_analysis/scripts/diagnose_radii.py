import pandas as pd
import numpy as np

print("=== Raw Data Analysis ===")
cols = ['x', 'y', 'z', 'radius', 'radius_ext', 'delta_min', 'delta_avg']
df_raw = pd.read_csv('../data/raw/voids_bitvf_aletheia_5p.txt', sep=r'\s+', names=cols, comment='#', dtype=float)

print("\nFirst 20 voids - radius vs radius_ext:")
print(df_raw[['radius', 'radius_ext']].head(20).to_string())

print(f"\nRaw data radius stats:")
print(f"  Min: {df_raw['radius'].min():.3f}, Max: {df_raw['radius'].max():.3f}")
print(f"  Mean: {df_raw['radius'].mean():.3f}, Median: {np.median(df_raw['radius']):.3f}")
print(f"  Mean ratio radius_ext/radius: {(df_raw['radius_ext'] / df_raw['radius']).mean():.3f}")

print(f"\nCleaned data analysis:")
df_clean = pd.read_csv('../data/cleaned/cleaned_void_cat_5p.csv')
radii = df_clean['radius'].values

print(f"  Count: {len(df_clean)}")
print(f"  Min: {radii.min():.3f}, Max: {radii.max():.3f}")
print(f"  Mean: {radii.mean():.3f}, Median: {np.median(radii):.3f}")

# Compute where peak should be
hist, edges = np.histogram(radii, bins=40)
centers = (edges[:-1] + edges[1:]) / 2
peak_idx = np.argmax(hist)
print(f"  Peak in histogram: {centers[peak_idx]:.2f} Mpc ({hist[peak_idx]} voids)")

# Show distribution
print("\n  Radius distribution (cleaned data):")
for i, (center, count) in enumerate(zip(centers[::4], hist[::4])):
    print(f"    {center:5.2f}: {'█' * (count // 5)}")
