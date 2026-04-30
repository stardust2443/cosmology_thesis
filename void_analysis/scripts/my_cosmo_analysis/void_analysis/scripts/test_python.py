#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np

print("Python works!")
print(f"Python version: {sys.version}")

# Quick test: read the cleaned CSV and compute histogram
try:
    df = pd.read_csv('../data/cleaned/cleaned_void_cat_5p.csv')
    radii = df['radius'].values
    print(f"Loaded {len(df)} voids")
    print(f"Radius range: {radii.min():.2f} - {radii.max():.2f}")

    # Compute histogram
    hist, edges = np.histogram(radii, bins=np.exp(np.linspace(np.log(1), np.log(20), 21)))
    centers = np.sqrt(edges[:-1] * edges[1:])
    peak_idx = np.argmax(hist)
    print(f"Peak of raw histogram: {centers[peak_idx]:.2f} Mpc")
except Exception as e:
    print(f"Error: {e}")
