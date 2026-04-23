#!/usr/bin/env python3
"""
lhs_sampling.py
---------------
Latin Hypercube Sampling for Bioprinting Parameter Optimization
48-well plate setup.

Parameters:
  Pressure_kPa    : 70 – 130 kPa  (step 5)
  NozzleSpeed_mms : 5  – 15  mm/s (step 1)
  Temperature_C   : 25 – 35  °C   (step 1)
  Zoffset_mm      : 0.4 – 0.9 mm  (step 0.1)

Output:
  lhs_bioprint_samples.csv         (comma-separated, for inspection)
  lhs_bioprint_samples_semicolon.csv  (semicolon-separated, for NC generator)
  lhs_pairplot.png
"""

import numpy as np
import pandas as pd
from scipy.stats import qmc
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

# ── 1. Parameter space ────────────────────────────────────────────────────────
PARAMS = {
    "Pressure_kPa":    {"min": 60,  "max": 120, "step": 5,  "decimals": 0},
    "NozzleSpeed_mms": {"min": 5,   "max": 15,  "step": 1,   "decimals": 1},
    "Temperature_C":   {"min": 25,  "max": 32,  "step": 1,   "decimals": 0},
    "Zoffset_mm":      {"min": 0.2, "max": 0.8, "step": 0.1, "decimals": 2},
}

N_SAMPLES   = 100
RANDOM_SEED = 42

OUTPUT_DIR = "."

# ── 2. Generate LHS in [0, 1]^4 ──────────────────────────────────────────────
sampler = qmc.LatinHypercube(d=len(PARAMS), seed=RANDOM_SEED)
lhs_raw = sampler.random(n=N_SAMPLES)

# ── 3. Scale and snap to nearest valid step ───────────────────────────────────
def snap_to_grid(value, min_val, max_val, step, decimals):
    scaled  = min_val + value * (max_val - min_val)
    snapped = round(round((scaled - min_val) / step) * step + min_val, decimals)
    return max(min_val, min(max_val, snapped))

samples = []
for row in lhs_raw:
    point = {}
    for i, (name, cfg) in enumerate(PARAMS.items()):
        point[name] = snap_to_grid(
            row[i], cfg["min"], cfg["max"], cfg["step"], cfg["decimals"]
        )
    samples.append(point)

df = pd.DataFrame(samples)

# ── 4. Deduplicate ────────────────────────────────────────────────────────────
n_before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
n_after  = len(df)
print(f"Samples after deduplication: {n_after} (removed {n_before - n_after} duplicates)")

# ── 5. Sort by Temperature ascending, then Pressure, then NozzleSpeed ─────────
# This ordering is preserved in the CSV so the NC generator can group by temp.
df = df.sort_values(
    by=["Temperature_C", "Pressure_kPa", "NozzleSpeed_mms", "Zoffset_mm"]
).reset_index(drop=True)
df.index.name = "Sample_ID"

print(f"\nSamples sorted by Temperature_C (ascending):")
print(df.groupby("Temperature_C").size().to_string())

# ── 6. Summary statistics ─────────────────────────────────────────────────────
print("\n── Parameter coverage summary ──────────────────────────────────────────")
print(df.describe().round(3).to_string())

print("\n── Unique levels per parameter ─────────────────────────────────────────")
for col in df.columns:
    vals = sorted(df[col].unique())
    print(f"  {col}: {len(vals)} levels → {vals}")

# ── 7. Export CSV ─────────────────────────────────────────────────────────────
csv_comma     = os.path.join(OUTPUT_DIR, "lhs_bioprint_samples.csv")
csv_semicolon = os.path.join(OUTPUT_DIR, "lhs_bioprint_samples_semicolon.csv")

df.to_csv(csv_comma,     sep=',')
df.to_csv(csv_semicolon, sep=';')

print(f"\nSaved {len(df)} samples → {csv_comma}")
print(f"Saved {len(df)} samples → {csv_semicolon}")

# ── 8. Pair-plot ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
fig.suptitle(f"LHS Parameter Space Coverage  (n={len(df)})", fontsize=14, fontweight="bold")

labels = {
    "Pressure_kPa":    "Pressure (kPa)",
    "NozzleSpeed_mms": "Nozzle Speed (mm/s)",
    "Temperature_C":   "Temperature (°C)",
    "Zoffset_mm":      "Z-offset (mm)",
}

cols_list = list(df.columns)
for i, row_param in enumerate(cols_list):
    for j, col_param in enumerate(cols_list):
        ax = axes[i][j]
        if i == j:
            ax.hist(df[row_param], bins=10, color="#2E75B6", edgecolor="white", alpha=0.85)
            ax.set_xlabel(labels[row_param], fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
        else:
            ax.scatter(df[col_param], df[row_param],
                       alpha=0.6, s=20, color="#2E75B6", edgecolors="none")
            ax.set_xlabel(labels[col_param], fontsize=8)
            ax.set_ylabel(labels[row_param], fontsize=8)
        ax.tick_params(labelsize=7)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "lhs_pairplot.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Saved pair-plot → {plot_path}")

# ── 9. Print full table ───────────────────────────────────────────────────────
print("\n── Full sample table (sorted by temperature) ───────────────────────────")
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 120)
print(df.to_string())
