"""
Latin Hypercube Sampling for Bioprinting Parameter Optimization
Parameters: Pressure, Nozzle Speed, Temperature, Z-offset
"""

import numpy as np
import pandas as pd
from scipy.stats import qmc
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── 1. Define parameter space ─────────────────────────────────────────────────
PARAMS = {
    "Pressure_kPa":    {"min": 60,  "max": 160, "step": 10,  "decimals": 0},
    "NozzleSpeed_mms": {"min": 5,   "max": 15,  "step": 1,   "decimals": 1},
    "Temperature_C":   {"min": 28,  "max": 38,  "step": 1,   "decimals": 1},
    "Zoffset_mm":      {"min": 0.4, "max": 0.9, "step": 0.1, "decimals": 2},
}

N_SAMPLES = 100          # adjust between 50–100 as needed
RANDOM_SEED = 42        # fix for reproducibility

# ── 2. Generate LHS in [0, 1]^4 ──────────────────────────────────────────────
sampler = qmc.LatinHypercube(d=len(PARAMS), seed=RANDOM_SEED)
lhs_raw = sampler.random(n=N_SAMPLES)          # shape: (N_SAMPLES, 4)

# ── 3. Scale to physical ranges and snap to nearest valid step ────────────────
def snap_to_grid(value, min_val, max_val, step, decimals):
    """Scale a [0,1] value to physical range and round to nearest step."""
    scaled = min_val + value * (max_val - min_val)
    snapped = round(round((scaled - min_val) / step) * step + min_val, decimals)
    # clip to ensure within bounds after rounding
    snapped = max(min_val, min(max_val, snapped))
    return snapped

param_names = list(PARAMS.keys())
samples = []

for row in lhs_raw:
    point = {}
    for i, (name, cfg) in enumerate(PARAMS.items()):
        point[name] = snap_to_grid(
            row[i], cfg["min"], cfg["max"], cfg["step"], cfg["decimals"]
        )
    samples.append(point)

df = pd.DataFrame(samples)

# ── 4. Remove any duplicate rows (can occur after grid snapping) ──────────────
n_before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
n_after = len(df)
print(f"Samples after deduplication: {n_after} (removed {n_before - n_after} duplicates)")

# ── 5. Summary statistics ─────────────────────────────────────────────────────
print("\n── Parameter coverage summary ──────────────────────────────────────────")
print(df.describe().round(3).to_string())

print("\n── Unique levels per parameter ─────────────────────────────────────────")
for col in df.columns:
    vals = sorted(df[col].unique())
    print(f"  {col}: {len(vals)} levels → {vals}")

# ── 6. Export to CSV ──────────────────────────────────────────────────────────
df.index.name = "Sample_ID"
csv_path = "/mnt/user-data/outputs/lhs_bioprint_samples.csv"
df.to_csv(csv_path)
print(f"\nSaved {len(df)} samples → {csv_path}")

# ── 7. Pair-plot of the sampled space ────────────────────────────────────────
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
fig.suptitle(f"LHS Parameter Space Coverage  (n={len(df)})", fontsize=14, fontweight="bold")

labels = {
    "Pressure_kPa":    "Pressure (kPa)",
    "NozzleSpeed_mms": "Nozzle Speed (mm/s)",
    "Temperature_C":   "Temperature (°C)",
    "Zoffset_mm":      "Z-offset (mm)",
}

cols = list(df.columns)
for i, row_param in enumerate(cols):
    for j, col_param in enumerate(cols):
        ax = axes[i][j]
        if i == j:
            # diagonal: histogram
            ax.hist(df[row_param], bins=10, color="#2E75B6", edgecolor="white", alpha=0.85)
            ax.set_xlabel(labels[row_param], fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
        else:
            # off-diagonal: scatter
            ax.scatter(df[col_param], df[row_param],
                       alpha=0.6, s=20, color="#2E75B6", edgecolors="none")
            ax.set_xlabel(labels[col_param], fontsize=8)
            ax.set_ylabel(labels[row_param], fontsize=8)
        ax.tick_params(labelsize=7)

plt.tight_layout()
plot_path = "/mnt/user-data/outputs/lhs_pairplot.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Saved pair-plot → {plot_path}")

# ── 8. Print full sample table ────────────────────────────────────────────────
print("\n── Full sample table ───────────────────────────────────────────────────")
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 120)
print(df.to_string())
