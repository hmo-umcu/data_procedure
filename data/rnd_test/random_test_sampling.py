"""
Random Test Set Sampling for Bioprinting Parameter Optimization
---------------------------------------------------------------
Purpose : Generate 20 purely random test samples (NOT LHS) to evaluate
          the surrogate model / BO optimization algorithm on unseen data.
Requires: numpy, pandas, openpyxl
Install : pip install numpy pandas openpyxl
"""

import numpy as np
import pandas as pd

# ── 1. Define parameter space ─────────────────────────────────────────────────
PARAMS = {
    "Pressure_kPa":    {"min": 60,  "max": 160, "step": 10},
    "NozzleSpeed_mms": {"min": 5,   "max": 15,  "step": 1},
    "Temperature_C":   {"min": 28,  "max": 38,  "step": 1},
    "Zoffset_mm":      {"min": 0.4, "max": 0.9, "step": 0.1},
}

N_TEST      = 20    # number of random test samples
RANDOM_SEED = 99    # different seed from LHS training set (which used 42)

# ── 2. Build valid discrete grid levels per parameter ─────────────────────────
rng = np.random.default_rng(RANDOM_SEED)

grids = {}
for name, cfg in PARAMS.items():
    levels = np.round(
        np.arange(cfg["min"], cfg["max"] + cfg["step"] / 2, cfg["step"]),
        decimals=2
    )
    grids[name] = levels

# ── 3. Pure random sampling (with replacement — intentional for test set) ─────
samples = {}
for name, levels in grids.items():
    samples[name] = rng.choice(levels, size=N_TEST, replace=True)

df = pd.DataFrame(samples)
df.index.name = "Sample_ID"

# ── 4. Optional: check overlap with LHS training set ─────────────────────────
# Uncomment if lhs_bioprint_samples.csv is in the same folder
# lhs_df = pd.read_csv('lhs_bioprint_samples.csv', index_col=0)
# merged = df.merge(lhs_df, on=list(df.columns), how='inner')
# print(f"Samples overlapping with LHS training set: {len(merged)}")

# ── 5. Print sample table ─────────────────────────────────────────────────────
print(f"Generated {N_TEST} random test samples (seed={RANDOM_SEED}):\n")
print(df.to_string())

# ── 6. Summary statistics ─────────────────────────────────────────────────────
print("\n── Summary statistics ──────────────────────────────────────────────────")
print(df.describe().round(3).to_string())

print("\n── Unique levels sampled per parameter ─────────────────────────────────")
for col in df.columns:
    vals = sorted(df[col].unique())
    print(f"  {col}: {len(vals)} unique → {vals}")

# ── 7. Export semicolon CSV (Excel-friendly for European locale) ──────────────
csv_path = "random_test_samples.csv"
df.to_csv(csv_path, sep=";")
print(f"\nSaved CSV  → {csv_path}")

# ── 8. Export Excel with two sheets ──────────────────────────────────────────
xlsx_path = "random_test_samples.xlsx"
with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    # Sheet 1: test samples
    df.to_excel(writer, sheet_name="Test_Samples", index=True)
    # Sheet 2: parameter space reference
    ref_data = {
        "Parameter": list(PARAMS.keys()),
        "Min":       [v["min"]  for v in PARAMS.values()],
        "Max":       [v["max"]  for v in PARAMS.values()],
        "Step":      [v["step"] for v in PARAMS.values()],
        "N_Levels":  [len(grids[k]) for k in PARAMS.keys()],
    }
    pd.DataFrame(ref_data).to_excel(writer, sheet_name="Parameter_Space", index=False)

print(f"Saved XLSX → {xlsx_path}")
