#!/usr/bin/env python3
"""
generate_nc_files.py
--------------------
Reads the LHS sample CSV and generates NC files for a 48-well plate.
Each NC file contains 8 columns (= 8 samples), each column has 6 constructs (rows A-F).

GEOMETRY: 0°–90° two-layer grid (field standard for shape fidelity assessment)
  Layer 1 (0°):  4 strands running in X-direction (horizontal)
  Layer 2 (90°): 4 strands running in Y-direction (vertical)
  → Produces a 3×3 grid of open square pores per construct
  → Enables Printability Index (Pr), pore area, filament width, and
    intersection quality metrics — all required for publication.

Grid dimensions (from centre of well, all in mm):
  Strand positions (layer 1, X-strands): Y = -1.35, -0.45, +0.45, +1.35
  Strand positions (layer 2, Y-strands): X = -1.35, -0.45, +0.45, +1.35
  Strand extent:  ±1.80 mm from well centre (= 3.60 mm total span)
  Nominal pore:   ~0.49 × 0.49 mm (spacing 0.9 mm − filament dia. ~0.41 mm)
  Layer 2 Z:      z_offset + LAYER_HEIGHT_MM above layer 1

Grouping strategy:
  - Samples are processed in CSV order (sequential batches of N_COLS)
  - Each NC file contains a mixed set of pressure/speed/z-offset values
  - Temperature is excluded from the optimization space; set manually in Architect UI

NC file naming:
  data_collection_48well_8cols_s{first}-s{last}.nc

48-well plate layout:
  Columns 1-8  → 8 samples (one per column)
  Rows    A-F  → 6 constructs per sample (one per well)

Parameters updated per column (sample):
  M200  : pressure     (kPa  → value × 10, unit 0.1 kPa)
  F     : nozzle speed (mm/s)
  Z     : z-offset     (mm, layer 1); layer 2 = z + LAYER_HEIGHT_MM

Usage:
  python generate_nc_files.py

Expects:
  - Template NC file : (same folder as this script) ai_poc_48_template.nc
  - CSV file         : data/lhs/lhs_bioprint_samples_semicolon.csv

Output:
  data/nc_files/   (created automatically if not existing)
"""

import os
import re
import math
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
TEMPLATE_FILE = BASE_DIR / "ai_poc_48_template.nc"
CSV_FILE      = BASE_DIR / "data" / "lhs" / "lhs_bioprint_samples_semicolon.csv"
OUTPUT_DIR    = BASE_DIR / "data" / "nc_files"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 48-well plate geometry ────────────────────────────────────────────────────
ROWS_ALPHA = ['A', 'B', 'C', 'D', 'E', 'F']
COLS       = [1, 2, 3, 4, 5, 6, 7, 8]
N_COLS     = 8

Z_IMAGING  = 2.24
Z_SAFE     = 22.340

X_A1, Y_A1 = -234.090,  95.780
X_A8        = -143.150
X_F1        = -234.090
Y_F1        =  30.630
X_STEP = (X_A8 - X_A1) / 7   # +12.991 mm per column
Y_STEP = (Y_F1 - Y_A1) / 5   # -13.030 mm per row

def build_well_map():
    wells = {}
    for r_idx, row in enumerate(ROWS_ALPHA):
        for c_idx, col in enumerate(COLS):
            x = round(X_A1 + c_idx * X_STEP, 3)
            y = round(Y_A1 + r_idx * Y_STEP, 3)
            wells[f"{row}{col}"] = (x, y, Z_IMAGING)
    return wells

WELL_MAP = build_well_map()

# ── G55 origin X positions per column (from original template) ────────────────
COL_X = {
    1: -45.785,
    2: -32.705,
    3: -19.625,
    4:  -6.545,
    5:   6.535,
    6:  19.615,
    7:  32.695,
    8:  45.775,
}

# ── G55 origin Y positions per row (A=top, F=bottom in G55 frame) ─────────────
ROW_Y = {
    'A':  32.720,
    'B':  19.640,
    'C':   6.560,
    'D':  -6.520,
    'E': -19.600,
    'F': -32.680,
}

# ── Grid geometry ─────────────────────────────────────────────────────────────
# All coordinates are relative to the G55 well centre (0, 0).
#
# 4 strands per layer, spaced 0.9 mm apart, centred on 0:
#   positions: -1.35, -0.45, +0.45, +1.35 mm
# Strand extent: ±1.80 mm from centre → 3.60 mm total span
# Travel Z (between wells): ~18.4 mm (kept from original template)

STRAND_POSITIONS = [-1.35, -0.45, 0.45, 1.35]   # mm from well centre
STRAND_EXTENT    =  1.80                          # mm half-length of each strand
Z_TRAVEL         = 18.400                         # mm safe travel Z between wells
LAYER_HEIGHT_MM  =  0.35                          # mm Z increment for layer 2


def generate_grid_toolpath(pressure_kpa, speed_mms, z_offset_mm):
    """
    Generate G-code lines for a single well: 0°–90° two-layer grid.

    Layer 1 (0°): 4 strands running in X-direction
      - Nozzle moves in X at fixed Y positions: STRAND_POSITIONS
      - Even strands: left→right (+X); odd strands: right→left (−X) [boustrophedon]
      - Start position before M160: X=−EXTENT, Y=strand_pos (or reversed)

    Layer 2 (90°): 4 strands running in Y-direction
      - Nozzle moves in Y at fixed X positions: STRAND_POSITIONS
      - Even strands: bottom→top (+Y); odd strands: top→bottom (−Y)
      - Z lifted by LAYER_HEIGHT_MM

    Each layer:  M160 (dispense on) at first strand, M161 (dispense off) after last
    Between layers: lift to Z_TRAVEL, reposition, lower to layer 2 Z

    Returns list of G-code strings (no trailing newlines).
    """
    m200_val  = int(round(pressure_kpa * 10))
    f_val     = f"{speed_mms:.3f}"
    z1        = f"{z_offset_mm:.3f}"
    z2        = f"{z_offset_mm + LAYER_HEIGHT_MM:.3f}"
    e         = STRAND_EXTENT

    lines = []

    # ── Layer 1: X-direction strands ─────────────────────────────────────────
    # Start position: before first strand (top-left corner, outside construct)
    # Boustrophedon: strand 0 left→right, strand 1 right→left, etc.
    start_x   = f"{-e:.3f}"
    start_y   = f"{STRAND_POSITIONS[0]:.3f}"

    lines.append(f"G00 G55 X{start_x} Y{start_y}    ; layer 1 start")
    lines.append(f"M151                               ; engage tool")
    lines.append(f"Z{z1}                              ; lower to layer 1 Z")
    lines.append(f"M160                               ; dispensing ON")
    lines.append(f"G01 X{e:.3f}                       ; strand L1-1  (+X)")

    for i, y_pos in enumerate(STRAND_POSITIONS[1:], start=1):
        y_str = f"{y_pos:.3f}"
        lines.append(f"Y{y_str}                              ; traverse to strand L1-{i+1}")
        if i % 2 == 1:
            lines.append(f"X{-e:.3f}                          ; strand L1-{i+1} (−X)")
        else:
            lines.append(f"X{e:.3f}                           ; strand L1-{i+1} (+X)")

    lines.append(f"M161                               ; dispensing OFF")

    # ── Lift and reposition for layer 2 ──────────────────────────────────────
    # Start position: before first Y-strand (bottom-left corner)
    start_x2  = f"{STRAND_POSITIONS[0]:.3f}"
    start_y2  = f"{-e:.3f}"

    lines.append(f"G00 Z{Z_TRAVEL:.3f}                ; lift to travel Z")
    lines.append(f"X{start_x2} Y{start_y2}            ; layer 2 start position")
    lines.append(f"Z{z2}                               ; lower to layer 2 Z")
    lines.append(f"M160                               ; dispensing ON")
    lines.append(f"G01 Y{e:.3f}                        ; strand L2-1  (+Y)")

    for i, x_pos in enumerate(STRAND_POSITIONS[1:], start=1):
        x_str = f"{x_pos:.3f}"
        lines.append(f"X{x_str}                              ; traverse to strand L2-{i+1}")
        if i % 2 == 1:
            lines.append(f"Y{-e:.3f}                          ; strand L2-{i+1} (−Y)")
        else:
            lines.append(f"Y{e:.3f}                           ; strand L2-{i+1} (+Y)")

    lines.append(f"M161                               ; dispensing OFF")

    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Build a complete column block for one sample (6 wells, rows A–F)
# ─────────────────────────────────────────────────────────────────────────────

def build_column_block(col_num, sample_id, pressure_kpa, speed_mms, z_offset_mm,
                       progress_start, progress_end):
    """
    Return list of G-code lines for one full column (6 wells = rows A-F).

    progress_start / progress_end: M110 range to distribute across the 6 wells.
    """
    m200_val  = int(round(pressure_kpa * 10))
    f_val     = f"{speed_mms:.3f}"
    col_x     = COL_X[col_num]

    lines = []
    lines.append("")
    lines.append(
        f"; ── Col {col_num} | Sample {sample_id} | "
        f"P={int(pressure_kpa)}kPa  F={speed_mms:.1f}mm/s  Z={z_offset_mm:.3f}mm ──"
    )

    # Parameters set once at start of column (inherited by all 6 wells)
    n_wells = len(ROWS_ALPHA)
    progress_step = max(1, (progress_end - progress_start) // n_wells)

    first_well = True
    for w_idx, row in enumerate(ROWS_ALPHA):
        row_y   = ROW_Y[row]
        prog    = progress_start + w_idx * progress_step

        lines.append(f"G805[{col_x:.3f}, {row_y:.3f}, 2.620]  ; G55 origin: {row}{col_num}")
        lines.append("G55")

        if first_well:
            # Full initialisation for first well of column
            lines.append("")
            lines.append("; Changing tool to 'PSD 1'")
            lines.append("#FLUSH WAIT")
            lines.append("T1")
            lines.append("G807[1, 0.002, 0.002]  ; time-based start/stop delays [s]")
            lines.append(f"M200={m200_val}         ; pressure {pressure_kpa}kPa")
            lines.append(f"F{f_val}               ; nozzle speed")
            first_well = False
        else:
            lines.append(f"G00 Z{Z_TRAVEL:.3f}       ; lift to travel Z")
            lines.append(f"M200={m200_val}         ; pressure {pressure_kpa}kPa")
            lines.append(f"F{f_val}")

        lines.append(f"M110={prog}              ; progress {prog}%")

        # Insert the grid toolpath
        toolpath = generate_grid_toolpath(pressure_kpa, speed_mms, z_offset_mm)
        lines.extend(toolpath)

        lines.append(f"M110={prog + progress_step - 1}  ; progress")

    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Parse template header (everything before first column content)
# ─────────────────────────────────────────────────────────────────────────────

def parse_template_header(template_path):
    """
    Extract only the initialization header from the template NC file.
    Returns list of header lines up to (not including) the first G805 line.
    """
    with open(template_path, 'r') as f:
        lines = f.read().splitlines()

    header = []
    for line in lines:
        if re.match(r'G805\[', line.strip()):
            break
        header.append(line.rstrip())
    return header


def update_header_pressure(header_lines, pressure_kpa):
    """Update M200 in initialization header with first sample pressure."""
    m200_val = int(round(pressure_kpa * 10))
    updated  = []
    for line in header_lines:
        if re.match(r'\s*M200=\d+', line):
            updated.append(f"M200={m200_val} ; Set pressure to {pressure_kpa}kPa")
        elif re.match(r'\s*M300=\d+', line) or re.match(r'\s*M302', line.strip()):
            pass  # strip temperature commands — set manually in Architect UI
        else:
            updated.append(line.rstrip())
    return updated


# ─────────────────────────────────────────────────────────────────────────────
# Generate imaging block
# ─────────────────────────────────────────────────────────────────────────────

def generate_imaging_block(n_active_cols=8):
    """Imaging G-code for all active columns, column-major order."""
    well_names = [f"{r}{c}"
                  for c in range(1, n_active_cols + 1)
                  for r in ROWS_ALPHA]

    lines = []
    lines.append("")
    lines.append("; ============================================================")
    lines.append("; CAMERA IMAGING POSITIONS")
    lines.append(f"; Wells: {', '.join(well_names)}")
    lines.append("; Column-major order: col1 (A1-F1) -> col2 (A2-F2) -> ...")
    lines.append("; Slot 1 selected (T1) — camera physically at slot 5.")
    lines.append(f"; Z safe travel = {Z_SAFE} mm  |  Z imaging = {Z_IMAGING} mm")
    lines.append("; ============================================================")
    lines.append("")
    lines.append("#FLUSH WAIT")
    lines.append("#CONTOUR MODE OFF          ; Exit tracking mode from printing")
    lines.append("#FLUSH WAIT")
    lines.append("")
    lines.append("T1                         ; Select slot 1 (coordinate frame reference)")
    lines.append("G803                       ; Move to system safe height")
    lines.append("")

    for well in well_names:
        x, y, z = WELL_MAP[well]
        lines.append(f"; --- Well {well} ---")
        lines.append(f"G00 G54 G90 Z{Z_SAFE:.3f}          ; Raise to safe Z first")
        lines.append(f"G00 X{x:.3f} Y{y:.3f}              ; Move XY to {well}")
        lines.append(f"G00 Z{z:.3f}                ; Lower to imaging height")
        lines.append(
            f'V.E.UserInteraction.Message = "Camera at {well} - trigger imaging, then click OK"'
        )
        lines.append(f"M121                       ; Pause for manual camera trigger")
        lines.append("")

    lines.append("; --- Return home ---")
    lines.append(f"G00 G54 G90 Z{Z_SAFE:.3f}          ; Safe Z before going home")
    lines.append("G800                       ; Go home")
    lines.append("M110=1000")
    lines.append("M30")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Batch samples (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def batch_samples(df):
    """
    Split samples into batches of N_COLS with stride-based interleaving,
    so each NC file covers the full pressure range.

    Example with 96 samples → 12 files of 8:
      File  1: indices  0, 12, 24, 36, 48, 60, 72, 84
      File  2: indices  1, 13, 25, 37, 49, 61, 73, 85
      ...
    """
    n       = len(df)
    n_files = math.ceil(n / N_COLS)
    all_s   = [df.iloc[i] for i in range(n)]
    batches = []
    for k in range(n_files):
        batch = [all_s[k + j * n_files]
                 for j in range(N_COLS)
                 if k + j * n_files < n]
        batches.append(batch)
    return batches


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Load CSV ──────────────────────────────────────────────────────────────
    if not CSV_FILE.is_file():
        print(f"[ERROR] CSV not found: {CSV_FILE}")
        print("  Run lhs_sampling.py first.")
        return

    df = pd.read_csv(CSV_FILE, sep=';', index_col=0)
    print(f"Loaded {len(df)} samples from:")
    print(f"  {CSV_FILE}")

    # ── Parse template header ─────────────────────────────────────────────────
    if not TEMPLATE_FILE.is_file():
        print(f"\n[ERROR] Template not found: {TEMPLATE_FILE}")
        return

    header_lines = parse_template_header(TEMPLATE_FILE)
    print(f"\nTemplate header parsed: {len(header_lines)} lines")

    # ── Batch samples ─────────────────────────────────────────────────────────
    batches = batch_samples(df)
    print(f"\nBatching into {len(batches)} NC files:")
    for idx, batch in enumerate(batches):
        sids = [int(s.name) for s in batch]
        print(f"  File {idx+1:>2}: {len(batch):>2} cols | sample IDs={sids}")

    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"\nGeometry: 0°–90° two-layer grid")
    print(f"  Layer 1 (0°):  4 X-strands at Y = {STRAND_POSITIONS} mm")
    print(f"  Layer 2 (90°): 4 Y-strands at X = {STRAND_POSITIONS} mm")
    print(f"  Strand extent: ±{STRAND_EXTENT} mm  |  Layer height: {LAYER_HEIGHT_MM} mm")
    print()

    # ── Generate NC files ─────────────────────────────────────────────────────
    for file_idx, batch in enumerate(batches):
        n_samples = len(batch)
        sids      = [int(s.name) for s in batch]
        fname     = f"data_collection_48well_8cols_s{sids[0]}-s{sids[-1]}.nc"
        fpath     = OUTPUT_DIR / fname

        first_pressure = int(batch[0]['Pressure_kPa'])

        # File comment block
        file_comments = [f"% {fname}"]
        file_comments.append(f"; Generated by generate_nc_files.py")
        file_comments.append(f"; 48-well plate | {n_samples} samples (columns)")
        file_comments.append(f"; Geometry: 0-90 two-layer grid | "
                             f"strand spacing 0.9mm | 4+4 strands | 3x3 pores")
        file_comments.append(f"; CSV: {CSV_FILE.name}")
        file_comments.append(f"; Temperature: set manually in Architect UI (not in G-code)")
        file_comments.append(f";")
        file_comments.append(
            f"; {'Col':<5} {'SampleID':<10} {'P(kPa)':<10} {'F(mm/s)':<10} {'Z(mm)':<8}"
        )
        for col_num, s in enumerate(batch, start=1):
            sid = int(s.name)
            p   = int(s['Pressure_kPa'])
            f_s = float(s['NozzleSpeed_mms'])
            z   = float(s['Zoffset_mm'])
            file_comments.append(
                f"; {col_num:<5} {sid:<10} {p:<10} {f_s:<10.1f} {z:<8}"
            )
        file_comments.append(f";")

        # Header (update M200 with first sample pressure)
        updated_header = update_header_pressure(header_lines, first_pressure)
        header_body    = updated_header[1:]   # skip first line (% filename)
        all_lines      = file_comments + header_body

        # Progress counter: distribute M110 values across all columns/wells
        total_wells    = n_samples * len(ROWS_ALPHA)
        prog_per_well  = max(1, 900 // total_wells)  # scale to 0–900 range
        global_prog    = 40  # start after header initialisation

        # Column blocks (one per sample)
        for col_idx, col_num in enumerate(COLS):
            if col_num > n_samples:
                break

            sample   = batch[col_num - 1]
            pressure = float(sample['Pressure_kPa'])
            speed    = float(sample['NozzleSpeed_mms'])
            z_off    = float(sample['Zoffset_mm'])
            sid      = int(sample.name)

            prog_start = global_prog
            prog_end   = prog_start + prog_per_well * len(ROWS_ALPHA)
            global_prog = prog_end

            col_block = build_column_block(
                col_num=col_num,
                sample_id=sid,
                pressure_kpa=pressure,
                speed_mms=speed,
                z_offset_mm=z_off,
                progress_start=prog_start,
                progress_end=prog_end,
            )
            all_lines.extend(col_block)

        # Imaging block
        all_lines.append("")
        all_lines.append("#FLUSH WAIT")
        all_lines.append(generate_imaging_block(n_active_cols=n_samples))

        # Write file
        with open(fpath, 'w', newline='\r\n', encoding='utf-8') as f:
            f.write("\n".join(all_lines))

        print(f"  [{file_idx+1:>2}] {fname}")

    print(f"\nDone. {len(batches)} NC files written to:")
    print(f"  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()