#!/usr/bin/env python3
"""
generate_nc_files.py
--------------------
Reads the LHS sample CSV and generates NC files for a 48-well plate.
Each NC file contains 8 columns (= 8 samples), each column has 6 constructs (rows A-F).

GEOMETRY: 2D cross-hatch grid (single layer, 0-90 degree pattern)
  Pass 1: 4 horizontal strands running in X-direction (at fixed Y positions)
  Pass 2: 4 vertical strands running in Y-direction (at fixed X positions)
  Both passes at the same Z height (z_offset).
  Dispensing is OFF during travel between strands -> open square pores.

  Strand positions (from well centre): -1.35, -0.45, +0.45, +1.35 mm
  Strand extent:  +/-1.80 mm from centre (3.60 mm total span)
  Nominal pore:   ~0.49 mm x 0.49 mm  (spacing 0.9 mm - nozzle ID ~0.41 mm)
  Grid:           3x3 = 9 open pores per construct

Key fixes vs. previous version:
  1. M151 (engage tool) called ONCE per column, at the first well only.
     Wells 2-6 reposition with G00 only. This prevents the
     "tool cannot be engaged at this height" collision error.
  2. Dispense (M160) is turned OFF (M161) between strands during travel,
     so cross-hatch pores are genuinely open, not filled in.
"""

import re
import math
import pandas as pd
from pathlib import Path

# == Paths ====================================================================
BASE_DIR      = Path(__file__).parent
TEMPLATE_FILE = BASE_DIR / "ai_poc_48_template.nc"
CSV_FILE      = BASE_DIR / "data" / "lhs" / "lhs_bioprint_samples_semicolon.csv"
OUTPUT_DIR    = BASE_DIR / "data" / "nc_files"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# == 48-well plate geometry ===================================================
ROWS_ALPHA = ['A', 'B', 'C', 'D', 'E', 'F']
COLS       = [1, 2, 3, 4, 5, 6, 7, 8]
N_COLS     = 8

Z_IMAGING = 2.24
Z_SAFE    = 22.340

X_A1, Y_A1 = -234.090,  95.780
X_A8        = -143.150
Y_F1        =  30.630
X_STEP = (X_A8 - X_A1) / 7
Y_STEP = (Y_F1 - Y_A1) / 5

def build_well_map():
    wells = {}
    for r_idx, row in enumerate(ROWS_ALPHA):
        for c_idx, col in enumerate(COLS):
            x = round(X_A1 + c_idx * X_STEP, 3)
            y = round(Y_A1 + r_idx * Y_STEP, 3)
            wells[f"{row}{col}"] = (x, y, Z_IMAGING)
    return wells

WELL_MAP = build_well_map()

# == G55 origin positions (from original template) ============================
COL_X = {
    1: -45.785, 2: -32.705, 3: -19.625, 4: -6.545,
    5:   6.535, 6:  19.615, 7:  32.695, 8: 45.775,
}

ROW_Y = {
    'A':  32.720, 'B':  19.640, 'C':   6.560,
    'D':  -6.520, 'E': -19.600, 'F': -32.680,
}

# == Grid geometry (mm, relative to G55 well centre = 0,0) ====================
# 4 strands per pass, 0.9 mm pitch, centred on zero:
#   -1.35, -0.45, +0.45, +1.35 mm
# Strand length: 3.60 mm total (+/-1.80 mm from centre)

STRAND_POS = [-2.25, -0.75, 0.75, 2.25]   # 1.5 mm pitch, 4 strands
STRAND_EXT =  3.00                         # strand half-length (mm)
Z_TRAVEL   = 18.400


# =============================================================================
def grid_toolpath(z_mm):
    """
    G-code lines for the 2D cross-hatch grid in one well.
    Tool is already engaged and Z is already correct when this runs.

    Matches the original RegenHU template pattern exactly:
      M160 once at the very start (dispensing ON)
      G01 for all printing moves
      G00 for repositioning between strands (fast traverse — no M161/M160 toggle)
      M161 once at the very end (dispensing OFF)

    Pores form because G00 is a rapid move: the brief pressure drop during
    the fast repositioning leaves a gap between strands. Toggling M160/M161
    per strand causes droplets/thick lines because the pneumatic system
    cannot respond fast enough.
    """
    e  = STRAND_EXT
    sp = STRAND_POS
    lines = []

    # M160 once — dispensing ON for the entire well
    lines.append("M160")

    # Pass 1: horizontal (X) strands
    # Start position already set before this block (at X=-e, Y=sp[0])
    lines.append(f"G01 X{e:.3f}             ; H-strand 1")
    for i in range(1, len(sp)):
        y = sp[i]
        # G00 rapid to next strand start (pressure briefly drops = gap between strands)
        x_start = -e if i % 2 == 0 else  e
        x_end   =  e if i % 2 == 0 else -e
        lines.append(f"G00 X{x_start:.3f} Y{y:.3f}")
        lines.append(f"G01 X{x_end:.3f}             ; H-strand {i+1}")

    # Reposition for Pass 2: move to start of first vertical strand
    # G00 so no extrusion during reposition
    y_start_v = -e if 0 % 2 == 0 else e
    lines.append(f"G00 X{sp[0]:.3f} Y{y_start_v:.3f}")

    # Pass 2: vertical (Y) strands
    y_end_v = e if 0 % 2 == 0 else -e
    lines.append(f"G01 Y{y_end_v:.3f}             ; V-strand 1")
    for i in range(1, len(sp)):
        x = sp[i]
        y_start = -e if i % 2 == 0 else  e
        y_end   =  e if i % 2 == 0 else -e
        lines.append(f"G00 X{x:.3f} Y{y_start:.3f}")
        lines.append(f"G01 Y{y_end:.3f}             ; V-strand {i+1}")

    # M161 once — dispensing OFF
    lines.append("M161")

    return lines


# =============================================================================
def build_column_block(col_num, sample_id, pressure_kpa, speed_mms, z_mm,
                       prog_start):
    """
    G-code for one full column (6 wells, rows A-F).

    Matches original template EXACTLY:

    First well (A):
        G805[x,y,z] / G55           <- set + activate G55 once
        G00 X{xs} Y{ys}             <- XY move (modal G00, G55 active)
        M151                         <- engage tool
        Z{z_mm}                      <- lower (modal G00)
        M110 / M160 / G01 / M161

    Wells B-F:
        G805[x,y,z] / G55           <- new G55 origin
        G00 Z18.400                  <- lift
        X{xs} Y{ys}                  <- XY (implicit G00, G55 modal — NO 'G00 G55')
        Z{z_mm}                      <- lower (implicit G00)
        M110 / M160 / G01 / M161
    """
    m200 = int(round(pressure_kpa * 10))
    fval = f"{speed_mms:.3f}"
    z    = f"{z_mm:.3f}"
    cx   = COL_X[col_num]
    xs   = f"{-STRAND_EXT:.3f}"
    ys   = f"{STRAND_POS[0]:.3f}"

    lines = []
    lines.append("")
    lines.append(
        f"; -- Col {col_num} | Sample {sample_id} | "
        f"P={int(pressure_kpa)}kPa  F={speed_mms:.1f}mm/s  Z={z_mm:.3f}mm --"
    )

    prog = prog_start

    for w_idx, row in enumerate(ROWS_ALPHA):
        ry = ROW_Y[row]

        lines.append(f"G805[{cx:.3f}, {ry:.3f}, 2.620] ; G55 origin: {row}{col_num}")
        lines.append("G55")

        if w_idx == 0:
            lines.append("")
            lines.append("; Changing tool to 'PSD 1'")
            lines.append("#FLUSH WAIT")
            lines.append("T1")
            lines.append("G807[1, 0.002, 0.002] ; time-based start/stop delays [s]")
            lines.append(f"M200={m200} ; pressure {pressure_kpa:.0f}kPa")
            lines.append(f"F{fval}")
            lines.append(f"G00 X{xs} Y{ys}")    # XY move — G55 is active, plain G00
            lines.append("M151 ; Engage tool for printing")
            lines.append(f"Z{z}")               # lower — modal G00
        else:
            lines.append(f"G00 Z{Z_TRAVEL:.3f}")  # lift
            lines.append(f"M200={m200}")
            lines.append(f"F{fval}")
            lines.append(f"X{xs} Y{ys}")          # XY — implicit G00, G55 modal
            lines.append(f"Z{z}")                  # lower — implicit G00

        lines.append(f"M110={prog}")
        lines.extend(grid_toolpath(z_mm))
        prog += 10

    lines.append(f"G00 Z{Z_TRAVEL:.3f} ; lift after column")
    return lines


# =============================================================================
def parse_template_header(template_path):
    raw    = template_path.read_text().splitlines()
    header = []
    for line in raw:
        if re.match(r'G805\[', line.strip()):
            break
        header.append(line.rstrip())
    return header


def update_header_pressure(header_lines, pressure_kpa):
    m200 = int(round(pressure_kpa * 10))
    out  = []
    for line in header_lines:
        s = line.strip()
        if re.match(r'M200=\d+', s):
            out.append(f"M200={m200} ; Set pressure to {pressure_kpa:.0f}kPa")
        elif re.match(r'M300=\d+', s) or re.match(r'M302', s):
            pass  # temperature set manually in Architect UI
        else:
            out.append(line.rstrip())
    return out


# =============================================================================
def generate_imaging_block(n_active_cols=8):
    well_names = [f"{r}{c}"
                  for c in range(1, n_active_cols + 1)
                  for r in ROWS_ALPHA]
    lines = []
    lines.append("")
    lines.append("; ============================================================")
    lines.append("; CAMERA IMAGING POSITIONS")
    lines.append(f"; Wells: {', '.join(well_names)}")
    lines.append("; Column-major order: col1 (A1-F1) -> col2 (A2-F2) -> ...")
    lines.append("; Slot 1 selected (T1) -- camera physically at slot 5.")
    lines.append(f"; Z safe = {Z_SAFE} mm  |  Z imaging = {Z_IMAGING} mm")
    lines.append("; ============================================================")
    lines.append("")
    lines.append("#FLUSH WAIT")
    lines.append("#CONTOUR MODE OFF          ; Exit tracking mode from printing")
    lines.append("#FLUSH WAIT")
    lines.append("")
    lines.append("T1                         ; Select slot 1")
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
        lines.append("M121                       ; Pause for manual camera trigger")
        lines.append("")

    lines.append("; --- Return home ---")
    lines.append(f"G00 G54 G90 Z{Z_SAFE:.3f}          ; Safe Z before going home")
    lines.append("G800                       ; Go home")
    lines.append("M110=1000")
    lines.append("M30")
    return "\n".join(lines)


# =============================================================================
def batch_samples(df):
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


# =============================================================================
def main():
    if not CSV_FILE.is_file():
        print(f"[ERROR] CSV not found: {CSV_FILE}")
        print("  Run lhs_sampling.py first.")
        return

    df = pd.read_csv(CSV_FILE, sep=';', index_col=0)
    print(f"Loaded {len(df)} samples from:\n  {CSV_FILE}")

    if not TEMPLATE_FILE.is_file():
        print(f"\n[ERROR] Template not found: {TEMPLATE_FILE}")
        return

    header_lines = parse_template_header(TEMPLATE_FILE)
    print(f"\nTemplate header parsed: {len(header_lines)} lines")

    batches = batch_samples(df)
    print(f"\nBatching into {len(batches)} NC files:")
    for idx, batch in enumerate(batches):
        sids = [int(s.name) for s in batch]
        print(f"  File {idx+1:>2}: {len(batch):>2} cols | sample IDs={sids}")

    print(f"\nGeometry: 2D cross-hatch grid (single layer)")
    print(f"  Strand positions: {STRAND_POS} mm")
    print(f"  Strand extent:    +/-{STRAND_EXT} mm ({2*STRAND_EXT:.1f} mm span)")
    print(f"  Pores:            {len(STRAND_POS)-1}x{len(STRAND_POS)-1} = {(len(STRAND_POS)-1)**2} open pores")
    print(f"\nOutput directory: {OUTPUT_DIR}\n")

    for file_idx, batch in enumerate(batches):
        n_samples = len(batch)
        sids      = [int(s.name) for s in batch]
        fname     = f"data_collection_48well_8cols_s{sids[0]}-s{sids[-1]}.nc"
        fpath     = OUTPUT_DIR / fname

        first_pressure = float(batch[0]['Pressure_kPa'])

        file_comments = [f"% {fname}"]
        file_comments.append("; Generated by generate_nc_files.py")
        file_comments.append(
            "; Geometry: 2D cross-hatch grid | 4+4 strands | 3x3 open pores | single layer"
        )
        file_comments.append(f"; CSV: {CSV_FILE.name}")
        file_comments.append("; Temperature: set manually in Architect UI (not in G-code)")
        file_comments.append(";")
        file_comments.append(
            f"; {'Col':<5} {'SampleID':<10} {'P(kPa)':<10} {'F(mm/s)':<10} {'Z(mm)':<8}"
        )
        for col_num, s in enumerate(batch, start=1):
            file_comments.append(
                f"; {col_num:<5} {int(s.name):<10} {int(s['Pressure_kPa']):<10} "
                f"{float(s['NozzleSpeed_mms']):<10.1f} {float(s['Zoffset_mm']):<8}"
            )
        file_comments.append(";")

        updated_header = update_header_pressure(header_lines, first_pressure)
        all_lines      = file_comments + updated_header[1:]

        for col_num in COLS:
            if col_num > n_samples:
                break
            sample = batch[col_num - 1]
            col_block = build_column_block(
                col_num=col_num,
                sample_id=int(sample.name),
                pressure_kpa=float(sample['Pressure_kPa']),
                speed_mms=float(sample['NozzleSpeed_mms']),
                z_mm=float(sample['Zoffset_mm']),
                prog_start=40 + (col_num - 1) * 10,
            )
            all_lines.extend(col_block)

        all_lines.append("")
        all_lines.append("#FLUSH WAIT")
        all_lines.append(generate_imaging_block(n_active_cols=n_samples))

        with open(fpath, 'w', newline='\r\n', encoding='utf-8') as f:
            f.write("\n".join(all_lines))

        print(f"  [{file_idx+1:>2}] {fname}")

    print(f"\nDone. {len(batches)} NC files written to:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
