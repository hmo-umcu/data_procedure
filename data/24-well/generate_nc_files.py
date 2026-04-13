#!/usr/bin/env python3
"""
generate_nc_files.py
--------------------
Reads the LHS sample CSV and generates 25 NC files, each containing
4 rows of printing (one sample per row = 6 wells per sample).

NC file naming:
  data_collection_24well_4rows_imaging_0-3.nc    (samples 0,1,2,3)
  data_collection_24well_4rows_imaging_4-7.nc    (samples 4,5,6,7)
  ...
  data_collection_24well_4rows_imaging_96-99.nc  (samples 96,97,98,99)

Each row maps to a plate row:
  Sample index 0 (within NC file) -> row A  (Y = 28.955)
  Sample index 1 (within NC file) -> row B  (Y =  9.655)
  Sample index 2 (within NC file) -> row C  (Y = -9.645)
  Sample index 3 (within NC file) -> row D  (Y = -28.945)

Parameters updated per row:
  - M200 : pressure (kPa -> value * 10 as per G-code dict 0.1kPa units)
  - F    : nozzle speed (mm/s)
  - Z    : z-offset (all printing Z occurrences within that row's block)

NOTE: Temperature (M300/M301/M302) is intentionally excluded.
      Set temperature manually in Architect UI before starting each NC file.
      Temperature is still logged in the summary comment for your records.

Usage:
  python generate_nc_files.py

Expects in same directory as script:
  - data_collection_24well_4rows_template.nc

CSV and output paths defined below.
"""

import os
import re
import pandas as pd

# -- Paths --------------------------------------------------------------------
TEMPLATE_FILE = "data_collection_24well_4rows_template.nc"
CSV_FILE      = r"C:\Users\hmo\hmo_workspace\data_procedure\data\lhs\lhs_bioprint_samples_3param.csv"
OUTPUT_DIR    = r"C:\Users\hmo\hmo_workspace\data_procedure\data\nc_files"

# -- Imaging constants (from nc_imaging.py) -----------------------------------
Z_IMAGING = 49.58
Z_SAFE    = 65.34

ROWS_ALPHA = ['A', 'B', 'C', 'D']
COLS       = [1, 2, 3, 4, 5, 6]

X_A1, Y_A1 = -236.800,  91.470
X_STEP      = ((-138.480) - X_A1) / 5   # +19.664 mm per column
Y_STEP      = (32.650 - Y_A1) / 3       # -19.607 mm per row


def build_well_map():
    wells = {}
    for r_idx, row in enumerate(ROWS_ALPHA):
        for c_idx, col in enumerate(COLS):
            x = round(X_A1 + c_idx * X_STEP, 3)
            y = round(Y_A1 + r_idx * Y_STEP, 3)
            wells[f"{row}{col}"] = (x, y, Z_IMAGING)
    return wells

WELL_MAP = build_well_map()

# -- Row Y-values in template (used to identify row blocks) -------------------
ROW_Y = {
    'A':  28.955,
    'B':   9.655,
    'C':  -9.645,
    'D': -28.945,
}


# -----------------------------------------------------------------------------
# STEP 1: Parse template into 4 row blocks + header
# -----------------------------------------------------------------------------

def parse_template(template_path):
    with open(template_path, 'r') as f:
        raw = f.read()

    lines = raw.splitlines()

    row_start = {}
    for row, y_val in ROW_Y.items():
        pattern = re.compile(
            r'G805\[.*?,\s*' + re.escape(f"{y_val:.3f}") + r'\s*,.*?\]'
        )
        for i, line in enumerate(lines):
            if pattern.search(line):
                row_start[row] = i
                break

    for row in ROWS_ALPHA:
        if row not in row_start:
            raise ValueError(f"Could not find row {row} (Y={ROW_Y[row]}) in template")

    sorted_starts = sorted(row_start.items(), key=lambda x: x[1])
    header_lines  = lines[:sorted_starts[0][1]]

    row_blocks = {}
    for idx, (row, start) in enumerate(sorted_starts):
        end   = sorted_starts[idx + 1][1] if idx + 1 < len(sorted_starts) else len(lines)
        block = lines[start:end]
        end_cmds = re.compile(r'^\s*(G800|M110=1000|M30|#FLUSH WAIT)\s*(;.*)?$')
        while block and (block[-1].strip() == '' or end_cmds.match(block[-1])):
            block.pop()
        row_blocks[row] = block

    return header_lines, row_blocks


# -----------------------------------------------------------------------------
# STEP 2: Update a row block — pressure, speed, Z only (NO temperature)
# -----------------------------------------------------------------------------

def update_row_block(block_lines, pressure_kpa, speed_mms, z_offset_mm):
    """
    Updates per row:
      M200  -> pressure_kpa * 10   (0.1 kPa units)
      F     -> speed_mms           (mm/s)
      Z     -> z_offset_mm         (printing Z only, travel Z18.4 unchanged)
      M300  -> REMOVED (temperature set manually in Architect UI)
      M302  -> REMOVED (no longer needed without M300)
    """
    m200_val = int(round(pressure_kpa * 10))
    f_val    = f"{speed_mms:.3f}"
    z_val    = f"{z_offset_mm:.3f}"

    new_lines = []
    for line in block_lines:
        stripped = line.strip()

        # Update M200 (pressure)
        if re.match(r'M200=\d+', stripped):
            new_lines.append(f"M200={m200_val} ; Set pressure to {pressure_kpa}kPa")
            continue

        # Update F (nozzle speed)
        if re.match(r'F\d+\.\d+', stripped):
            new_lines.append(f"F{f_val}")
            continue

        # REMOVE M300 (temperature setpoint) — handled manually in UI
        if re.match(r'M300=\d+', stripped):
            continue

        # REMOVE M302 (wait for tool temperature) — not needed without M300
        if re.match(r'M302', stripped):
            continue

        # Update printing Z only (< 5mm threshold excludes travel Z ~18.4)
        if re.match(r'Z\d+\.\d+', stripped):
            z_num = float(re.match(r'Z(\d+\.\d+)', stripped).group(1))
            if z_num < 5.0:
                new_lines.append(f"Z{z_val}")
                continue

        new_lines.append(line.rstrip())

    return new_lines


def update_header(header_lines, pressure_kpa):
    """Update M200 in the initialization header."""
    m200_val = int(round(pressure_kpa * 10))
    new_lines = []
    for line in header_lines:
        if re.match(r'\s*M200=\d+', line):
            new_lines.append(f"M200={m200_val} ; Set pressure to {pressure_kpa}kPa")
        else:
            new_lines.append(line.rstrip())
    return new_lines


# -----------------------------------------------------------------------------
# STEP 3: Generate imaging G-code block
# -----------------------------------------------------------------------------

def generate_imaging_block(well_names):
    lines = []
    lines.append("")
    lines.append("; ============================================================")
    lines.append("; CAMERA IMAGING POSITIONS")
    lines.append(f"; Wells: {', '.join(well_names)}")
    lines.append("; Slot 1 selected (T1) to match coordinate frame of recorded")
    lines.append("; positions. Camera physically mounted at slot 5.")
    lines.append(f"; Z safe travel = {Z_SAFE} mm  |  Z imaging = {Z_IMAGING} mm")
    lines.append("; ============================================================")
    lines.append("")
    lines.append("#FLUSH WAIT")
    lines.append("#CONTOUR MODE OFF          ; Exit tracking mode from printing")
    lines.append("#FLUSH WAIT")
    lines.append("")
    lines.append("T1                         ; Select slot 1 (matches recorded coordinate frame)")
    lines.append("G803                       ; Move to system safe height")
    lines.append("")

    for well in well_names:
        x, y, z = WELL_MAP[well]
        lines.append(f"; --- Well {well} ---")
        lines.append(f"G00 G54 G90 Z{Z_SAFE:.3f}          ; Raise to safe Z first")
        lines.append(f"G00 X{x:.3f} Y{y:.3f}              ; Move XY to {well} camera position")
        lines.append(f"G00 Z{z:.3f}                ; Lower to imaging height")
        lines.append(f'V.E.UserInteraction.Message = "Camera at {well} - trigger imaging, then click OK"')
        lines.append(f"M121                       ; Pause for manual camera trigger")
        lines.append("")

    lines.append("; --- Return home ---")
    lines.append(f"G00 G54 G90 Z{Z_SAFE:.3f}          ; Safe Z before going home")
    lines.append("G800                       ; Go home")
    lines.append("M110=1000")
    lines.append("M30")

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# STEP 4: Main
# -----------------------------------------------------------------------------

def main():
    df = pd.read_csv(CSV_FILE, sep=';', index_col=0)
    print(f"Loaded {len(df)} samples from {CSV_FILE}")

    header_lines, row_blocks = parse_template(TEMPLATE_FILE)
    print(f"Template parsed: header={len(header_lines)} lines, "
          f"row blocks: { {r: len(b) for r, b in row_blocks.items()} }")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n_files = 0
    for file_idx in range(25):
        sample_indices = [file_idx * 4 + i for i in range(4)]

        valid = [i for i in sample_indices if i < len(df)]
        if len(valid) < 4:
            print(f"  [WARNING] File {file_idx}: only {len(valid)} samples — skipping")
            continue

        samples   = [df.iloc[i] for i in sample_indices]
        first_idx = sample_indices[0]
        last_idx  = sample_indices[3]
        fname     = f"data_collection_24well_4rows_imaging_{first_idx}-{last_idx}.nc"
        fpath     = os.path.join(OUTPUT_DIR, fname)

        # Header
        row_a_pressure = float(samples[0]['Pressure_kPa'])
        updated_header = update_header(header_lines, row_a_pressure)

        # Parameter summary comment at top of file
        param_comments = [f"% {fname}"]
        param_comments.append(f"; Generated by generate_nc_files.py")
        param_comments.append(f"; Samples {first_idx}-{last_idx} from CSV")
        param_comments.append(f";")
        param_comments.append(
            f"; {'Row':<6} {'SampleID':<10} {'P(kPa)':<10} "
            f"{'F(mm/s)':<10} {'Z(mm)':<8}"
        )
        for row_letter, s in zip(ROWS_ALPHA, samples):
            param_comments.append(
                f"; {row_letter:<6} {int(s.name):<10} "
                f"{int(s['Pressure_kPa']):<10} "
                f"{float(s['NozzleSpeed_mms']):<10.1f} "
                f"{float(s['Zoffset_mm']):<8}"
            )
        param_comments.append(f";")
        param_comments.append(
            f"; !! Set temperature manually in Architect UI before clicking Start !!"
        )
        param_comments.append(f";")

        # Skip original % line from header
        all_lines = param_comments + updated_header[1:]

        # Row blocks
        for row_letter, sample in zip(ROWS_ALPHA, samples):
            pressure = float(sample['Pressure_kPa'])
            speed    = float(sample['NozzleSpeed_mms'])
            z_off    = float(sample['Zoffset_mm'])

            updated_block = update_row_block(
                list(row_blocks[row_letter]),
                pressure, speed, z_off
            )

            all_lines.append("")
            all_lines.append(
                f"; -- Row {row_letter} | Sample {int(sample.name)} | "
                f"P={int(pressure)}kPa  F={speed:.1f}mm/s  "
                f"Z={z_off:.3f}mm --"
            )
            all_lines.extend(updated_block)

        # Imaging block
        all_lines.append("")
        all_lines.append("#FLUSH WAIT")
        all_well_names = [f"{r}{c}" for r in ROWS_ALPHA for c in COLS]
        all_lines.append(generate_imaging_block(all_well_names))

        # Write
        with open(fpath, 'w', newline='\r\n', encoding='utf-8') as f:
            f.write("\n".join(all_lines))

        print(f"  Written: {fname}  "
              f"P={[int(s['Pressure_kPa']) for s in samples]}kPa  "
              f"F={[float(s['NozzleSpeed_mms']) for s in samples]}mm/s  "
              f"Z={[float(s['Zoffset_mm']) for s in samples]}mm")
        n_files += 1

    print(f"\nDone. {n_files} NC files written to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
