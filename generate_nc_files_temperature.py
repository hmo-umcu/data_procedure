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
  Sample index 0 (within NC file) → row A  (Y = 28.955)
  Sample index 1 (within NC file) → row B  (Y =  9.655)
  Sample index 2 (within NC file) → row C  (Y = -9.645)
  Sample index 3 (within NC file) → row D  (Y = -28.945)

Parameters updated per row:
  - M200   : pressure (kPa → value * 10 as per G-code dict 0.1kPa units)
  - F      : nozzle speed (mm/s)
  - M300   : printhead temperature (°C → value * 10 as per G-code dict 0.1°C units)
  - Z      : z-offset (all Z0.600 occurrences within that row's block)

Usage:
  python generate_nc_files.py

Expects in same directory:
  - data_collection_24well_4rows_template.nc
  - lhs_bioprint_samples_semicolon.csv

Output folder:
  nc_files/   (created if not present)
"""

import os
import re
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
TEMPLATE_FILE = "data_collection_24well_4rows_template.nc"
CSV_FILE      = r"C:\Users\hmo\hmo_workspace\data_procedure\data\lhs\lhs_bioprint_samples_semicolon.csv"
OUTPUT_DIR    = r"C:\Users\hmo\hmo_workspace\data_procedure\data\nc_files"

# ── Imaging constants (from nc_imaging.py) ────────────────────────────────────
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

# ── Row Y-values in template (used to identify row blocks) ───────────────────
# Each row in the 24-well plate corresponds to a G55 Y origin value
ROW_Y = {
    'A': 28.955,
    'B':  9.655,
    'C': -9.645,
    'D': -28.945,
}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Parse template into 4 row blocks + header
# ─────────────────────────────────────────────────────────────────────────────

def parse_template(template_path):
    """
    Split the template NC file into:
      - header   : lines before first G805 origin of row A
      - row_blocks: dict {row_letter: [lines]} for rows A, B, C, D
      - (end commands stripped — will be re-added by imaging block)

    Row block starts at the first G805 line whose Y matches ROW_Y[row]
    and ends just before the next row's first G805 line.
    """
    with open(template_path, 'r') as f:
        raw = f.read()

    lines = raw.splitlines()

    # Find line indices where each row starts (first G805 of that row's Y)
    row_start = {}
    for row, y_val in ROW_Y.items():
        pattern = re.compile(
            r'G805\[.*?,\s*' + re.escape(f"{y_val:.3f}") + r'\s*,.*?\]'
        )
        for i, line in enumerate(lines):
            if pattern.search(line):
                row_start[row] = i
                break

    # Verify all 4 rows found
    for row in ROWS_ALPHA:
        if row not in row_start:
            raise ValueError(f"Could not find row {row} (Y={ROW_Y[row]}) in template")

    sorted_starts = sorted(row_start.items(), key=lambda x: x[1])
    # header = everything before row A starts
    header_end = sorted_starts[0][1]  # line index of row A's first G805
    header_lines = lines[:header_end]

    # Each row block = lines from its start to just before the next row's start
    row_blocks = {}
    for idx, (row, start) in enumerate(sorted_starts):
        if idx + 1 < len(sorted_starts):
            end = sorted_starts[idx + 1][1]
        else:
            # last row — go until end, stripping trailing G800/M110=1000/M30/#FLUSH WAIT
            end = len(lines)
        block = lines[start:end]
        # strip trailing blank lines and end commands from last row
        end_cmds = re.compile(r'^\s*(G800|M110=1000|M30|#FLUSH WAIT)\s*(;.*)?$')
        while block and (block[-1].strip() == '' or end_cmds.match(block[-1])):
            block.pop()
        row_blocks[row] = block

    return header_lines, row_blocks


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Update a row block with new parameters
# ─────────────────────────────────────────────────────────────────────────────

def update_row_block(block_lines, pressure_kpa, speed_mms, temp_c, z_offset_mm,
                     is_first_row=False):
    """
    Return a new list of lines with updated parameters.

    For the first row (row A):
      - M200 in initialization section is also updated (line 10 in header,
        handled separately — see update_header)
      - M200, F, M300, M302, Z updated in the block itself

    For subsequent rows (B, C, D):
      - M200 updated (pressure may change between rows)
      - F updated
      - M300 + M302 inserted before first Z0.xxx if not present
      - Z updated

    G-code encoding:
      M200 value = pressure_kpa * 10   (unit: 0.1 kPa)
      M300 value = temp_c * 10         (unit: 0.1 °C)
      F value    = speed_mms directly  (unit: mm/s)
      Z value    = z_offset_mm directly
    """
    m200_val = int(round(pressure_kpa * 10))
    m300_val = int(round(temp_c * 10))
    f_val    = f"{speed_mms:.3f}"
    z_val    = f"{z_offset_mm:.3f}"

    new_lines = []
    m300_inserted = False
    found_m302    = False

    for line in block_lines:
        stripped = line.strip()

        # Update M200 (pressure)
        if re.match(r'M200=\d+', stripped):
            comment = re.search(r';.*', line)
            c = f" ; Set pressure to {pressure_kpa}kPa"
            new_lines.append(f"M200={m200_val}{c}")
            continue

        # Update F (nozzle speed)
        if re.match(r'F\d+\.\d+', stripped):
            new_lines.append(f"F{f_val}")
            continue

        # Update existing M300
        if re.match(r'M300=\d+', stripped):
            new_lines.append(f"M300={m300_val} ; Set tool temperature to {temp_c}°C")
            m300_inserted = True
            continue

        # Track M302
        if re.match(r'M302', stripped):
            found_m302 = True
            # Insert M300 before M302 if not already inserted
            if not m300_inserted:
                new_lines.append(f"M300={m300_val} ; Set tool temperature to {temp_c}°C")
                m300_inserted = True
            new_lines.append(line.rstrip())
            continue

        # Update Z offset (printing height only — not Z18.400 travel height)
        if re.match(r'Z\d+\.\d+', stripped):
            z_num = float(re.match(r'Z(\d+\.\d+)', stripped).group(1))
            if z_num < 5.0:   # printing Z values are small (< 5mm); travel Z ~18.4
                new_lines.append(f"Z{z_val}")
                continue

        new_lines.append(line.rstrip())

    # If M302 never appeared (row B/C/D don't have M302 in template),
    # insert M300+M302 before the first Z printing line
    if not m300_inserted:
        final_lines = []
        inserted = False
        for line in new_lines:
            stripped = line.strip()
            if not inserted and re.match(r'Z\d+\.\d+', stripped):
                z_num = float(re.match(r'Z(\d+\.\d+)', stripped).group(1))
                if z_num < 5.0:
                    final_lines.append(f"M300={m300_val} ; Set tool temperature to {temp_c}°C")
                    final_lines.append(f"M302 ; Wait for tool temperature")
                    inserted = True
            final_lines.append(line)
        return final_lines

    return new_lines


def update_header(header_lines, pressure_kpa):
    """Update the M200 in the initialization header (line with 'Set pressure')."""
    m200_val = int(round(pressure_kpa * 10))
    new_lines = []
    for line in header_lines:
        if re.match(r'\s*M200=\d+', line):
            new_lines.append(f"M200={m200_val} ; Set pressure to {pressure_kpa}kPa")
        else:
            new_lines.append(line.rstrip())
    return new_lines


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Generate imaging G-code block (from nc_imaging.py logic)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Main — generate all 25 NC files
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Load CSV
    df = pd.read_csv(CSV_FILE, sep=';', index_col=0)
    print(f"Loaded {len(df)} samples from {CSV_FILE}")

    # Parse template
    header_lines, row_blocks = parse_template(TEMPLATE_FILE)
    print(f"Template parsed: header={len(header_lines)} lines, "
          f"row blocks: { {r: len(b) for r, b in row_blocks.items()} }")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate 25 NC files (4 samples each)
    n_files = 0
    for file_idx in range(25):
        sample_indices = [file_idx * 4 + i for i in range(4)]  # e.g. [0,1,2,3]

        # Verify all 4 sample indices exist
        valid = [i for i in sample_indices if i < len(df)]
        if len(valid) < 4:
            print(f"  [WARNING] File {file_idx}: only {len(valid)} samples available "
                  f"(indices {sample_indices}) — skipping incomplete file")
            continue

        # Get 4 samples
        samples = [df.iloc[i] for i in sample_indices]

        # File name: e.g. data_collection_24well_4rows_imaging_0-3.nc
        first_idx = sample_indices[0]
        last_idx  = sample_indices[3]
        fname = f"data_collection_24well_4rows_imaging_{first_idx}-{last_idx}.nc"
        fpath = os.path.join(OUTPUT_DIR, fname)

        # ── Build NC content ──────────────────────────────────────────────────

        # Header — use pressure of row A (first sample) for init M200
        row_a_pressure = int(samples[0]['Pressure_kPa'])
        updated_header = update_header(header_lines, row_a_pressure)

        # Add file-level comment showing all 4 sample parameters
        param_comments = [f"% {fname}"]
        param_comments.append(f"; Generated by generate_nc_files.py")
        param_comments.append(f"; Samples {first_idx}–{last_idx} from {CSV_FILE}")
        param_comments.append(f";")
        param_comments.append(f"; {'Row':<6} {'SampleID':<10} {'P(kPa)':<10} "
                               f"{'F(mm/s)':<10} {'T(°C)':<8} {'Z(mm)':<8}")
        for row_letter, s in zip(ROWS_ALPHA, samples):
            sid = int(s.name)
            p   = int(s['Pressure_kPa'])
            f_s = float(s['NozzleSpeed_mms'])
            t   = int(s['Temperature_C'])
            z   = float(s['Zoffset_mm'])
            param_comments.append(
                f"; {row_letter:<6} {sid:<10} {p:<10} {f_s:<10.1f} {t:<8} {z:<8}"
            )
        param_comments.append(f";")

        # Replace the first line (% filename) of header with our new header
        # The template starts with "% data_collection_24well_4rows"
        header_body = updated_header[1:]  # skip original % line

        all_lines = param_comments + header_body

        # Append each row block with updated parameters
        for row_letter, sample in zip(ROWS_ALPHA, samples):
            pressure = float(sample['Pressure_kPa'])
            speed    = float(sample['NozzleSpeed_mms'])
            temp     = float(sample['Temperature_C'])
            z_off    = float(sample['Zoffset_mm'])

            is_first = (row_letter == 'A')
            updated_block = update_row_block(
                list(row_blocks[row_letter]),   # pass a copy
                pressure, speed, temp, z_off,
                is_first_row=is_first
            )

            # Add a separator comment before each row
            all_lines.append("")
            all_lines.append(f"; ── Row {row_letter} │ Sample {int(sample.name)} │ "
                              f"P={int(pressure)}kPa  F={speed:.1f}mm/s  "
                              f"T={int(temp)}°C  Z={z_off:.3f}mm ──")
            all_lines.extend(updated_block)

        # Add flush wait before imaging
        all_lines.append("")
        all_lines.append("#FLUSH WAIT")

        # Imaging block — all 24 wells (row1 row2 row3 row4 = all wells)
        all_well_names = [f"{r}{c}" for r in ROWS_ALPHA for c in COLS]
        imaging_block = generate_imaging_block(all_well_names)
        all_lines.append(imaging_block)

        # Write file
        with open(fpath, 'w', newline='\r\n', encoding='utf-8') as f:
            f.write("\n".join(all_lines))

        print(f"  Written: {fname}  "
              f"(samples {first_idx}–{last_idx}: "
              f"P={[int(s['Pressure_kPa']) for s in samples]}kPa  "
              f"F={[float(s['NozzleSpeed_mms']) for s in samples]}mm/s  "
              f"T={[int(s['Temperature_C']) for s in samples]}°C  "
              f"Z={[float(s['Zoffset_mm']) for s in samples]}mm)")
        n_files += 1

    print(f"\nDone. {n_files} NC files written to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
