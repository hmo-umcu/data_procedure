#!/usr/bin/env python3
"""
nc_imaging.py
-------------
Loads a baseline R-GEN 100 NC print file and appends G-code camera
imaging positions for selected wells on a 24-well plate.

Camera is mounted at slot 5. Positions are recorded with slot 1 selected.
Z=49.58 is confirmed imaging height. Z=65.34 is confirmed safe travel height.

Usage examples:
  python nc_imaging.py sqr_corner.nc A1 B3 C4
  python nc_imaging.py sqr_corner.nc row1
  python nc_imaging.py sqr_corner.nc col6
  python nc_imaging.py sqr_corner.nc row1 row4 B3 C5
  python nc_imaging.py sqr_corner.nc all
"""

import sys
import os
import re

# ─────────────────────────────────────────────────────────────────────────────
# WELL COORDINATE MAP
# Derived by interpolating from 4 confirmed corner positions (slot 1 selected,
# camera physically at slot 5 over the well):
#   A1: X=-236.800, Y= 91.470
#   A6: X=-138.480, Y= 91.470
#   D1: X=-236.800, Y= 32.650
#   D6: X=-138.470, Y= 32.650
#
# X step per column = (−138.480 − (−236.800)) / 5 = +19.664 mm
# Y step per row    = ( 32.650  −   91.470  ) / 3 = −19.607 mm
# Z imaging height  = 49.58 mm  (confirmed)
# Z safe travel     = 65.34 mm  (confirmed)
# ─────────────────────────────────────────────────────────────────────────────

ROWS    = ['A', 'B', 'C', 'D']
COLS    = [1, 2, 3, 4, 5, 6]

# Confirmed corner anchors
X_A1, Y_A1 = -236.800,  91.470
X_A6, Y_A6 = -138.480,  91.470
X_D1, Y_D1 = -236.800,  32.650
X_D6, Y_D6 = -138.470,  32.650

X_STEP = (X_A6 - X_A1) / 5   # per column increment (+19.664 mm)
Y_STEP = (Y_D1 - Y_A1) / 3   # per row increment    (−19.607 mm)

Z_IMAGING = 49.58
Z_SAFE    = 65.34


def build_well_map():
    """Build dict of well_name -> (X, Y, Z) for all 24 wells."""
    wells = {}
    for r_idx, row in enumerate(ROWS):
        for c_idx, col in enumerate(COLS):
            x = round(X_A1 + c_idx * X_STEP, 3)
            y = round(Y_A1 + r_idx * Y_STEP, 3)
            wells[f"{row}{col}"] = (x, y, Z_IMAGING)
    return wells


WELL_MAP = build_well_map()


def parse_targets(args):
    """
    Parse input arguments into a list of well names.
    Accepts:
      - Well addresses : A1, B3, C4  (case-insensitive)
      - Row selectors  : row1, row2, row3, row4  (= rowA, rowB, rowC, rowD)
      - Col selectors  : col1 .. col6
      - 'all'          : all 24 wells
    Returns ordered list of well names, preserving input order, no duplicates.
    """
    selected = []
    seen = set()

    def add(well):
        w = well.upper()
        if w in WELL_MAP and w not in seen:
            selected.append(w)
            seen.add(w)

    for arg in args:
        arg_lower = arg.lower().strip()

        if arg_lower == 'all':
            for r in ROWS:
                for c in COLS:
                    add(f"{r}{c}")

        elif re.match(r'^row[1-4]$', arg_lower):
            row_idx = int(arg_lower[3]) - 1          # row1 → index 0 → 'A'
            row_letter = ROWS[row_idx]
            for c in COLS:
                add(f"{row_letter}{c}")

        elif re.match(r'^col[1-6]$', arg_lower):
            col_num = int(arg_lower[3])
            for r in ROWS:
                add(f"{r}{col_num}")

        elif re.match(r'^[abcdABCD][1-6]$', arg):
            add(arg)

        else:
            print(f"  [WARNING] Unrecognised argument '{arg}' — skipped.")

    return selected


def generate_imaging_gcode(well_names):
    """Generate the G-code block for camera imaging of given wells."""
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

    for i, well in enumerate(well_names):
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


def strip_end_commands(nc_content):
    """
    Remove the trailing G800 / M110=1000 / M30 from the print file
    so we can append imaging block with its own ending.
    """
    lines = nc_content.splitlines()
    # Walk backwards and remove G800, M110=1000, M30 and trailing blanks
    remove_patterns = re.compile(r'^\s*(G800|M110=1000|M30)\s*(;.*)?$')
    while lines and (lines[-1].strip() == '' or remove_patterns.match(lines[-1])):
        lines.pop()
    return "\n".join(lines)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    nc_arg = sys.argv[1]
    target_args = sys.argv[2:]

    # Always resolve input file from nc_files/ subfolder
    NC_DIR = "nc_files"
    nc_file = os.path.join(NC_DIR, os.path.basename(nc_arg))

    if not os.path.isfile(nc_file):
        print(f"[ERROR] File not found: {nc_file}")
        print(f"  Make sure the file exists in the '{NC_DIR}/' subfolder.")
        sys.exit(1)

    with open(nc_file, 'r') as f:
        nc_content = f.read()

    # Parse targets
    well_names = parse_targets(target_args)

    if not well_names:
        print("[ERROR] No valid wells selected. Check your input arguments.")
        print("  Valid wells: A1-D6")
        print("  Row selectors: row1 row2 row3 row4")
        print("  Col selectors: col1 col2 col3 col4 col5 col6")
        print("  Or: all")
        sys.exit(1)

    print(f"\n  Source file  : {nc_file}")
    print(f"  Wells to image: {', '.join(well_names)}")

    # Strip end commands from print file
    nc_stripped = strip_end_commands(nc_content)

    # Generate imaging G-code
    imaging_block = generate_imaging_gcode(well_names)

    # Combine
    output_content = nc_stripped + "\n" + imaging_block + "\n"

    # Build output filename — saved in same nc_files/ subfolder
    basename = os.path.basename(nc_file)
    base, ext = os.path.splitext(basename)
    output_file = os.path.join(NC_DIR, f"{base}_imaging{ext}")

    with open(output_file, 'w') as f:
        f.write(output_content)

    print(f"  Output file  : {output_file}")
    print(f"  Done. {len(well_names)} well(s) appended.\n")


if __name__ == "__main__":
    main()