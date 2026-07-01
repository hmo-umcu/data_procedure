"""
generate_bo_print_nc.py
========================
Generate a single-column (6-well: A1-F1) print+imaging NC file for a
Bayesian-Optimization-recommended parameter set, by reusing the EXACT
G-code structure of Column 1 from an existing multi-column template NC
file (origins, strand sequences, travel heights, imaging moves — all of
it) and substituting only the pressure/speed/Z-offset values.

This deliberately does NOT regenerate G-code geometry from scratch. It
parses the template's own Column-1 block, reads off what pressure/speed/
Z it was originally written for (from the column's own header comment
line), and does targeted string substitution to retarget those exact
same moves to new values — minimizing the chance of introducing a new
geometry bug versus a script that recomputes strand coordinates itself.

What gets changed from the template's Column 1:
    - M200=<pressure in 0.1 kPa units>   (every occurrence within the block)
    - F<speed in mm/s>                    (every occurrence within the block)
    - Z<offset in mm>                     (the press height, every occurrence)
    - Z<offset + 1.0>                     (the "lift 1.0mm" travel height,
                                           which is always offset+1.0 in the
                                           template — every occurrence)
    - The column's own header comment line (cosmetic, records new values)
    - The file's top comment table (cosmetic, records new values)
    - The INITIALIZATION block's pressure preload (M200, so slot 1 starts
      at the correct pressure before the first well)

What is explicitly left untouched (per the "change as little as possible"
instruction):
    - All G805 well origins, strand G01/G00 moves, M160/M161 toggling
    - The "G00 Z18.400 ; lift after well X" travel height — confirmed from
      the template to be a FIXED system safe-clearance height, independent
      of Z-offset (same value at Z=0.300 and Z=0.600 in the source file),
      so it is not touched
    - M110=<percent> progress markers — cosmetic print-progress percentages
      only, carried over as-is from the template's Column 1 numbering
    - All camera imaging G-code (no P/F/Z involved there at all)
    - The "Return home" / M30 footer

What gets TRIMMED (not substituted, just cut down to one column):
    - Only Column 1's print block is kept (Columns 2-8 are dropped)
    - Only the 6 imaging blocks for wells A1, B1, C1, D1, E1, F1 are kept
      (the other 42 wells' imaging blocks are dropped)
    - The imaging section's "Wells:" comment list is shortened to match

Usage
-----
    python generate_bo_print_nc.py \\
        --bo_log       bo_recommendation_log.csv \\
        --iteration    1 \\
        --template_nc  data_collection_48well_8cols_s0-s49.nc \\
        --output_nc    bo_iter1_col1.nc
"""

import argparse
import re
import pandas as pd


COL_HEADER_RE = re.compile(
    r'^; -- Col (\d+) \| Sample (\S+) \| P=([\d.]+)kPa\s+F=([\d.]+)mm/s\s+Z=([\d.]+)mm --'
)
WELL_BLOCK_RE = re.compile(r'^; --- Well (\S+) ---')
TOP_TABLE_ROW_RE = re.compile(
    r'^;\s*\d+\s+\S+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s*$'
)


def load_bo_row(bo_log_path, iteration):
    df = pd.read_csv(bo_log_path, sep=';')
    if 'iteration' not in df.columns:
        raise ValueError(f"'{bo_log_path}' has no 'iteration' column — "
                         f"found: {list(df.columns)}")
    row = df[df['iteration'] == iteration]
    if row.empty:
        raise ValueError(
            f"iteration={iteration} not found in {bo_log_path}. "
            f"Available iterations: {sorted(df['iteration'].unique().tolist())}")
    row = row.iloc[-1]   # if duplicated, take the most recent entry
    return {
        'iteration':         int(row['iteration']),
        'pressure_kpa':      float(row['Pressure_kPa']),
        'speed_mms':         float(row['NozzleSpeed_mms']),
        'zoffset_mm':        float(row['Zoffset_mm']),
        'predicted_SF_mean': float(row['predicted_SF_mean'])
                             if 'predicted_SF_mean' in row else None,
    }


def find_column_blocks(lines):
    """
    Returns dict: col_num -> {
        'start': line idx of the "; -- Col N | ..." header line,
        'end':   line idx (exclusive) where this column's block ends
                 (= start of next column's header, or start of the
                 '#FLUSH WAIT' / imaging section if this is the last column),
        'sample_id': str, 'pressure_kpa': float, 'speed_mms': float,
        'zoffset_mm': float,
    }
    """
    headers = []
    for i, line in enumerate(lines):
        m = COL_HEADER_RE.match(line)
        if m:
            headers.append((i, m))
    if not headers:
        raise ValueError("No '; -- Col N | Sample ... | P=...kPa F=...mm/s "
                         "Z=...mm --' header lines found in template — "
                         "is this the right NC file?")

    blocks = {}
    for idx, (start_i, m) in enumerate(headers):
        end_i = headers[idx + 1][0] if idx + 1 < len(headers) else None
        if end_i is None:
            end_i = start_i
            for j in range(start_i + 1, len(lines)):
                if 'CAMERA IMAGING POSITIONS' in lines[j] or \
                   lines[j].strip() == '#FLUSH WAIT':
                    end_i = j
                    break
            else:
                end_i = len(lines)
        col_num = int(m.group(1))
        blocks[col_num] = {
            'start': start_i, 'end': end_i,
            'sample_id': m.group(2),
            'pressure_kpa': float(m.group(3)),
            'speed_mms': float(m.group(4)),
            'zoffset_mm': float(m.group(5)),
        }
    return blocks


def find_well_blocks(lines):
    """Returns dict: well_name -> (start_idx, end_idx_exclusive)."""
    markers = []
    for i, line in enumerate(lines):
        m = WELL_BLOCK_RE.match(line.strip())
        if m:
            markers.append((i, m.group(1)))
    blocks = {}
    for idx, (start_i, name) in enumerate(markers):
        real_start = start_i - 1 if start_i > 0 and lines[start_i - 1].strip() == '' else start_i
        end_i = markers[idx + 1][0] - 1 if idx + 1 < len(markers) else None
        if end_i is None:
            end_i = len(lines)
            for j in range(start_i + 1, len(lines)):
                if lines[j].strip().startswith('; --- Return home'):
                    end_i = j - 1
                    break
        blocks[name] = (real_start, end_i)
    return blocks


def substitute_block(block_lines, orig, new_pressure_kpa, new_speed_mms,
                     new_zoffset_mm, col_to_keep, new_iteration):
    """
    Targeted substitution within a column's print block. Uses the
    block's OWN original parameter values (parsed from its header line)
    to build exact-match patterns, rather than guessing.
    """
    orig_m200 = round(orig['pressure_kpa'] * 10)
    new_m200  = round(new_pressure_kpa * 10)

    orig_f_str = f"{orig['speed_mms']:.3f}"
    new_f_str  = f"{new_speed_mms:.3f}"

    orig_z_str = f"{orig['zoffset_mm']:.3f}"
    new_z_str  = f"{new_zoffset_mm:.3f}"

    orig_lift_z_str = f"{orig['zoffset_mm'] + 1.0:.3f}"
    new_lift_z_str  = f"{new_zoffset_mm + 1.0:.3f}"

    out = []
    for line in block_lines:
        new_line = line
        new_line = re.sub(rf'M200={orig_m200}\b', f'M200={new_m200}', new_line)
        new_line = new_line.replace(
            f"; pressure {orig['pressure_kpa']:g}kPa",
            f"; pressure {new_pressure_kpa:g}kPa")
        new_line = re.sub(rf'\bF{re.escape(orig_f_str)}\b', f'F{new_f_str}', new_line)
        # lift height (offset+1.0) BEFORE the bare press-Z substitution, so
        # e.g. "1.300" is not also matched by a "0.300" pattern below
        new_line = re.sub(rf'\bZ{re.escape(orig_lift_z_str)}\b',
                          f'Z{new_lift_z_str}', new_line)
        new_line = re.sub(rf'\bZ{re.escape(orig_z_str)}\b',
                          f'Z{new_z_str}', new_line)
        out.append(new_line)

    out[0] = (f"; -- Col {col_to_keep} | BO iteration {new_iteration} | "
             f"P={new_pressure_kpa:g}kPa  F={new_speed_mms:.1f}mm/s  "
             f"Z={new_zoffset_mm:.3f}mm --\n")
    return out


def main():
    parser = argparse.ArgumentParser(
        description='Generate a 1-column (6-well) NC file for a BO-recommended '
                    'print, reusing Column 1 of an existing template NC file.')
    parser.add_argument('--bo_log', required=True,
        help='Path to bo_recommendation_log.csv')
    parser.add_argument('--iteration', type=int, required=True,
        help='Which iteration (row) in bo_log to use')
    parser.add_argument('--template_nc', required=True,
        help='Path to the existing multi-column NC file to extract '
             'Column 1 (and wells A1-F1 imaging) from')
    parser.add_argument('--source_column', type=int, default=1,
        help='Which column in the template to use as the print/imaging '
             'template (default: 1, i.e. wells A1-F1)')
    parser.add_argument('--output_nc', default=None,
        help='Output path (default: bo_iter<N>_col<source_column>.nc)')
    args = parser.parse_args()

    bo_row = load_bo_row(args.bo_log, args.iteration)
    print(f"BO iteration {bo_row['iteration']}: "
         f"Pressure={bo_row['pressure_kpa']:g}kPa  "
         f"Speed={bo_row['speed_mms']:g}mm/s  "
         f"Zoffset={bo_row['zoffset_mm']:.3f}mm"
         + (f"  (predicted SF={bo_row['predicted_SF_mean']:.4f})"
            if bo_row['predicted_SF_mean'] is not None else ''))

    with open(args.template_nc) as f:
        lines = f.readlines()

    col_blocks = find_column_blocks(lines)
    if args.source_column not in col_blocks:
        raise ValueError(f"--source_column {args.source_column} not found "
                         f"in template. Columns present: "
                         f"{sorted(col_blocks.keys())}")
    cb = col_blocks[args.source_column]
    print(f"\nUsing template Column {args.source_column} "
         f"(original: Sample {cb['sample_id']}, "
         f"P={cb['pressure_kpa']:g}kPa, F={cb['speed_mms']:g}mm/s, "
         f"Z={cb['zoffset_mm']:.3f}mm) as the geometry template.")

    column_block = lines[cb['start']:cb['end']]
    new_column_block = substitute_block(
        column_block, cb, bo_row['pressure_kpa'], bo_row['speed_mms'],
        bo_row['zoffset_mm'], args.source_column, bo_row['iteration'])

    col_letter_wells = [f"{r}{args.source_column}" for r in 'ABCDEF']
    well_blocks = find_well_blocks(lines)
    missing = [w for w in col_letter_wells if w not in well_blocks]
    if missing:
        raise ValueError(f"Imaging blocks not found for wells: {missing}")

    imaging_well_lines = []
    for w in col_letter_wells:
        s, e = well_blocks[w]
        imaging_well_lines.extend(lines[s:e])

    # ── preamble: trim top comment table to one row, update INITIALIZATION
    #    pressure preload ───────────────────────────────────────────────────
    first_col_start = min(b['start'] for b in col_blocks.values())
    preamble = lines[:first_col_start]

    new_preamble = []
    in_table = False
    table_written = False
    for line in preamble:
        if line.startswith('; Col') and 'SampleID' in line:
            new_preamble.append(line)
            in_table = True
            continue
        if in_table and TOP_TABLE_ROW_RE.match(line):
            if not table_written:
                new_preamble.append(
                    f"; {args.source_column:<5} BO-iter{bo_row['iteration']:<3} "
                    f"{bo_row['pressure_kpa']:<10g} {bo_row['speed_mms']:<10g} "
                    f"{bo_row['zoffset_mm']:<7.3f}\n")
                table_written = True
            continue
        if in_table and not TOP_TABLE_ROW_RE.match(line):
            in_table = False

        m200_init = re.search(r'M200=(\d+) ; Set pressure to ([\d.]+)kPa', line)
        if m200_init:
            new_m200 = round(bo_row['pressure_kpa'] * 10)
            line = f"M200={new_m200} ; Set pressure to {bo_row['pressure_kpa']:g}kPa\n"

        new_preamble.append(line)

    # ── camera section header, trimmed well list ───────────────────────────
    cam_header_start = None
    for i, line in enumerate(lines):
        if 'CAMERA IMAGING POSITIONS' in line:
            cam_header_start = i - 2
            break
    cam_header_end = well_blocks[col_letter_wells[0]][0]
    camera_header = lines[cam_header_start:cam_header_end]
    new_camera_header = []
    for line in camera_header:
        if line.strip().startswith('; Wells:'):
            new_camera_header.append(f"; Wells: {', '.join(col_letter_wells)}\n")
        elif 'Column-major order' in line:
            new_camera_header.append(
                f"; Single column: {' -> '.join(col_letter_wells)}\n")
        else:
            new_camera_header.append(line)

    # ── footer: "Return home" through end of file ─────────────────────────
    footer_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('; --- Return home'):
            footer_start = i
            break
    footer = lines[footer_start:] if footer_start is not None else []

    output_lines = (new_preamble + new_column_block + ['\n']
                    + new_camera_header + imaging_well_lines + ['\n'] + footer)

    output_path = args.output_nc or f"bo_iter{args.iteration}_col{args.source_column}.nc"
    with open(output_path, 'w') as f:
        f.writelines(output_lines)

    print(f"\nWritten: {output_path}")
    print(f"  Wells printed + imaged: {', '.join(col_letter_wells)}")
    print(f"  Pressure: {bo_row['pressure_kpa']:g} kPa  "
         f"(M200={round(bo_row['pressure_kpa']*10)})")
    print(f"  Speed:    {bo_row['speed_mms']:g} mm/s")
    print(f"  Zoffset:  {bo_row['zoffset_mm']:.3f} mm")
    print(f"\n  Set Temperature manually in Architect UI before printing "
         f"(not controlled by this G-code, per the original template).")


if __name__ == '__main__':
    main()
