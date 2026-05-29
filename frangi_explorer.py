"""
Frangi Binary Mask Explorer
============================
Usage:
    python frangi_explorer.py --image path/to/image.tif

Controls (keyboard):
    Q / A  →  increase / decrease threshold (coarse: ±0.02)
    W / S  →  increase / decrease threshold (fine:   ±0.005)
    E / D  →  increase / decrease min object size (±50)
    R / F  →  increase / decrease dilation disk radius (±1)
    P      →  print current parameters to console
    SPACE  →  save overlay + mask to disk (same folder as image)
    ESC    →  quit

Display (4 panels):
    Top-left:     Raw image
    Top-right:    Frangi heatmap (hot colormap)
    Bottom-left:  Binary mask (current threshold)
    Bottom-right: Overlay: raw + green mask + yellow skeleton
"""

import argparse
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')          # change to 'Qt5Agg' if TkAgg not available
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
from PIL import Image
from skimage import exposure, morphology, measure
from skimage.filters import frangi
from skimage.morphology import disk, skeletonize
import warnings
warnings.filterwarnings('ignore')

# ── Parse arguments ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Frangi threshold explorer')
parser.add_argument('--image', required=True, help='Path to .tif image')
parser.add_argument('--sigmas', nargs='+', type=int, default=[2,4,6,8,10,12,14],
                    help='Frangi sigma values (default: 2 4 6 8 10 12 14)')
parser.add_argument('--clip',   type=float, default=0.02,
                    help='CLAHE clip limit (default: 0.02)')
parser.add_argument('--dark',   type=float, default=50,
                    help='Exclude pixels darker than this (default: 50)')
parser.add_argument('--bright', type=float, default=230,
                    help='Exclude pixels brighter than this (default: 230)')
args = parser.parse_args()

# ── Load and precompute Frangi (expensive, only done once) ───────────────────
print(f"Loading: {args.image}")
img = np.array(Image.open(args.image))
h, w = img.shape[:2]

R = img[:,:,0].astype(float)
G = img[:,:,1].astype(float)
B = img[:,:,2].astype(float)
gray = 0.299*R + 0.587*G + 0.114*B

valid = (gray > args.dark) & (gray < args.bright)
inv_eq = exposure.equalize_adapthist(1.0 - gray/255.0, clip_limit=args.clip)

print(f"Computing Frangi (sigmas={args.sigmas}) ... this may take ~10-20s ...")
fmap = frangi(inv_eq, sigmas=args.sigmas, alpha=0.5, beta=0.5, black_ridges=False)
fmax = fmap.max()
fnorm = (fmap / fmax) * valid if fmax > 0 else fmap
print(f"Done. Frangi range: [{fnorm.min():.4f}, {fnorm.max():.4f}]")
print(f"Nonzero pixel stats:")
nz = fnorm[fnorm > 0.001]
for pct in [50, 75, 85, 90, 92, 94, 95, 96, 97, 98, 99]:
    print(f"  pct {pct:3d}: {np.percentile(nz, pct):.4f}")

# ── State ────────────────────────────────────────────────────────────────────
state = {
    'thresh':   np.percentile(nz, 96),
    'min_size': 400,
    'dil_r':    2,
}

# ── Segmentation function ────────────────────────────────────────────────────
def apply_threshold(thresh, min_size, dil_r):
    binary = fnorm > thresh
    if min_size > 0:
        binary = morphology.remove_small_objects(binary, min_size=max(1, min_size))
    if dil_r > 0:
        binary = morphology.binary_dilation(binary, disk(dil_r))
        binary = morphology.remove_small_objects(binary, min_size=max(1, min_size))
    skel = skeletonize(binary) if binary.sum() > 0 else np.zeros_like(binary)
    return binary, skel

def make_overlay(binary, skel):
    ov = img.copy().astype(np.float32)
    ov[binary, 0] = ov[binary, 0] * 0.25
    ov[binary, 1] = np.clip(ov[binary, 1] * 0.4 + 130, 0, 255)
    ov[binary, 2] = ov[binary, 2] * 0.25
    ov = ov.astype(np.uint8)
    # Yellow skeleton (thickened)
    sk2 = morphology.binary_dilation(skel, disk(1))
    ov[sk2, 0] = 255
    ov[sk2, 1] = 240
    ov[sk2, 2] = 20
    return ov

# ── Build figure ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.patch.set_facecolor('#0a0a14')
plt.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.18, hspace=0.08, wspace=0.05)

ax_raw  = axes[0, 0]
ax_heat = axes[0, 1]
ax_mask = axes[1, 0]
ax_ovl  = axes[1, 1]

im_raw  = ax_raw.imshow(img)
im_heat = ax_heat.imshow(fnorm, cmap='hot', vmin=0, vmax=fnorm.max())
plt.colorbar(im_heat, ax=ax_heat, fraction=0.03, pad=0.02)

binary0, skel0 = apply_threshold(state['thresh'], state['min_size'], state['dil_r'])
mask_rgb0 = np.zeros((*binary0.shape, 3), dtype=np.uint8)
mask_rgb0[binary0] = [120, 255, 120]
im_mask = ax_mask.imshow(mask_rgb0)
im_ovl  = ax_ovl.imshow(make_overlay(binary0, skel0))

for ax, ttl in zip([ax_raw, ax_heat, ax_mask, ax_ovl],
                   ['① Raw Image', '② Frangi Heatmap',
                    '③ Binary Mask', '④ Overlay + Skeleton']):
    ax.set_title(ttl, color='white', fontsize=12, fontweight='bold')
    ax.axis('off')

fname_base = os.path.splitext(os.path.basename(args.image))[0]
title_obj = fig.suptitle('', color='#aaffaa', fontsize=11, y=0.97)

def update_title():
    t  = state['thresh']
    ms = state['min_size']
    dr = state['dil_r']
    b, _ = apply_threshold(t, ms, dr)
    nc   = measure.label(b).max()
    cov  = b.sum() / b.size * 100
    title_obj.set_text(
        f"{fname_base}  |  threshold={t:.4f}  min_size={ms}  dil_r={dr}  "
        f"|  coverage={cov:.2f}%  components={nc}\n"
        f"Keys: Q/A=thresh±0.02  W/S=thresh±0.005  E/D=min_size±50  R/F=dil±1  "
        f"SPACE=save  P=print  ESC=quit"
    )

def redraw():
    t  = state['thresh']
    ms = state['min_size']
    dr = state['dil_r']
    binary, skel = apply_threshold(t, ms, dr)

    mask_rgb = np.zeros((*binary.shape, 3), dtype=np.uint8)
    mask_rgb[binary] = [120, 255, 120]
    im_mask.set_data(mask_rgb)
    im_ovl.set_data(make_overlay(binary, skel))
    update_title()
    fig.canvas.draw_idle()

# ── Sliders ──────────────────────────────────────────────────────────────────
ax_sl_thresh   = plt.axes([0.10, 0.11, 0.80, 0.025])
ax_sl_minsize  = plt.axes([0.10, 0.07, 0.80, 0.025])
ax_sl_dil      = plt.axes([0.10, 0.03, 0.80, 0.025])

sl_thresh  = mwidgets.Slider(ax_sl_thresh, 'Threshold',  0.0, 1.0,
                              valinit=state['thresh'],  valstep=0.001, color='#4e9af1')
sl_minsize = mwidgets.Slider(ax_sl_minsize,'Min size',   0,   2000,
                              valinit=state['min_size'], valstep=10,   color='#f1a44e')
sl_dil     = mwidgets.Slider(ax_sl_dil,    'Dil radius', 0,   8,
                              valinit=state['dil_r'],    valstep=1,    color='#4ef18a')

for sl in [sl_thresh, sl_minsize, sl_dil]:
    sl.label.set_color('white')
    sl.valtext.set_color('white')
    sl.ax.set_facecolor('#1a1a2e')

def on_thresh(val):
    state['thresh'] = val
    redraw()

def on_minsize(val):
    state['min_size'] = int(val)
    redraw()

def on_dil(val):
    state['dil_r'] = int(val)
    redraw()

sl_thresh.on_changed(on_thresh)
sl_minsize.on_changed(on_minsize)
sl_dil.on_changed(on_dil)

# ── Keyboard ─────────────────────────────────────────────────────────────────
def on_key(event):
    t  = state['thresh']
    ms = state['min_size']
    dr = state['dil_r']
    changed = True
    if   event.key == 'q': state['thresh']   = min(1.0, t + 0.02)
    elif event.key == 'a': state['thresh']   = max(0.0, t - 0.02)
    elif event.key == 'w': state['thresh']   = min(1.0, t + 0.005)
    elif event.key == 's': state['thresh']   = max(0.0, t - 0.005)
    elif event.key == 'e': state['min_size'] = min(2000, ms + 50)
    elif event.key == 'd': state['min_size'] = max(0, ms - 50)
    elif event.key == 'r': state['dil_r']    = min(8, dr + 1)
    elif event.key == 'f': state['dil_r']    = max(0, dr - 1)
    elif event.key == 'p':
        print(f"\nCurrent params: thresh={state['thresh']:.4f}  "
              f"min_size={state['min_size']}  dil_r={state['dil_r']}")
        changed = False
    elif event.key == ' ':
        save_results()
        changed = False
    elif event.key == 'escape':
        plt.close()
        changed = False
    else:
        changed = False

    if changed:
        # sync sliders
        sl_thresh.set_val(state['thresh'])
        sl_minsize.set_val(state['min_size'])
        sl_dil.set_val(state['dil_r'])
        redraw()

fig.canvas.mpl_connect('key_press_event', on_key)

# ── Save function ─────────────────────────────────────────────────────────────
def save_results():
    t  = state['thresh']
    ms = state['min_size']
    dr = state['dil_r']
    binary, skel = apply_threshold(t, ms, dr)
    ov = make_overlay(binary, skel)

    out_dir  = os.path.dirname(os.path.abspath(args.image))
    tag      = f"thresh{t:.4f}_ms{ms}_dil{dr}"
    out_mask = os.path.join(out_dir, f"{fname_base}_mask_{tag}.png")
    out_ovl  = os.path.join(out_dir, f"{fname_base}_overlay_{tag}.png")
    out_heat = os.path.join(out_dir, f"{fname_base}_frangi_heatmap.png")

    Image.fromarray((mask_rgb := np.zeros((*binary.shape, 3), dtype=np.uint8)) or mask_rgb).save(out_mask)
    mask_rgb[binary] = [120, 255, 120]
    Image.fromarray(mask_rgb).save(out_mask)
    Image.fromarray(ov).save(out_ovl)

    # Save heatmap as PNG
    import matplotlib.cm as cm
    hmap_norm = (fnorm / fnorm.max() * 255).astype(np.uint8) if fnorm.max() > 0 else fnorm.astype(np.uint8)
    hmap_colored = (cm.hot(hmap_norm / 255.0)[:,:,:3] * 255).astype(np.uint8)
    Image.fromarray(hmap_colored).save(out_heat)

    print(f"\n✓ Saved:")
    print(f"  Mask:    {out_mask}")
    print(f"  Overlay: {out_ovl}")
    print(f"  Heatmap: {out_heat}")
    print(f"  Params:  thresh={t:.4f}  min_size={ms}  dil_r={dr}")

# ── Initial draw ──────────────────────────────────────────────────────────────
update_title()
redraw()
plt.show()
