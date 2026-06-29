"""
shape_collage.py
----------------
Extracts white/bright regions from an image and composites them
into a line, grid, or random scatter arrangement.

Requirements:
    pip install opencv-python numpy Pillow
"""

import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ==============================================================================
#  QUICK-RUN COMMANDS  (copy one into PowerShell to run)
# ==============================================================================
#
#  Line arrangement:
#    python shape_collage.py --input "C:\path\to\your.jpg" --mode line --threshold 80 --min-size 30 --scale-min 0.5 --scale-max 2.0
#
#  Grid arrangement:
#    python shape_collage.py --input "C:\path\to\your.jpg" --mode grid --cols 8 --scale-min 0.5 --scale-max 2.0
#
#  Random scatter:
#    python shape_collage.py --input "C:\path\to\your.jpg" --mode scatter --canvas-w 3000 --canvas-h 1000 --seed 99 --scale-min 0.5 --scale-max 2.0
#
# ==============================================================================


# ==============================================================================
#  CONFIGURATION  — edit the values in this section
# ==============================================================================

# ------------------------------------------------------------------------------
#  PATHS
#  OUTPUT_DIR is a folder, not a file. Every run creates a new numbered file
#  inside it so previous results are never overwritten:
#    e.g. output_folder/your_image_line_001.png
#         output_folder/your_image_line_002.png  (next run, same input+mode)
# ------------------------------------------------------------------------------
DEFAULT_INPUT       = r"C:\Users\rymki\OneDrive\Documents\GRADUATION 2026\3D visual language project\experiments\ffmpgeg_selection\frame extraction\ffmpeg_edited\output2.jpg"   # ← change this
DEFAULT_OUTPUT_DIR  = r"C:\Users\rymki\OneDrive\Documents\GRADUATION 2026\3D visual language project\experiments\ffmpgeg_selection\frame extraction\ffmpeg_edited"    # ← change this

# ------------------------------------------------------------------------------
#  MODE
#  Choose the arrangement: "line" | "grid" | "scatter"
# ------------------------------------------------------------------------------
DEFAULT_MODE        = "line"

# ------------------------------------------------------------------------------
#  EXTRACTION  — controls which pixels count as "white" and what gets kept
#
#  THRESHOLD   0–255  Lower = picks up dimmer shapes (try 60–120).
#                     Higher = only the very brightest regions (try 180–220).
#
#  MIN_SIZE    px²    Minimum contour area to keep. Raise to filter out
#                     tiny noise specks. Lower to keep fine detail.
# ------------------------------------------------------------------------------
DEFAULT_THRESHOLD   = 80
DEFAULT_MIN_SIZE    = 30

# ------------------------------------------------------------------------------
#  SHAPE TRANSFORMS
#
#  Each shape gets a randomly chosen scale between MIN and MAX.
#  SCALE_MIN   smallest a shape can be scaled  (e.g. 0.5 = half size)
#  SCALE_MAX   largest a shape can be scaled   (e.g. 3.0 = triple size)
# ------------------------------------------------------------------------------
DEFAULT_SCALE_MIN   = 0.5
DEFAULT_SCALE_MAX   = 2.0

# ------------------------------------------------------------------------------
#  LINE & GRID MODE
#
#  PADDING     Pixels of space between shapes / grid cells.
#  COLS        Grid columns. 0 = automatic (roughly square).
# ------------------------------------------------------------------------------
DEFAULT_PADDING     = 20
DEFAULT_COLS        = 0     # 0 = auto

# ------------------------------------------------------------------------------
#  SCATTER MODE
#
#  CANVAS_W / CANVAS_H   Output canvas size in pixels.
#  SEED                  Change this number to get a different random layout.
#  ALLOW_OVERLAP         True = shapes can stack. False = tries to avoid it.
# ------------------------------------------------------------------------------
DEFAULT_CANVAS_W      = 2400
DEFAULT_CANVAS_H      = 1600
DEFAULT_SEED          = 42
DEFAULT_ALLOW_OVERLAP = False
DEFAULT_OVERLAP_CHANCE = 0.5   # 0.0 = never overlap | 1.0 = always overlap | 0.5 = 50% chance

# ------------------------------------------------------------------------------
#  COLORS
#
#  CANVAS_BG   Background color as (R, G, B).
#              Black = (0,0,0)  |  White = (255,255,255)
#
#  SHAPE_COLOR Set to (R, G, B) to recolor all shapes flat, e.g. (255,0,0)
#              for red.  None = keep original colors from the source image.
# ------------------------------------------------------------------------------
DEFAULT_CANVAS_BG   = (0, 0, 0)
DEFAULT_SHAPE_COLOR = None

# ==============================================================================
#  END OF CONFIGURATION  — no need to edit below this line
# ==============================================================================


# ------------------------------------------------------------------------------
#  Extraction
# ------------------------------------------------------------------------------

def extract_shapes(img_path, threshold, min_size):
    with open(img_path, 'rb') as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    src = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if src is None:
        sys.exit(f"[ERROR] Could not read image: {img_path}")

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shapes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_size:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        region_color = src[y:y+h, x:x+w]
        region_mask  = mask[y:y+h, x:x+w]
        local_mask   = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(local_mask, [cnt - [x, y]], -1, 255, thickness=cv2.FILLED)
        alpha = cv2.bitwise_and(region_mask, local_mask)
        rgba  = cv2.cvtColor(region_color, cv2.COLOR_BGR2RGBA)
        rgba[:, :, 3] = alpha
        shapes.append(Image.fromarray(rgba, "RGBA"))

    print(f"[INFO] Found {len(contours)} contours, kept {len(shapes)} shapes "
          f"(threshold={threshold}, min_size={min_size}px)")
    return shapes


# ------------------------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------------------------

def recolor_shapes(shapes, color):
    r, g, b = color
    out = []
    for img in shapes:
        arr = np.array(img, dtype=np.uint8)
        alpha = arr[:, :, 3].copy()
        arr[:, :, :3] = [r, g, b]
        arr[:, :, 3]  = alpha
        out.append(Image.fromarray(arr, "RGBA"))
    return out


def scale_shape(img, factor):
    if factor == 1.0:
        return img
    return img.resize((max(1, int(img.width*factor)), max(1, int(img.height*factor))), Image.LANCZOS)


def unique_output_path(folder, stem, mode):
    """Returns folder/stem_mode_001.png, incrementing until unused."""
    out_dir = Path(folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = "".join(c if (c.isascii() and (c.isalnum() or c in "-_. ")) else "_" for c in stem)
    i = 1
    while True:
        p = out_dir / f"{safe_stem}_{mode}_{i:03d}.png"
        try:
            exists = p.exists()
        except Exception:
            exists = False
        if not exists:
            return p
        i += 1


# ------------------------------------------------------------------------------
#  Layout modes
# ------------------------------------------------------------------------------

def make_line(shapes, padding, bg):
    total_w = sum(s.width for s in shapes) + padding * (len(shapes)-1)
    max_h   = max(s.height for s in shapes)
    canvas  = Image.new("RGBA", (total_w, max_h), bg+(255,))
    x = 0
    for s in shapes:
        canvas.alpha_composite(s, (x, (max_h-s.height)//2))
        x += s.width + padding
    return canvas


def make_grid(shapes, padding, cols, bg):
    if cols <= 0:
        cols = max(1, int(np.ceil(np.sqrt(len(shapes)))))
    rows   = int(np.ceil(len(shapes)/cols))
    cell_w = max(s.width  for s in shapes) + padding
    cell_h = max(s.height for s in shapes) + padding
    canvas = Image.new("RGBA", (cols*cell_w+padding, rows*cell_h+padding), bg+(255,))
    for i, s in enumerate(shapes):
        col = i % cols
        row = i // cols
        x = padding + col*cell_w + (cell_w-padding-s.width)//2
        y = padding + row*cell_h + (cell_h-padding-s.height)//2
        canvas.alpha_composite(s, (x, y))
    return canvas


def make_scatter(shapes, canvas_w, canvas_h, bg, seed, allow_overlap, overlap_chance=0.5):
    rng    = random.Random(seed)
    canvas = Image.new("RGBA", (canvas_w, canvas_h), bg+(255,))
    placed = []
    skipped = 0
    for s in shapes:
        sw, sh = s.width, s.height
        max_x  = max(0, canvas_w-sw)
        max_y  = max(0, canvas_h-sh)
        force_overlap = allow_overlap or (rng.random() < overlap_chance)
        if force_overlap:
            canvas.alpha_composite(s, (rng.randint(0, max_x), rng.randint(0, max_y)))
            placed.append((rng.randint(0, max_x), rng.randint(0, max_y), sw, sh))
        else:
            ok = False
            for _ in range(300):
                x, y = rng.randint(0, max_x), rng.randint(0, max_y)
                if not any(x < px+pw and x+sw > px and y < py+ph and y+sh > py
                           for px,py,pw,ph in placed):
                    canvas.alpha_composite(s, (x, y))
                    placed.append((x, y, sw, sh))
                    ok = True
                    break
            if not ok:
                skipped += 1
    if skipped:
        print(f"[WARN] Could not place {skipped} shapes — try --allow-overlap or a larger canvas.")
    return canvas


# ------------------------------------------------------------------------------
#  CLI  (command-line flags override the config defaults above)
# ------------------------------------------------------------------------------

def parse_color(s):
    try:
        parts = [int(v.strip()) for v in s.split(",")]
        assert len(parts) == 3
        return tuple(parts)
    except Exception:
        sys.exit(f"[ERROR] Color must be R,G,B e.g. 255,255,255 — got: {s}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",          default=DEFAULT_INPUT)
    ap.add_argument("--output-dir",     default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--mode",           default=DEFAULT_MODE,           choices=["line","grid","scatter"])
    ap.add_argument("--threshold",      type=int,   default=DEFAULT_THRESHOLD)
    ap.add_argument("--min-size",       type=int,   default=DEFAULT_MIN_SIZE)
    ap.add_argument("--scale-min",      type=float, default=DEFAULT_SCALE_MIN)
    ap.add_argument("--scale-max",      type=float, default=DEFAULT_SCALE_MAX)
    ap.add_argument("--padding",        type=int,   default=DEFAULT_PADDING)
    ap.add_argument("--cols",           type=int,   default=DEFAULT_COLS)
    ap.add_argument("--canvas-w",       type=int,   default=DEFAULT_CANVAS_W)
    ap.add_argument("--canvas-h",       type=int,   default=DEFAULT_CANVAS_H)
    ap.add_argument("--canvas-bg",      default=None)
    ap.add_argument("--shape-color",    default=None)
    ap.add_argument("--seed",           type=int,   default=DEFAULT_SEED)
    ap.add_argument("--allow-overlap",  action="store_true", default=DEFAULT_ALLOW_OVERLAP)
    ap.add_argument("--overlap-chance", type=float, default=DEFAULT_OVERLAP_CHANCE)
    ap.add_argument("--white-bg",       action="store_true")
    args = ap.parse_args()

    canvas_bg   = (255,255,255) if args.white_bg else (parse_color(args.canvas_bg) if args.canvas_bg else DEFAULT_CANVAS_BG)
    shape_color = parse_color(args.shape_color) if args.shape_color else DEFAULT_SHAPE_COLOR

    out_path = unique_output_path(args.output_dir, Path(args.input).stem, args.mode)

    shapes = extract_shapes(args.input, args.threshold, args.min_size)
    if not shapes:
        sys.exit("[ERROR] No shapes found. Try lowering --threshold or --min-size.")

    shapes = [scale_shape(s, random.uniform(args.scale_min, args.scale_max)) for s in shapes]
    if shape_color:
        shapes = recolor_shapes(shapes, shape_color)

    if args.mode == "line":
        canvas = make_line(shapes, args.padding, canvas_bg)
    elif args.mode == "grid":
        canvas = make_grid(shapes, args.padding, args.cols, canvas_bg)
    else:
        canvas = make_scatter(shapes, args.canvas_w, args.canvas_h, canvas_bg, args.seed, args.allow_overlap, args.overlap_chance)

    bg = Image.new("RGB", canvas.size, canvas_bg)
    bg.paste(canvas, mask=canvas.split()[3])
    bg.save(str(out_path), "PNG")
    print(f"[DONE] Saved → {out_path}  ({canvas.width}x{canvas.height}px)")


if __name__ == "__main__":
    main()
