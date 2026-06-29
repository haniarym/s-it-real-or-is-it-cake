"""
video_filter.py
---------------
Applies customizable image filters to every frame of a video.
Never overwrites previous results — creates a new numbered file each run.

Requirements:
    pip install opencv-python numpy
    ffmpeg must be installed: winget install ffmpeg

QUICK-RUN COMMANDS:

  Basic B&W + contrast:
    python video_filter.py --input "C:\path\to\video.mp4"

  High contrast B&W with grain:
    python video_filter.py --input "C:\path\to\video.mp4" --contrast 2.0 --gamma 0.65 --noise 20 --grain

  Full custom:
    python video_filter.py --input "C:\path\to\video.mp4" --color-mode bw --exposure 0.1 --contrast 1.8 --gamma 0.7 --blur 1.5 --sharpen 1.5 --noise 20 --grain --vignette 0.5
"""

import sys
import subprocess
from pathlib import Path
import cv2
import numpy as np


# ==============================================================================
#  CONFIGURATION  -- edit the values in this section
# ==============================================================================

# PATHS -- OUTPUT_DIR is a folder, new numbered file created each run
DEFAULT_INPUT      = r"C:\path\to\your_video.mp4"   # change this
DEFAULT_OUTPUT_DIR = r"C:\path\to\output_folder"     # change this

# COLOR MODE: "bw" = black and white | "color" = keep original
DEFAULT_COLOR_MODE = "bw"

# EXPOSURE: -1.0 (dark) to 1.0 (bright) | 0.0 = no change
DEFAULT_EXPOSURE = 0.0

# CONTRAST: 0.0 (flat) to 3.0 (extreme) | 1.0 = no change
DEFAULT_CONTRAST = 1.8

# GAMMA: 0.1 (dark mids) to 4.0 (bright mids) | 1.0 = no change
# Below 1.0 pushes midtones toward black
DEFAULT_GAMMA = 0.65

# LEVELS -- clips shadows/highlights. Set LEVELS_ON = True to enable
# INPUT_BLACK  0.0-0.45: pixels below this -> pure black
# INPUT_WHITE  0.55-1.0: pixels above this -> pure white
# OUTPUT_BLACK 0.0-0.45: darkest output value
# OUTPUT_WHITE 0.55-1.0: brightest output value
DEFAULT_LEVELS_ON        = False
DEFAULT_LEVELS_IN_BLACK  = 0.0
DEFAULT_LEVELS_IN_WHITE  = 1.0
DEFAULT_LEVELS_OUT_BLACK = 0.0
DEFAULT_LEVELS_OUT_WHITE = 1.0

# BLUR: 0.0 = off | 0.5 (subtle) to 20.0 (very blurry)
DEFAULT_BLUR = 0.0

# SHARPEN: 0.0 = off | 0.5 (subtle) to 3.0 (strong)
DEFAULT_SHARPEN = 0.0

# NOISE / GRAIN
# NOISE_AMOUNT: 0 = off | 1-80 = strength
# GRAIN: True = add fine film grain on top
DEFAULT_NOISE_AMOUNT = 0
DEFAULT_GRAIN        = False

# VIGNETTE: 0.0 = off | 0.1 (subtle) to 1.0 (strong edge darkening)
DEFAULT_VIGNETTE = 0.0

# THRESHOLD -- hard crush to pure black/white. Set THRESHOLD_ON = True to enable
# THRESHOLD_LEVEL: 0-255. 128 = midpoint | lower = more white | higher = more black
DEFAULT_THRESHOLD_ON    = False
DEFAULT_THRESHOLD_LEVEL = 128

# OUTPUT FORMAT and QUALITY
# FORMAT: "mp4" or "avi"
# QUALITY: CRF -- 18 = high quality | 23 = balanced | 28 = smaller file
DEFAULT_FORMAT  = "mp4"
DEFAULT_QUALITY = 23

# ==============================================================================
#  END OF CONFIGURATION  -- no need to edit below this line
# ==============================================================================


def apply_exposure(img, amount):
    if amount == 0.0:
        return img
    shift = int(amount * 255)
    return np.clip(img.astype(np.int16) + shift, 0, 255).astype(np.uint8)

def apply_contrast(img, factor):
    if factor == 1.0:
        return img
    return np.clip(128 + (img.astype(np.float32) - 128) * factor, 0, 255).astype(np.uint8)

def apply_gamma(img, gamma):
    if gamma == 1.0:
        return img
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
    return lut[img]

def apply_levels(img, in_black, in_white, out_black, out_white):
    in_b  = int(in_black  * 255)
    in_w  = int(in_white  * 255)
    out_b = int(out_black * 255)
    out_w = int(out_white * 255)
    in_range  = max(in_w - in_b, 1)
    out_range = out_w - out_b
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        val = np.clip((i - in_b) / in_range, 0.0, 1.0)
        lut[i] = int(out_b + val * out_range)
    return lut[img]

def apply_blur(img, sigma):
    if sigma <= 0:
        return img
    ksize = int(sigma * 6) | 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)

def apply_sharpen(img, strength):
    if strength <= 0:
        return img
    blurred = cv2.GaussianBlur(img, (5, 5), 1.0)
    return np.clip(img.astype(np.float32) + strength * (img.astype(np.float32) - blurred), 0, 255).astype(np.uint8)

def apply_noise(img, amount, grain):
    if amount <= 0 and not grain:
        return img
    result = img.astype(np.float32)
    if amount > 0:
        noise = np.random.randint(-amount, amount + 1, img.shape, dtype=np.int16)
        result = np.clip(result + noise, 0, 255)
    if grain:
        g = np.random.normal(0, amount * 0.4 if amount > 0 else 8, img.shape).astype(np.float32)
        result = np.clip(result + g, 0, 255)
    return result.astype(np.uint8)

def apply_vignette(img, strength):
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    mask = (1.0 - np.clip(dist * strength, 0, 1)) ** 1.5
    if len(img.shape) == 3:
        mask = mask[:, :, np.newaxis]
    return np.clip(img.astype(np.float32) * mask, 0, 255).astype(np.uint8)

def apply_threshold_filter(img, level):
    return np.where(img >= level, 255, 0).astype(np.uint8)

def process_frame(frame, args):
    img = frame.copy()
    if args.color_mode == "bw":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img = apply_exposure(img, args.exposure)
    img = apply_contrast(img, args.contrast)
    img = apply_gamma(img, args.gamma)
    if args.levels:
        img = apply_levels(img, args.levels_in_black, args.levels_in_white,
                           args.levels_out_black, args.levels_out_white)
    img = apply_blur(img, args.blur)
    img = apply_sharpen(img, args.sharpen)
    img = apply_noise(img, args.noise, args.grain)
    img = apply_vignette(img, args.vignette)
    if args.threshold:
        gray_t = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_t = apply_threshold_filter(gray_t, args.threshold_level)
        img    = cv2.cvtColor(gray_t, cv2.COLOR_GRAY2BGR)
    return img

def unique_output_path(folder, stem, fmt):
    out_dir = Path(folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = "".join(c if (c.isascii() and (c.isalnum() or c in "-_. ")) else "_" for c in stem)
    i = 1
    while True:
        p = out_dir / f"{safe_stem}_filtered_{i:03d}.{fmt}"
        try:
            exists = p.exists()
        except Exception:
            exists = False
        if not exists:
            return p
        i += 1

def run(args):
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Could not open video: {args.input}")
    fps      = cap.get(cv2.CAP_PROP_FPS)
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps if fps > 0 else 0
    print(f"[INFO] Opened: {args.input}")
    print(f"       {width}x{height}  {fps:.2f}fps  {total} frames  ({duration:.1f}s)")
    print(f"[INFO] Filters: color={args.color_mode}  exposure={args.exposure:+.2f}  contrast={args.contrast:.2f}  gamma={args.gamma:.2f}")
    if args.blur     > 0: print(f"         blur={args.blur:.1f}")
    if args.sharpen  > 0: print(f"         sharpen={args.sharpen:.1f}")
    if args.noise    > 0: print(f"         noise={args.noise}  grain={args.grain}")
    if args.vignette > 0: print(f"         vignette={args.vignette:.2f}")
    if args.threshold:    print(f"         threshold={args.threshold_level}")
    stem     = Path(args.input).stem
    out_path = unique_output_path(args.output_dir, stem, args.format)
    tmp_path = str(out_path).replace(f".{args.format}", "_tmp.avi")
    fourcc   = cv2.VideoWriter_fourcc(*"MJPG")
    writer   = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        sys.exit("[ERROR] Could not open video writer.")
    frame_idx = 0
    print(f"[INFO] Processing {total} frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(process_frame(frame, args))
        frame_idx += 1
        if frame_idx % 30 == 0:
            pct = frame_idx / total * 100 if total > 0 else 0
            print(f"       {frame_idx}/{total}  ({pct:.0f}%)", end="\r")
    cap.release()
    writer.release()
    print(f"\n[INFO] Remuxing to {args.format}...")
    result = subprocess.run([
        "ffmpeg", "-y", "-i", tmp_path,
        "-c:v", "libx264", "-crf", str(args.quality),
        "-preset", "fast", "-pix_fmt", "yuv420p",
        str(out_path)
    ], capture_output=True, text=True)
    Path(tmp_path).unlink(missing_ok=True)
    if result.returncode != 0:
        sys.exit(f"[ERROR] ffmpeg failed:\n{result.stderr}")
    print(f"[DONE] Saved -> {out_path}  ({width}x{height}  {fps:.2f}fps)")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",              default=DEFAULT_INPUT)
    ap.add_argument("--output-dir",         default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--color-mode",         default=DEFAULT_COLOR_MODE,   choices=["bw","color"])
    ap.add_argument("--exposure",           type=float, default=DEFAULT_EXPOSURE)
    ap.add_argument("--contrast",           type=float, default=DEFAULT_CONTRAST)
    ap.add_argument("--gamma",              type=float, default=DEFAULT_GAMMA)
    ap.add_argument("--levels",             action="store_true",          default=DEFAULT_LEVELS_ON)
    ap.add_argument("--levels-in-black",    type=float, default=DEFAULT_LEVELS_IN_BLACK)
    ap.add_argument("--levels-in-white",    type=float, default=DEFAULT_LEVELS_IN_WHITE)
    ap.add_argument("--levels-out-black",   type=float, default=DEFAULT_LEVELS_OUT_BLACK)
    ap.add_argument("--levels-out-white",   type=float, default=DEFAULT_LEVELS_OUT_WHITE)
    ap.add_argument("--blur",               type=float, default=DEFAULT_BLUR)
    ap.add_argument("--sharpen",            type=float, default=DEFAULT_SHARPEN)
    ap.add_argument("--noise",              type=int,   default=DEFAULT_NOISE_AMOUNT)
    ap.add_argument("--grain",              action="store_true",          default=DEFAULT_GRAIN)
    ap.add_argument("--vignette",           type=float, default=DEFAULT_VIGNETTE)
    ap.add_argument("--threshold",          action="store_true",          default=DEFAULT_THRESHOLD_ON)
    ap.add_argument("--threshold-level",    type=int,   default=DEFAULT_THRESHOLD_LEVEL)
    ap.add_argument("--format",             default=DEFAULT_FORMAT,       choices=["mp4","avi"])
    ap.add_argument("--quality",            type=int,   default=DEFAULT_QUALITY)
    args = ap.parse_args()
    run(args)

if __name__ == "__main__":
    main()
