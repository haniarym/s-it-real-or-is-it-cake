"""
image_filter.py
---------------
Applies customizable image filters using OpenCV + PIL.
Every run saves a new numbered file — nothing is ever overwritten.

Requirements:
    pip install opencv-python numpy Pillow scipy

QUICK-RUN COMMANDS (copy one into PowerShell):

  Basic black & white + contrast:
    python image_filter.py --input "C:\path\to\image.jpg"

  High contrast B&W with grain:
    python image_filter.py --input "C:\path\to\image.jpg" --mode bw --contrast 2.0 --noise 25 --grain

  Blur + exposure boost:
    python image_filter.py --input "C:\path\to\image.jpg" --blur 3.0 --exposure 0.3

  Full custom:
    python image_filter.py --input "C:\path\to\image.jpg" --mode bw --exposure 0.1 --contrast 1.8 --gamma 0.7 --blur 1.5 --sharpen 1.5 --noise 20 --grain --vignette 0.6
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter


# ==============================================================================
#  CONFIGURATION  — edit the values in this section
# ==============================================================================

# ------------------------------------------------------------------------------
#  PATHS
#  OUTPUT_DIR is a folder. Every run saves a new numbered file inside it:
#    e.g.  output_folder/photo_filtered_001.jpg
#          output_folder/photo_filtered_002.jpg  (next run, same input)
# ------------------------------------------------------------------------------
DEFAULT_INPUT       = r"C:\path\to\input.png"  # ← change this
DEFAULT_OUTPUT_DIR  = r"C:\path\to\output_folder"     # ← change this

# ------------------------------------------------------------------------------
#  COLOR MODE
#  "bw"    = black and white
#  "color" = keep original colors
# ------------------------------------------------------------------------------
DEFAULT_MODE        = "bw"

# ------------------------------------------------------------------------------
#  EXPOSURE  — overall brightness
#
#  Range:  -1.0  (very dark)  to  1.0  (very bright)
#  0.0 = no change
# ------------------------------------------------------------------------------
DEFAULT_EXPOSURE    = 0.0

# ------------------------------------------------------------------------------
#  CONTRAST
#
#  Range:  0.0  (flat gray)  to  3.0  (extreme contrast)
#  1.0 = no change
# ------------------------------------------------------------------------------
DEFAULT_CONTRAST    = 1.0

# ------------------------------------------------------------------------------
#  GAMMA  — midtone brightness (non-linear)
#
#  Range:  0.1  (very dark midtones)  to  4.0  (very bright midtones)
#  1.0 = no change
#  Below 1.0 = darker midtones (pushes grays toward black)
#  Above 1.0 = brighter midtones (lifts shadows)
# ------------------------------------------------------------------------------
DEFAULT_GAMMA       = 1.0

# ------------------------------------------------------------------------------
#  LEVELS  — clip the darkest and brightest parts of the image
#
#  INPUT_BLACK   0.0–0.45  — pixels below this become pure black
#  INPUT_WHITE   0.55–1.0  — pixels above this become pure white
#  OUTPUT_BLACK  0.0–0.45  — how dark the darkest output pixel is
#  OUTPUT_WHITE  0.55–1.0  — how bright the brightest output pixel is
#
#  All at defaults (0.0 / 1.0 / 0.0 / 1.0) = no levels adjustment
# ------------------------------------------------------------------------------
DEFAULT_LEVELS_ON           = False
DEFAULT_LEVELS_INPUT_BLACK  = 0.0
DEFAULT_LEVELS_INPUT_WHITE  = 1.0
DEFAULT_LEVELS_OUTPUT_BLACK = 0.0
DEFAULT_LEVELS_OUTPUT_WHITE = 1.0

# ------------------------------------------------------------------------------
#  BLUR  — gaussian blur
#
#  0.0 = no blur
#  Range: 0.5 (very subtle) to 20.0 (very blurry)
# ------------------------------------------------------------------------------
DEFAULT_BLUR        = 0.0

# ------------------------------------------------------------------------------
#  SHARPEN  — unsharp mask sharpening
#
#  0.0 = no sharpening
#  Range: 0.5 (subtle) to 3.0 (strong)
# ------------------------------------------------------------------------------
DEFAULT_SHARPEN     = 0.0

# ------------------------------------------------------------------------------
#  NOISE / GRAIN
#
#  NOISE_AMOUNT  — amount of random pixel noise added
#                  0 = none | Range: 1–80
#  GRAIN         — True adds a fine film grain texture on top of noise
# ------------------------------------------------------------------------------
DEFAULT_NOISE_AMOUNT = 0      # 0 = disabled
DEFAULT_GRAIN        = False

# ------------------------------------------------------------------------------
#  VIGNETTE  — darkens the edges of the image
#
#  0.0 = no vignette
#  Range: 0.1 (very subtle) to 1.0 (strong dark edges)
# ------------------------------------------------------------------------------
DEFAULT_VIGNETTE    = 0.0

# ------------------------------------------------------------------------------
#  THRESHOLD / POSTERIZE  — crushes image to pure black & white (no grays)
#
#  THRESHOLD_ON    True = enable hard B&W threshold
#  THRESHOLD_LEVEL 0–255: pixels above this = white, below = black
#                  128 = midpoint  |  lower = more white  |  higher = more black
# ------------------------------------------------------------------------------
DEFAULT_THRESHOLD_ON    = False
DEFAULT_THRESHOLD_LEVEL = 128

# ------------------------------------------------------------------------------
#  OUTPUT QUALITY
#
#  FORMAT  "jpg" or "png"
#  QUALITY 1–95  (jpg only — higher = better quality, larger file)
# ------------------------------------------------------------------------------
DEFAULT_FORMAT      = "jpg"
DEFAULT_QUALITY     = 92

# ==============================================================================
#  END OF CONFIGURATION  — no need to edit below this line
# ==============================================================================


def unique_output_path(folder, stem, fmt):
    out_dir = Path(folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Sanitize stem: replace special/non-ASCII chars so file checks work on Windows
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


def apply_exposure(img, amount):
    """Shift brightness linearly. amount in -1.0 to 1.0."""
    if amount == 0.0:
        return img
    shift = int(amount * 255)
    return np.clip(img.astype(np.int16) + shift, 0, 255).astype(np.uint8)


def apply_contrast(img, factor):
    """Scale contrast around midpoint 128. factor: 0=flat, 1=unchanged, 3=extreme."""
    if factor == 1.0:
        return img
    return np.clip(128 + (img.astype(np.float32) - 128) * factor, 0, 255).astype(np.uint8)


def apply_gamma(img, gamma):
    """Non-linear brightness correction via lookup table."""
    if gamma == 1.0:
        return img
    inv = 1.0 / gamma
    lut = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8)
    return lut[img]


def apply_levels(img, in_black, in_white, out_black, out_white):
    """Remap pixel values: crush blacks/whites and remap output range."""
    in_b  = int(in_black  * 255)
    in_w  = int(in_white  * 255)
    out_b = int(out_black * 255)
    out_w = int(out_white * 255)
    in_range  = max(in_w - in_b, 1)
    out_range = out_w - out_b
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        val = (i - in_b) / in_range
        val = np.clip(val, 0.0, 1.0)
        lut[i] = int(out_b + val * out_range)
    return lut[img]


def apply_blur(img, sigma):
    """Gaussian blur. sigma = blur radius in pixels."""
    if sigma <= 0:
        return img
    ksize = int(sigma * 6) | 1  # must be odd
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def apply_sharpen(img, strength):
    """Unsharp mask sharpening."""
    if strength <= 0:
        return img
    blurred = cv2.GaussianBlur(img, (5, 5), 1.0)
    return np.clip(img.astype(np.float32) + strength * (img.astype(np.float32) - blurred.astype(np.float32)), 0, 255).astype(np.uint8)


def apply_noise(img, amount, grain):
    """Add random pixel noise and optional fine grain texture."""
    if amount <= 0 and not grain:
        return img
    result = img.astype(np.float32)
    if amount > 0:
        noise = np.random.randint(-amount, amount + 1, img.shape, dtype=np.int16)
        result = np.clip(result + noise, 0, 255)
    if grain:
        # Fine grain: smaller, subtler high-frequency noise
        g = np.random.normal(0, amount * 0.4 if amount > 0 else 8, img.shape).astype(np.float32)
        result = np.clip(result + g, 0, 255)
    return result.astype(np.uint8)


def apply_vignette(img, strength):
    """Darken edges with an elliptical gradient."""
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    # Create a vignette mask
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    # Normalized distance from center
    dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    # Smooth falloff
    mask = 1.0 - np.clip(dist * strength, 0, 1)
    mask = mask ** 1.5  # soften the curve
    if len(img.shape) == 3:
        mask = mask[:, :, np.newaxis]
    return np.clip(img.astype(np.float32) * mask, 0, 255).astype(np.uint8)


def apply_threshold(img, level):
    """Hard crush to pure black and white — no grays."""
    return np.where(img >= level, 255, 0).astype(np.uint8)


def process_image(
    input_path,
    output_dir,
    mode             = DEFAULT_MODE,
    exposure         = DEFAULT_EXPOSURE,
    contrast         = DEFAULT_CONTRAST,
    gamma            = DEFAULT_GAMMA,
    levels_on        = DEFAULT_LEVELS_ON,
    levels_in_black  = DEFAULT_LEVELS_INPUT_BLACK,
    levels_in_white  = DEFAULT_LEVELS_INPUT_WHITE,
    levels_out_black = DEFAULT_LEVELS_OUTPUT_BLACK,
    levels_out_white = DEFAULT_LEVELS_OUTPUT_WHITE,
    blur             = DEFAULT_BLUR,
    sharpen          = DEFAULT_SHARPEN,
    noise_amount     = DEFAULT_NOISE_AMOUNT,
    grain            = DEFAULT_GRAIN,
    vignette         = DEFAULT_VIGNETTE,
    threshold_on     = DEFAULT_THRESHOLD_ON,
    threshold_level  = DEFAULT_THRESHOLD_LEVEL,
    fmt              = DEFAULT_FORMAT,
    quality          = DEFAULT_QUALITY,
):
# Use numpy to read the file first — handles special characters in paths on Windows
    with open(input_path, 'rb') as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"[ERROR] Could not read image: {input_path}")

    print(f"[INFO] Loaded {input_path}  ({img.shape[1]}x{img.shape[0]}px)")

    # Convert to B&W if needed
    if mode == "bw":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # --- Apply filters in order ---
    img = apply_exposure(img, exposure)
    print(f"  exposure      {exposure:+.2f}")

    img = apply_contrast(img, contrast)
    print(f"  contrast      {contrast:.2f}")

    img = apply_gamma(img, gamma)
    print(f"  gamma         {gamma:.2f}")

    if levels_on:
        img = apply_levels(img, levels_in_black, levels_in_white, levels_out_black, levels_out_white)
        print(f"  levels        in({levels_in_black:.2f}–{levels_in_white:.2f})  out({levels_out_black:.2f}–{levels_out_white:.2f})")

    img = apply_blur(img, blur)
    if blur > 0:
        print(f"  blur          sigma={blur:.1f}")

    img = apply_sharpen(img, sharpen)
    if sharpen > 0:
        print(f"  sharpen       strength={sharpen:.1f}")

    img = apply_noise(img, noise_amount, grain)
    if noise_amount > 0 or grain:
        print(f"  noise         amount={noise_amount}  grain={grain}")

    img = apply_vignette(img, vignette)
    if vignette > 0:
        print(f"  vignette      strength={vignette:.2f}")

    if threshold_on:
        # Apply threshold to grayscale version, then expand back
        gray_t = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_t = apply_threshold(gray_t, threshold_level)
        img    = cv2.cvtColor(gray_t, cv2.COLOR_GRAY2BGR)
        print(f"  threshold     level={threshold_level}")

    # Save
    stem     = Path(input_path).stem
    out_path = unique_output_path(output_dir, stem, fmt)

    # Use imencode + open() to handle special characters in output path on Windows
    if fmt == "png":
        success, buf = cv2.imencode(".png", img)
    else:
        success, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        sys.exit("[ERROR] Failed to encode image.")
    with open(str(out_path), "wb") as f:
        f.write(buf.tobytes())

    print(f"[DONE] Saved → {out_path}")
    return str(out_path)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Apply image filters with auto-numbered output.")
    ap.add_argument("--input",               default=DEFAULT_INPUT)
    ap.add_argument("--output-dir",          default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--mode",                default=DEFAULT_MODE,             choices=["bw","color"])
    ap.add_argument("--exposure",            type=float, default=DEFAULT_EXPOSURE)
    ap.add_argument("--contrast",            type=float, default=DEFAULT_CONTRAST)
    ap.add_argument("--gamma",               type=float, default=DEFAULT_GAMMA)
    ap.add_argument("--levels",              action="store_true",              default=DEFAULT_LEVELS_ON)
    ap.add_argument("--levels-in-black",     type=float, default=DEFAULT_LEVELS_INPUT_BLACK)
    ap.add_argument("--levels-in-white",     type=float, default=DEFAULT_LEVELS_INPUT_WHITE)
    ap.add_argument("--levels-out-black",    type=float, default=DEFAULT_LEVELS_OUTPUT_BLACK)
    ap.add_argument("--levels-out-white",    type=float, default=DEFAULT_LEVELS_OUTPUT_WHITE)
    ap.add_argument("--blur",                type=float, default=DEFAULT_BLUR)
    ap.add_argument("--sharpen",             type=float, default=DEFAULT_SHARPEN)
    ap.add_argument("--noise",               type=int,   default=DEFAULT_NOISE_AMOUNT)
    ap.add_argument("--grain",               action="store_true",              default=DEFAULT_GRAIN)
    ap.add_argument("--vignette",            type=float, default=DEFAULT_VIGNETTE)
    ap.add_argument("--threshold",           action="store_true",              default=DEFAULT_THRESHOLD_ON)
    ap.add_argument("--threshold-level",     type=int,   default=DEFAULT_THRESHOLD_LEVEL)
    ap.add_argument("--format",              default=DEFAULT_FORMAT,           choices=["jpg","png"])
    ap.add_argument("--quality",             type=int,   default=DEFAULT_QUALITY)
    args = ap.parse_args()

    process_image(
        input_path       = args.input,
        output_dir       = args.output_dir,
        mode             = args.mode,
        exposure         = args.exposure,
        contrast         = args.contrast,
        gamma            = args.gamma,
        levels_on        = args.levels,
        levels_in_black  = args.levels_in_black,
        levels_in_white  = args.levels_in_white,
        levels_out_black = args.levels_out_black,
        levels_out_white = args.levels_out_white,
        blur             = args.blur,
        sharpen          = args.sharpen,
        noise_amount     = args.noise,
        grain            = args.grain,
        vignette         = args.vignette,
        threshold_on     = args.threshold,
        threshold_level  = args.threshold_level,
        fmt              = args.format,
        quality          = args.quality,
    )


if __name__ == "__main__":
    main()
