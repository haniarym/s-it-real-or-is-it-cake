"""
video_tracked_collage.py
------------------------
Extracts white shapes from a reference frame, tracks each shape's
region through the video, and composites them at fixed canvas positions.
The content inside each shape follows the tracked region -- when the
object moves in the video, what you see inside the porthole moves too.
When a tracked region leaves the frame it goes dark.

Every run picks a different random reference frame and canvas layout.
Never overwrites previous results.

Requirements:
    pip install opencv-python numpy Pillow
    ffmpeg must be installed: winget install ffmpeg

QUICK-RUN COMMANDS:

  Basic run:
    python video_tracked_collage.py --input "C:\path\to\video.mp4"

  Larger shapes, more overlap:
    python video_tracked_collage.py --input "C:\path\to\video.mp4" --scale-min 2.0 --scale-max 6.0 --overlap-chance 0.5 --min-size 80

  Specific reference frame:
    python video_tracked_collage.py --input "C:\path\to\video.mp4" --ref-time 8.5
"""

import random
import sys
import subprocess
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


# ==============================================================================
#  CONFIGURATION  -- edit the values in this section
# ==============================================================================

# PATHS
DEFAULT_INPUT      = r"C:\path\to\your_video.mp4"   # change this
DEFAULT_OUTPUT_DIR = r"C:\path\to\output_folder"     # change this

# ------------------------------------------------------------------------------
#  REFERENCE FRAME
#  Shapes (masks) are extracted from this frame. Each shape's region is then
#  tracked forward through the video.
#  REF_TIME  time in seconds. -1 = pick a random time each run.
# ------------------------------------------------------------------------------
DEFAULT_REF_TIME = -1       # -1 = random | or set e.g. 8.5

# ------------------------------------------------------------------------------
#  SHAPE EXTRACTION
#  THRESHOLD  0-255: pixels brighter than this become shapes
#             Lower = more shapes | Higher = only very bright regions
#  MIN_SIZE   minimum contour area in pixels (filters out noise speckles)
# ------------------------------------------------------------------------------
DEFAULT_THRESHOLD = 80
DEFAULT_MIN_SIZE  = 100

# ------------------------------------------------------------------------------
#  SHAPE SCALE
#  Each shape porthole is scaled randomly between MIN and MAX
# ------------------------------------------------------------------------------
DEFAULT_SCALE_MIN = 1.0
DEFAULT_SCALE_MAX = 4.0

# ------------------------------------------------------------------------------
#  CANVAS
#  CANVAS_W / CANVAS_H  output frame size in pixels
#  CANVAS_BG            background color (R, G, B) -- shown when a shape goes dark
# ------------------------------------------------------------------------------
DEFAULT_CANVAS_W  = 3000
DEFAULT_CANVAS_H  = 1000
DEFAULT_CANVAS_BG = (0, 0, 0)    # black | (255,255,255) for white

# ------------------------------------------------------------------------------
#  OVERLAP
#  OVERLAP_CHANCE  0.0 = shapes never overlap | 1.0 = always overlap
# ------------------------------------------------------------------------------
DEFAULT_OVERLAP_CHANCE = 0.4

# ------------------------------------------------------------------------------
#  TRACKING
#  TRACKER_TYPE  algorithm used to track each region through frames.
#                "CSRT"  -- most accurate, slower
#                "KCF"   -- fast, good for simple motion
#                "MOSSE" -- fastest, least accurate
#  MAX_SHAPES    maximum number of shapes to track simultaneously
#                (more = slower processing)
# ------------------------------------------------------------------------------
DEFAULT_TRACKER_TYPE = "CSRT"
DEFAULT_MAX_SHAPES   = 12

# ------------------------------------------------------------------------------
#  OUTPUT
#  OUTPUT_FPS  -1 = match source video fps
#  FORMAT      "mp4" or "avi"
#  QUALITY     CRF: 18 = high quality | 23 = balanced | 28 = smaller file
# ------------------------------------------------------------------------------
DEFAULT_OUTPUT_FPS = -1
DEFAULT_FORMAT     = "mp4"
DEFAULT_QUALITY    = 23

# ==============================================================================
#  END OF CONFIGURATION  -- no need to edit below this line
# ==============================================================================


def unique_output_path(folder, stem, fmt):
    out_dir = Path(folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = "".join(c if (c.isascii() and (c.isalnum() or c in "-_. ")) else "_" for c in stem)
    i = 1
    while True:
        p = out_dir / f"{safe_stem}_tracked_{i:03d}.{fmt}"
        try:
            exists = p.exists()
        except Exception:
            exists = False
        if not exists:
            return p
        i += 1


def make_tracker(tracker_type):
    t = tracker_type.upper()
    try:
        if t == "CSRT":
            return cv2.legacy.TrackerCSRT_create()
        elif t == "KCF":
            return cv2.legacy.TrackerKCF_create()
        elif t == "MOSSE":
            return cv2.legacy.TrackerMOSSE_create()
        else:
            return cv2.legacy.TrackerCSRT_create()
    except AttributeError:
        # Fallback for some OpenCV builds
        return cv2.TrackerKCF_create()


def extract_shape_regions(frame, threshold, min_size, max_shapes):
    """
    Returns list of (x, y, w, h) bounding boxes of detected white regions,
    limited to max_shapes largest ones.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_size:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # Build per-contour mask
        local_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(local_mask, [cnt - [x, y]], -1, 255, thickness=cv2.FILLED)
        regions.append({"bbox": (x, y, w, h), "mask": local_mask, "area": area})

    # Keep only the largest max_shapes
    regions.sort(key=lambda r: r["area"], reverse=True)
    return regions[:max_shapes]


def scatter_placements(regions, canvas_w, canvas_h, scale_min, scale_max,
                       overlap_chance, rng):
    """
    Assigns each region a fixed position on the canvas and a scale.
    Returns list of placement dicts.
    """
    placements = []
    placed_rects = []

    for r in regions:
        x, y, w, h = r["bbox"]
        scale = rng.uniform(scale_min, scale_max)
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))

        # Scale the contour mask
        scaled_mask = cv2.resize(r["mask"], (nw, nh), interpolation=cv2.INTER_LINEAR)
        _, scaled_mask = cv2.threshold(scaled_mask, 127, 255, cv2.THRESH_BINARY)

        max_cx = max(0, canvas_w - nw)
        max_cy = max(0, canvas_h - nh)
        if max_cx == 0 and max_cy == 0:
            continue

        placed = False
        if rng.random() >= overlap_chance:
            for _ in range(300):
                cx = rng.randint(0, max_cx)
                cy = rng.randint(0, max_cy)
                if not any(cx < rx+rw and cx+nw > rx and cy < ry+rh and cy+nh > ry
                           for rx, ry, rw, rh in placed_rects):
                    placed_rects.append((cx, cy, nw, nh))
                    placed = True
                    placements.append({
                        "canvas_pos": (cx, cy),
                        "canvas_size": (nw, nh),
                        "orig_bbox":  (x, y, w, h),
                        "scale":      scale,
                        "mask":       scaled_mask,
                    })
                    break

        if not placed:
            cx = rng.randint(0, max_cx)
            cy = rng.randint(0, max_cy)
            placed_rects.append((cx, cy, nw, nh))
            placements.append({
                "canvas_pos": (cx, cy),
                "canvas_size": (nw, nh),
                "orig_bbox":  (x, y, w, h),
                "scale":      scale,
                "mask":       scaled_mask,
            })

    return placements


def render_frame(video_frame, placements, tracker_states,
                 canvas_w, canvas_h, bg):
    """
    For each shape: sample the current tracked region from the video frame,
    scale it, apply the shape mask, and composite at the fixed canvas position.
    tracker_states[i] = (x, y, w, h) current tracked bbox, or None if lost.
    """
    src_h, src_w = video_frame.shape[:2]
    canvas = np.full((canvas_h, canvas_w, 3), bg, dtype=np.uint8)

    for i, p in enumerate(placements):
        tracked = tracker_states[i]
        if tracked is None:
            # Shape lost -- porthole stays dark (bg color)
            continue

        tx, ty, tw, th = tracked
        # Clamp to frame
        tx = max(0, min(tx, src_w - 1))
        ty = max(0, min(ty, src_h - 1))
        tx2 = min(tx + tw, src_w)
        ty2 = min(ty + th, src_h)
        if tx2 - tx <= 0 or ty2 - ty <= 0:
            continue

        nw, nh = p["canvas_size"]
        cx, cy = p["canvas_pos"]
        mask   = p["mask"]

        # Crop tracked region from video and scale to porthole size
        crop = video_frame[ty:ty2, tx:tx2]
        if crop.size == 0:
            continue
        scaled_crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)

        # Clamp canvas region
        cx2 = min(cx + nw, canvas_w)
        cy2 = min(cy + nh, canvas_h)
        pw = cx2 - cx
        ph = cy2 - cy
        if pw <= 0 or ph <= 0:
            continue

        crop_region   = scaled_crop[:ph, :pw]
        mask_region   = mask[:ph, :pw]
        canvas_region = canvas[cy:cy2, cx:cx2]

        mask_3 = cv2.cvtColor(mask_region, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        canvas[cy:cy2, cx:cx2] = np.clip(
            canvas_region.astype(np.float32) * (1.0 - mask_3) +
            crop_region.astype(np.float32)   * mask_3,
            0, 255
        ).astype(np.uint8)

    return canvas


def run(args):
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Could not open video: {args.input}")

    fps      = cap.get(cv2.CAP_PROP_FPS)
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total / fps if fps > 0 else 0
    out_fps  = fps if args.output_fps <= 0 else args.output_fps

    print(f"[INFO] Opened: {args.input}")
    print(f"       {src_w}x{src_h}  {fps:.2f}fps  {total} frames  ({duration:.1f}s)")

    # Pick reference frame
    ref_time = args.ref_time if args.ref_time >= 0 else random.uniform(0, max(1, duration - 2))
    ref_frame_idx = int(ref_time * fps)
    print(f"[INFO] Reference frame: {ref_time:.2f}s (frame {ref_frame_idx})")

    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_frame_idx)
    ret, ref_frame = cap.read()
    if not ret:
        sys.exit("[ERROR] Could not read reference frame.")

    # Extract shape regions from reference frame
    regions = extract_shape_regions(ref_frame, args.threshold, args.min_size, args.max_shapes)
    print(f"[INFO] Found {len(regions)} shapes to track")
    if not regions:
        sys.exit("[ERROR] No shapes found. Try lowering --threshold or --min-size.")

    # Assign canvas positions
    rng = random.Random()
    placements = scatter_placements(
        regions, args.canvas_w, args.canvas_h,
        args.scale_min, args.scale_max,
        args.overlap_chance, rng
    )
    print(f"[INFO] Placed {len(placements)} shapes on canvas")

    # Initialise one tracker per shape, starting at the reference frame
    trackers = []
    tracker_states = []
    for p in placements:
        x, y, w, h = p["orig_bbox"]
        bbox = (x, y, w, h)
        t = make_tracker(args.tracker_type)
        t.init(ref_frame, bbox)
        trackers.append(t)
        tracker_states.append((x, y, w, h))

    # Set up video writer
    stem     = Path(args.input).stem
    out_path = unique_output_path(args.output_dir, stem, args.format)
    tmp_path = str(out_path).replace(f".{args.format}", "_tmp.avi")
    fourcc   = cv2.VideoWriter_fourcc(*"MJPG")
    writer   = cv2.VideoWriter(tmp_path, fourcc, out_fps, (args.canvas_w, args.canvas_h))
    if not writer.isOpened():
        sys.exit("[ERROR] Could not open video writer.")

    bg_tuple = args.canvas_bg

    # Seek back to reference frame and process forward
    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_frame_idx)
    frame_idx = ref_frame_idx
    written   = 0
    remaining = total - ref_frame_idx

    print(f"[INFO] Rendering from {ref_time:.1f}s to end ({remaining} frames)...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Update each tracker
        for i, t in enumerate(trackers):
            if tracker_states[i] is None:
                continue
            ok, bbox = t.update(frame)
            if ok:
                x, y, w, h = [int(v) for v in bbox]
                # Check if still in frame
                if x >= src_w or y >= src_h or x + w <= 0 or y + h <= 0:
                    tracker_states[i] = None
                else:
                    tracker_states[i] = (x, y, w, h)
            else:
                tracker_states[i] = None

        out_frame = render_frame(frame, placements, tracker_states,
                                 args.canvas_w, args.canvas_h, bg_tuple)
        writer.write(out_frame)
        written   += 1
        frame_idx += 1

        if written % 30 == 0:
            pct = written / remaining * 100 if remaining > 0 else 0
            active = sum(1 for s in tracker_states if s is not None)
            print(f"       frame {written}/{remaining}  ({pct:.0f}%)  active tracks: {active}", end="\r")

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

    print(f"[DONE] Saved -> {out_path}  ({args.canvas_w}x{args.canvas_h}  {out_fps:.1f}fps)")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",           default=DEFAULT_INPUT)
    ap.add_argument("--output-dir",      default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--ref-time",        type=float, default=DEFAULT_REF_TIME)
    ap.add_argument("--threshold",       type=int,   default=DEFAULT_THRESHOLD)
    ap.add_argument("--min-size",        type=int,   default=DEFAULT_MIN_SIZE)
    ap.add_argument("--scale-min",       type=float, default=DEFAULT_SCALE_MIN)
    ap.add_argument("--scale-max",       type=float, default=DEFAULT_SCALE_MAX)
    ap.add_argument("--canvas-w",        type=int,   default=DEFAULT_CANVAS_W)
    ap.add_argument("--canvas-h",        type=int,   default=DEFAULT_CANVAS_H)
    ap.add_argument("--canvas-bg",       default=None)
    ap.add_argument("--overlap-chance",  type=float, default=DEFAULT_OVERLAP_CHANCE)
    ap.add_argument("--tracker-type",    default=DEFAULT_TRACKER_TYPE, choices=["CSRT","KCF","MOSSE"])
    ap.add_argument("--max-shapes",      type=int,   default=DEFAULT_MAX_SHAPES)
    ap.add_argument("--output-fps",      type=float, default=DEFAULT_OUTPUT_FPS)
    ap.add_argument("--format",          default=DEFAULT_FORMAT, choices=["mp4","avi"])
    ap.add_argument("--quality",         type=int,   default=DEFAULT_QUALITY)
    args = ap.parse_args()

    if args.canvas_bg:
        parts = [int(v.strip()) for v in args.canvas_bg.split(",")]
        args.canvas_bg = tuple(parts)
    else:
        args.canvas_bg = DEFAULT_CANVAS_BG

    run(args)

if __name__ == "__main__":
    main()
