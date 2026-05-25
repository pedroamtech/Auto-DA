"""
Convert Manipal-UAV dataset annotations to per-frame YOLO label files.

The script auto-detects one of two input formats:

  FORMAT A — MOT-style (pixel coordinates, space or comma separated):
    frame_id  track_id  x_left  y_top  width  height  [conf  class  visibility]
    Example: 1 0 764 309 59 132 1 1 1

  FORMAT B — Already YOLO (normalized, 5 fields):
    class_id  cx  cy  w  h
    Example:  0 0.409703 0.786944 0.042828 0.189583

  For FORMAT B the script simply validates ranges and copies annotations
  into per-frame files (useful when a single big file contains all frames).

YOLO output (one .txt file per image frame):
  class_id cx cy w h   (all values normalized 0-1)

Usage:
  python tools/convert_manipal_to_yolo.py \
      --labels_dir path/to/Manipal/labels \
      --output_dir path/to/Manipal/yolo_labels \
      [--img_width 3840] [--img_height 2160]

  # If each .txt file is already a single-frame YOLO file (FORMAT B, no frame_id):
  python tools/convert_manipal_to_yolo.py \
      --labels_dir path/to/Manipal/labels \
      --output_dir path/to/Manipal/yolo_labels \
      --already_yolo

Expected input structure for MOT format (one file per sequence/split):
  labels/
    Train/
      seq001.txt
      seq002.txt
    Test/
      seq003.txt

Output mirrors input structure:
  yolo_labels/
    Train/
      seq001/
        000001.txt
        000002.txt
      ...
"""

import os
import argparse
from pathlib import Path
from collections import defaultdict

DEFAULT_WIDTH  = 3840   # Manipal-UAV 4K
DEFAULT_HEIGHT = 2160

PERSON_CLASS_ID = 0


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def _is_normalized(values):
    """True if all float values are in [0, 1] — consistent with YOLO format."""
    return all(0.0 <= v <= 1.0 for v in values)


def detect_format(ann_path):
    """
    Peek at first non-empty line to decide if the file is MOT or YOLO format.
    Returns 'yolo' or 'mot'.
    """
    with open(ann_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Try comma or space split
            sep = "," if "," in line else None
            parts = line.split(sep)
            if len(parts) < 5:
                continue
            try:
                floats = [float(p) for p in parts]
            except ValueError:
                continue

            if len(floats) == 5 and _is_normalized(floats[1:]):
                return "yolo"
            return "mot"
    return "mot"


# ---------------------------------------------------------------------------
# MOT format parser
# ---------------------------------------------------------------------------

def parse_mot_line(line):
    """
    Parse MOT-style annotation line (space or comma separated).
    Expected fields: frame_id track_id x_left y_top width height [conf class vis]
    Returns (frame_id, x_left, y_top, w_px, h_px) or None.
    """
    line = line.strip()
    if not line:
        return None
    sep = "," if "," in line else None
    parts = line.split(sep)
    if len(parts) < 6:
        return None
    try:
        frame_id = int(float(parts[0]))
        x_left   = float(parts[2])
        y_top    = float(parts[3])
        w_px     = float(parts[4])
        h_px     = float(parts[5])
    except (ValueError, IndexError):
        return None
    return frame_id, x_left, y_top, w_px, h_px


def mot_to_yolo(x_left, y_top, w_px, h_px, img_w, img_h):
    cx = (x_left + w_px / 2) / img_w
    cy = (y_top  + h_px / 2) / img_h
    w  = w_px / img_w
    h  = h_px / img_h
    return cx, cy, w, h


def convert_mot_file(ann_path, out_dir, img_w, img_h):
    frames = defaultdict(list)

    with open(ann_path, encoding="utf-8") as f:
        for line in f:
            parsed = parse_mot_line(line)
            if parsed is None:
                continue
            frame_id, x_left, y_top, w_px, h_px = parsed

            # Clamp
            x_left = max(0.0, x_left)
            y_top  = max(0.0, y_top)
            w_px   = min(w_px, img_w - x_left)
            h_px   = min(h_px, img_h - y_top)

            if w_px <= 0 or h_px <= 0:
                continue

            cx, cy, w, h = mot_to_yolo(x_left, y_top, w_px, h_px, img_w, img_h)
            frames[frame_id].append((PERSON_CLASS_ID, cx, cy, w, h))

    _write_frames(frames, out_dir)
    return len(frames)


# ---------------------------------------------------------------------------
# YOLO format (already normalized) — validate + write per-frame if needed
# ---------------------------------------------------------------------------

def parse_yolo_line(line):
    """
    Parse a YOLO-format line: class_id cx cy w h
    Returns (class_id, cx, cy, w, h) or None.
    """
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        class_id = int(parts[0])
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    except ValueError:
        return None
    if not _is_normalized([cx, cy, w, h]):
        return None
    return class_id, cx, cy, w, h


def copy_yolo_file(ann_path, out_dir):
    """
    Validate and copy a single already-YOLO file (no frame_id field).
    Used when each input file corresponds to one image frame.
    """
    valid_lines = []
    with open(ann_path, encoding="utf-8") as f:
        for line in f:
            parsed = parse_yolo_line(line)
            if parsed is not None:
                class_id, cx, cy, w, h = parsed
                valid_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / ann_path.name
    with open(out_file, "w") as f:
        f.write("\n".join(valid_lines) + ("\n" if valid_lines else ""))

    return len(valid_lines)


# ---------------------------------------------------------------------------
# Shared write helper
# ---------------------------------------------------------------------------

def _write_frames(frames, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for frame_id in sorted(frames):
        label_file = out_dir / f"{frame_id:06d}.txt"
        with open(label_file, "w") as f:
            for class_id, cx, cy, w, h in frames[frame_id]:
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert Manipal-UAV annotations to YOLO format"
    )
    parser.add_argument("--labels_dir", required=True,
                        help="Root directory with Manipal annotation files")
    parser.add_argument("--output_dir", required=True,
                        help="Root directory for output YOLO label files")
    parser.add_argument("--img_width",  type=int, default=DEFAULT_WIDTH,
                        help=f"Image width in pixels (default: {DEFAULT_WIDTH})")
    parser.add_argument("--img_height", type=int, default=DEFAULT_HEIGHT,
                        help=f"Image height in pixels (default: {DEFAULT_HEIGHT})")
    parser.add_argument("--already_yolo", action="store_true",
                        help="Skip conversion: input files are already per-frame YOLO labels")
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir)
    ann_files  = sorted(labels_dir.rglob("*.txt"))

    if not ann_files:
        print(f"No .txt annotation files found in: {labels_dir}")
        return

    total = 0
    for ann_path in ann_files:
        rel      = ann_path.relative_to(labels_dir)
        out_base = Path(args.output_dir) / rel.parent

        if args.already_yolo:
            # Input is one file per frame — copy to output folder as-is
            n = copy_yolo_file(ann_path, out_base)
            print(f"  {rel}: {n} detections (YOLO copy) -> {out_base}")
        else:
            fmt = detect_format(ann_path)
            if fmt == "yolo":
                # Single-frame YOLO file without frame_id column
                n = copy_yolo_file(ann_path, out_base)
                print(f"  {rel}: {n} detections (YOLO, auto-detected) -> {out_base}")
            else:
                # MOT-style multi-frame file
                out_seq = out_base / rel.stem
                n = convert_mot_file(ann_path, out_seq, args.img_width, args.img_height)
                print(f"  {rel}: {n} frames (MOT->YOLO) -> {out_seq}")
        total += 1

    print(f"\nTotal: {total} annotation files processed.")


if __name__ == "__main__":
    main()
