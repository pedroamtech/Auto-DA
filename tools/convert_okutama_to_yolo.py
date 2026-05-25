"""
Convert Okutama-Action dataset annotations to per-frame YOLO label files.

Okutama format (one file per video sequence):
  track_id x1 y1 x2 y2 frame_id lost occluded generated "label" "action"

YOLO output (one .txt file per image frame):
  class_id cx cy w h   (all values normalized 0-1)

Usage:
  python tools/convert_okutama_to_yolo.py \
      --labels_dir path/to/Okutama/labels \
      --output_dir path/to/Okutama/yolo_labels \
      [--img_width 1920] [--img_height 1080] [--keep_lost]

Expected input structure:
  labels/
    Train/
      Morning/1.1.txt
      Noon/1.2.txt
      ...
    Test/
      ...

Output mirrors input structure:
  yolo_labels/
    Train/
      Morning/
        1.1/
          000000.txt
          000001.txt
          ...
"""

import os
import argparse
from pathlib import Path
from collections import defaultdict

DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080

CLASS_MAP = {"Person": 0}


def parse_line(line):
    """
    Parse one Okutama annotation line.
    Returns (frame_id, x1, y1, x2, y2, lost, class_id) or None.
    """
    line = line.strip()
    if not line:
        return None

    parts = line.split()
    if len(parts) < 10:
        return None

    try:
        x1 = int(parts[1])
        y1 = int(parts[2])
        x2 = int(parts[3])
        y2 = int(parts[4])
        frame_id = int(parts[5])
        lost = int(parts[6])
        label = parts[9].strip('"')
    except (ValueError, IndexError):
        return None

    return frame_id, x1, y1, x2, y2, lost, CLASS_MAP.get(label, 0)


def to_yolo(x1, y1, x2, y2, img_w, img_h):
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return cx, cy, w, h


def convert_sequence(ann_path, out_dir, img_w, img_h, skip_lost):
    frames = defaultdict(list)

    with open(ann_path, encoding="utf-8") as f:
        for line in f:
            parsed = parse_line(line)
            if parsed is None:
                continue
            frame_id, x1, y1, x2, y2, lost, class_id = parsed

            if skip_lost and lost == 1:
                continue

            # Clamp to image bounds
            x1 = max(0, min(x1, img_w - 1))
            x2 = max(0, min(x2, img_w))
            y1 = max(0, min(y1, img_h - 1))
            y2 = max(0, min(y2, img_h))

            if x2 <= x1 or y2 <= y1:
                continue

            cx, cy, w, h = to_yolo(x1, y1, x2, y2, img_w, img_h)
            frames[frame_id].append((class_id, cx, cy, w, h))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for frame_id in sorted(frames):
        label_file = out_dir / f"{frame_id:06d}.txt"
        with open(label_file, "w") as f:
            for class_id, cx, cy, w, h in frames[frame_id]:
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    return len(frames)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Okutama-Action annotations to YOLO format"
    )
    parser.add_argument("--labels_dir", required=True,
                        help="Root directory with Okutama .txt annotation files")
    parser.add_argument("--output_dir", required=True,
                        help="Root directory for output YOLO label files")
    parser.add_argument("--img_width",  type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--img_height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--keep_lost", action="store_true",
                        help="Include bboxes marked as lost (lost=1)")
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir)
    ann_files = sorted(labels_dir.rglob("*.txt"))

    if not ann_files:
        print(f"No .txt annotation files found in: {labels_dir}")
        return

    total_frames = 0
    for ann_path in ann_files:
        rel = ann_path.relative_to(labels_dir)
        out_seq = Path(args.output_dir) / rel.parent / rel.stem
        n = convert_sequence(ann_path, out_seq, args.img_width, args.img_height,
                             skip_lost=not args.keep_lost)
        total_frames += n
        print(f"  {rel}: {n} frames -> {out_seq}")

    print(f"\nTotal: {len(ann_files)} sequences | {total_frames} label files written.")


if __name__ == "__main__":
    main()
