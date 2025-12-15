#!/usr/bin/env python3
"""
Viewer for Mars 2020 EDL_RDCAM (Rover Down-Look Camera) images.

Usage:
    python3 view_rdc.py                           # View all RDC images
    python3 view_rdc.py --path data/m2020/rdcam/images  # Custom path
    python3 view_rdc.py --fps 30                  # Faster playback

Controls:
    Space   - Pause/resume
    Q/Esc   - Quit
    Left    - Previous frame
    Right   - Next frame
    Home    - First frame
    End     - Last frame
    +/=     - Speed up
    -       - Slow down
"""

import argparse
import glob
import re
import sys
from pathlib import Path

import cv2
import numpy as np

# Add repo root to path for pds_img import
sys.path.insert(0, str(Path(__file__).parent))
from pds_img import infer_pds3_layout, read_image_header_text


def extract_sclk(filename: str) -> float:
    """Extract spacecraft clock from filename for sorting."""
    # EDF_0000_0666952787_126FDR_... -> 666952787.126
    m = re.search(r'_(\d{10})_(\d{3})FDR_', filename)
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")
    return 0.0


def read_rdc_image(path: str) -> tuple[np.ndarray | None, dict]:
    """Read an RDC .IMG file and return (image_bgr, header_dict)."""
    try:
        layout, label = infer_pds3_layout(path)

        with open(path, 'rb') as f:
            f.seek(layout.image_offset_bytes)
            raw = f.read(layout.image_size_bytes)

        # RDC is 1024x1280x3 bands, interleaved by band (BSQ)
        img = np.frombuffer(raw, dtype=np.uint8)
        img = img.reshape((layout.bands, layout.lines, layout.line_samples))
        # Transpose from (bands, lines, samples) to (lines, samples, bands)
        img = np.transpose(img, (1, 2, 0))
        # RGB to BGR for OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Get header metadata
        try:
            _, hdr = read_image_header_text(path)
        except Exception:
            hdr = {}

        return img, hdr
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None, {}


def main():
    parser = argparse.ArgumentParser(description="View Mars 2020 EDL_RDCAM descent images")
    parser.add_argument("--path", default="data/m2020/rdcam",
                        help="Path to RDC image directory")
    parser.add_argument("--fps", type=int, default=15, help="Playback FPS (default: 15)")
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    args = parser.parse_args()

    # Find all .IMG files
    pattern = str(Path(args.path) / "*.IMG")
    filenames = sorted(glob.glob(pattern), key=lambda f: extract_sclk(Path(f).name))

    if not filenames:
        print(f"No .IMG files found in {args.path}")
        print("Download with: python3 download.py rdcam")
        return 1

    print(f"Found {len(filenames)} RDC images")
    print("Controls: Space=pause, Q=quit, Left/Right=step, +/-=speed, Home/End=jump")

    cv2.namedWindow("RDC View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RDC View", 1280, 1024)

    idx = min(args.start, len(filenames) - 1)
    paused = False
    delay = max(1, 1000 // args.fps)

    while True:
        fname = filenames[idx]
        img, hdr = read_rdc_image(fname)

        if img is not None:
            # Add overlay info
            display = img.copy()
            sclk = extract_sclk(Path(fname).name)
            frame_id = hdr.get("FRAME_ID", "?")
            info = f"Frame {idx+1}/{len(filenames)} | SCLK: {sclk:.3f} | {Path(fname).name}"

            cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
            if paused:
                cv2.putText(display, "PAUSED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)

            cv2.imshow("RDC View", display)

        key = cv2.waitKey(delay if not paused else 0) & 0xFF

        if key == ord('q') or key == 27:  # Q or Esc
            break
        elif key == ord(' '):  # Space
            paused = not paused
        elif key == 81 or key == 2:  # Left arrow
            idx = max(0, idx - 1)
            paused = True
        elif key == 83 or key == 3:  # Right arrow
            idx = min(len(filenames) - 1, idx + 1)
            paused = True
        elif key == 80 or key == 0:  # Home (up arrow as fallback)
            idx = 0
        elif key == 87 or key == 1:  # End (down arrow as fallback)
            idx = len(filenames) - 1
        elif key in (ord('+'), ord('=')):
            delay = max(1, delay - 10)
            print(f"Speed: {1000/delay:.1f} fps")
        elif key == ord('-'):
            delay = min(1000, delay + 10)
            print(f"Speed: {1000/delay:.1f} fps")
        elif not paused:
            idx = (idx + 1) % len(filenames)

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
