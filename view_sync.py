#!/usr/bin/env python3
"""
Synchronized multi-camera Mars EDL image viewer.

Loads multiple camera directories and plays them back on a shared timeline,
aligned by `SPACECRAFT_CLOCK_START_COUNT` (SCLK start).

Usage:
    python3 view_sync.py data/m2020/rdcam data/m2020/lcam data/m2020/ddcam

Notes:
- The first directory is the master timeline.
- Alignment uses nearest-neighbor match in SCLK with optional tolerance.
- Requires OpenCV + numpy. SPICE is not required.
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pds3 import infer_pds3_layout_fast

try:
    from bsq_cython_neon import bsq_to_bgr_neon_ultimate as _bsq_to_bgr_fast
except (ImportError, OSError):
    _bsq_to_bgr_fast = None


_SCLK_START_RE = re.compile(
    rb'(?m)^\s*SPACECRAFT_CLOCK_START_COUNT\s*=\s*"?([0-9]+(?:\.[0-9]+)?)"?\s*$'
)


@dataclass(frozen=True)
class CameraStream:
    path: Path
    mission: str
    camera: str
    entries: list["FrameEntry"]


@dataclass(frozen=True)
class FrameEntry:
    sclk_start: float
    img_path: Path


def auto_detect_camera(path: Path) -> Tuple[str, str]:
    img_files = list(path.glob("*.IMG")) + list(path.glob("*.img"))
    lbl_files = list(path.glob("*.LBL")) + list(path.glob("*.lbl"))
    if not img_files and not lbl_files:
        raise ValueError(f"No .IMG or .LBL files found in {path}")

    sample_file = (img_files + lbl_files)[0].name.upper()
    if sample_file.startswith("ELM_"):
        return ("lcam", "m2020")
    if sample_file.startswith("EDF_"):
        return ("rdcam", "m2020")
    if sample_file.startswith("ESF_"):
        return ("ddcam", "m2020")
    if sample_file.startswith("0000MD"):
        return ("mardi", "msl")
    raise ValueError(f"Unknown camera type from filename: {sample_file}")


def extract_sclk_start_from_label(img_path: Path) -> float:
    """
    Fast SCLK_START extractor from the product label.

    - For detached labels: uses sibling .LBL/.lbl
    - For embedded labels: reads up to 128KB from .IMG
    """
    lbl_upper = img_path.with_suffix(".LBL")
    lbl_lower = img_path.with_suffix(".lbl")
    if lbl_upper.exists() or lbl_lower.exists():
        lbl_path = lbl_upper if lbl_upper.exists() else lbl_lower
        data = lbl_path.read_bytes()
    else:
        with open(img_path, "rb") as f:
            data = f.read(131072)
    m = _SCLK_START_RE.search(data)
    if not m:
        raise ValueError(f"{img_path}: missing SPACECRAFT_CLOCK_START_COUNT")
    return float(m.group(1))


def load_stream(path: Path) -> CameraStream:
    camera, mission = auto_detect_camera(path)

    if camera == "mardi":
        lbl_pattern_upper = str(path / "0000MD*.LBL")
        lbl_pattern_lower = str(path / "0000MD*.lbl")
        lbl_files = sorted(glob.glob(lbl_pattern_upper) + glob.glob(lbl_pattern_lower))
        img_paths: list[Path] = []
        for lbl in lbl_files:
            img_path = Path(lbl).with_suffix(".IMG")
            if not img_path.exists():
                img_path = Path(lbl).with_suffix(".img")
            img_paths.append(img_path)
    else:
        img_pattern_upper = str(path / "*.IMG")
        img_pattern_lower = str(path / "*.img")
        img_paths = [Path(p) for p in (glob.glob(img_pattern_upper) + glob.glob(img_pattern_lower))]

    entries: list[FrameEntry] = []
    for p in img_paths:
        s = extract_sclk_start_from_label(p)
        entries.append(FrameEntry(sclk_start=float(s), img_path=p))
    entries.sort(key=lambda e: (e.sclk_start, e.img_path.name))

    return CameraStream(path=path, mission=mission, camera=camera, entries=entries)


def read_image(img_path: Path, camera: str) -> Optional[np.ndarray]:
    try:
        layout = infer_pds3_layout_fast(str(img_path))
        with open(img_path, "rb") as f:
            f.seek(layout.image_offset_bytes)
            raw = f.read(layout.image_size_bytes)

        img = np.frombuffer(raw, dtype=np.uint8)
        if camera == "lcam":
            img = img.reshape((layout.lines, layout.line_samples))
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if camera in ("rdcam", "ddcam"):
            img = img.reshape((layout.bands, layout.lines, layout.line_samples))
            if _bsq_to_bgr_fast and layout.bands == 3:
                return _bsq_to_bgr_fast(img)
            return np.stack([img[2], img[1], img[0]], axis=-1)

        if camera == "mardi":
            if layout.bands == 1:
                img = img.reshape((layout.lines, layout.line_samples))
                return cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)
            img = img.reshape((layout.bands, layout.lines, layout.line_samples))
            if _bsq_to_bgr_fast and layout.bands == 3:
                return _bsq_to_bgr_fast(img)
            return np.stack([img[2], img[1], img[0]], axis=-1)

        raise ValueError(f"Unknown camera: {camera}")
    except Exception as e:
        print(f"Error reading {img_path}: {e}")
        return None


def overlay_text(img: np.ndarray, lines: list[str]) -> None:
    x, y = 10, 24
    for line in lines:
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 22


def blank_frame(size: Tuple[int, int], text: str) -> np.ndarray:
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    overlay_text(img, [text])
    return img


def choose_grid(n: int) -> Tuple[int, int]:
    if n <= 1:
        return (1, 1)
    if n == 2:
        return (1, 2)
    if n <= 4:
        return (2, 2)
    if n <= 6:
        return (2, 3)
    return (3, 3)


def main() -> int:
    ap = argparse.ArgumentParser(description="Synchronized multi-camera EDL viewer (SCLK_START)")
    ap.add_argument("paths", nargs="+", type=Path, help="One or more camera directories (first is master)")
    ap.add_argument("--fps", type=int, default=15, help="Playback FPS (default: 15)")
    ap.add_argument("--start", type=int, default=0, help="Start index on master timeline")
    ap.add_argument("--max-delta", type=float, default=0.10, help="Max |ΔSCLK| allowed for match (default: 0.10)")
    args = ap.parse_args()

    for p in args.paths:
        if not p.exists():
            print(f"Error: path not found: {p}")
            return 1

    print("Loading streams (SCLK_START from labels)...")
    streams = [load_stream(p) for p in args.paths]

    missions = {s.mission for s in streams}
    if len(missions) != 1:
        print(f"Error: mixed missions not supported: {sorted(missions)}")
        return 1

    master = streams[0]
    master_sclks = [e.sclk_start for e in master.entries]
    if not master_sclks:
        print("Error: master stream has no frames")
        return 1

    # Precompute nearest-neighbor index mapping for each stream to master timeline.
    import bisect

    maps: list[list[Optional[int]]] = []
    for s in streams:
        sclks = [e.sclk_start for e in s.entries]
        idxs: list[Optional[int]] = []
        for t in master_sclks:
            j = bisect.bisect_left(sclks, t)
            candidates = []
            if 0 <= j < len(sclks):
                candidates.append(j)
            if 0 <= j - 1 < len(sclks):
                candidates.append(j - 1)
            best: Optional[int] = None
            best_d = None
            for k in candidates:
                d = abs(sclks[k] - t)
                if best_d is None or d < best_d:
                    best_d = d
                    best = k
            if best is None or (best_d is not None and best_d > args.max_delta):
                idxs.append(None)
            else:
                idxs.append(best)
        maps.append(idxs)

    # Determine a nominal frame size for blanks from the first readable frame of each stream.
    sizes: list[Tuple[int, int]] = []
    for s in streams:
        sample_img = None
        for e in s.entries[:5]:
            sample_img = read_image(e.img_path, s.camera)
            if sample_img is not None:
                break
        if sample_img is None:
            sizes.append((480, 640))
        else:
            h, w = sample_img.shape[:2]
            sizes.append((h, w))

    rows, cols = choose_grid(len(streams))
    cv2.namedWindow("EDL Sync Viewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("EDL Sync Viewer", 1600, 1000)

    idx = min(args.start, len(master_sclks) - 1)
    paused = False
    delay = max(1, 1000 // args.fps)

    print("Controls: Space=pause, Q/Esc=quit, Left/Right=step, Home/End, +/- speed")

    while True:
        t_master = master_sclks[idx]

        tiles: list[np.ndarray] = []
        for si, s in enumerate(streams):
            mi = maps[si][idx]
            if mi is None:
                img = blank_frame(sizes[si], f"{s.camera.upper()}: no match")
                overlay_text(img, [f"SCLK(master) {t_master:.3f}"])
                tiles.append(img)
                continue

            entry = s.entries[mi]
            img = read_image(entry.img_path, s.camera)
            if img is None:
                img = blank_frame(sizes[si], f"{s.camera.upper()}: read error")
                tiles.append(img)
                continue

            d = entry.sclk_start - t_master
            overlay_text(
                img,
                [
                    f"{s.camera.upper()}  (idx {mi+1}/{len(s.entries)})",
                    f"SCLK {entry.sclk_start:.3f}  Δ {d:+.3f}",
                    f"master {idx+1}/{len(master_sclks)}  {t_master:.3f}",
                ],
            )
            tiles.append(img)

        # Assemble grid (pad with blanks).
        while len(tiles) < rows * cols:
            tiles.append(blank_frame((480, 640), ""))

        row_imgs: list[np.ndarray] = []
        for r in range(rows):
            row = tiles[r * cols : (r + 1) * cols]
            # Normalize heights per row.
            max_h = max(im.shape[0] for im in row)
            resized = []
            for im in row:
                h, w = im.shape[:2]
                if h != max_h:
                    new_w = int(w * (max_h / max(1, h)))
                    im = cv2.resize(im, (new_w, max_h), interpolation=cv2.INTER_AREA)
                resized.append(im)
            row_imgs.append(cv2.hconcat(resized))

        canvas = cv2.vconcat(row_imgs)
        cv2.imshow("EDL Sync Viewer", canvas)

        key = cv2.waitKey(delay if not paused else 0) & 0xFF
        if key == ord("q") or key == 27:
            break
        if key == ord(" "):
            paused = not paused
            continue
        if key == 81 or key == 2:  # Left
            idx = max(0, idx - 1)
            paused = True
            continue
        if key == 83 or key == 3:  # Right
            idx = min(len(master_sclks) - 1, idx + 1)
            paused = True
            continue
        if key == 80 or key == 0:  # Home
            idx = 0
            continue
        if key == 87 or key == 1:  # End
            idx = len(master_sclks) - 1
            continue
        if key in (ord("+"), ord("=")):
            delay = max(1, delay - 10)
            continue
        if key == ord("-"):
            delay = min(1000, delay + 10)
            continue

        if not paused:
            idx = (idx + 1) % len(master_sclks)

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
