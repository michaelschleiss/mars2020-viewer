#!/usr/bin/env python3
"""
Unified Mars EDL Image Viewer

Supports:
- Mars 2020 LCAM (Lander Vision System)
- Mars 2020 RDCAM (Rover Downlook Camera)
- Mars 2020 DDCAM (Descent Stage Downlook Camera)
- MSL MARDI (Mars Descent Imager)

Auto-detects camera type from filename patterns.
Displays pose overlay with trajectory mini-map using SPICE kernels.

Usage:
    python3 view.py                      # Default: data/m2020/lcam
    python3 view.py data/m2020/lcam      # Auto-detects LCAM
    python3 view.py data/m2020/rdcam     # Auto-detects RDCAM
    python3 view.py data/m2020/ddcam     # Auto-detects DDCAM
    python3 view.py data/msl/rdr         # Auto-detects MARDI
    python3 view.py --help

Controls:
    Space     - Pause/resume
    Q/Esc     - Quit
    Left      - Previous frame
    Right     - Next frame
    Home      - First frame
    End       - Last frame
    +/=       - Speed up
    -         - Slow down
    M         - Toggle mini-map
    O         - Toggle overlay
    T         - Toggle timing stats
"""

import argparse
import glob
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


class FrameProfiler:
    """Simple profiler for frame timing stats."""

    def __init__(self, window: int = 30):
        self.window = window
        self.timings: dict[str, deque] = {}
        self._start: float = 0

    def start(self) -> None:
        self._start = time.perf_counter()

    def mark(self, name: str) -> None:
        elapsed = (time.perf_counter() - self._start) * 1000
        if name not in self.timings:
            self.timings[name] = deque(maxlen=self.window)
        self.timings[name].append(elapsed)
        self._start = time.perf_counter()

    def clear(self, *names: str) -> None:
        for name in names:
            self.timings.pop(name, None)

    def stats(self) -> dict[str, float]:
        return {name: sum(times) / len(times) for name, times in self.timings.items() if times}

    def total(self) -> float:
        return sum(self.stats().values())


# Import PDS3 parser
sys.path.insert(0, str(Path(__file__).parent))
from pds3 import (
    infer_pds3_layout_fast,
    read_image_header_text,
    FrameMetadata,
    load_frame_metadata,
)

# Optional: fast BSQ (RGB) -> BGR converter (Apple Silicon NEON Cython extension).
try:
    from bsq_cython_neon import bsq_to_bgr_neon_ultimate as _bsq_to_bgr_fast
except (ImportError, OSError):
    _bsq_to_bgr_fast = None

# Try to import SPICE
try:
    import spiceypy as spice
    SPICE_AVAILABLE = True
except ImportError:
    SPICE_AVAILABLE = False
    print("Warning: spiceypy not installed. SPICE pose queries disabled.")


@dataclass
class PoseData:
    """Pose data for a frame."""
    position_xyz: Optional[np.ndarray]  # Mars-centered XYZ (meters)
    lat_deg: Optional[float]
    lon_deg: Optional[float]
    altitude_m: Optional[float]
    source: str  # "HEADER", "SPICE", or "UNAVAILABLE"


@dataclass(frozen=True)
class MinimapCache:
    """Pre-rendered mini-map background and per-frame pixel coordinates."""

    base: np.ndarray
    px_by_index: list[Optional[tuple[int, int]]]
    map_size: int = 300


def build_minimap_cache(
    trajectory: list[tuple[Optional[float], Optional[float]]],
    *,
    map_size: int = 300,
) -> Optional[MinimapCache]:
    """Precompute mini-map background for a trajectory to keep per-frame work constant."""
    valid_pts = [(lat, lon) for lat, lon in trajectory if lat is not None and lon is not None]
    if not valid_pts:
        return None

    lats = [p[0] for p in valid_pts]
    lons = [p[1] for p in valid_pts]
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)

    lat_margin = (lat_max - lat_min) * 0.1
    lon_margin = (lon_max - lon_min) * 0.1
    lat_min -= lat_margin
    lat_max += lat_margin
    lon_min -= lon_margin
    lon_max += lon_margin

    def to_px(lat: float, lon: float) -> tuple[int, int]:
        x = int((lon - lon_min) / (lon_max - lon_min + 1e-9) * (map_size - 20) + 10)
        y = int((1 - (lat - lat_min) / (lat_max - lat_min + 1e-9)) * (map_size - 20) + 10)
        return (x, y)

    px_by_index: list[Optional[tuple[int, int]]] = []
    for lat, lon in trajectory:
        if lat is None or lon is None:
            px_by_index.append(None)
        else:
            px_by_index.append(to_px(lat, lon))

    minimap = np.zeros((map_size, map_size, 3), dtype=np.uint8)
    minimap[:] = (40, 40, 40)

    valid_px = [p for p in px_by_index if p is not None]
    for i in range(len(valid_px) - 1):
        cv2.line(minimap, valid_px[i], valid_px[i + 1], (0, 255, 0), 2)

    landing_pt = next((p for p in reversed(valid_px) if p is not None), None)
    if landing_pt is not None:
        cv2.drawMarker(minimap, landing_pt, (255, 255, 255), cv2.MARKER_CROSS, 15, 2)

    cv2.rectangle(minimap, (0, 0), (map_size - 1, map_size - 1), (200, 200, 200), 2)

    return MinimapCache(base=minimap, px_by_index=px_by_index, map_size=map_size)


class SPICEManager:
    """Manage SPICE kernels and pose queries."""

    def __init__(self, mission: str, kernel_dir: Path = Path("spice_kernels")):
        self.mission = mission
        self.kernel_dir = kernel_dir
        self.loaded = False

        if SPICE_AVAILABLE:
            self._load_kernels()

    def _load_kernels(self):
        """Load SPICE kernels for mission."""
        try:
            # Try meta-kernel first
            meta_kernel_map = {
                'm2020': self.kernel_dir / 'm2020.tm',
                'msl': self.kernel_dir / 'msl_edl.tm',
            }

            mk_path = meta_kernel_map.get(self.mission)
            if mk_path and mk_path.exists():
                spice.furnsh(str(mk_path))
                self.loaded = True
                print(f"Loaded {self.mission} SPICE kernels from {mk_path.name}")
            else:
                print(f"Warning: SPICE meta-kernel not found: {mk_path}")
        except Exception as e:
            print(f"Warning: Failed to load SPICE kernels: {e}")

    def query_pose(self, sclk_ticks: int, camera: str, *, sclk_partition: Optional[int] = None) -> Optional[PoseData]:
        """Query pose from SPICE kernels using SCLK ticks."""
        if not self.loaded:
            return None

        try:
            # Convert SCLK to ephemeris time
            sc_id = -168 if self.mission == 'm2020' else -76
            if self.mission in ("m2020", "msl"):
                # M2020/MSL SCLK format is "SSSSSSSSSS-FFFFF", where FFFFF are ticks of 1/65536 sec.
                ticks_per_sec = 65536
                coarse = int(sclk_ticks // ticks_per_sec)
                fine = int(sclk_ticks % ticks_per_sec)

                partition = 1
                if sclk_partition is not None:
                    try:
                        parsed_partition = int(sclk_partition)
                        if parsed_partition > 0:
                            partition = parsed_partition
                    except Exception:
                        pass
                sclk_prefix = f"{partition}/"
                candidates = (
                    f"{sclk_prefix}{coarse}-{fine:05d}",
                    f"{sclk_prefix}{coarse}.{fine:05d}",
                )

                last_err: Optional[Exception] = None
                for sclk_str in candidates:
                    try:
                        et = spice.scs2e(sc_id, sclk_str)
                        return self.query_pose_et(et, camera)
                    except Exception as e:
                        last_err = e
                raise last_err if last_err is not None else RuntimeError("SPICE SCLK conversion failed")

            raise RuntimeError(f"Unsupported mission for SCLK query: {self.mission}")
        except Exception as e:
            print(f"SPICE query failed: {e}")
            return None

    def query_pose_et(self, et: float, camera: str) -> Optional[PoseData]:
        """Query pose from SPICE kernels using ephemeris time (ET)."""
        if not self.loaded:
            return None

        try:
            # Query spacecraft position
            target = 'M2020' if self.mission == 'm2020' else 'MSL'
            state, _ = spice.spkezr(target, et, 'IAU_MARS', 'NONE', 'MARS')
            # SPICE returns km; convert to meters to match label convention
            position = np.array(state[:3], dtype=np.float64) * 1000.0

            # Convert to lat/lon/alt
            radius, lon_rad, lat_rad = spice.reclat(position)
            lat_deg = np.degrees(lat_rad)
            lon_deg = np.degrees(lon_rad)

            # Altitude above reference
            ref_radius = 3394507.0 if self.mission == 'm2020' else 3390000.0
            altitude_m = radius - ref_radius

            return PoseData(
                position_xyz=position,
                lat_deg=lat_deg,
                lon_deg=lon_deg,
                altitude_m=altitude_m,
                source="SPICE"
            )
        except Exception as e:
            print(f"SPICE query failed: {e}")
            return None


def auto_detect_camera(path: Path) -> Tuple[str, str]:
    """
    Auto-detect camera and mission from path.

    Returns:
        (camera, mission) tuple, e.g., ("lcam", "m2020")
    """
    # Check first .IMG file
    img_files = list(path.glob("*.IMG")) + list(path.glob("*.img"))
    lbl_files = list(path.glob("*.LBL")) + list(path.glob("*.lbl"))

    if not img_files and not lbl_files:
        raise ValueError(f"No .IMG or .LBL files found in {path}")

    sample_file = (img_files + lbl_files)[0].name.upper()

    # Mars 2020 LCAM
    if sample_file.startswith('ELM_'):
        return ("lcam", "m2020")

    # Mars 2020 RDCAM
    if sample_file.startswith('EDF_'):
        return ("rdcam", "m2020")

    # Mars 2020 DDCAM
    if sample_file.startswith('ESF_'):
        return ("ddcam", "m2020")

    # MSL MARDI
    if sample_file.startswith('0000MD'):
        return ("mardi", "msl")

    raise ValueError(f"Unknown camera type from filename: {sample_file}")


def get_pose(
    meta: FrameMetadata,
    spice_mgr: Optional[SPICEManager],
) -> PoseData:
    """
    Get pose data for a frame.

    Priority: embedded header → SPICE → unavailable
    """
    # Try embedded header first (LCAM has it)
    if meta.camera == 'lcam':
        try:
            _, header = read_image_header_text(str(meta.filepath))
            pos_str = header.get('ORIGIN_OFFSET_VECTOR', '')
            if pos_str:
                # Parse "(x, y, z)" format
                pos_xyz = np.array([float(x.strip()) for x in pos_str.strip('()').split(',')])

                # Convert to lat/lon/alt
                radius = np.linalg.norm(pos_xyz)
                lat_deg = np.degrees(np.arcsin(pos_xyz[2] / radius))
                lon_deg = np.degrees(np.arctan2(pos_xyz[1], pos_xyz[0]))
                ref_radius = 3394507.0
                altitude_m = radius - ref_radius

                return PoseData(
                    position_xyz=pos_xyz,
                    lat_deg=lat_deg,
                    lon_deg=lon_deg,
                    altitude_m=altitude_m,
                    source="HEADER"
                )
        except Exception:
            pass

    # Fallback to SPICE
    if spice_mgr:
        # Convention: use START (time or SCLK).
        time_str = meta.start_time or meta.stop_time
        if time_str:
            try:
                et = spice.str2et(time_str)
                pose = spice_mgr.query_pose_et(et, meta.camera)
                if pose:
                    return pose
            except Exception:
                pass

        sclk_ticks = meta.sclk_start_ticks
        if sclk_ticks is not None and sclk_ticks > 0:
            pose = spice_mgr.query_pose(int(sclk_ticks), meta.camera, sclk_partition=meta.sclk_partition)
            if pose:
                return pose

    # No pose available
    return PoseData(None, None, None, None, "UNAVAILABLE")


def read_image(img_path: Path, camera: str, profiler: Optional[FrameProfiler] = None) -> Optional[np.ndarray]:
    """Read and decode image based on camera type."""
    try:
        layout = infer_pds3_layout_fast(str(img_path))
        if profiler:
            profiler.mark("read.label")

        with open(img_path, 'rb') as f:
            f.seek(layout.image_offset_bytes)
            raw = f.read(layout.image_size_bytes)
        if profiler:
            profiler.mark("read.io")

        img = np.frombuffer(raw, dtype=np.uint8)
        if profiler:
            profiler.mark("read.frombuf")

        if camera == 'lcam':
            # 1-band grayscale
            img = img.reshape((layout.lines, layout.line_samples))
            # Convert to BGR for overlay
            result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if profiler:
                profiler.mark("read.convert")
            return result

        elif camera in ('rdcam', 'ddcam'):
            # 3-band BSQ (RGB) → HWC BGR: stack in reverse order to skip cvtColor
            img = img.reshape((layout.bands, layout.lines, layout.line_samples))
            if _bsq_to_bgr_fast and layout.bands == 3:
                result = _bsq_to_bgr_fast(img)
            else:
                result = np.stack([img[2], img[1], img[0]], axis=-1)
            if profiler:
                profiler.mark("read.convert")
            return result

        elif camera == 'mardi':
            # 1-band Bayer mosaic or 3-band
            if layout.bands == 1:
                img = img.reshape((layout.lines, layout.line_samples))
                # Demosaic assuming RGGB pattern (MSL MARDI default)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)
                if profiler:
                    profiler.mark("read.convert")
                return img_bgr
            else:
                # 3-band BSQ (RGB) → HWC BGR: stack in reverse order to skip cvtColor
                img = img.reshape((layout.bands, layout.lines, layout.line_samples))
                if _bsq_to_bgr_fast and layout.bands == 3:
                    return _bsq_to_bgr_fast(img)
                return np.stack([img[2], img[1], img[0]], axis=-1)

    except Exception as e:
        print(f"Error reading {img_path}: {e}")
        return None


def render_overlay(img: np.ndarray, pose: PoseData, idx: int, total: int,
                   meta: FrameMetadata, sclk_display: str, paused: bool,
                   spice_mgr: Optional[SPICEManager] = None) -> None:
    """Render metadata overlay on image (in-place)."""
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Color based on state
    if paused:
        color = (0, 165, 255)  # Orange
    elif pose.source == "UNAVAILABLE":
        color = (0, 0, 255)    # Red
    elif pose.source == "SPICE":
        color = (0, 255, 255)  # Yellow
    else:
        color = (0, 255, 0)    # Green

    # Semi-transparent background
    h, w = img.shape[:2]
    overlay_h = 260 if meta.camera == "lcam" else 150
    overlay_bg = np.zeros((overlay_h, 600, 3), dtype=np.uint8)
    overlay_bg[:] = (20, 20, 20)
    roi = img[10:10 + overlay_h, 10:610]
    cv2.addWeighted(overlay_bg, 0.75, roi, 0.25, 0, roi)

    # Line 1: Frame info
    extra = ""
    if meta.frame_index is not None:
        extra = f" | N{meta.frame_index}"
    text = f"Frame {idx+1}/{total} | SCLK: {sclk_display} | Camera: {meta.camera.upper()}{extra}"
    cv2.putText(img, text, (20, 35), font, 0.6, color, 2)

    # Line 2: Filename
    short_name = meta.filename if len(meta.filename) < 60 else meta.filename[:57] + "..."
    cv2.putText(img, f"File: {short_name}", (20, 65), font, 0.5, (255, 255, 255), 1)

    # Pose (if available)
    y = 95
    if meta.camera == "lcam" and spice_mgr and spice_mgr.loaded:
        # For LCAM, compare embedded header pose vs SPICE at multiple timing choices.
        spice_entries: list[tuple[str, Optional[PoseData]]] = []
        try:
            et_start = spice.str2et(meta.start_time) if meta.start_time else None
            et_stop = spice.str2et(meta.stop_time) if meta.stop_time else None
        except Exception:
            et_start = None
            et_stop = None

        if et_start is not None:
            spice_entries.append(("start", spice_mgr.query_pose_et(et_start, meta.camera)))
        if et_start is not None and et_stop is not None:
            spice_entries.append(("mid", spice_mgr.query_pose_et((et_start + et_stop) / 2.0, meta.camera)))
        if et_stop is not None:
            spice_entries.append(("stop", spice_mgr.query_pose_et(et_stop, meta.camera)))

        if pose.lat_deg is not None:
            cv2.putText(img, f"Header: {pose.lat_deg:.4f}°N, {pose.lon_deg:.4f}°E  alt {pose.altitude_m:.0f} m", (20, y), font, 0.5, (0, 255, 0), 1)
            y += 25
        else:
            cv2.putText(img, "Header: UNAVAILABLE", (20, y), font, 0.5, (0, 0, 255), 1)
            y += 25

        if not spice_entries:
            cv2.putText(img, "SPICE:  UNAVAILABLE", (20, y), font, 0.5, (0, 0, 255), 1)
            y += 25
        else:
            for label, spice_pose in spice_entries:
                if spice_pose and spice_pose.lat_deg is not None:
                    cv2.putText(
                        img,
                        f"SPICE({label}): {spice_pose.lat_deg:.4f}°N, {spice_pose.lon_deg:.4f}°E  alt {spice_pose.altitude_m:.0f} m",
                        (20, y),
                        font,
                        0.5,
                        (0, 255, 255),
                        1,
                    )
                    y += 25
                    if pose.position_xyz is not None and spice_pose.position_xyz is not None:
                        d_m = float(np.linalg.norm(pose.position_xyz - spice_pose.position_xyz))
                        d_alt = None
                        if pose.altitude_m is not None and spice_pose.altitude_m is not None:
                            d_alt = float(spice_pose.altitude_m - pose.altitude_m)
                        if d_alt is None:
                            cv2.putText(img, f"  |Δpos|: {d_m:.1f} m", (20, y), font, 0.5, (255, 255, 255), 1)
                        else:
                            cv2.putText(img, f"  |Δpos|: {d_m:.1f} m  Δalt: {d_alt:+.1f} m", (20, y), font, 0.5, (255, 255, 255), 1)
                        y += 25
                else:
                    cv2.putText(img, f"SPICE({label}): UNAVAILABLE", (20, y), font, 0.5, (0, 0, 255), 1)
                    y += 25
    else:
        if pose.lat_deg is not None:
            text = f"Position: {pose.lat_deg:.4f}°N, {pose.lon_deg:.4f}°E"
            cv2.putText(img, text, (20, y), font, 0.5, color, 1)
            y += 30
            text = f"Altitude: {pose.altitude_m:.0f}m  |  Source: {pose.source}"
            cv2.putText(img, text, (20, y), font, 0.5, color, 1)
        else:
            cv2.putText(img, "Pose: UNAVAILABLE", (20, y), font, 0.5, color, 1)

    # Paused indicator
    if paused:
        cv2.putText(img, "PAUSED", (w - 150, 40), font, 1.0, (0, 165, 255), 3)


def render_minimap(img: np.ndarray, cache: Optional[MinimapCache], current_idx: int, show: bool) -> None:
    """Render trajectory mini-map (in-place)."""
    if not show or cache is None:
        return

    # Mini-map setup
    map_size = cache.map_size
    h, w = img.shape[:2]
    map_x = 10
    map_y = h - map_size - 10

    if map_y < 0 or map_x + map_size > w or map_y + map_size > h:
        return

    # Draw current position
    minimap = cache.base.copy()
    if 0 <= current_idx < len(cache.px_by_index):
        curr_pt = cache.px_by_index[current_idx]
        if curr_pt is not None:
            cv2.circle(minimap, curr_pt, 8, (0, 0, 255), -1)
            cv2.circle(minimap, curr_pt, 4, (255, 255, 255), -1)

    # Composite onto image
    roi = img[map_y:map_y+map_size, map_x:map_x+map_size]
    cv2.addWeighted(minimap, 0.85, roi, 0.15, 0, roi)


def main():
    parser = argparse.ArgumentParser(
        description="Unified Mars EDL descent image viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("data/m2020/lcam"),
        help="Path to dataset directory (default: data/m2020/lcam)",
    )
    parser.add_argument("--fps", type=int, default=15, help="Playback FPS (default: 15)")
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--no-spice", action="store_true", help="Disable SPICE pose queries")
    parser.add_argument("--no-minimap", action="store_true", help="Disable trajectory mini-map")
    args = parser.parse_args()

    if not args.path.exists():
        print(f"Error: Path not found: {args.path}")
        return 1

    # Auto-detect camera
    try:
        camera, mission = auto_detect_camera(args.path)
        print(f"Detected: {mission.upper()} {camera.upper()}")
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Find image files
    metas: list[FrameMetadata]
    if camera == 'mardi':
        lbl_pattern_upper = str(args.path / "0000MD*.LBL")
        lbl_pattern_lower = str(args.path / "0000MD*.lbl")
        lbl_files = sorted(glob.glob(lbl_pattern_upper) + glob.glob(lbl_pattern_lower))
        img_paths: list[Path] = []
        for lbl in lbl_files:
            img_path = Path(lbl).with_suffix(".IMG")
            if not img_path.exists():
                img_path = Path(lbl).with_suffix(".img")
            img_paths.append(img_path)
    else:
        img_pattern_upper = str(args.path / "*.IMG")
        img_pattern_lower = str(args.path / "*.img")
        img_paths = [Path(p) for p in (glob.glob(img_pattern_upper) + glob.glob(img_pattern_lower))]

    if not img_paths:
        print(f"No images found in {args.path}")
        return 1

    print(f"Found {len(img_paths)} frames")

    # Load metadata and sort deterministically.
    metas = [load_frame_metadata(p, mission=mission, camera=camera) for p in img_paths]
    if camera == "lcam" and mission == "m2020":
        metas.sort(key=lambda m: (0, m.frame_index) if m.frame_index is not None else (1, m.filename))
    else:
        metas.sort(key=lambda m: (m.sclk_start_ticks if m.sclk_start_ticks is not None else 0, m.filename))

    # Load SPICE kernels
    spice_mgr = None
    if not args.no_spice and SPICE_AVAILABLE:
        spice_mgr = SPICEManager(mission)

    # Pre-compute trajectory for mini-map
    print("Loading trajectory...")
    poses = []
    trajectory = []
    for meta in metas:
        pose = get_pose(meta, spice_mgr)
        poses.append(pose)
        trajectory.append((pose.lat_deg, pose.lon_deg))
    minimap_cache = build_minimap_cache(trajectory)

    # Setup viewer
    print("Controls: Space=pause, Q=quit, Left/Right=step, +/-=speed, M=minimap, O=overlay, T=timing")
    cv2.namedWindow("EDL Viewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("EDL Viewer", 1280, 960)

    idx = min(args.start, len(metas) - 1)
    paused = False
    delay = max(1, 1000 // args.fps)
    show_minimap = not args.no_minimap
    show_overlay = True
    show_timing = False
    profiler = FrameProfiler(window=30)

    # Main loop
    while True:
        if show_timing:
            profiler.start()
        meta = metas[idx]
        img = read_image(meta.filepath, camera, profiler if show_timing else None)

        if img is not None:
            display = img

            # Get pose for current frame
            pose = poses[idx]
            if meta.sclk_start_raw:
                sclk_display = meta.sclk_start_raw
            elif meta.sclk_start_ticks is not None:
                coarse = meta.sclk_start_ticks // 65536
                fine = meta.sclk_start_ticks % 65536
                sclk_display = f"{coarse}-{fine:05d}"
            else:
                sclk_display = "N/A"
            if show_timing:
                profiler.mark("pose")

            # Render overlays
            if show_overlay:
                render_overlay(
                    display,
                    pose,
                    idx,
                    len(metas),
                    meta,
                    sclk_display,
                    paused,
                    spice_mgr,
                )
                if show_timing:
                    profiler.mark("overlay")

            if show_minimap:
                render_minimap(display, minimap_cache, idx, show_minimap)
                if show_timing:
                    profiler.mark("minimap")

            # Timing overlay
            if show_timing:
                stats = profiler.stats()
                h = img.shape[0]
                y = h - 20
                for name, ms in reversed(stats.items()):
                    cv2.putText(display, f"{name}: {ms:.1f}ms", (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    y -= 20
                cv2.putText(display, f"total: {profiler.total():.1f}ms ({1000/max(0.1, profiler.total()):.0f}fps)",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow("EDL Viewer", display)
            if show_timing:
                profiler.mark("display")

        # Handle keyboard input
        key = cv2.waitKey(delay if not paused else 0) & 0xFF

        if key == ord('q') or key == 27:  # Q or Esc
            break
        elif key == ord(' '):  # Space
            paused = not paused
        elif key == 81 or key == 2:  # Left arrow
            idx = max(0, idx - 1)
            paused = True
        elif key == 83 or key == 3:  # Right arrow
            idx = min(len(metas) - 1, idx + 1)
            paused = True
        elif key == 80 or key == 0:  # Home
            idx = 0
        elif key == 87 or key == 1:  # End
            idx = len(metas) - 1
        elif key in (ord('+'), ord('=')):
            delay = max(1, delay - 10)
            print(f"Speed: {1000/delay:.1f} fps")
        elif key == ord('-'):
            delay = min(1000, delay + 10)
            print(f"Speed: {1000/delay:.1f} fps")
        elif key == ord('m'):
            show_minimap = not show_minimap
            if not show_minimap:
                profiler.clear("minimap")
            print(f"Mini-map: {'ON' if show_minimap else 'OFF'}")
        elif key == ord('o'):
            show_overlay = not show_overlay
            if not show_overlay:
                profiler.clear("overlay")
            print(f"Overlay: {'ON' if show_overlay else 'OFF'}")
        elif key == ord('t'):
            show_timing = not show_timing
            if show_timing:
                profiler.timings.clear()
            print(f"Timing: {'ON' if show_timing else 'OFF'}")
        elif not paused:
            idx = (idx + 1) % len(metas)

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
