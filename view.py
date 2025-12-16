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
"""

import argparse
import glob
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# Import PDS3 parser
sys.path.insert(0, str(Path(__file__).parent))
from pds_img import infer_pds3_layout, read_image_header_text

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

    def query_pose(self, sclk: float, camera: str) -> Optional[PoseData]:
        """Query pose from SPICE kernels using SCLK."""
        if not self.loaded:
            return None

        try:
            # Convert SCLK to ephemeris time
            sc_id = -168 if self.mission == 'm2020' else -76
            sclk_str = f"{int(sclk)}.{int((sclk % 1) * 1000):03d}"
            et = spice.scs2e(sc_id, sclk_str)
            return self.query_pose_et(et, camera)
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


def extract_sclk(filepath: str, mission: str) -> float:
    """Extract SCLK from filename or label."""
    filename = Path(filepath).name

    if mission == 'm2020':
        # ELM_0000_0666952774_000FDR_... or EDF_0000_0666952787_126FDR_...
        m = re.search(r'_(\d{10})_(\d{3})FDR_', filename)
        if m:
            return float(f"{m.group(1)}.{m.group(2)}")

    elif mission == 'msl':
        # Read from .LBL file for MARDI
        lbl_path = Path(filepath).with_suffix('.LBL')
        if lbl_path.exists():
            try:
                with open(lbl_path, 'r') as f:
                    for line in f:
                        if 'SPACECRAFT_CLOCK_START_COUNT' in line:
                            # Extract value: SPACECRAFT_CLOCK_START_COUNT = "397501987.0000"
                            match = re.search(r'"([0-9.]+)"', line)
                            if match:
                                return float(match.group(1))
            except Exception:
                pass

    return 0.0


def get_pose(img_path: Path, camera: str, mission: str, spice_mgr: Optional[SPICEManager]) -> PoseData:
    """
    Get pose data for a frame.

    Priority: embedded header → SPICE → unavailable
    """
    # Try embedded header first (LCAM has it)
    if camera == 'lcam':
        try:
            _, header = read_image_header_text(str(img_path))
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
        if camera == "lcam" and mission == "m2020":
            try:
                _, header = read_image_header_text(str(img_path))
                start_time = header.get("START_TIME")
                stop_time = header.get("STOP_TIME")
                if start_time and stop_time:
                    et0 = spice.str2et(start_time)
                    et1 = spice.str2et(stop_time)
                    pose = spice_mgr.query_pose_et((et0 + et1) / 2.0, camera)
                    if pose:
                        return pose
            except Exception:
                pass
            # If header timing can't be parsed, fall through to SCLK-based SPICE.

        sclk = extract_sclk(str(img_path), mission)
        if sclk > 0:
            pose = spice_mgr.query_pose(sclk, camera)
            if pose:
                return pose

    # No pose available
    return PoseData(None, None, None, None, "UNAVAILABLE")


def read_image(img_path: Path, camera: str) -> Optional[np.ndarray]:
    """Read and decode image based on camera type."""
    try:
        layout, _ = infer_pds3_layout(str(img_path))

        with open(img_path, 'rb') as f:
            f.seek(layout.image_offset_bytes)
            raw = f.read(layout.image_size_bytes)

        img = np.frombuffer(raw, dtype=np.uint8)

        if camera == 'lcam':
            # 1-band grayscale
            img = img.reshape((layout.lines, layout.line_samples))
            # Convert to BGR for overlay
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        elif camera in ('rdcam', 'ddcam'):
            # 3-band BSQ
            img = img.reshape((layout.bands, layout.lines, layout.line_samples))
            img = np.transpose(img, (1, 2, 0))  # BSQ → HWC
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        elif camera == 'mardi':
            # 1-band Bayer mosaic or 3-band
            if layout.bands == 1:
                img = img.reshape((layout.lines, layout.line_samples))
                # Demosaic assuming RGGB pattern (MSL MARDI default)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)
                return img_bgr
            else:
                img = img.reshape((layout.bands, layout.lines, layout.line_samples))
                img = np.transpose(img, (1, 2, 0))
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    except Exception as e:
        print(f"Error reading {img_path}: {e}")
        return None


def render_overlay(img: np.ndarray, pose: PoseData, idx: int, total: int,
                   img_path: Path, filename: str, sclk: float, camera: str, paused: bool,
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
    overlay_h = 200 if camera == "lcam" else 150
    overlay_bg = np.zeros((overlay_h, 600, 3), dtype=np.uint8)
    overlay_bg[:] = (20, 20, 20)
    roi = img[10:10 + overlay_h, 10:610]
    cv2.addWeighted(overlay_bg, 0.75, roi, 0.25, 0, roi)

    # Line 1: Frame info
    text = f"Frame {idx+1}/{total} | SCLK: {sclk:.3f} | Camera: {camera.upper()}"
    cv2.putText(img, text, (20, 35), font, 0.6, color, 2)

    # Line 2: Filename
    short_name = filename if len(filename) < 60 else filename[:57] + "..."
    cv2.putText(img, f"File: {short_name}", (20, 65), font, 0.5, (255, 255, 255), 1)

    # Pose (if available)
    y = 95
    if camera == "lcam" and spice_mgr and spice_mgr.loaded:
        # For LCAM, compare embedded header pose vs SPICE using:
        # 1) mid-point of START_TIME/STOP_TIME only (no fallback).
        spice_pose = None
        try:
            _, header = read_image_header_text(str(img_path))
            start_time = header.get("START_TIME")
            stop_time = header.get("STOP_TIME")
            if start_time and stop_time:
                et0 = spice.str2et(start_time)
                et1 = spice.str2et(stop_time)
                spice_pose = spice_mgr.query_pose_et((et0 + et1) / 2.0, camera)
        except Exception:
            spice_pose = None

        if pose.lat_deg is not None:
            cv2.putText(img, f"Header: {pose.lat_deg:.4f}°N, {pose.lon_deg:.4f}°E  alt {pose.altitude_m:.0f} m", (20, y), font, 0.5, (0, 255, 0), 1)
            y += 25
        else:
            cv2.putText(img, "Header: UNAVAILABLE", (20, y), font, 0.5, (0, 0, 255), 1)
            y += 25

        if spice_pose and spice_pose.lat_deg is not None:
            cv2.putText(img, f"SPICE:  {spice_pose.lat_deg:.4f}°N, {spice_pose.lon_deg:.4f}°E  alt {spice_pose.altitude_m:.0f} m", (20, y), font, 0.5, (0, 255, 255), 1)
            y += 25
        else:
            cv2.putText(img, "SPICE:  UNAVAILABLE", (20, y), font, 0.5, (0, 0, 255), 1)
            y += 25

        if pose.position_xyz is not None and spice_pose and spice_pose.position_xyz is not None:
            d_m = float(np.linalg.norm(pose.position_xyz - spice_pose.position_xyz))
            cv2.putText(img, f"|Δpos|: {d_m:.1f} m", (20, y), font, 0.5, (255, 255, 255), 1)
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


def render_minimap(img: np.ndarray, trajectory: list, current_idx: int, show: bool) -> None:
    """Render trajectory mini-map (in-place)."""
    if not show or not trajectory:
        return

    # Mini-map setup
    map_size = 300
    h, w = img.shape[:2]
    map_x = 10
    map_y = h - map_size - 10

    # Create mini-map
    minimap = np.zeros((map_size, map_size, 3), dtype=np.uint8)
    minimap[:] = (40, 40, 40)

    # Extract valid positions
    valid_pts = [(lat, lon) for lat, lon in trajectory if lat is not None]
    if not valid_pts:
        return

    # Compute bounds
    lats = [p[0] for p in valid_pts]
    lons = [p[1] for p in valid_pts]
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)

    # Add margin
    lat_margin = (lat_max - lat_min) * 0.1
    lon_margin = (lon_max - lon_min) * 0.1
    lat_min -= lat_margin
    lat_max += lat_margin
    lon_min -= lon_margin
    lon_max += lon_margin

    def to_px(lat, lon):
        x = int((lon - lon_min) / (lon_max - lon_min + 1e-9) * (map_size - 20) + 10)
        y = int((1 - (lat - lat_min) / (lat_max - lat_min + 1e-9)) * (map_size - 20) + 10)
        return (x, y)

    # Draw trajectory
    for i in range(len(valid_pts) - 1):
        pt1 = to_px(valid_pts[i][0], valid_pts[i][1])
        pt2 = to_px(valid_pts[i+1][0], valid_pts[i+1][1])
        cv2.line(minimap, pt1, pt2, (0, 255, 0), 2)

    # Draw current position
    if current_idx < len(trajectory):
        lat, lon = trajectory[current_idx]
        if lat is not None:
            curr_pt = to_px(lat, lon)
            cv2.circle(minimap, curr_pt, 8, (0, 0, 255), -1)
            cv2.circle(minimap, curr_pt, 4, (255, 255, 255), -1)

    # Draw landing site
    if valid_pts:
        landing_pt = to_px(valid_pts[-1][0], valid_pts[-1][1])
        cv2.drawMarker(minimap, landing_pt, (255, 255, 255), cv2.MARKER_CROSS, 15, 2)

    # Border
    cv2.rectangle(minimap, (0, 0), (map_size-1, map_size-1), (200, 200, 200), 2)

    # Composite onto image
    roi = img[map_y:map_y+map_size, map_x:map_x+map_size]
    cv2.addWeighted(minimap, 0.85, roi, 0.15, 0, roi)


def main():
    parser = argparse.ArgumentParser(
        description="Unified Mars EDL descent image viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("path", type=Path, help="Path to dataset directory")
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
    if camera == 'mardi':
        pattern = str(args.path / "0000MD*.LBL")
        files = sorted(glob.glob(pattern))
        # Convert .LBL to .IMG paths
        filenames = [f.replace('.LBL', '.IMG').replace('.lbl', '.IMG') for f in files]
    else:
        pattern = str(args.path / "*.IMG")
        filenames = sorted(glob.glob(pattern),
                          key=lambda f: extract_sclk(Path(f).name, mission))

    if not filenames:
        print(f"No images found in {args.path}")
        return 1

    print(f"Found {len(filenames)} frames")

    # Load SPICE kernels
    spice_mgr = None
    if not args.no_spice and SPICE_AVAILABLE:
        spice_mgr = SPICEManager(mission)

    # Pre-compute trajectory for mini-map
    print("Loading trajectory...")
    trajectory = []
    for fpath in filenames:
        pose = get_pose(Path(fpath), camera, mission, spice_mgr)
        trajectory.append((pose.lat_deg, pose.lon_deg))

    # Setup viewer
    print("Controls: Space=pause, Q=quit, Left/Right=step, +/-=speed, M=minimap, O=overlay")
    cv2.namedWindow("EDL Viewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("EDL Viewer", 1280, 960)

    idx = min(args.start, len(filenames) - 1)
    paused = False
    delay = max(1, 1000 // args.fps)
    show_minimap = not args.no_minimap
    show_overlay = True

    # Main loop
    while True:
        fpath = Path(filenames[idx])
        img = read_image(fpath, camera)

        if img is not None:
            display = img.copy()

            # Get pose for current frame
            pose = get_pose(fpath, camera, mission, spice_mgr)
            sclk = extract_sclk(str(fpath), mission)

            # Render overlays
            if show_overlay:
                render_overlay(
                    display,
                    pose,
                    idx,
                    len(filenames),
                    fpath,
                    fpath.name,
                    sclk,
                    camera,
                    paused,
                    spice_mgr,
                )

            if show_minimap:
                render_minimap(display, trajectory, idx, show_minimap)

            cv2.imshow("EDL Viewer", display)

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
            idx = min(len(filenames) - 1, idx + 1)
            paused = True
        elif key == 80 or key == 0:  # Home
            idx = 0
        elif key == 87 or key == 1:  # End
            idx = len(filenames) - 1
        elif key in (ord('+'), ord('=')):
            delay = max(1, delay - 10)
            print(f"Speed: {1000/delay:.1f} fps")
        elif key == ord('-'):
            delay = min(1000, delay + 10)
            print(f"Speed: {1000/delay:.1f} fps")
        elif key == ord('m'):
            show_minimap = not show_minimap
            print(f"Mini-map: {'ON' if show_minimap else 'OFF'}")
        elif key == ord('o'):
            show_overlay = not show_overlay
            print(f"Overlay: {'ON' if show_overlay else 'OFF'}")
        elif not paused:
            idx = (idx + 1) % len(filenames)

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
