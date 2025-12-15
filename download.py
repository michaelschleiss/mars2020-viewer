#!/usr/bin/env python3
"""Download Mars EDL descent imagery from PDS archives.

Usage:
    python download.py              # Download all cameras
    python download.py lcam         # Mars 2020 LCAM (87 images, ~91 MB)
    python download.py mardi        # MSL MARDI (1504 images, ~165 MB)
    python download.py --dry-run    # Show sizes without downloading
"""

import argparse
import re
import threading
import time
import urllib.request
from pathlib import Path

from tqdm import tqdm


class ByteProgress:
    """Thread-safe byte counter for live download rate display."""

    def __init__(self, pbar, total_files: int, est_file_size: int = 0):
        self.total_bytes = 0
        self.file_count = 0
        self.total_files = total_files
        self.est_file_size = est_file_size  # from HEAD request
        self.pbar = pbar
        self.start_time = time.perf_counter()
        self._lock = threading.Lock()

    def add(self, file_size: int) -> None:
        with self._lock:
            self.total_bytes += file_size
            self.file_count += 1
            elapsed = time.perf_counter() - self.start_time
            mb = self.total_bytes / 1_000_000
            rate = mb / elapsed if elapsed > 0 else 0

            # Estimate total and ETA based on average file size so far
            avg_size = self.total_bytes / self.file_count
            est_total_mb = avg_size * self.total_files / 1_000_000
            remaining_mb = est_total_mb - mb
            eta_sec = remaining_mb / rate if rate > 0 else 0
            eta_str = f"{int(eta_sec // 60)}:{int(eta_sec % 60):02d}"

            # Format elapsed/ETA
            elapsed_str = f"{int(elapsed // 60)}:{int(elapsed % 60):02d}"
            time_str = f"{elapsed_str}/{eta_str}"

            if est_total_mb >= 1000:
                self.pbar.set_description_str(
                    f"⏱ {time_str} | ↓ {mb / 1000:.1f}/{est_total_mb / 1000:.1f}GB @ {rate:.1f}MB/s"
                )
            else:
                self.pbar.set_description_str(
                    f"⏱ {time_str} | ↓ {mb:.0f}/{est_total_mb:.0f}MB @ {rate:.1f}MB/s"
                )


# Dataset configurations
#
# Inventory format: P,urn:nasa:pds:bundle:collection:product_id::version
# Product ID → filename: uppercase + "01.IMG" (inventory omits the "01" version suffix)
#
# LCAM products:
#   elm_0000_{sclk}_{seq}fdr_n{frame:07d}lvs_04000_0000luj
#   All 87 frames are lossless grayscale 1024x1024
#
# RDCAM products (4 variants per frame):
#   edf_0000_{sclk}_{seq}fdr_n{frame}edlc{seq}_0000luj  # full-frame lossless (want this)
#   edf_0000_{sclk}_{seq}fdr_t{frame}edlc{seq}_0000luj  # thumbnail lossless
#   edf_0000_{sclk}_{seq}fdr_n{frame}edlc{seq}_0000fhj  # full-frame compressed
#   edf_0000_{sclk}_{seq}fdr_t{frame}edlc{seq}_0000fhj  # thumbnail compressed
#   Filter: "_n" (full-frame) + "luj" (lossless) → 7141 of 19712 products
#
_M2020_BASE = "https://planetarydata.jpl.nasa.gov/img/data/mars2020/mars2020_edlcam_ops_calibrated"
_MSL_BASE = "https://planetarydata.jpl.nasa.gov/img/data/msl"
_LOSSLESS_FULL = lambda pid: "_n" in pid and "luj" in pid

DATASETS = {
    # === MSL Curiosity ===
    "mardi": {
        "name": "MARDI - Mars Descent Imager (1504 lossy)",
        "directory": f"{_MSL_BASE}/MSLMRD_0001/DATA/EDR/SURFACE/0000/",
        "output": "data/msl/mardi",
        "pattern": r"0000MD\d+E01_XXXX\.DAT",  # EDR lossy JPEG
    },
    "mardi_lossless": {
        "name": "MARDI Lossless - 635 frames across 3 volumes",
        "directories": [
            f"{_MSL_BASE}/MSLMRD_0001/DATA/EDR/SURFACE/0000/",
            f"{_MSL_BASE}/MSLMRD_0002/DATA/EDR/SURFACE/0000/",
            f"{_MSL_BASE}/MSLMRD_0003/DATA/EDR/SURFACE/0000/",
        ],
        "output": "data/msl/mardi_lossless",
        "pattern": r"0000MD\d+C0[01]_XXXX\.DAT",  # EDR lossless (C00/C01)
    },
    # === Mars 2020 Perseverance ===
    "lcam": {
        "name": "LCAM - Lander Vision System (87 images)",
        "inventory": f"{_M2020_BASE}/data_sol0_lcam/collection_data_sol0_lcam_inventory.csv",
        "base": f"{_M2020_BASE}/data_sol0_lcam/sol/00000/ids/fdr/edl/",
        "output": "data/m2020/lcam",
        "suffix": "01.IMG",
        "filter": None,
    },
    "rdcam": {
        "name": "RDCAM - Rover Downlook (7141 lossless full-frames)",
        "inventory": f"{_M2020_BASE}/data_sol0_rdc/collection_data_sol0_rdc_inventory.csv",
        "base": f"{_M2020_BASE}/data_sol0_rdc/sol/00000/ids/fdr/edl/",
        "output": "data/m2020/rdcam",
        "suffix": "01.IMG",
        "filter": _LOSSLESS_FULL,
    },
    "ddcam": {
        "name": "DDCAM - Descent Stage Downlook (1985 products)",
        "inventory": f"{_M2020_BASE}/data_sol0_ddc/collection_data_sol0_ddc_inventory.csv",
        "base": f"{_M2020_BASE}/data_sol0_ddc/sol/00000/ids/fdr/edl/",
        "output": "data/m2020/ddcam",
        "suffix": "01.IMG",
        "filter": _LOSSLESS_FULL,
    },
    "rucam": {
        "name": "RUCAM - Rover Uplook (8987 products)",
        "inventory": f"{_M2020_BASE}/data_sol0_ruc/collection_data_sol0_ruc_inventory.csv",
        "base": f"{_M2020_BASE}/data_sol0_ruc/sol/00000/ids/fdr/edl/",
        "output": "data/m2020/rucam",
        "suffix": "01.IMG",
        "filter": _LOSSLESS_FULL,
    },
    "pucam1": {
        "name": "PUCAM1 - Parachute Uplook 1 (10686 products)",
        "inventory": f"{_M2020_BASE}/data_sol0_puc1/collection_data_sol0_puc1_inventory.csv",
        "base": f"{_M2020_BASE}/data_sol0_puc1/sol/00000/ids/fdr/edl/",
        "output": "data/m2020/pucam1",
        "suffix": "01.IMG",
        "filter": _LOSSLESS_FULL,
    },
    "pucam2": {
        "name": "PUCAM2 - Parachute Uplook 2",
        "inventory": f"{_M2020_BASE}/data_sol0_puc2/collection_data_sol0_puc2_inventory.csv",
        "base": f"{_M2020_BASE}/data_sol0_puc2/sol/00000/ids/fdr/edl/",
        "output": "data/m2020/pucam2",
        "suffix": "01.IMG",
        "filter": _LOSSLESS_FULL,
    },
    "pucam3": {
        "name": "PUCAM3 - Parachute Uplook 3",
        "inventory": f"{_M2020_BASE}/data_sol0_puc3/collection_data_sol0_puc3_inventory.csv",
        "base": f"{_M2020_BASE}/data_sol0_puc3/sol/00000/ids/fdr/edl/",
        "output": "data/m2020/pucam3",
        "suffix": "01.IMG",
        "filter": _LOSSLESS_FULL,
    },
}


def fetch_directory(url: str, pattern: str) -> list[str]:
    """Fetch directory listing and return files matching pattern."""
    with urllib.request.urlopen(url, timeout=30) as resp:
        html = resp.read().decode("utf-8")

    # Extract filenames from directory listing
    files = re.findall(pattern, html)
    return sorted(set(files))


def fetch_inventory(url: str, filter_fn=None) -> list[str]:
    """Fetch inventory CSV and return list of product IDs."""
    with urllib.request.urlopen(url, timeout=30) as resp:
        text = resp.read().decode("utf-8")

    product_ids = []
    for line in text.strip().splitlines():
        # Format: P,urn:nasa:pds:...:product_id::version
        if "," not in line:
            continue
        urn = line.split(",", 1)[1]
        product_id = urn.split("::")[0].rsplit(":", 1)[-1]
        if filter_fn is None or filter_fn(product_id):
            product_ids.append(product_id)
    return product_ids


def download_file(
    url: str, dest: Path, byte_progress: ByteProgress | None = None
) -> bool:
    """Download a single file. Returns True if downloaded, False if skipped.

    Uses .partial suffix during download to prevent corrupt files from
    being skipped on retry if download is interrupted.
    """
    if dest.exists():
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")

    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            data = resp.read()
        partial.write_bytes(data)
        partial.rename(dest)  # atomic on same filesystem
        if byte_progress:
            byte_progress.add(len(data))
        return True
    except Exception as e:
        partial.unlink(missing_ok=True)  # clean up partial on failure
        print(f"  Error: {e}")
        return False


def get_file_size(url: str) -> int:
    """Get file size via HEAD request."""
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return int(resp.headers.get("Content-Length", 0))


def download_dataset(name: str, config: dict, dry_run: bool = False) -> None:
    """Download all files for a dataset."""
    print(f"\n{config['name']}")

    output_dir = Path(config["output"])

    # Determine file list based on dataset type
    if "inventory" in config:
        # Mars 2020 style: inventory CSV + base URL + suffix
        print("  Fetching inventory...")
        product_ids = fetch_inventory(config["inventory"], config.get("filter"))
        base_url = config["base"]
        suffix = config["suffix"]
        files = [
            (base_url + pid.upper() + suffix, pid.upper() + suffix)
            for pid in product_ids
        ]
    elif "directory" in config:
        # MSL style: single directory listing + pattern
        print("  Fetching directory listing...")
        filenames = fetch_directory(config["directory"], config["pattern"])
        base_url = config["directory"]
        files = [(base_url + fn, fn) for fn in filenames]
    elif "directories" in config:
        # MSL style: multiple directories (e.g., across PDS volumes)
        files = []
        seen = set()
        for i, dir_url in enumerate(config["directories"], 1):
            print(f"  Fetching directory {i}/{len(config['directories'])}...")
            filenames = fetch_directory(dir_url, config["pattern"])
            for fn in filenames:
                if fn not in seen:
                    seen.add(fn)
                    files.append((dir_url + fn, fn))
        files.sort(key=lambda x: x[1])  # Sort by filename
    else:
        print("  Error: unknown dataset config")
        return

    print(f"  Found {len(files)} files")

    if not files:
        return

    # Estimate total size from first file
    first_url = files[0][0]
    sample_size = get_file_size(first_url)
    total_estimate = sample_size * len(files)
    if total_estimate >= 1e9:
        print(
            f"  Estimated size: {total_estimate / 1e9:.1f} GB ({sample_size / 1e6:.2f} MB × {len(files)})"
        )
    else:
        print(
            f"  Estimated size: {total_estimate / 1e6:.0f} MB ({sample_size / 1e6:.2f} MB × {len(files)})"
        )

    if dry_run:
        return

    downloaded = 0
    skipped = 0

    # Initial description with estimate
    est_total_mb = sample_size * len(files) / 1_000_000
    if est_total_mb >= 1000:
        init_desc = f"⏱ 0:00/--:-- | ↓ 0/{est_total_mb / 1000:.1f}GB @ --MB/s"
    else:
        init_desc = f"⏱ 0:00/--:-- | ↓ 0/{est_total_mb:.0f}MB @ --MB/s"

    with tqdm(
        total=len(files),
        desc=init_desc,
        unit="file",
        bar_format="Downloading: [{bar:25}] {n}/{total} | {desc}",
    ) as pbar:
        byte_progress = ByteProgress(pbar, len(files), sample_size)

        for url, filename in files:
            dest = output_dir / filename

            if download_file(url, dest, byte_progress):
                downloaded += 1
            else:
                skipped += 1
            pbar.update(1)

    print(f"  Done: {downloaded} downloaded, {skipped} skipped")


def main():
    parser = argparse.ArgumentParser(description="Download Mars EDL descent imagery")
    parser.add_argument(
        "dataset",
        nargs="?",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Dataset to download (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Show what would be downloaded without downloading",
    )
    args = parser.parse_args()

    if args.dataset == "all":
        for name, config in DATASETS.items():
            download_dataset(name, config, dry_run=args.dry_run)
    else:
        download_dataset(args.dataset, DATASETS[args.dataset], dry_run=args.dry_run)


if __name__ == "__main__":
    main()
