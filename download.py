#!/usr/bin/env python3
"""Download Mars EDL data: descent imagery, orbital maps, and SPICE kernels.

Usage:
    python download.py              # List all datasets
    python download.py lcam         # Mars 2020 LCAM (87 images)
    python download.py ctx          # CTX orbital maps (44 MB)
    python download.py m2020_spice  # Mars 2020 SPICE kernels
    python download.py all          # Download everything
    python download.py --dry-run    # Show sizes without downloading
"""

import argparse
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from tqdm import tqdm

try:
    import urllib3
except Exception:  # pragma: no cover
    urllib3 = None


_USER_AGENT = "mars2020-viewer/1.0"


class ByteProgress:
    """Thread-safe byte counter for live download rate display."""

    def __init__(self, pbar, total_files: int, est_total_bytes: int = 0):
        self.total_bytes = 0
        self.file_count = 0
        self.total_files = total_files
        self.est_total_bytes = est_total_bytes  # from HEAD request(s)
        self.pbar = pbar
        self.start_time = time.perf_counter()
        self._lock = threading.Lock()
        self._last_update = 0.0

    def add_bytes(self, chunk_size: int) -> None:
        """Update progress incrementally as bytes are received."""
        with self._lock:
            self.total_bytes += chunk_size

            # Throttle updates to avoid excessive overhead
            now = time.perf_counter()
            if now - self._last_update < 0.1:
                return
            self._last_update = now

            self._update_display()

    def file_done(self) -> None:
        """Mark a file as complete (for file count tracking)."""
        with self._lock:
            self.file_count += 1
            self._update_display()

    def _update_display(self) -> None:
        """Update the progress bar description."""
        elapsed = time.perf_counter() - self.start_time
        mb = self.total_bytes / 1_000_000
        rate = mb / elapsed if elapsed > 0 else 0

        # Use known total if available, else estimate from average
        if self.est_total_bytes > 0:
            est_total_mb = self.est_total_bytes / 1_000_000
        elif self.file_count > 0:
            avg_size = self.total_bytes / self.file_count
            est_total_mb = avg_size * self.total_files / 1_000_000
        else:
            est_total_mb = mb  # fallback

        remaining_mb = max(0, est_total_mb - mb)
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


# =============================================================================
# BASE URLs for data archives
# =============================================================================
#
# PDS (Planetary Data System) - NASA's official archive
# USGS (US Geological Survey) - Processed mosaics and DEMs
# NAIF (Navigation and Ancillary Information Facility) - SPICE kernels
#
_M2020_BASE = "https://planetarydata.jpl.nasa.gov/img/data/mars2020/mars2020_edlcam_ops_calibrated"
_MSL_BASE = "https://planetarydata.jpl.nasa.gov/img/data/msl"
_USGS_M2020 = "https://planetarymaps.usgs.gov/mosaic/mars2020_trn"
_USGS_MSL = "https://planetarymaps.usgs.gov/mosaic/Mars/MSL"
_NAIF_M2020 = "https://naif.jpl.nasa.gov/pub/naif/pds/pds4/mars2020/mars2020_spice/spice_kernels"
_NAIF_MSL = "https://naif.jpl.nasa.gov/pub/naif/MSL/kernels"
_NAIF_GENERIC = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels"

# Filter for Mars 2020 lossless full-frame products (vs thumbnails and compressed)
_LOSSLESS_FULL = lambda pid: "_n" in pid and "luj" in pid

# =============================================================================
# DATASET CONFIGURATIONS
# =============================================================================
#
# Each dataset uses one of four download patterns:
#
# 1. "inventory" + "base" + "suffix"
#    Mars 2020 PDS4 style: fetch CSV inventory, construct URLs from product IDs
#    Example: LCAM, RDCAM, etc.
#
# 2. "directory" + "pattern"
#    MSL PDS3 style: scrape HTML directory listing, match regex pattern
#    Example: MARDI lossy
#
# 3. "directories" + "pattern"
#    Like #2 but across multiple PDS volumes (deduplicates)
#    Example: MARDI lossless (spread across volumes 1-3)
#
# 4. "files"
#    Static file list: direct URL to filename mapping
#    Example: Orbital maps, SPICE kernels
#
# Optional keys:
#   "meta_kernel" - Generate SPICE .tm file after download
#   "filter" - Function to filter product IDs (for inventory-based)

DATASETS = {
    # === MSL Curiosity ===
    "mardi": {
        "name": "MARDI - Mars Descent Imager (1504 lossy)",
        "directory": f"{_MSL_BASE}/MSLMRD_0001/DATA/EDR/SURFACE/0000/",
        "output": "data/msl/mardi",
        "pattern": r"0000MD\d+E01_XXXX\.DAT",  # EDR lossy JPEG
    },
    "mardi_rdr": {
        "name": "MARDI RDR - Sol 0000 (E01_DRCL label+image pairs)",
        "directory": f"{_MSL_BASE}/MSLMRD_0001/DATA/RDR/SURFACE/0000/",
        "output": "data/msl/rdr",
        "pattern": r"0000MD\d+E01_DRCL\.(LBL|IMG)",
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
    # === MSL Curiosity - Orbital Maps ===
    # Citation: Grindrod & Davis (2018) https://doi.org/10.6084/m9.figshare.6584984
    "msl_ctx": {
        "name": "MSL Gale CTX stereo DTMs + orthos (1.1 GB)",
        "files": [
            ("https://ndownloader.figshare.com/files/12096770", "DTMs.zip"),
            ("https://ndownloader.figshare.com/files/12096776", "FOMs.zip"),
            ("https://ndownloader.figshare.com/files/12096839", "orthos.zip"),
            ("https://ndownloader.figshare.com/files/12096842", "Readme.pdf"),
        ],
        "output": "data/msl/ctx",
    },
    "msl_orbital": {
        "name": "MSL Gale merged ortho + DEM (27 GB)",
        "files": [
            (
                "https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/MSL_Gale_Orthophoto_Mosaic_25cm_v3.tif",
                "MSL_Gale_Orthophoto_Mosaic_25cm_v3.tif",
            ),
            (
                f"{_USGS_MSL}/MSL_Gale_DEM_Mosaic_1m_v3.tif",
                "MSL_Gale_DEM_Mosaic_1m_v3.tif",
            ),
        ],
        "output": "data/msl/orbital",
    },
    # === MSL Curiosity - SPICE Kernels ===
    "msl_spice": {
        "name": "MSL EDL SPICE kernels",
        "files": [
            (f"{_NAIF_MSL}/spk/msl_edl_v01.bsp", "spk/msl_edl_v01.bsp"),
            (f"{_NAIF_MSL}/spk/msl_struct_v02.bsp", "spk/msl_struct_v02.bsp"),
            (f"{_NAIF_MSL}/ck/msl_edl_v01.bc", "ck/msl_edl_v01.bc"),
            (f"{_NAIF_MSL}/fk/msl.tf", "fk/msl.tf"),
            (f"{_NAIF_MSL}/ik/msl_mardi_20120731_c02.ti", "ik/msl_mardi.ti"),
            (f"{_NAIF_MSL}/sclk/msl.tsc", "sclk/msl.tsc"),
            (f"{_NAIF_MSL}/lsk/naif0012.tls", "lsk/naif0012.tls"),
            (f"{_NAIF_GENERIC}/pck/pck00010.tpc", "pck/pck00010.tpc"),
        ],
        "output": "data/msl/spice",
        "meta_kernel": "msl_edl.tm",
    },
    # === Mars 2020 - Orbital Maps ===
    "ctx": {
        "name": "CTX Jezero ortho + DTM (TRN reference)",
        "files": [
            (
                f"{_USGS_M2020}/CTX/JEZ_ctx_B_soc_008_orthoMosaic_6m_Eqc_latTs0_lon0.tif",
                "JEZ_ctx_ortho_6m.tif",
            ),
            (
                f"{_USGS_M2020}/CTX/JEZ_ctx_B_soc_008_DTM_MOLAtopography_DeltaGeoid_20m_Eqc_latTs0_lon0.tif",
                "JEZ_ctx_dtm_20m.tif",
            ),
        ],
        "output": "data/m2020/orbital/ctx",
    },
    "hirise": {
        "name": "HiRISE Jezero ortho + DTM (hazard map)",
        "files": [
            (
                f"{_USGS_M2020}/HiRISE/JEZ_hirise_soc_006_orthoMosaic_25cm_Eqc_latTs0_lon0_first.tif",
                "JEZ_hirise_ortho_25cm.tif",
            ),
            (
                f"{_USGS_M2020}/HiRISE/JEZ_hirise_soc_006_DTM_MOLAtopography_DeltaGeoid_1m_Eqc_latTs0_lon0_blend40.tif",
                "JEZ_hirise_dtm_1m.tif",
            ),
        ],
        "output": "data/m2020/orbital/hirise",
    },
    # === Mars 2020 - SPICE Kernels ===
    "m2020_spice": {
        "name": "Mars 2020 EDL SPICE kernels",
        "files": [
            (f"{_NAIF_M2020}/lsk/naif0012.tls", "lsk/naif0012.tls"),
            (f"{_NAIF_M2020}/pck/pck00010.tpc", "pck/pck00010.tpc"),
            (f"{_NAIF_M2020}/fk/m2020_v04.tf", "fk/m2020_v04.tf"),
            (f"{_NAIF_M2020}/sclk/m2020_168_sclkscet_refit_v14.tsc", "sclk/m2020_sclk.tsc"),
            (f"{_NAIF_M2020}/spk/de438s.bsp", "spk/de438s.bsp"),
            (f"{_NAIF_M2020}/spk/mar097.bsp", "spk/mar097.bsp"),
            (f"{_NAIF_M2020}/spk/m2020_edl_v01.bsp", "spk/m2020_edl_v01.bsp"),
            (f"{_NAIF_M2020}/spk/m2020_ls_ops210303_iau2000_v1.bsp", "spk/m2020_ls.bsp"),
            (f"{_NAIF_M2020}/ck/m2020_edl_v01.bc", "ck/m2020_edl_v01.bc"),
        ],
        "output": "data/m2020/spice",
        "meta_kernel": "m2020_edl.tm",
    },
}


def write_meta_kernel(output_dir: Path, kernel_files: list[str], name: str) -> Path:
    """Generate a SPICE meta-kernel (.tm) file for the downloaded kernels."""
    mk_path = output_dir / f"{name}"
    abs_root = output_dir.resolve()

    lines = [
        "KPL/MK",
        "",
        f"Meta-kernel for {output_dir.name} (generated by download.py)",
        "",
        "\\begindata",
        "",
        f"   PATH_VALUES     = ( '{abs_root.as_posix()}' )",
        "   PATH_SYMBOLS    = ( 'KERNELS' )",
        "",
        "   KERNELS_TO_LOAD = (",
    ]
    for kf in kernel_files:
        lines.append(f"      '$KERNELS/{kf}'")
    lines.extend([
        "   )",
        "",
        "\\begintext",
        "",
    ])

    mk_path.write_text("\n".join(lines), encoding="utf-8")
    return mk_path


def _request(url: str, *, method: str = "GET", headers: dict[str, str] | None = None) -> urllib.request.Request:
    req_headers = {"User-Agent": _USER_AGENT}
    if headers:
        req_headers.update(headers)
    return urllib.request.Request(url, method=method, headers=req_headers)


def _urllib3_pool() -> "urllib3.PoolManager":
    if urllib3 is None:  # pragma: no cover
        raise RuntimeError("urllib3 is not available")
    timeout = urllib3.Timeout(connect=30.0, read=120.0)
    return urllib3.PoolManager(headers={"User-Agent": _USER_AGENT}, timeout=timeout)


_URLLIB3_POOL: "urllib3.PoolManager | None" = None


def _get_urllib3_pool() -> "urllib3.PoolManager":
    global _URLLIB3_POOL
    if _URLLIB3_POOL is None:
        _URLLIB3_POOL = _urllib3_pool()
    return _URLLIB3_POOL


def fetch_directory(url: str, pattern: str) -> list[str]:
    """Fetch directory listing and return files matching pattern."""
    with urllib.request.urlopen(_request(url), timeout=30) as resp:
        html = resp.read().decode("utf-8")

    # Extract filenames from directory listing
    files = re.findall(pattern, html)
    return sorted(set(files))


def fetch_inventory(url: str, filter_fn=None) -> list[str]:
    """Fetch inventory CSV and return list of product IDs."""
    with urllib.request.urlopen(_request(url), timeout=30) as resp:
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
    url: str, dest: Path, byte_progress: ByteProgress | None = None, retries: int = 3
) -> bool:
    """Download a single file. Returns True if downloaded, False if skipped."""
    if dest.exists() and dest.stat().st_size > 0:
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")

    for attempt in range(retries):
        try:
            resume_from = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={resume_from}-"} if resume_from > 0 else None

            if urllib3 is not None:
                pool = _get_urllib3_pool()
                resp = pool.request("GET", url, headers=headers, preload_content=False)
                try:
                    if resp.status == 416 and resume_from > 0:
                        partial.unlink(missing_ok=True)
                        continue
                    if resp.status not in (200, 206):
                        raise RuntimeError(f"HTTP {resp.status}")

                    mode = "ab" if resp.status == 206 and resume_from > 0 else "wb"
                    with partial.open(mode) as f:
                        while True:
                            chunk = resp.read(1024 * 1024)
                            if not chunk:
                                break
                            f.write(chunk)
                            if byte_progress:
                                byte_progress.add_bytes(len(chunk))
                finally:
                    resp.release_conn()
            else:
                req = _request(url, headers=headers or {})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    status = getattr(resp, "status", resp.getcode())
                    mode = "ab" if status == 206 and resume_from > 0 else "wb"
                    with partial.open(mode) as f:
                        while True:
                            chunk = resp.read(1024 * 1024)
                            if not chunk:
                                break
                            f.write(chunk)
                            if byte_progress:
                                byte_progress.add_bytes(len(chunk))

            partial.rename(dest)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 416 and partial.exists():
                partial.unlink(missing_ok=True)
                continue
            if e.code in {400, 401, 403, 404}:
                print(f"  Error: HTTP {e.code} for {url}")
                return False
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # 1s, 2s, 4s...
            else:
                print(f"  Error: {e}")
    return False


def get_file_size(url: str) -> int:
    """Get file size via HEAD request, following redirects."""
    if urllib3 is not None:
        pool = _get_urllib3_pool()
        resp = pool.request("HEAD", url, preload_content=False, redirect=False)
        try:
            size = int(resp.headers.get("Content-Length") or 0)
            # Handle redirects (e.g., Figshare -> S3 signed URLs)
            if size == 0 and resp.status in (301, 302, 303, 307, 308):
                redirect_url = resp.headers.get("Location")
                if redirect_url:
                    # S3 signed URLs don't return Content-Length on HEAD,
                    # but do return Content-Range on range requests
                    resp2 = pool.request(
                        "GET", redirect_url, preload_content=False,
                        headers={"Range": "bytes=0-0"}
                    )
                    try:
                        # Parse "bytes 0-0/12345" to get total size
                        cr = resp2.headers.get("Content-Range", "")
                        if "/" in cr:
                            size = int(cr.split("/")[-1])
                    finally:
                        resp2.release_conn()
            return size
        finally:
            resp.release_conn()
    req = _request(url, method="HEAD")
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
    elif "files" in config:
        # Static file list (orbital maps, SPICE kernels)
        files = [(url, fn) for url, fn in config["files"]]
    else:
        print("  Error: unknown dataset config")
        return

    print(f"  {len(files)} files")

    if not files:
        return

    # For small file lists (static files), get actual sizes; else estimate from first
    if len(files) <= 10:
        print("  Checking sizes...")
        file_sizes = [get_file_size(url) for url, _ in files]
        total_estimate = sum(file_sizes)
    else:
        first_url = files[0][0]
        sample_size = get_file_size(first_url)
        total_estimate = sample_size * len(files)

    if total_estimate >= 1e9:
        print(f"  Size: ~{total_estimate / 1e9:.1f} GB")
    else:
        print(f"  Size: ~{total_estimate / 1e6:.0f} MB")

    if dry_run:
        return

    downloaded = 0
    skipped = 0

    # Initial description with estimate
    est_total_mb = total_estimate / 1_000_000
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
        byte_progress = ByteProgress(pbar, len(files), total_estimate)

        for url, filename in files:
            dest = output_dir / filename

            if download_file(url, dest, byte_progress):
                downloaded += 1
            else:
                skipped += 1
                # For skipped files, count their size as already done
                if dest.exists():
                    byte_progress.add_bytes(dest.stat().st_size)
            byte_progress.file_done()
            pbar.update(1)

    print(f"  Done: {downloaded} downloaded, {skipped} skipped")

    # Generate meta-kernel for SPICE datasets
    if "meta_kernel" in config and downloaded > 0:
        kernel_files = [fn for _, fn in files]
        mk = write_meta_kernel(output_dir, kernel_files, config["meta_kernel"])
        print(f"  Meta-kernel: {mk}")


def list_datasets() -> None:
    """Print all available datasets grouped by mission."""
    print("\nAvailable datasets:\n")

    # Group by mission
    msl = [(k, v) for k, v in DATASETS.items() if k.startswith("msl") or k.startswith("mardi")]
    m2020 = [(k, v) for k, v in DATASETS.items() if k not in dict(msl)]

    print("MSL Curiosity:")
    for key, cfg in msl:
        print(f"  {key:18} {cfg['name']}")

    print("\nMars 2020 Perseverance:")
    for key, cfg in m2020:
        print(f"  {key:18} {cfg['name']}")

    print("\nUsage: python download.py <dataset>")
    print("       python download.py <dataset> -n  # Dry run (show size)")
    print("       python download.py all           # Download everything")


def main():
    parser = argparse.ArgumentParser(
        description="Download Mars EDL data: imagery, orbital maps, SPICE kernels"
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Dataset(s) to download (or 'all')",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available datasets",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be downloaded without downloading",
    )
    args = parser.parse_args()

    if args.list or not args.datasets:
        list_datasets()
        return

    # Expand 'all' and validate dataset names
    datasets = []
    for name in args.datasets:
        if name == "all":
            datasets.extend(DATASETS.keys())
        elif name in DATASETS:
            datasets.append(name)
        else:
            print(f"Unknown dataset: {name}")
            return

    for name in datasets:
        download_dataset(name, DATASETS[name], dry_run=args.dry_run)


if __name__ == "__main__":
    main()
