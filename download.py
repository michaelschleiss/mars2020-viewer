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
import time
import urllib.error
import urllib.request
from pathlib import Path

from tqdm import tqdm

_USER_AGENT = "mars2020-viewer/1.0"


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

def _pds4_edlcam(key: str, short: str, name: str, filter_fn) -> tuple[str, dict]:
    return (
        key,
        {
            "name": name,
            "inventory": f"{_M2020_BASE}/data_sol0_{short}/collection_data_sol0_{short}_inventory.csv",
            "base": f"{_M2020_BASE}/data_sol0_{short}/sol/00000/ids/fdr/edl/",
            "output": f"data/m2020/{key}",
            "suffix": "01.IMG",
            "filter": filter_fn,
        },
    )


def _files_from_base(base: str, rels: list[str | tuple[str, str]]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for rel in rels:
        if isinstance(rel, tuple):
            remote, local = rel
        else:
            remote = local = rel
        out.append((f"{base}/{remote}", local))
    return out


_M2020_CAMS = [
    _pds4_edlcam("lcam", "lcam", "LCAM - Lander Vision System (87 images)", None),
    _pds4_edlcam("rdcam", "rdc", "RDCAM - Rover Downlook (lossless full-frames)", _LOSSLESS_FULL),
    _pds4_edlcam("ddcam", "ddc", "DDCAM - Descent Stage Downlook (lossless full-frames)", _LOSSLESS_FULL),
    _pds4_edlcam("rucam", "ruc", "RUCAM - Rover Uplook (lossless full-frames)", _LOSSLESS_FULL),
    _pds4_edlcam("pucam1", "puc1", "PUCAM1 - Parachute Uplook 1 (lossless full-frames)", _LOSSLESS_FULL),
    _pds4_edlcam("pucam2", "puc2", "PUCAM2 - Parachute Uplook 2 (lossless full-frames)", _LOSSLESS_FULL),
    _pds4_edlcam("pucam3", "puc3", "PUCAM3 - Parachute Uplook 3 (lossless full-frames)", _LOSSLESS_FULL),
]

_MSL_SPICE_FILES = (
    _files_from_base(
        _NAIF_MSL,
        [
            "spk/msl_edl_v01.bsp",
            "spk/msl_struct_v02.bsp",
            "ck/msl_edl_v01.bc",
            "fk/msl.tf",
            ("ik/msl_mardi_20120731_c02.ti", "ik/msl_mardi.ti"),
            "sclk/msl.tsc",
            "lsk/naif0012.tls",
        ],
    )
    + _files_from_base(_NAIF_GENERIC, ["pck/pck00010.tpc"])
)

_M2020_SPICE_FILES = _files_from_base(
    _NAIF_M2020,
    [
        "lsk/naif0012.tls",
        "pck/pck00010.tpc",
        "fk/m2020_v04.tf",
        ("sclk/m2020_168_sclkscet_refit_v14.tsc", "sclk/m2020_sclk.tsc"),
        "spk/de438s.bsp",
        "spk/mar097.bsp",
        "spk/m2020_edl_v01.bsp",
        ("spk/m2020_ls_ops210303_iau2000_v1.bsp", "spk/m2020_ls.bsp"),
        "ck/m2020_edl_v01.bc",
    ],
)

DATASETS = {
    "mardi": {
        "name": "MARDI RDR - Sol 0000 (E01_DRCL label+image pairs)",
        "directory": f"{_MSL_BASE}/MSLMRD_0001/DATA/RDR/SURFACE/0000/",
        "output": "data/msl/rdr",
        "pattern": r"0000MD\d+E01_DRCL\.(?:LBL|IMG)",
    },
    "mardi_lossless": {
        "name": "MARDI RDR lossless subset - Sol 0000 (C00/C01_DRCL pairs across volumes)",
        "directories": [
            f"{_MSL_BASE}/MSLMRD_0001/DATA/RDR/SURFACE/0000/",
            f"{_MSL_BASE}/MSLMRD_0002/DATA/RDR/SURFACE/0000/",
            f"{_MSL_BASE}/MSLMRD_0003/DATA/RDR/SURFACE/0000/",
        ],
        "output": "data/msl/rdr",
        "pattern": r"0000MD\d+C0[01]_DRCL\.(?:LBL|IMG)",
    },
    **dict(_M2020_CAMS),
    "msl_orbital": {
        "name": "MSL Gale merged ortho + DEM",
        "files": [
            (
                "https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/MSL_Gale_Orthophoto_Mosaic_25cm_v3.tif",
                "MSL_Gale_Orthophoto_Mosaic_25cm_v3.tif",
            ),
            (f"{_USGS_MSL}/MSL_Gale_DEM_Mosaic_1m_v3.tif", "MSL_Gale_DEM_Mosaic_1m_v3.tif"),
        ],
        "output": "data/msl/orbital",
    },
    "msl_spice": {
        "name": "MSL EDL SPICE kernels",
        "files": _MSL_SPICE_FILES,
        "output": "data/msl/spice",
        "meta_kernel": "msl_edl.tm",
    },
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
    "m2020_spice": {
        "name": "Mars 2020 EDL SPICE kernels",
        "files": _M2020_SPICE_FILES,
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
    url: str, dest: Path, pbar: "tqdm | None" = None, retries: int = 3
) -> bool:
    """Download a single file. Returns True if downloaded, False if skipped."""
    if dest.exists() and dest.stat().st_size > 0:
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    partial.unlink(missing_ok=True)

    for attempt in range(retries):
        try:
            req = _request(url)
            with urllib.request.urlopen(req, timeout=120) as resp:
                status = int(getattr(resp, "status", resp.getcode()))
                if status != 200:
                    raise RuntimeError(f"HTTP {status}")
                with partial.open("wb") as f:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        if pbar:
                            pbar.update(len(chunk))

            partial.rename(dest)
            return True
        except urllib.error.HTTPError as e:
            partial.unlink(missing_ok=True)
            if e.code in {400, 401, 403, 404}:
                print(f"  Error: HTTP {e.code} for {url}")
                return False
        except Exception as e:
            partial.unlink(missing_ok=True)
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # 1s, 2s, 4s...
            else:
                print(f"  Error: {e}")
    return False


def get_file_size(url: str) -> int:
    """Get file size via HEAD request."""
    req = _request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=10) as resp:
        status = getattr(resp, "status", None)
        if status is None:
            status = resp.getcode()
        status = int(status)
        if status < 200 or status >= 300:
            raise RuntimeError(f"HEAD {url} failed with HTTP {status}")
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
        file_sizes = []
        for url, filename in files:
            size = get_file_size(url)
            file_sizes.append(size)
            if size >= 1_000_000_000:
                print(f"    {filename}: {size / 1_000_000_000:.1f} GB")
            elif size >= 1_000_000:
                print(f"    {filename}: {size / 1_000_000:.0f} MB")
            elif size > 0:
                print(f"    {filename}: {size} bytes")
            else:
                print(f"    {filename}: unknown size (no Content-Length)")
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
    total = total_estimate if total_estimate > 0 else None
    with tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1000, desc="Downloading") as pbar:
        for i, (url, filename) in enumerate(files, 1):
            dest = output_dir / filename

            if download_file(url, dest, pbar):
                downloaded += 1
            else:
                skipped += 1
                if dest.exists():
                    pbar.update(dest.stat().st_size)
            pbar.set_postfix_str(f"{i}/{len(files)} files", refresh=False)

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
