#!/usr/bin/env python3
"""
Profile startup costs for view.py without importing OpenCV.

Measures:
- label/metadata extraction time (load_frame_metadata)
- optional SPICE kernel load + per-frame query time (if spiceypy installed)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pds_metadata import load_frame_metadata


def iter_imgs(path: Path) -> list[Path]:
    imgs = sorted(list(path.glob("*.IMG")) + list(path.glob("*.img")))
    return imgs


def main() -> int:
    ap = argparse.ArgumentParser(description="Profile view.py startup costs")
    ap.add_argument("path", type=Path, help="Dataset directory (e.g. data/m2020/rdcam)")
    ap.add_argument("--mission", required=True, help="m2020 or msl")
    ap.add_argument("--camera", required=True, help="lcam/rdcam/ddcam/mardi")
    ap.add_argument("--n", type=int, default=500, help="Number of frames to sample (default: 500)")
    ap.add_argument("--spice", action="store_true", help="Also profile SPICE (requires spiceypy + kernels)")
    args = ap.parse_args()

    imgs = iter_imgs(args.path)
    if not imgs:
        raise SystemExit(f"No .IMG files found in {args.path}")

    sample = imgs[: min(args.n, len(imgs))]
    print(f"files: {len(imgs)} (sampling {len(sample)})")

    t0 = time.perf_counter()
    metas = [load_frame_metadata(p, mission=args.mission, camera=args.camera) for p in sample]
    dt = time.perf_counter() - t0
    total_label_mb = sum(len(m.label_text) for m in metas) / 1e6
    print(f"metadata: {dt:.3f}s total  ({dt/len(sample)*1000:.2f} ms/frame)  label_text={total_label_mb:.1f}MB")

    if not args.spice:
        return 0

    try:
        import spiceypy as spice  # type: ignore
    except Exception as e:
        raise SystemExit(f"--spice requested but spiceypy not available: {e}")

    mk = Path("spice_kernels/m2020.tm" if args.mission == "m2020" else "spice_kernels/msl_edl.tm")
    if not mk.exists():
        raise SystemExit(f"SPICE meta-kernel not found: {mk}")

    t0 = time.perf_counter()
    spice.furnsh(str(mk))
    dt_load = time.perf_counter() - t0
    print(f"spice: furnsh {mk.name}: {dt_load:.3f}s")

    # Query position using ET from START/STOP midpoint when present, else SCLK.
    sc_id = -168 if args.mission == "m2020" else -76
    target = "M2020" if args.mission == "m2020" else "MSL"

    def sclk_to_et(sclk: float) -> float:
        sclk_str = f"{int(sclk)}.{int((sclk % 1) * 1000):03d}"
        return spice.scs2e(sc_id, sclk_str)

    t0 = time.perf_counter()
    ok = 0
    for m in metas:
        try:
            if m.start_time and m.stop_time:
                et0 = spice.str2et(m.start_time)
                et1 = spice.str2et(m.stop_time)
                et = (et0 + et1) / 2.0
            else:
                sclk = m.sclk_mid if m.sclk_mid is not None else m.sclk_start
                if sclk is None:
                    continue
                et = sclk_to_et(float(sclk))
            spice.spkezr(target, et, "IAU_MARS", "NONE", "MARS")
            ok += 1
        except Exception:
            continue
    dt_query = time.perf_counter() - t0
    if ok:
        print(f"spice: spkezr {ok} queries: {dt_query:.3f}s total ({dt_query/ok*1000:.2f} ms/query)")
    else:
        print("spice: no successful queries")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
