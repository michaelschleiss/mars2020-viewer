#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class Frame:
    product_id: str
    start_time_utc: str
    img_path: Path
    lines: int
    line_samples: int
    bands: int
    sample_bits: int
    sample_type: str


def _unquote(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    t = s.strip()
    if len(t) >= 2 and t[0] == '"' and t[-1] == '"':
        return t[1:-1]
    return t


def parse_lbl_scalar(path: Path, key: str) -> Optional[str]:
    for line in path.read_text(errors="ignore").splitlines():
        if line.lstrip().startswith("/*"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == key:
            return v.strip()
    return None


def load_frames(labels_glob: str, *, images_dir: Optional[Path] = None) -> List[Frame]:
    import sys

    tools_dir = Path(__file__).resolve().parent
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    try:
        from msl_mardi_inventory import parse_image_object, parse_pds3_lbl  # type: ignore
    except Exception as e:
        raise SystemExit(f"Failed to import msl_mardi_inventory helpers: {e}")

    frames: List[Frame] = []
    for lbl_path in sorted(Path().glob(labels_glob)):
        label = parse_pds3_lbl(lbl_path)
        img_obj = parse_image_object(lbl_path)

        product_id = _unquote(label.get("PRODUCT_ID")) or lbl_path.stem
        start_time = _unquote(label.get("START_TIME")) or ""
        ptr = label.get("^IMAGE") or label.get("^IMAGE_FILE") or ""
        img_name = None
        if ptr:
            import re

            m = re.search(r'"([^"]+\.(IMG|DAT))"', ptr, flags=re.IGNORECASE)
            if m:
                img_name = m.group(1)
        if not img_name:
            # EDR labels describe a COMPRESSED_FILE. The field name is "FILE_NAME",
            # but it may appear multiple times; our simple parser may capture the
            # UNCOMPRESSED_FILE's FILE_NAME instead. Prefer the actual payload if present.
            for candidate in [label.get("^MINIHEADER_TABLE"), label.get("REQUIRED_STORAGE_BYTES"), label.get("UNCOMPRESSED_FILE_NAME")]:
                _ = candidate  # ignored; retained for future improvement
            # Fallback: infer the compressed payload name from the label filename.
            if lbl_path.suffix.upper() == ".LBL" and lbl_path.name.upper().endswith(".LBL"):
                dat = lbl_path.with_suffix(".DAT").name
                if (images_dir / dat).exists():
                    img_name = dat
        if not img_name:
            # Default to IMG, but some EDR products store payloads as .DAT
            img_name = f"{product_id}.IMG"

        # Prefer canonical detached-label layout: payload next to label.
        img_path = lbl_path.parent / img_name
        if not img_path.exists():
            # Back-compat: allow split layout (labels/ + images/) via --images-dir.
            if images_dir is None:
                continue
            img_path = images_dir / img_name
            if not img_path.exists():
                continue

        lines = int(img_obj.get("LINES") or 0)
        line_samples = int(img_obj.get("LINE_SAMPLES") or 0)
        bands = int(img_obj.get("BANDS") or 1)
        sample_bits = int(img_obj.get("SAMPLE_BITS") or 8)
        sample_type = (img_obj.get("SAMPLE_TYPE") or "UNSIGNED_INTEGER").strip().strip('"')
        if lines <= 0 or line_samples <= 0:
            continue

        frames.append(
            Frame(
                product_id=product_id,
                start_time_utc=start_time,
                img_path=img_path,
                lines=lines,
                line_samples=line_samples,
                bands=bands,
                sample_bits=sample_bits,
                sample_type=sample_type,
            )
        )

    frames.sort(key=lambda f: f.start_time_utc)
    return frames


def main() -> int:
    ap = argparse.ArgumentParser(description="Quick interactive viewer for MSL/MARDI detached-label IMG frames.")
    ap.add_argument("--labels", default="data/msl/rdr/*.LBL", help="Glob for PDS3 labels.")
    ap.add_argument(
        "--images-dir",
        default="",
        help="Optional: directory containing IMG payloads when labels are in a separate directory.",
    )
    ap.add_argument("--stride", type=int, default=1, help="Show every Nth frame (default: 1).")
    ap.add_argument("--start", type=int, default=0, help="Start frame index (after stride).")
    ap.add_argument("--fps", type=float, default=8.0, help="Playback FPS when running (default: 8).")
    ap.add_argument(
        "--demosaic",
        choices=["off", "auto", "infer", "rggb", "bggr", "grbg", "gbrg"],
        default="auto",
        help=(
            "If the payload is 1-band Bayer mosaic, debayer to RGB for viewing. "
            "`auto` uses the MSL MMM SIS Bayer layout (top-left pixel is Red: RGGB). "
            "Use `infer` to guess from image content. Use `off` for raw mosaic."
        ),
    )
    ap.add_argument(
        "--cmap",
        default="gray",
        help="Matplotlib colormap for 1-band display when not demosaicing (default: gray).",
    )
    ap.add_argument(
        "--autocrop",
        action="store_true",
        help="Auto-crop uniform/invalid borders (useful for black frame edges).",
    )
    ap.add_argument(
        "--crop-photoactive",
        action="store_true",
        help=(
            "Crop known MMM 'dark columns' for full-frame 1648-wide products "
            "(keeps columns 24–1631 per the MMM DPSIS; 0-based slice [23:1631])."
        ),
    )
    ap.add_argument(
        "--crop-threshold",
        type=float,
        default=1.0,
        help="Threshold used by --autocrop (default: 1.0 for 8-bit imagery).",
    )
    ap.add_argument(
        "--mask-invalid",
        action="store_true",
        help="Replace pixels equal to INVALID_CONSTANT/MISSING_CONSTANT (typically 255) with 0 for display.",
    )
    ap.add_argument(
        "--autocontrast",
        action="store_true",
        help="Apply a simple percentile stretch per frame for display (helps when most pixels are near-black).",
    )
    ap.add_argument("--export-csv", default=None, help="Optional: write the resolved frame list to CSV.")
    args = ap.parse_args()

    try:
        import numpy as np
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit(f"Missing dependencies. Install `numpy matplotlib`. Import error: {e}")

    images_dir = Path(args.images_dir) if args.images_dir else None
    frames = load_frames(args.labels, images_dir=images_dir)
    if not frames:
        hint = f" with images under {images_dir}" if images_dir is not None else ""
        raise SystemExit(f"No frames found for {args.labels}{hint}")

    s = max(int(args.stride), 1)
    frames = frames[::s]

    if args.export_csv:
        outp = Path(args.export_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["index", "product_id", "start_time_utc", "img_path", "lines", "line_samples", "bands"],
            )
            w.writeheader()
            for i, fr in enumerate(frames):
                w.writerow(
                    {
                        "index": i,
                        "product_id": fr.product_id,
                        "start_time_utc": fr.start_time_utc,
                        "img_path": str(fr.img_path),
                        "lines": fr.lines,
                        "line_samples": fr.line_samples,
                        "bands": fr.bands,
                    }
                )

    idx = max(int(args.start), 0)
    idx = min(idx, len(frames) - 1)
    playing = True
    interval_s = 1.0 / max(float(args.fps), 0.1)

    fig, ax = plt.subplots()
    im = None
    inferred_bayer: Optional[str] = None

    def demosaic_bilinear(m: "np.ndarray", pattern: str) -> "np.ndarray":
        """
        Very small, dependency-free Bayer demosaic for quick visualization.
        Pattern is one of: rggb, bggr, grbg, gbrg and refers to the 2x2 tile at (row0,col0).
        """
        if m.ndim != 2:
            raise ValueError("demosaic expects a single-channel mosaic image")
        if pattern not in {"rggb", "bggr", "grbg", "gbrg"}:
            raise ValueError(f"Unsupported Bayer pattern: {pattern}")

        # Work in float32 for filtering; convert back at the end.
        in_dtype = m.dtype
        mf = m.astype("float32", copy=False)

        # Pad by 1 pixel so we can use simple neighbor averaging at borders.
        p = np.pad(mf, ((1, 1), (1, 1)), mode="edge")
        c = p[1:-1, 1:-1]
        up = p[0:-2, 1:-1]
        dn = p[2:, 1:-1]
        lf = p[1:-1, 0:-2]
        rt = p[1:-1, 2:]
        ul = p[0:-2, 0:-2]
        ur = p[0:-2, 2:]
        dl = p[2:, 0:-2]
        dr = p[2:, 2:]

        # Common interpolants.
        interp_hv = 0.25 * (up + dn + lf + rt)
        interp_diag = 0.25 * (ul + ur + dl + dr)
        interp_h = 0.5 * (lf + rt)
        interp_v = 0.5 * (up + dn)

        h, w = mf.shape
        rr = np.arange(h)[:, None]
        cc = np.arange(w)[None, :]

        # Build masks for R/G/B sites in the mosaic.
        if pattern == "rggb":
            is_r = (rr % 2 == 0) & (cc % 2 == 0)
            is_g1 = (rr % 2 == 0) & (cc % 2 == 1)
            is_g2 = (rr % 2 == 1) & (cc % 2 == 0)
            is_b = (rr % 2 == 1) & (cc % 2 == 1)
        elif pattern == "bggr":
            is_b = (rr % 2 == 0) & (cc % 2 == 0)
            is_g1 = (rr % 2 == 0) & (cc % 2 == 1)
            is_g2 = (rr % 2 == 1) & (cc % 2 == 0)
            is_r = (rr % 2 == 1) & (cc % 2 == 1)
        elif pattern == "grbg":
            is_g1 = (rr % 2 == 0) & (cc % 2 == 0)
            is_r = (rr % 2 == 0) & (cc % 2 == 1)
            is_b = (rr % 2 == 1) & (cc % 2 == 0)
            is_g2 = (rr % 2 == 1) & (cc % 2 == 1)
        else:  # gbrg
            is_g1 = (rr % 2 == 0) & (cc % 2 == 0)
            is_b = (rr % 2 == 0) & (cc % 2 == 1)
            is_r = (rr % 2 == 1) & (cc % 2 == 0)
            is_g2 = (rr % 2 == 1) & (cc % 2 == 1)

        # Green at R/B sites: interpolate from 4-neighborhood.
        g = np.where(is_g1 | is_g2, c, interp_hv)

        # Red channel:
        # - At R sites: known
        # - At G sites: interpolate from horizontal/vertical depending on stripe orientation
        # - At B sites: interpolate from diagonal
        r = np.empty_like(c)
        r[is_r] = c[is_r]
        # For green sites, which direction contains red depends on pattern parity.
        # Determine orientation by checking where red is on even rows vs even cols.
        red_on_even_row = bool(is_r[0, :2].any())
        if pattern in {"rggb", "bggr"}:
            # G sites alternate between horizontal-red and vertical-red depending on row parity.
            r[is_g1] = interp_h[is_g1] if red_on_even_row else interp_v[is_g1]
            r[is_g2] = interp_v[is_g2] if red_on_even_row else interp_h[is_g2]
        else:
            # grbg/gbrg swap the G roles
            r[is_g1] = interp_h[is_g1] if pattern == "gbrg" else interp_v[is_g1]
            r[is_g2] = interp_v[is_g2] if pattern == "gbrg" else interp_h[is_g2]
        r[is_b] = interp_diag[is_b]

        # Blue channel: symmetric to red.
        b = np.empty_like(c)
        b[is_b] = c[is_b]
        if pattern in {"rggb", "bggr"}:
            b[is_g1] = interp_v[is_g1] if red_on_even_row else interp_h[is_g1]
            b[is_g2] = interp_h[is_g2] if red_on_even_row else interp_v[is_g2]
        else:
            b[is_g1] = interp_v[is_g1] if pattern == "gbrg" else interp_h[is_g1]
            b[is_g2] = interp_h[is_g2] if pattern == "gbrg" else interp_v[is_g2]
        b[is_r] = interp_diag[is_r]

        rgb = np.stack([r, g, b], axis=-1)
        if in_dtype == np.uint8:
            return np.clip(rgb + 0.5, 0, 255).astype(np.uint8)
        if in_dtype == np.dtype(">u2") or in_dtype == np.uint16:
            # Return float for display; preserve dynamic range.
            mx = float(np.max(rgb)) if rgb.size else 1.0
            return (rgb / mx) if mx > 0 else rgb
        return rgb

    def infer_bayer_pattern(m: "np.ndarray") -> str:
        """
        Heuristically choose the Bayer pattern that minimizes chroma artifacts.
        This is not guaranteed, but works well for natural scenes.
        """
        # Use a central crop to avoid borders/invalid masks; also downsample for speed.
        h, w = m.shape
        y0 = max(0, h // 2 - 256)
        y1 = min(h, h // 2 + 256)
        x0 = max(0, w // 2 - 256)
        x1 = min(w, w // 2 + 256)
        crop = m[y0:y1, x0:x1]
        if crop.shape[0] < 32 or crop.shape[1] < 32:
            crop = m
        # Downsample by 2 to reduce compute while preserving Bayer parity.
        crop = crop[0::2, 0::2]

        best_p = "rggb"
        best_score = float("inf")
        for p in ("rggb", "bggr", "grbg", "gbrg"):
            rgb = demosaic_bilinear(crop, p).astype("float32", copy=False)
            # Chroma channels; large spatial variation indicates wrong pattern.
            g = rgb[:, :, 1]
            cr = rgb[:, :, 0] - g
            cb = rgb[:, :, 2] - g
            # Edge energy (simple finite differences).
            score = (
                float(np.mean(np.abs(np.diff(cr, axis=0))))
                + float(np.mean(np.abs(np.diff(cr, axis=1))))
                + float(np.mean(np.abs(np.diff(cb, axis=0))))
                + float(np.mean(np.abs(np.diff(cb, axis=1))))
            )
            if score < best_score:
                best_score = score
                best_p = p
        return best_p

    def read_frame(fr: Frame):
        nonlocal inferred_bayer
        raw = fr.img_path.read_bytes()
        if fr.sample_bits == 8 and fr.sample_type.upper().startswith("UNSIGNED"):
            dtype = np.uint8
        elif fr.sample_bits == 16 and "MSB" in fr.sample_type.upper():
            dtype = ">u2"
        elif fr.sample_bits == 16 and fr.sample_type.upper().startswith("UNSIGNED"):
            dtype = "<u2"
        else:
            raise ValueError(f"Unsupported sample type/bits: {fr.sample_type} {fr.sample_bits}")

        arr = np.frombuffer(raw, dtype=np.dtype(dtype))
        expected = fr.lines * fr.line_samples * fr.bands
        if arr.size < expected:
            raise ValueError(f"{fr.img_path} too small: {arr.size} < {expected}")
        arr = arr[:expected]
        if fr.bands == 1:
            img = arr.reshape((fr.lines, fr.line_samples))
        else:
            # BAND_SEQUENTIAL: [band, line, sample] -> [line, sample, band]
            img = arr.reshape((fr.bands, fr.lines, fr.line_samples)).transpose(1, 2, 0)

        def _autocrop(im: "np.ndarray") -> "np.ndarray":
            thr = float(args.crop_threshold)
            if im.ndim == 2:
                mask = im.astype("float32") > thr
            else:
                mask = (im.astype("float32") > thr).any(axis=2)
            if not mask.any():
                return im
            ys = np.where(mask.any(axis=1))[0]
            xs = np.where(mask.any(axis=0))[0]
            y0, y1 = int(ys[0]), int(ys[-1] + 1)
            x0, x1 = int(xs[0]), int(xs[-1] + 1)
            return im[y0:y1, x0:x1]

        if args.mask_invalid:
            if fr.bands == 1:
                img = img.copy()
                img[img == 255] = 0
            else:
                img = img.copy()
                m = (img[:, :, 0] == 255) & (img[:, :, 1] == 255) & (img[:, :, 2] == 255)
                img[m] = 0

        if args.crop_photoactive and img.shape[1] == 1648:
            # MSL MMM DPSIS: dark columns 1-23 and 1632-1648, photoactive 24-1631.
            # Convert to 0-based half-open slice [23:1631].
            img = img[:, 23:1631] if img.ndim == 2 else img[:, 23:1631, :]

        if args.autocrop:
            img = _autocrop(img)

        if args.autocontrast:
            if fr.bands == 1:
                p1 = float(np.percentile(img, 1))
                p99 = float(np.percentile(img, 99))
                if p99 > p1:
                    img = np.clip((img.astype("float32") - p1) / (p99 - p1), 0.0, 1.0)
            else:
                fimg = img.astype("float32")
                for c in range(min(fr.bands, 3)):
                    ch = fimg[:, :, c]
                    p1 = float(np.percentile(ch, 1))
                    p99 = float(np.percentile(ch, 99))
                    if p99 > p1:
                        fimg[:, :, c] = (ch - p1) / (p99 - p1)
                img = np.clip(fimg, 0.0, 1.0)

        # Demosaic (EDR uncompressed products are typically 1-band Bayer mosaics).
        if fr.bands == 1 and args.demosaic != "off":
            pattern = args.demosaic
            if pattern == "auto":
                pattern = "rggb"
            elif pattern == "infer":
                if inferred_bayer is None:
                    inferred_bayer = infer_bayer_pattern(img)
                    print(f"[msl_mardi_view] inferred Bayer pattern: {inferred_bayer}")
                pattern = inferred_bayer
            img = demosaic_bilinear(img, pattern)
        return img

    def draw():
        nonlocal im
        fr = frames[idx]
        img = read_frame(fr)
        if im is None:
            if img.ndim == 2:
                im = ax.imshow(img, interpolation="nearest", cmap=args.cmap)
            else:
                im = ax.imshow(img, interpolation="nearest")
            ax.set_axis_off()
        else:
            im.set_data(img)
        ax.set_title(f"[{idx+1}/{len(frames)}] {fr.product_id}  {fr.start_time_utc}")
        fig.canvas.draw_idle()

    def on_key(event):
        nonlocal idx, playing
        if event.key in ("q", "escape"):
            plt.close(fig)
            return
        if event.key == " ":
            playing = not playing
            return
        if event.key in ("right", "down", "n"):
            idx = min(idx + 1, len(frames) - 1)
            draw()
            return
        if event.key in ("left", "up", "p"):
            idx = max(idx - 1, 0)
            draw()
            return
        if event.key == "home":
            idx = 0
            draw()
            return
        if event.key == "end":
            idx = len(frames) - 1
            draw()
            return

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw()

    while plt.fignum_exists(fig.number):
        if playing:
            if idx < len(frames) - 1:
                idx += 1
                draw()
            plt.pause(interval_s)
        else:
            plt.pause(0.05)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
