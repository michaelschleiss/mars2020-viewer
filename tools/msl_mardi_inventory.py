#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


_RE_KEYVAL = re.compile(r"^\s*([A-Z0-9_:^]+)\s*=\s*(.+?)\s*$")
_RE_OBJECT_IMAGE = re.compile(r"^\s*OBJECT\s*=\s*IMAGE\s*$", re.IGNORECASE)
_RE_END_OBJECT_IMAGE = re.compile(r"^\s*END_OBJECT\s*=\s*IMAGE\s*$", re.IGNORECASE)


def _strip_comments(line: str) -> str:
    if "/*" in line:
        return line.split("/*", 1)[0].rstrip()
    return line.rstrip()


def _collect_multiline_value(first: str, lines: List[str], i: int) -> Tuple[str, int]:
    """
    Collect multi-line tuple or quoted string values.

    Returns (value_text, next_index).
    """
    text = first.strip()
    if text.count('"') % 2 == 1:
        parts = [text]
        i += 1
        while i < len(lines):
            s = _strip_comments(lines[i]).strip()
            parts.append(s)
            if s.count('"') % 2 == 1:
                i += 1
                break
            i += 1
        return " ".join(parts).strip(), i

    if "(" in text and text.count("(") > text.count(")"):
        parts = [text]
        i += 1
        while i < len(lines):
            s = _strip_comments(lines[i]).strip()
            parts.append(s)
            joined = " ".join(parts)
            if joined.count(")") >= joined.count("("):
                i += 1
                break
            if ")" in s:
                i += 1
                break
            i += 1
        return " ".join(parts).strip(), i

    return text, i + 1


def parse_pds3_lbl(path: Path) -> Dict[str, str]:
    lines = path.read_text(errors="ignore").splitlines()
    out: Dict[str, str] = {}
    i = 0
    while i < len(lines):
        s = _strip_comments(lines[i]).strip()
        if not s:
            i += 1
            continue
        m = _RE_KEYVAL.match(s)
        if not m:
            i += 1
            continue
        key = m.group(1).strip()
        rest = m.group(2).strip()
        val, i = _collect_multiline_value(rest, lines, i)
        out[key] = val
    return out


def _unquote(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    t = s.strip()
    if len(t) >= 2 and t[0] == '"' and t[-1] == '"':
        return t[1:-1]
    return t


def parse_image_object(path: Path) -> Dict[str, str]:
    lines = path.read_text(errors="ignore").splitlines()
    in_img = False
    img: Dict[str, str] = {}
    i = 0
    while i < len(lines):
        s = _strip_comments(lines[i]).strip()
        if not s:
            i += 1
            continue
        if not in_img and _RE_OBJECT_IMAGE.match(s):
            in_img = True
            i += 1
            continue
        if in_img and _RE_END_OBJECT_IMAGE.match(s):
            break
        if not in_img:
            i += 1
            continue
        m = _RE_KEYVAL.match(s)
        if not m:
            i += 1
            continue
        key = m.group(1).strip()
        rest = m.group(2).strip()
        val, i = _collect_multiline_value(rest, lines, i)
        img[key] = val
    return img


def _parse_int(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    v = value.strip().strip('"')
    try:
        return int(v)
    except Exception:
        return None


@dataclass(frozen=True)
class MardiProduct:
    product_id: str
    lbl_path: str
    img_path: str
    start_time_utc: Optional[str]
    sclk_start: Optional[str]
    record_bytes: Optional[int]
    file_records: Optional[int]
    lines: Optional[int]
    line_samples: Optional[int]
    sample_bits: Optional[int]
    sample_type: Optional[str]
    bands: Optional[int]
    model_type: Optional[str]

    @property
    def expected_img_bytes(self) -> Optional[int]:
        if self.record_bytes is None or self.file_records is None:
            return None
        return int(self.record_bytes) * int(self.file_records)


def main() -> int:
    ap = argparse.ArgumentParser(description="Inventory local MSL/MARDI PDS3 detached-label products.")
    ap.add_argument(
        "--dir",
        default="data/msl/rdr",
        help="Directory containing paired *.LBL/*.IMG files (preferred layout).",
    )
    ap.add_argument(
        "--labels-dir",
        default=None,
        help="Back-compat: directory containing *.LBL labels if stored separately.",
    )
    ap.add_argument(
        "--images-dir",
        default=None,
        help="Back-compat: directory containing *.IMG images if stored separately.",
    )
    ap.add_argument("--out", default="out/msl_mardi_inventory.json", help="Write JSON summary to this path.")
    args = ap.parse_args()

    labels_dir = Path(args.labels_dir) if args.labels_dir else Path(args.dir)
    images_dir = Path(args.images_dir) if args.images_dir else labels_dir
    if not labels_dir.exists():
        raise SystemExit(f"labels dir not found: {labels_dir}")
    if not images_dir.exists():
        raise SystemExit(f"images dir not found: {images_dir}")

    products: List[MardiProduct] = []
    missing_imgs: List[str] = []

    for lbl in sorted(labels_dir.glob("*.LBL")):
        label = parse_pds3_lbl(lbl)
        img_obj = parse_image_object(lbl)
        ptr = label.get("^IMAGE")
        img_name = None
        if ptr:
            m = re.search(r'"([^"]+\\.IMG)"', ptr, flags=re.IGNORECASE)
            if m:
                img_name = m.group(1)
            else:
                m2 = re.search(r"\(?\s*([^\s\)]+\.IMG)\s*\)?", ptr, flags=re.IGNORECASE)
                if m2:
                    img_name = m2.group(1).strip().strip('"')
        if not img_name:
            continue
        img_path = lbl.parent / img_name
        if not img_path.exists():
            img_path = images_dir / img_name
            if not img_path.exists():
                missing_imgs.append(img_name)
                continue

        products.append(
            MardiProduct(
                product_id=_unquote(label.get("PRODUCT_ID")) or img_path.stem,
                lbl_path=str(lbl),
                img_path=str(img_path),
                start_time_utc=_unquote(label.get("START_TIME")),
                sclk_start=_unquote(label.get("SPACECRAFT_CLOCK_START_COUNT")),
                record_bytes=_parse_int(label.get("RECORD_BYTES")),
                file_records=_parse_int(label.get("FILE_RECORDS")),
                lines=_parse_int(img_obj.get("LINES")),
                line_samples=_parse_int(img_obj.get("LINE_SAMPLES")),
                sample_bits=_parse_int(img_obj.get("SAMPLE_BITS")),
                sample_type=_unquote(img_obj.get("SAMPLE_TYPE")),
                bands=_parse_int(img_obj.get("BANDS")),
                model_type=_unquote(label.get("MODEL_TYPE")),
            )
        )

    summary = {
        "count_labels": len(list(labels_dir.glob("*.LBL"))),
        "count_products_with_images": len(products),
        "missing_images": sorted(set(missing_imgs)),
        "products": [asdict(p) for p in products],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({k: summary[k] for k in ["count_labels", "count_products_with_images"]}, indent=2))
    if summary["missing_images"]:
        print(f"missing_images: {len(summary['missing_images'])} (examples: {summary['missing_images'][:5]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
