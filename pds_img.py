"""PDS3 image reader using pdr library.

Provides backward-compatible functions for reading Mars 2020 EDL camera products.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pdr

_INLINE_KV_RE = re.compile(r"(\w+)=([^=]+?)(?=\s+\w+=|$)")

# Fast regex patterns for lightweight label parsing
# Note: Use [0-9] instead of \d for Python 3.12+ compatibility with raw byte strings
_RECORD_BYTES_RE = re.compile(rb"RECORD_BYTES\s*=\s*([0-9]+)")
_IMAGE_PTR_RE = re.compile(rb"\^IMAGE\s*=\s*([0-9]+)")
# For LINES/LINE_SAMPLES/BANDS/SAMPLE_BITS, find ALL numeric values and use the last one
# (IMAGE object values come after metadata "NULL" values)
_LINES_RE = re.compile(rb"LINES\s*=\s*([0-9]+)")
_LINE_SAMPLES_RE = re.compile(rb"LINE_SAMPLES\s*=\s*([0-9]+)")
_BANDS_RE = re.compile(rb"BANDS\s*=\s*([0-9]+)")
_SAMPLE_BITS_RE = re.compile(rb"SAMPLE_BITS\s*=\s*([0-9]+)")


@dataclass(frozen=True)
class Pds3ImageLayout:
    """Image layout information extracted from PDS3 label."""
    record_bytes: int
    label_records: int
    image_record: int
    lines: int
    line_samples: int
    bands: int
    sample_bits: int
    sample_type: str

    @property
    def image_offset_bytes(self) -> int:
        return (self.image_record - 1) * self.record_bytes

    @property
    def image_size_bytes(self) -> int:
        pixels = self.lines * self.line_samples * self.bands
        return pixels * (self.sample_bits // 8)


def infer_pds3_layout_fast(path: str | Path) -> Pds3ImageLayout:
    """Fast layout extraction reading only the label header.

    ~100x faster than infer_pds3_layout() for video playback.
    Does not return full label dict - use infer_pds3_layout() when metadata needed.
    Handles both embedded and detached (.LBL) labels.
    """
    path = Path(path)

    # Check for detached label (.LBL file)
    lbl_path = path.with_suffix(".LBL")
    if lbl_path.exists():
        with open(lbl_path, "rb") as f:
            # Read full detached label (MSL MARDI labels are ~22KB)
            header = f.read()
    else:
        with open(path, "rb") as f:
            # Read first 32KB - sufficient for most embedded PDS3 labels
            header = f.read(32768)

    def extract_first(pattern: re.Pattern, default: int = 0) -> int:
        m = pattern.search(header)
        return int(m.group(1)) if m else default

    def extract_last(pattern: re.Pattern, default: int = 0) -> int:
        # For IMAGE object values, take the last match (after metadata "NULL" values)
        matches = pattern.findall(header)
        return int(matches[-1]) if matches else default

    record_bytes = extract_first(_RECORD_BYTES_RE)
    image_record = extract_first(_IMAGE_PTR_RE, 1)  # Default to 1 for detached

    return Pds3ImageLayout(
        record_bytes=record_bytes,
        label_records=0,  # Not needed for reading
        image_record=image_record,
        lines=extract_last(_LINES_RE),
        line_samples=extract_last(_LINE_SAMPLES_RE),
        bands=extract_last(_BANDS_RE, 1),
        sample_bits=extract_last(_SAMPLE_BITS_RE),
        sample_type="",  # Not needed for reading
    )


def infer_pds3_layout(path: str | Path) -> Tuple[Pds3ImageLayout, Dict[str, str]]:
    """Extract image layout and label from a PDS3 file."""
    data = pdr.read(str(path))
    meta = data.metadata

    def get_int(key: str, default: int = 0) -> int:
        val = meta.get(key)
        if val is None:
            return default
        # Handle detached labels where ^IMAGE points to filename
        if isinstance(val, str) and not val.isdigit():
            return default
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    # IMAGE properties are nested under 'IMAGE' group
    img_meta = meta.get("IMAGE", {})

    def get_img_int(key: str, default: int = 0) -> int:
        val = img_meta.get(key) if img_meta else None
        return int(val) if val is not None else default

    # For detached labels, ^IMAGE contains filename - default to record 1
    image_record_val = meta.get("^IMAGE")
    if isinstance(image_record_val, str) and not image_record_val.isdigit():
        image_record = 1  # Image starts at byte 0 in detached file
    else:
        image_record = get_int("^IMAGE")

    layout = Pds3ImageLayout(
        record_bytes=get_int("RECORD_BYTES"),
        label_records=get_int("LABEL_RECORDS"),
        image_record=image_record,
        lines=get_img_int("LINES"),
        line_samples=get_img_int("LINE_SAMPLES"),
        bands=get_img_int("BANDS", 1),
        sample_bits=get_img_int("SAMPLE_BITS"),
        sample_type=str(img_meta.get("SAMPLE_TYPE", "") if img_meta else ""),
    )

    # Convert metadata to flat dict (top-level only)
    label = {str(k): str(v) for k, v in meta.items() if not isinstance(v, dict)}

    return layout, label


def read_image_bytes(path: str | Path) -> Tuple[bytes, Pds3ImageLayout, Dict[str, str]]:
    """Read raw image bytes from a PDS3 file.

    Returns (image_bytes, layout, label).
    """
    data = pdr.read(str(path))
    layout, label = infer_pds3_layout(path)

    # Preserve the native dtype to avoid truncating >8-bit products.
    image_array = np.asarray(data["IMAGE"])
    image_bytes = image_array.tobytes()

    return image_bytes, layout, label


def read_image_header_text(path: str | Path) -> Tuple[str, Dict[str, str]]:
    """Read IMAGE_HEADER object containing pose and CAHVORE data.

    Returns (raw_text, parsed_key_value_dict).
    """
    data = pdr.read(str(path))

    if "IMAGE_HEADER" not in data.keys():
        raise ValueError(f"{path} has no IMAGE_HEADER object")

    text = data["IMAGE_HEADER"]

    # Parse inline KEY=VALUE pairs
    items = _INLINE_KV_RE.findall(text)
    parsed = {k.strip(): v.split("\x00", 1)[0].strip().strip("'") for k, v in items}

    return text, parsed


def image_stats_u8(image_bytes: bytes) -> Dict[str, Any]:
    """Compute basic statistics for 8-bit image data."""
    if not image_bytes:
        return {"min": None, "max": None, "mean": None}

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    return {
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
    }
