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


def infer_pds3_layout(path: str | Path) -> Tuple[Pds3ImageLayout, Dict[str, str]]:
    """Extract image layout and label from a PDS3 file."""
    data = pdr.read(str(path))
    meta = data.metadata

    def get_int(key: str, default: int = 0) -> int:
        val = meta.get(key)
        return int(val) if val is not None else default

    # IMAGE properties are nested under 'IMAGE' group
    img_meta = meta.get("IMAGE", {})

    def get_img_int(key: str, default: int = 0) -> int:
        val = img_meta.get(key) if img_meta else None
        return int(val) if val is not None else default

    layout = Pds3ImageLayout(
        record_bytes=get_int("RECORD_BYTES"),
        label_records=get_int("LABEL_RECORDS"),
        image_record=get_int("^IMAGE"),
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

    # Convert numpy array to bytes
    image_array = data["IMAGE"]
    image_bytes = image_array.astype(np.uint8).tobytes()

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
