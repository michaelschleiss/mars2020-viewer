"""
Standardized metadata extraction for PDS3 EDL camera products.

This module centralizes label reading/parsing and produces a single metadata
structure that viewer code can consume.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from pds_label import (
    ODLNode,
    ODLValue,
    find_blocks,
    find_key_occurrences,
    parse_odl,
    read_detached_label_text,
    read_embedded_label_text,
    require_unique_value,
)

LabelSource = Literal["embedded_img", "detached_lbl"]

_LCAM_FRAME_RE = re.compile(r"_N(\d{7})LVS_", re.IGNORECASE)


@dataclass(frozen=True)
class FrameMetadata:
    mission: str
    camera: str
    filepath: Path
    filename: str

    label_source: LabelSource
    label_text: str
    label_tree: ODLNode

    # Canonical timing
    sclk_partition: Optional[int]
    sclk_start: Optional[float]
    sclk_stop: Optional[float]
    sclk_mid: Optional[float]
    start_time: Optional[str]
    stop_time: Optional[str]

    # Identity / provenance
    product_id: Optional[str]
    product_type: Optional[str]
    instrument_id: Optional[str]
    instrument_name: Optional[str]
    target_name: Optional[str]
    producer_id: Optional[str]
    processing_level: Optional[str]

    # Acquisition / encoding
    exposure_duration: Optional[float]
    exposure_type: Optional[str]
    filter_name: Optional[str]
    compression_type: Optional[str]
    encoding_type: Optional[str]

    # Layout essentials (for debugging decode)
    record_bytes: Optional[int]
    image_ptr: Optional[Any]
    lines: Optional[int]
    line_samples: Optional[int]
    bands: Optional[int]
    sample_bits: Optional[int]
    sample_type: Optional[str]

    # Derived ordering key (e.g. LCAM N#######)
    frame_index: Optional[int]


def _get_optional_unique(tree: ODLNode, key: str) -> Optional[ODLValue]:
    vals = find_key_occurrences(tree, key)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    first = vals[0]
    if all(v.value == first.value and v.unit == first.unit for v in vals[1:]):
        return first
    raise ValueError(f"Expected unique/consistent {key} when present, found {len(vals)} differing values")


def _get_optional_unique_in(tree: ODLNode, key: str, within: tuple[str, str]) -> Optional[ODLValue]:
    kind, name = within
    blocks = find_blocks(tree, kind=kind, name=name)
    if not blocks:
        return None
    if len(blocks) != 1:
        raise ValueError(f"Expected 1 {kind}={name} block, found {len(blocks)}")
    vals = blocks[0].keywords.get(key.upper(), [])
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    first = vals[0]
    if all(v.value == first.value and v.unit == first.unit for v in vals[1:]):
        return first
    raise ValueError(f"Expected unique/consistent {key} within {kind}={name}, found {len(vals)} differing values")


def _as_str(v: Optional[ODLValue]) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v.value, str):
        return v.value
    return str(v.value)


def _as_int(v: Optional[ODLValue]) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v.value, int):
        return v.value
    try:
        return int(str(v.value))
    except Exception:
        return None


def _as_float(v: Optional[ODLValue]) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v.value, float):
        return v.value
    if isinstance(v.value, int):
        return float(v.value)
    try:
        return float(str(v.value))
    except Exception:
        return None


def _derive_frame_index(camera: str, filename: str) -> Optional[int]:
    if camera.lower() != "lcam":
        return None
    m = _LCAM_FRAME_RE.search(filename)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def load_frame_metadata(img_path: str | Path, *, mission: str, camera: str) -> FrameMetadata:
    img_path = Path(img_path)
    filename = img_path.name

    lbl_path = img_path.with_suffix(".LBL")
    lbl_path_lower = img_path.with_suffix(".lbl")
    if lbl_path.exists() or lbl_path_lower.exists():
        label_source: LabelSource = "detached_lbl"
        label_text = read_detached_label_text(lbl_path if lbl_path.exists() else lbl_path_lower)
    else:
        label_source = "embedded_img"
        label_text = read_embedded_label_text(img_path)

    tree = parse_odl(label_text)

    # Canonical timing for sorting/pose: fail fast if ambiguous.
    sclk_partition = _as_int(_get_optional_unique(tree, "SPACECRAFT_CLOCK_CNT_PARTITION"))
    try:
        sclk_start = _as_float(_get_optional_unique(tree, "SPACECRAFT_CLOCK_START_COUNT"))
    except Exception as e:
        raise ValueError(f"{img_path}: missing/ambiguous SPACECRAFT_CLOCK_START_COUNT ({e})") from e
    try:
        sclk_stop = _as_float(_get_optional_unique(tree, "SPACECRAFT_CLOCK_STOP_COUNT"))
    except Exception as e:
        raise ValueError(f"{img_path}: missing/ambiguous SPACECRAFT_CLOCK_STOP_COUNT ({e})") from e
    sclk_mid = _as_float(_get_optional_unique(tree, "SPACECRAFT_CLOCK_MID_COUNT"))

    if sclk_mid is None and sclk_start is not None and sclk_stop is not None:
        sclk_mid = (sclk_start + sclk_stop) / 2.0

    start_time = _as_str(_get_optional_unique(tree, "START_TIME"))
    stop_time = _as_str(_get_optional_unique(tree, "STOP_TIME"))

    # Optional identity/provenance.
    product_id = _as_str(_get_optional_unique(tree, "PRODUCT_ID"))
    product_type = _as_str(_get_optional_unique(tree, "PRODUCT_TYPE"))
    instrument_id = _as_str(_get_optional_unique(tree, "INSTRUMENT_ID"))
    instrument_name = _as_str(_get_optional_unique(tree, "INSTRUMENT_NAME"))
    target_name = _as_str(_get_optional_unique(tree, "TARGET_NAME"))
    producer_id = _as_str(_get_optional_unique(tree, "PRODUCER_ID"))
    processing_level = _as_str(_get_optional_unique(tree, "PROCESSING_LEVEL_ID"))

    exposure_duration = _as_float(_get_optional_unique(tree, "EXPOSURE_DURATION"))
    exposure_type = _as_str(_get_optional_unique(tree, "EXPOSURE_TYPE"))
    filter_name = _as_str(_get_optional_unique(tree, "FILTER_NAME"))
    compression_type = _as_str(_get_optional_unique(tree, "COMPRESSION_TYPE"))
    encoding_type = _as_str(_get_optional_unique(tree, "ENCODING_TYPE"))

    # Layout essentials.
    record_bytes = _as_int(_get_optional_unique(tree, "RECORD_BYTES"))
    image_ptr = _get_optional_unique(tree, "^IMAGE")
    image_ptr_val: Optional[Any] = image_ptr.value if image_ptr is not None else None
    lines = _as_int(_get_optional_unique_in(tree, "LINES", ("OBJECT", "IMAGE")))
    line_samples = _as_int(_get_optional_unique_in(tree, "LINE_SAMPLES", ("OBJECT", "IMAGE")))
    bands = _as_int(_get_optional_unique_in(tree, "BANDS", ("OBJECT", "IMAGE")))
    sample_bits = _as_int(_get_optional_unique_in(tree, "SAMPLE_BITS", ("OBJECT", "IMAGE")))
    sample_type = _as_str(_get_optional_unique_in(tree, "SAMPLE_TYPE", ("OBJECT", "IMAGE")))

    frame_index = _derive_frame_index(camera, filename)

    return FrameMetadata(
        mission=mission,
        camera=camera,
        filepath=img_path,
        filename=filename,
        label_source=label_source,
        label_text=label_text,
        label_tree=tree,
        sclk_partition=sclk_partition,
        sclk_start=sclk_start,
        sclk_stop=sclk_stop,
        sclk_mid=sclk_mid,
        start_time=start_time,
        stop_time=stop_time,
        product_id=product_id,
        product_type=product_type,
        instrument_id=instrument_id,
        instrument_name=instrument_name,
        target_name=target_name,
        producer_id=producer_id,
        processing_level=processing_level,
        exposure_duration=exposure_duration,
        exposure_type=exposure_type,
        filter_name=filter_name,
        compression_type=compression_type,
        encoding_type=encoding_type,
        record_bytes=record_bytes,
        image_ptr=image_ptr_val,
        lines=lines,
        line_samples=line_samples,
        bands=bands,
        sample_bits=sample_bits,
        sample_type=sample_type,
        frame_index=frame_index,
    )
