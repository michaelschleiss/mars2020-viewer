"""
PDS3 image and metadata handling for Mars EDL camera products.

Provides:
- Fast regex-based layout extraction for video playback (infer_pds3_layout_fast)
- Full ODL label parsing and metadata extraction (load_frame_metadata)
- Image data reading via pdr library (read_image_bytes, read_image_header_text)

Combines functionality from former pds_label, pds_metadata, and pds_img modules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional, Tuple

import numpy as np
import pdr

# ============================================================================
# Regex Patterns
# ============================================================================

# Inline KEY=VALUE parsing for IMAGE_HEADER objects
_INLINE_KV_RE = re.compile(r"(\w+)=([^=]+?)(?=\s+\w+=|$)")

# Fast regex patterns for lightweight label parsing (binary mode)
# Note: Use [0-9] instead of \d for Python 3.12+ compatibility with raw byte strings.
# IMPORTANT: Anchor to start-of-line to avoid matching substrings like DETECTOR_LINES.
_RECORD_BYTES_RE = re.compile(rb"^\s*RECORD_BYTES\s*=\s*([0-9]+)\s*$", re.MULTILINE)
_IMAGE_PTR_RE = re.compile(rb"^\s*\^IMAGE\s*=\s*([0-9]+)\s*$", re.MULTILINE)
# For LINES/LINE_SAMPLES/BANDS/SAMPLE_BITS, find ALL numeric values and use the last one
# (IMAGE object values come after metadata "NULL" values).
_LINES_RE = re.compile(rb"^\s*LINES\s*=\s*([0-9]+)\s*$", re.MULTILINE)
_LINE_SAMPLES_RE = re.compile(rb"^\s*LINE_SAMPLES\s*=\s*([0-9]+)\s*$", re.MULTILINE)
_BANDS_RE = re.compile(rb"^\s*BANDS\s*=\s*([0-9]+)\s*$", re.MULTILINE)
_SAMPLE_BITS_RE = re.compile(rb"^\s*SAMPLE_BITS\s*=\s*([0-9]+)\s*$", re.MULTILINE)

# ODL parsing patterns (text mode)
_ASSIGN_RE = re.compile(r"^\s*([A-Za-z0-9_^\-]+)\s*=\s*(.*?)\s*$")
_BLOCK_START_RE = re.compile(r"^\s*(OBJECT|GROUP)\s*=\s*([A-Za-z0-9_^\-]+)\s*$", re.IGNORECASE)
_BLOCK_END_RE = re.compile(r"^\s*END_(OBJECT|GROUP)\s*=\s*([A-Za-z0-9_^\-]+)\s*$", re.IGNORECASE)
_END_RE = re.compile(r"^\s*END\s*$", re.IGNORECASE | re.MULTILINE)

# Frame index extraction (LCAM)
_LCAM_FRAME_RE = re.compile(r"_N(\d{7})LVS_", re.IGNORECASE)

# ============================================================================
# Dataclasses
# ============================================================================

@dataclass(frozen=True)
class ODLValue:
    """A parsed ODL value with optional unit."""
    raw: str
    value: Any
    unit: Optional[str] = None


@dataclass
class ODLNode:
    """A node in the ODL tree (ROOT, OBJECT, or GROUP)."""
    kind: str  # ROOT | OBJECT | GROUP
    name: Optional[str] = None
    keywords: dict[str, list[ODLValue]] = None
    children: list["ODLNode"] = None

    def __post_init__(self) -> None:
        if self.keywords is None:
            self.keywords = {}
        if self.children is None:
            self.children = []


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


LabelSource = Literal["embedded_img", "detached_lbl"]


@dataclass(frozen=True)
class FrameMetadata:
    """Standardized metadata for PDS3 EDL camera products."""
    mission: str
    camera: str
    filepath: Path
    filename: str

    label_source: LabelSource
    label_text: str
    label_tree: ODLNode

    # Canonical timing
    sclk_partition: Optional[int]
    sclk_start_ticks: Optional[int]
    sclk_stop_ticks: Optional[int]
    sclk_start_raw: Optional[str]
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


# ============================================================================
# Label I/O Functions
# ============================================================================

def read_embedded_label_text(img_path: str | Path, *, max_bytes: int = 2_000_000) -> str:
    """
    Read the embedded PDS3 label from the start of a .IMG file (up to END).
    """
    img_path = Path(img_path)
    chunk = 65536
    data = bytearray()
    with open(img_path, "rb") as f:
        while len(data) < max_bytes:
            part = f.read(chunk)
            if not part:
                break
            data.extend(part)
            text = data.decode("ascii", errors="ignore")
            if _END_RE.search(text):
                # Keep only through END line to avoid decoding binary payload.
                lines = text.splitlines()
                out: list[str] = []
                for line in lines:
                    out.append(line)
                    if _END_RE.match(line):
                        return "\n".join(out) + "\n"
    raise ValueError(f"Failed to locate END in embedded label: {img_path}")


def read_detached_label_text(lbl_path: str | Path) -> str:
    """Read a detached PDS3 label from a .LBL file."""
    lbl_path = Path(lbl_path)
    return lbl_path.read_text(encoding="ascii", errors="ignore")


# ============================================================================
# ODL Parser
# ============================================================================

def _strip_c_style_comments(lines: Iterable[str]) -> list[str]:
    """Remove C-style /* */ comments from ODL text."""
    out: list[str] = []
    in_comment = False
    for line in lines:
        s = line
        while True:
            if in_comment:
                end = s.find("*/")
                if end == -1:
                    s = ""
                    break
                s = s[end + 2 :]
                in_comment = False
                continue
            start = s.find("/*")
            if start == -1:
                break
            end = s.find("*/", start + 2)
            if end == -1:
                s = s[:start]
                in_comment = True
                break
            s = s[:start] + s[end + 2 :]
        out.append(s)
    return out


def _split_unit(raw: str) -> tuple[str, Optional[str]]:
    """Split a value string from its unit (e.g., '123 <km>' -> ('123', 'km'))."""
    s = raw.strip()
    m = re.match(r"^(.*?)(?:\s*<\s*([^>]+)\s*>\s*)$", s)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return s, None


def _parse_scalar(raw: str) -> Any:
    """Parse a scalar ODL value (string, int, or float)."""
    s = raw.strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        return s[1:-1]
    if re.fullmatch(r"[+-]?[0-9]+", s):
        try:
            return int(s)
        except ValueError:
            return s
    if re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[Ee][+-]?[0-9]+)?", s):
        try:
            return float(s)
        except ValueError:
            return s
    return s


def _parse_value(raw: str) -> ODLValue:
    """Parse an ODL value (scalar, tuple, or set)."""
    val_s, unit = _split_unit(raw)
    s = val_s.strip()
    if s.startswith("(") and s.endswith(")"):
        inner = s[1:-1].strip()
        if not inner:
            return ODLValue(raw=raw, value=tuple(), unit=unit)
        parts = [p.strip() for p in inner.split(",")]
        return ODLValue(raw=raw, value=tuple(_parse_scalar(p) for p in parts), unit=unit)
    if s.startswith("{") and s.endswith("}"):
        inner = s[1:-1].strip()
        if not inner:
            return ODLValue(raw=raw, value=[], unit=unit)
        parts = [p.strip() for p in inner.split(",")]
        return ODLValue(raw=raw, value=[_parse_scalar(p) for p in parts], unit=unit)
    return ODLValue(raw=raw, value=_parse_scalar(s), unit=unit)


def parse_odl(label_text: str) -> ODLNode:
    """
    Parse a PDS3 ODL label into a tree preserving GROUP/OBJECT structure and repeats.
    """
    lines = _strip_c_style_comments(label_text.splitlines())
    root = ODLNode(kind="ROOT", name=None)
    stack: list[ODLNode] = [root]

    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\r\n")
        i += 1
        if not line.strip():
            continue
        if _END_RE.match(line):
            break

        m = _BLOCK_START_RE.match(line)
        if m:
            kind, name = m.group(1).upper(), m.group(2)
            node = ODLNode(kind=kind, name=name)
            stack[-1].children.append(node)
            stack.append(node)
            continue

        m = _BLOCK_END_RE.match(line)
        if m:
            kind = m.group(1).upper()
            if len(stack) > 1 and stack[-1].kind == kind:
                stack.pop()
            continue

        m = _ASSIGN_RE.match(line)
        if not m:
            continue
        key, value = m.group(1).strip().upper(), m.group(2).strip()

        # Handle multiline values (tuple/list spanning lines).
        def balance(s: str) -> tuple[int, int]:
            paren = s.count("(") - s.count(")")
            brace = s.count("{") - s.count("}")
            return paren, brace

        paren, brace = balance(value)
        while (paren > 0 or brace > 0) and i < len(lines):
            nxt = lines[i].rstrip("\r\n")
            i += 1
            value = value + " " + nxt.strip()
            p2, b2 = balance(nxt)
            paren += p2
            brace += b2

        stack[-1].keywords.setdefault(key, []).append(_parse_value(value))

    return root


def iter_nodes(node: ODLNode) -> Iterable[ODLNode]:
    """Iterate over all nodes in the ODL tree."""
    yield node
    for child in node.children:
        yield from iter_nodes(child)


def find_blocks(node: ODLNode, *, kind: str, name: str) -> list[ODLNode]:
    """Find all OBJECT or GROUP blocks with a given name."""
    kind_u = kind.upper()
    out: list[ODLNode] = []
    for n in iter_nodes(node):
        if n.kind == kind_u and (n.name or "").upper() == name.upper():
            out.append(n)
    return out


def find_key_occurrences(node: ODLNode, key: str) -> list[ODLValue]:
    """Find all occurrences of a key in the ODL tree."""
    out: list[ODLValue] = []
    key_u = key.upper()
    for n in iter_nodes(node):
        for k, vals in n.keywords.items():
            if k.upper() == key_u:
                out.extend(vals)
    return out


def require_unique_value(node: ODLNode, key: str, *, within: Optional[tuple[str, str]] = None) -> ODLValue:
    """
    Require exactly one occurrence of a key, optionally constrained to a block (OBJECT/GROUP).
    """
    if within:
        kind, name = within
        blocks = find_blocks(node, kind=kind, name=name)
        if len(blocks) != 1:
            raise ValueError(f"Expected 1 {kind}={name} block, found {len(blocks)}")
        vals = blocks[0].keywords.get(key.upper(), [])
    else:
        vals = find_key_occurrences(node, key)
    if len(vals) != 1:
        raise ValueError(f"Expected unique {key}, found {len(vals)}")
    return vals[0]


# ============================================================================
# Type Conversion Helpers
# ============================================================================

def _strip_outer_quotes(s: str) -> str:
    """Strip outer quotes from a string."""
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        return s[1:-1]
    return s


def _as_str(v: Optional[ODLValue]) -> Optional[str]:
    """Convert ODLValue to string."""
    if v is None:
        return None
    if isinstance(v.value, str):
        return v.value
    return str(v.value)


def _as_raw_str(v: Optional[ODLValue]) -> Optional[str]:
    """Get the raw string value from ODLValue without conversion."""
    if v is None:
        return None
    return _strip_outer_quotes(v.raw)


def _as_int(v: Optional[ODLValue]) -> Optional[int]:
    """Convert ODLValue to int."""
    if v is None:
        return None
    if isinstance(v.value, int):
        return v.value
    try:
        return int(str(v.value))
    except Exception:
        return None


def _as_float(v: Optional[ODLValue]) -> Optional[float]:
    """Convert ODLValue to float."""
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


def _get_optional_unique(tree: ODLNode, key: str) -> Optional[ODLValue]:
    """Get a unique or consistently-valued key from anywhere in the tree."""
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
    """Get a unique or consistently-valued key from within a specific block."""
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


def _sclk_decimal_seconds_to_ticks(sclk_decimal_seconds: str, *, ticks_per_sec: int = 65536) -> Optional[int]:
    """
    Convert a label SCLK like "666952834.305263901" (decimal seconds) into integer ticks.

    For M2020 and MSL, the SCLK fine field is ticks of 1/65536 sec (see SCLK01_MODULI_* = (... 65536)).
    """
    s = sclk_decimal_seconds.strip()
    if not s:
        return None
    try:
        d = Decimal(s)
    except Exception:
        return None

    coarse = int(d.to_integral_value(rounding=ROUND_FLOOR))
    frac = d - Decimal(coarse)
    fine = int((frac * Decimal(ticks_per_sec)).to_integral_value(rounding=ROUND_HALF_UP))
    if fine >= ticks_per_sec:
        coarse += 1
        fine -= ticks_per_sec
    if fine < 0:
        fine = 0
    return coarse * ticks_per_sec + fine


# ============================================================================
# Fast Layout Extraction (Regex-based, ~100x faster)
# ============================================================================

def infer_pds3_layout_fast(path: str | Path) -> Pds3ImageLayout:
    """Fast layout extraction reading only the label header.

    ~100x faster than infer_pds3_layout() for video playback.
    Does not return full label dict - use infer_pds3_layout() when metadata needed.
    Handles both embedded and detached (.LBL) labels.
    """
    path = Path(path)

    # Check for detached label (.LBL or .lbl file)
    lbl_path = path.with_suffix(".LBL")
    lbl_path_lower = path.with_suffix(".lbl")
    if lbl_path.exists():
        with open(lbl_path, "rb") as f:
            # Read full detached label (MSL MARDI labels are ~22KB)
            header = f.read()
    elif lbl_path_lower.exists():
        with open(lbl_path_lower, "rb") as f:
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


# ============================================================================
# Full Layout Extraction (pdr-based)
# ============================================================================

def infer_pds3_layout(path: str | Path) -> Tuple[Pds3ImageLayout, Dict[str, str]]:
    """Extract image layout and label from a PDS3 file using pdr."""
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


# ============================================================================
# High-Level Metadata Extraction
# ============================================================================

def _derive_frame_index(camera: str, filename: str) -> Optional[int]:
    """Extract frame index from LCAM filename."""
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
    """Load standardized metadata for a PDS3 EDL camera product.

    Extracts timing, identity, acquisition parameters, and layout information
    from either embedded or detached PDS3 labels.
    """
    img_path = Path(img_path)
    filename = img_path.name

    # Read label (embedded or detached)
    lbl_path = img_path.with_suffix(".LBL")
    lbl_path_lower = img_path.with_suffix(".lbl")
    if lbl_path.exists():
        label_source: LabelSource = "detached_lbl"
        label_text = read_detached_label_text(lbl_path)
    elif lbl_path_lower.exists():
        label_source = "detached_lbl"
        label_text = read_detached_label_text(lbl_path_lower)
    else:
        label_source = "embedded_img"
        label_text = read_embedded_label_text(img_path)

    tree = parse_odl(label_text)

    # Canonical timing for sorting/pose: fail fast if ambiguous.
    sclk_partition = _as_int(_get_optional_unique(tree, "SPACECRAFT_CLOCK_CNT_PARTITION"))
    try:
        sclk_start_val = _get_optional_unique(tree, "SPACECRAFT_CLOCK_START_COUNT")
    except Exception as e:
        raise ValueError(f"{img_path}: missing/ambiguous SPACECRAFT_CLOCK_START_COUNT ({e})") from e
    try:
        sclk_stop_val = _get_optional_unique(tree, "SPACECRAFT_CLOCK_STOP_COUNT")
    except Exception as e:
        raise ValueError(f"{img_path}: missing/ambiguous SPACECRAFT_CLOCK_STOP_COUNT ({e})") from e

    sclk_start_raw = _as_raw_str(sclk_start_val)
    sclk_stop_raw = _as_raw_str(sclk_stop_val)
    sclk_start_ticks = _sclk_decimal_seconds_to_ticks(sclk_start_raw) if sclk_start_raw else None
    sclk_stop_ticks = _sclk_decimal_seconds_to_ticks(sclk_stop_raw) if sclk_stop_raw else None

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
        sclk_start_ticks=sclk_start_ticks,
        sclk_stop_ticks=sclk_stop_ticks,
        sclk_start_raw=sclk_start_raw,
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
