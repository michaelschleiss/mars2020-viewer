"""
Lightweight PDS3 ODL label reader/parser.

Goals:
- Preserve OBJECT/GROUP structure and repeated keys.
- Provide strict querying helpers for canonical metadata extraction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class ODLValue:
    raw: str
    value: Any
    unit: Optional[str] = None


@dataclass
class ODLNode:
    kind: str  # ROOT | OBJECT | GROUP
    name: Optional[str] = None
    keywords: dict[str, list[ODLValue]] = None
    children: list["ODLNode"] = None

    def __post_init__(self) -> None:
        if self.keywords is None:
            self.keywords = {}
        if self.children is None:
            self.children = []


_ASSIGN_RE = re.compile(r"^\s*([A-Za-z0-9_^\-]+)\s*=\s*(.*?)\s*$")
_BLOCK_START_RE = re.compile(r"^\s*(OBJECT|GROUP)\s*=\s*([A-Za-z0-9_^\-]+)\s*$", re.IGNORECASE)
_BLOCK_END_RE = re.compile(r"^\s*END_(OBJECT|GROUP)\s*=\s*([A-Za-z0-9_^\-]+)\s*$", re.IGNORECASE)
_END_RE = re.compile(r"^\s*END\s*$", re.IGNORECASE | re.MULTILINE)


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
    lbl_path = Path(lbl_path)
    return lbl_path.read_text(encoding="ascii", errors="ignore")


def _strip_c_style_comments(lines: Iterable[str]) -> list[str]:
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
    s = raw.strip()
    m = re.match(r"^(.*?)(?:\s*<\s*([^>]+)\s*>\s*)$", s)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return s, None


def _parse_scalar(raw: str) -> Any:
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
    yield node
    for child in node.children:
        yield from iter_nodes(child)


def find_blocks(node: ODLNode, *, kind: str, name: str) -> list[ODLNode]:
    kind_u = kind.upper()
    out: list[ODLNode] = []
    for n in iter_nodes(node):
        if n.kind == kind_u and (n.name or "").upper() == name.upper():
            out.append(n)
    return out


def find_key_occurrences(node: ODLNode, key: str) -> list[ODLValue]:
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
