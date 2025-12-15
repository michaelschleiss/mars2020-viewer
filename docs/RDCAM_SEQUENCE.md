# RDCAM Image Sequence — Detailed Findings

---

## 1. Sensor Specifications

| Parameter | Value |
|-----------|-------|
| Resolution | 1280 × 1024 pixels |
| FOV | ~35° × 30° |
| Focal length | ~9.5 mm |
| Pixel pitch | ~4.8 µm |
| Bit depth | 8-bit color (Bayer) |

**Note:** No CAHVORE model in headers. Intrinsics approximated from Camera SIS or fitted from data.

### Pinhole Approximation

| Parameter | Value |
|-----------|-------|
| fx, fy | ~1979 pixels |
| cx | 639.5 pixels |
| cy | 511.5 pixels |

Derived from: fx = fy = 9.5 mm / 4.8 µm ≈ 1979, cx = (1280-1)/2, cy = (1024-1)/2. Distortion unknown.

---

## 2. Sequence Overview

| Parameter | Value |
|-----------|-------|
| Frames | ~7000 |
| Frame rate | 30 fps |
| Camera model | Not in headers |
| Pose | Not in headers — use SPICE kernels |

---

## 3. Products

| Type | Description |
|------|-------------|
| Compressed | `data_edl_rdcam/` |
| Lossless (LU) | `data_edl_rdcam_lu/` (~100GB total) |

---

## 4. Pose Recovery

RDCAM does not embed pose in headers like LCAM. Use SPICE kernels:

- `spk/m2020_edl_v01.bsp` — Trajectory
- `ck/m2020_edl_v01.bc` — Attitude

See [DATA_INVENTORY.md](DATA_INVENTORY.md) Section 6 for full kernel list.

---

## 5. References

- **Mars2020_Camera_SIS.pdf** — Sensor specs, FOV, focal length
