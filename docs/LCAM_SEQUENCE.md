# LCAM Image Sequence — Detailed Findings

Consolidated from literature, local file analysis, Camera SIS, and codebase tools.

---

## 1. Sequence Overview

| Parameter | Value |
|-----------|-------|
| Total frames | 87 calibrated images |
| First frame | 2021-02-18 20:41:04.447849 UTC (15:50 LMST) |
| Last frame | 2021-02-18 20:44:11.249523 UTC (15:53 LMST) |
| Duration | 186.8 seconds (3.1 minutes) |
| Frame rate | 0.29–1.00 fps (avg 0.47 fps) |
| Distance travelled | 13.55 km |
| Ground track | 6.21 km |
| Speed | 3–191 m/s (avg 73 m/s) |

**Source:** Computed from `urn:nasa:pds:mars2020_edlcam_ops_calibrated` (87 LCAM FDR products). Timestamps from `START_TIME`, positions from `MODEL_COMPONENT_1`.

---

## 2. Altitude Profile

| Parameter | Value |
|-----------|-------|
| Start altitude | 6,970 m (6.97 km) — frame 1 |
| End altitude | -4,253 m (below MOLA reference) — frame 87 |
| Altitude span | 11,223 m |
| TRN active range | 4,200m → 500m AGL |
| Average descent rate | 60.1 m/s (216 km/h) |

**Note:** Negative altitudes indicate descent below Mars reference ellipsoid (radius 3,396,190 m). Jezero crater floor is ~2.5 km below MOLA datum.

**Source:** Altitudes computed from `MODEL_COMPONENT_1` (camera position in MCMF) minus Mars reference radius. TRN active range from Johnson et al. (2022).

### Descent Profile by EDL Phase

| Phase | Frames | Count | Duration | Altitude Range (AGL) | Altitude Change | Descent Rate |
|-------|--------|-------|----------|----------------------|-----------------|--------------|
| **Parachute Descent** | 1–47 | 47 | 104.2 s | 11.23 km → 2.08 km | 9.14 km | 87.7 m/s |
| **Powered Descent** | 48–74 | 27 | 37.6 s | 1.99 km → 0.04 km | 1.95 km | 51.9 m/s |
| **Sky Crane** | 75–87 | 13 | 40.5 s | 20 m → 0 m | 20 m | 0.5 m/s |

**Total:** 11.23 km descent, 186.8 seconds, average 60.1 m/s

**Notes:**
- Altitudes shown as AGL (Above Ground Level), estimated from final frame position at touchdown
- Parachute phase: From heat shield separation (~11 km AGL) to backshell separation (~2 km AGL)
- Powered descent: Backshell separation to sky crane deployment (~20 m AGL)
- Sky crane: Rover lowering on bridle to touchdown
- Ground level: -4254.6 m relative to Mars reference ellipsoid (3,396,190 m radius)

---

## 3. Frame Rate Strategy

LCAM used a **variable frame rate** to optimize storage while maximizing TRN performance:

| Phase | Frame Rate | Altitude | Purpose |
|-------|------------|----------|---------|
| Pre-localization | ~0.3 Hz | Heat shield sep → 4200m | Storage optimization |
| TRN active | ~1 Hz | 4200m → 500m | Landmark matching |
| Post-TRN ("Tourist") | ~0.3 Hz | 500m → surface | Continued capture for analysis |

**Source:** Frame rates from local data analysis (measured 0.29–1.00 fps). "1 Hz MRL update" and "TOURIST" mode from Johnson et al. (2022). LCAM capability "up to 2 Hz" from Maki et al. (2020).

---

## 4. TRN Processing Breakdown

### Coarse vs Fine Phase

| Parameter | Coarse | Fine |
|-----------|--------|------|
| LCAM frames | 25–27 | 28–69 |
| Frames processed | 3 | 42 |
| Map resolution | 12 m/pixel (binned) | 6 m/pixel |
| Landmarks/frame | 5 large patches | Up to 150 small patches |
| Duration | 4.75 s | ~45 s |
| Result | 15/15 matched | 80+ inliers/frame |
| Position error | ~200 m | 40 m |

**Position error (design):** up to 3.2 km initial → ~200 m after coarse → 40 m after fine [Johnson et al. 2022, Fig. 1]
**Actual flight:** 137 m coarse correction applied, 5 m touchdown error [Johnson et al. 2022, §V]

### Frame-to-Stage Mapping

| Stage | LCAM Frames | Count |
|-------|-------------|-------|
| Pre-TRN | 1–24 | 24 |
| **Coarse** | 25–27 | 3 |
| **Fine** | 28–69 | 42 |
| Tourist | 70–87 | 18 |
| **Total TRN** | 25–69 | **45** |

**Key frame:** Frame 43 = Fine image 16 = 19th MRL image — position used by STS to select landing site.

**Source:** Johnson et al. (2022). Frame assignments verified via supplementary video (DOI: 10.2514/6.2022-1214.vid).

---

## 5. Per-Frame Metadata

Every LCAM .IMG file contains embedded metadata in the `IMAGE_HEADER` object:

### Position & Orientation
| Field | Format | Description |
|-------|--------|-------------|
| `ORIGIN_OFFSET_VECTOR` | (x, y, z) meters | Camera position in MCMF frame |
| `ORIGIN_ROTATION_QUATERNION` | (w, x, y, z) | Camera orientation quaternion |
| `REFERENCE_COORD_SYSTEM_NAME` | string | Always "MCMF_FRAME" |

### CAHVORE Camera Model
| Component | Field | Description |
|-----------|-------|-------------|
| C | `MODEL_COMPONENT_1` | Camera center (meters) |
| A | `MODEL_COMPONENT_2` | Optical axis unit vector |
| H | `MODEL_COMPONENT_3` | Horizontal image plane vector |
| V | `MODEL_COMPONENT_4` | Vertical image plane vector |
| O | `MODEL_COMPONENT_5` | Optical axis for distortion |
| R | `MODEL_COMPONENT_6` | Radial distortion coefficients |
| E | `MODEL_COMPONENT_7` | Entrance pupil parameters |

### Timing
| Field | Description |
|-------|-------------|
| `START_TIME` | UTC timestamp (shutter open) |
| `STOP_TIME` | UTC timestamp (shutter close) |
| `SPACECRAFT_CLOCK_START_COUNT` | SCLK value |
| `EXPOSURE_DURATION` | ~95 ms in labels (see note) |

**Note on EXPOSURE_DURATION:** PDS labels report ~95 ms, but Johnson et al. (2022) specifies actual sensor integration time of **150 µs**. The label value likely includes readout/transfer overhead. For motion blur calculations, use 150 µs (yields ~1 cm blur at descent speeds, consistent with TRN requirements for "crisp images").

---

## 6. Key Timeline (Mission Elapsed Time)

| Time (s) | Event |
|----------|-------|
| 0 | LVS power-on |
| 753 | Imaging starts (just before heat shield separation) |
| 831.6 | MRL (Map-Relative Localization) processing begins |
| 833.3 | Coarse image 1 processed |
| 834.8 | Coarse image 2 processed |
| 836.3 | Coarse image 3 processed |
| 836.4 | Fine processing begins |
| 837.5 | First VALID fine position estimate (5.8s after MRL start) |
| 853.8 | Fine image 16 (Frame 43) — used by Safe Target Selection (STS) |
| 882.3 | Fine image 42 — last fine estimate |
| 882.3 | Enter TOURIST mode |
| 943.3 | Start writing data products |
| ~1170 | LVS power-off (417s total operation) |

**Source:** Johnson et al. (2022)

---

## 7. Processing Timing Margins

All timing deadlines met with significant margin:

| Operation | Allocated | Actual | Margin |
|-----------|-----------|--------|--------|
| LCAM image capture | 171.9 ms | 109.9 ms | 36% |
| Coarse image processing | 1296.9 ms | 979.5 ms | 24% |
| Fine image processing | 750.0 ms | 555.7 ms | 26% |
| Fine position update | 93.8 ms | 55.7 ms | 41% |

---

## 8. File Structure

### Filename Convention
```
ELM_0000_[SCLK]_[TYPE]_N[FRAME][CAM]_[SEQ]_[PROD].IMG
```

Example: `ELM_0000_0666952774_000FDR_N0000001LVS_04000_0000LUJ01.IMG`

| Component | Example | Meaning |
|-----------|---------|---------|
| Instrument | ELM | EDL camera module |
| SCLK | 0666952774 | Spacecraft clock (10 digits) |
| Type | FDR | Flight Data Record (calibrated) |
| Frame | N0000001 | Frame number (1-87) |
| Camera | LVS | Lander Vision System |
| Sequence | 04000 | Sequence ID |
| Product | 0000LUJ01 | Unique product ID |

### File Format (PDS3)
1. **ASCII Label** — Variable-length PDS3 keywords
2. **IMAGE_HEADER** — Binary key=value metadata (pose, CAHVORE)
3. **Image Data** — 1024×1024 raw pixels (8-bit grayscale, BSQ)

---

## 9. Sample Frames

### Frame 1 (Sequence Start)
| Parameter | Value |
|-----------|-------|
| Time | 2021-02-18 20:41:04.447849 UTC |
| SCLK | 666952774.005264 |
| Altitude | 6,970 m |
| Position (MCMF) | (706498.6, 3149249.8, 1079158.9) m |
| Orientation (quat) | (0.614, 0.217, -0.753, -0.098) |

### Frame 44 (Mid-Sequence)
| Parameter | Value |
|-----------|-------|
| Time | 2021-02-18 20:42:45.348760 UTC |
| SCLK | 666952874.905280 |
| Altitude | -1,905 m |
| Position (MCMF) | (700204.2, 3142714.1, 1074352.6) m |
| Orientation (quat) | (0.541, 0.706, 0.269, -0.370) |

### Frame 87 (Sequence End)
| Parameter | Value |
|-----------|-------|
| Time | 2021-02-18 20:44:11.249523 UTC |
| SCLK | 666952960.805264 |
| Altitude | -4,252 m |
| Position (MCMF) | (699129.2, 3140823.4, 1073168.4) m |
| Orientation (quat) | (0.505, 0.778, 0.247, -0.279) |

---

## 10. What Makes LCAM Unique Among EDL Cameras

| Feature | LCAM | RDCAM | Other EDL Cams |
|---------|------|-------|----------------|
| Embedded CAHVORE | Yes | No | No |
| Embedded pose | Yes | No | No |
| TRN processing | Yes (45 frames) | No | No |
| Frame rate | Variable 0.3-1 Hz | Fixed 30 fps | Various |
| Total frames | 87 | ~7000 | Varies |

**LCAM is the only EDL camera with complete 6-DOF pose in every frame header**, enabling direct 3D reconstruction and orbital imagery matching without external trajectory data.

---

## 11. References

- **Maki et al. (2020)** — Camera specifications, frame rate strategy
- **Johnson et al. (2017)** — LVS system design, requirements
- **Johnson et al. (2022)** — Flight performance, timeline, processing results
- **Mohan et al. (2021)** — TRN accuracy analysis
- **Mars2020_Camera_SIS.pdf** — PDS data format, label keywords

---

## 12. Constants Used in Analysis

| Constant | Value | Source |
|----------|-------|--------|
| Mars sphere radius | 3,396,190 m | IAU |
| Jezero areoid radius | 3,394,507.076 m | Camera SIS |
| UTC format | `%Y-%m-%dT%H:%M:%S.%f` | PDS3 |

---

## 13. Frame-to-Stage Mapping (Detailed)

See **Section 4** for the frame-to-stage summary table.

For detailed mapping with verification methodology:
- **`LCAM_FRAME_TO_STAGE_MAPPING.md`** — Complete frame assignments, conversion formulas, Python code
- **`LCAM_FRAME_MAPPING_REFERENCES.md`** — Citations, timeline correlation, verification methodology
