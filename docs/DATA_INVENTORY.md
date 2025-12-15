# Data Inventory (Mars 2020)

## Overview

| Parameter | Value |
|-----------|-------|
| Landing site | Jezero Crater (18.4447°N, 77.4508°E) |
| Landing date | 2021-02-18 20:55 UTC |
| Elevation | -2.5 km (MOLA datum) |
| TRN status | First Mars mission with active Terrain Relative Navigation |

---

## 1. LCAM (Lander Vision System Camera)

The primary TRN sensor — matched descent images to onboard CTX-derived map.

### Sensor

| Parameter | Value |
|-----------|-------|
| Resolution | 1024 × 1024 pixels |
| FOV | 90° × 90° |
| Pixel scale | 1.67 mrad/pixel |
| Focal length | 5.8 mm, f/2.7 |
| Detector | On Semi Python 5000 (2×2 binned) |
| Bit depth | 8-bit grayscale |

**Reference:** Maki et al. (2020), Table 5 — `literature/Maki2020_Mars2020_Engineering_Cameras.pdf`

### EDL Sequence

| Parameter | Value |
|-----------|-------|
| Frames | 87 calibrated images |
| Altitude range | ~4 km to backshell separation |
| Camera model | CAHVORE (embedded in each .IMG header) |
| Pose | `ORIGIN_OFFSET_VECTOR` + `ORIGIN_ROTATION_QUATERNION` in MCMF frame |

**Source:** `urn:nasa:pds:mars2020_edlcam_ops_calibrated`
- https://pds-imaging.jpl.nasa.gov/data/mars2020/mars2020_edlcam_ops_calibrated/

### Filename Convention

Example: `ELM_0000_0666952774_000FDR_N0000001LVS_04000_0000LUJ01.IMG`

| Position | Field | Example | Description |
|----------|-------|---------|-------------|
| 01–02 | Instrument | `EL` | EDL Lander Vision System (LCAM) |
| 03 | Color | `M` | Monochrome/Panchromatic |
| 04 | Special | `_` | Nominal processing |
| 05–08 | Sol | `0000` | Sol number (0 for EDL) |
| 09 | Venue | `_` | Flight |
| 10–19 | SCLK | `0666952774` | Spacecraft clock (integer seconds only) |
| 20 | — | `_` | Separator |
| 21–23 | Milliseconds | `000` | SCLK fractional part |
| 24–26 | Product type | `FDR` | Flight Data Record (calibrated) |
| 27 | Geometry | `_` | Raw (non-linearized) |
| 28 | Thumbnail | `N` | Non-thumbnail |
| 29–31 | Site/Set | `000` | Always 000 for LCAM |
| 32–35 | Frame | `0001` | Frame number (1–87) |
| 36–38 | Camera | `LVS` | Lander Vision System |
| 39 | — | `_` | Separator |
| 40–44 | Sequence | `04000` | Sequence ID |
| 45 | — | `_` | Separator |
| 46–54 | Product ID | `0000LUJ01` | Unique product identifier |

**Reference:** Mars2020_Camera_SIS.pdf, Section 18 "FILE NAMING STANDARDS", Table 18-1 (pages 198–210)

---

## 2. RDCAM (Rover Downlook Camera)

Captured continuous video from powered descent through touchdown.

### Sensor

| Parameter | Value |
|-----------|-------|
| Resolution | 1280 × 1024 pixels |
| FOV | ~35° × 30° |
| Focal length | ~9.5 mm |
| Pixel pitch | ~4.8 µm |
| Bit depth | 8-bit color (Bayer) |

### EDL Sequence

| Parameter | Value |
|-----------|-------|
| Frames | ~7000 |
| Frame rate | 30 fps |
| Camera model | **Not in headers** — no CAHVORE embedded |
| Pose | **Not in headers** — must use SPICE kernels |

**Source:** `urn:nasa:pds:mars2020_edlcam_ops_calibrated`
- Compressed: `data_edl_rdcam/`
- Lossless (LU): `data_edl_rdcam_lu/` (~100GB total)

**Note:** Intrinsics must be approximated from Camera SIS or fitted from data.

---

## 3. Other EDL Cameras

Also in the EDL camera bundle (not primary focus of this repo):
- `EDL_DDCAM` — Descent Stage Downlook
- `EDL_RUCAM` — Rover Uplook
- `EDL_PUCAM1/2/3` — Parachute Uplook cameras

---

## 4. CTX Orbital Maps (TRN Reference)

The **LVS appearance map** was generated from three CTX orthorectified images — this is the "truth" dataset TRN matched against during descent.

### Onboard Map Specifications

| Parameter | Value | Source |
|-----------|-------|--------|
| Coverage | 30 × 30 km | Johnson et al. (2022) |
| Appearance map | 6 m/pixel | Cheng et al. (2021), Fergason et al. (2020) |
| DTM (delivered) | 20 m/pixel | Fergason et al. (2020) |
| Source imagery | 3 CTX stereo pairs | Fergason et al. (2020) |
| Horizontal co-registration | <6 m (99%) | Cheng et al. (2021) |
| Horizontal displacement (individual CTX) | 9.6 m average | Fergason et al. (2020) |

### TRN Processing Resolutions

| Phase | Appearance Map | Description | Source |
|-------|----------------|-------------|--------|
| **Coarse** | 12 m/pixel | 6 m map binned 2× | Johnson et al. (2022) |
| **Fine** | 6 m/pixel | Native resolution | Johnson et al. (2022) |

**Note:** The 12 m/pixel coarse resolution is the appearance (orthoimage) map binned for faster processing, not the elevation map.

### Orthomosaic (6 m/pixel)

| Parameter | Value |
|-----------|-------|
| Resolution | 6 m/pixel |
| Projection | Equirectangular (Mars 2000 sphere) |
| Coverage | ~30 × 30 km around landing site |

**Source:** USGS Astrogeology
- https://astrogeology.usgs.gov/search/map/mars_2020_terrain_relative_navigation_context_camera_orthorectified_image_mosaic
- DOI: [10.5066/P9GV1ND3](https://doi.org/10.5066/P9GV1ND3)

### DTM (20 m/pixel)

| Parameter | Value |
|-----------|-------|
| Resolution | 20 m/pixel |
| Vertical datum | MOLA areoid |

**Source:** USGS Astrogeology
- https://astrogeology.usgs.gov/search/map/mars_2020_terrain_relative_navigation_ctx_dtm_mosaic

### Ambiguity: Onboard Elevation Map Resolution

The Cheng et al. (2021) **requirement** states:
> "The LVS elevation map shall be 12 m/pixel and be derived from DEMs with a ground sample distance of no more than 20 m/pixel."

However:
- Fergason et al. (2020) reports CTX DTMs were **exported at 20 m/pixel**
- The USGS-delivered product is **20 m/pixel**
- No explicit statement found confirming the DTM was resampled to 12 m/pixel for the onboard product

The 12 m/pixel value in Johnson et al. (2022) specifically refers to **binning the appearance map** for coarse landmark matching, not the elevation map resolution.

### References

| Paper | Citation | Key Content |
|-------|----------|-------------|
| Cheng et al. (2021) | "Making an Onboard Reference Map From MRO/CTX Imagery for Mars 2020 LVS", *Earth and Space Science*, DOI: [10.1029/2020EA001560](https://doi.org/10.1029/2020EA001560) | Map requirements, CTX camera model refinement, jitter correction, validation |
| Fergason et al. (2020) | "Mars 2020 TRN Flight Product Generation", 51st LPSC | USGS map production, native resolutions (6 m ortho, 20 m DTM) |
| Johnson et al. (2022) | "The Lander Vision System for Mars 2020", AIAA SciTech, DOI: [10.2514/6.2022-1214](https://doi.org/10.2514/6.2022-1214) | Onboard processing, coarse/fine phases, 12 m binned map |

**Local copies:** `literature/Cheng2021_CTX_Onboard_Map.pdf`, `literature/Fergason2020_TRN_Flight_Products_LPSC.pdf`, `literature/Mars2020_LVS_Flight_Performance_Johnson2022.pdf`

---

## 5. HiRISE Orbital Maps

Used as the **hazard basemap** for Safe Target Selection (STS). Not the appearance map for landmark matching.

### Orthomosaic (25 cm/pixel)

| Parameter | Value |
|-----------|-------|
| Resolution | 25 cm/pixel |
| Size | ~3.2 GB |

**Source:** USGS Astrogeology
- https://astrogeology.usgs.gov/search/map/mars_2020_terrain_relative_navigation_hirise_orthorectified_image_mosaic
- DTM: https://doi.org/10.5066/P9REJ9JN

---

## 6. SPICE Kernels

EDL trajectory and attitude reconstruction.

**Source:** NAIF PDS Archive
- https://naif.jpl.nasa.gov/pub/naif/pds/pds4/mars2020/mars2020_spice/

**Minimal EDL kernel set:**
| Kernel | Purpose |
|--------|---------|
| `spk/m2020_edl_v01.bsp` | Trajectory |
| `ck/m2020_edl_v01.bc` | Attitude |
| `fk/m2020_v04.tf` | Frame definitions |
| `sclk/m2020_168_sclkscet_*.tsc` | Spacecraft clock |
| `pck/pck00010.tpc` | Mars orientation (IAU_MARS) |
| `lsk/naif0012.tls` | Leap seconds |

**Helper:** `tools/spice_fetch_mars2020_edl.py`

---

## 7. Camera Calibration & Documentation

### CAHVORE Model Files

Pre-flight calibration files for LCAM (not needed if using embedded models from .IMG headers).

**Source:** `urn:nasa:pds:mars2020_mission:calibration_camera`
- https://pds-geosciences.wustl.edu/m2020/urn-nasa-pds-mars2020_mission/calibration_camera/camera_models/

### Camera SIS Documentation

- **Mars2020_Camera_SIS.pdf** — Full specification including CAHVORE model
  - https://pds-geosciences.wustl.edu/m2020/urn-nasa-pds-mars2020_mission/document_camera/Mars2020_Camera_SIS.pdf
- **Mars2020_Camera_Characteristics.pdf** — Sensor property tables
  - https://pds-geosciences.wustl.edu/m2020/urn-nasa-pds-mars2020_mission/document_camera/Mars2020_Camera_Characteristics.pdf

---

## 8. Coordinate System

LCAM products use `REFERENCE_COORD_SYSTEM_NAME=MCMF_FRAME` — Mars-Centered, Mars-Fixed (equivalent to SPICE `IAU_MARS`).

See **[COORDINATES.md](COORDINATES.md)** for:
- MCMF → equirectangular projection formulas
- Vertical datum (MOLA areoid vs map sphere radius)

---

## See Also

- **[DATA_INVENTORY_MSL.md](DATA_INVENTORY_MSL.md)** — MSL/MARDI descent imagery
- **[../literature/README.md](../literature/README.md)** — Annotated paper summaries

---

<!--
================================================================================
ARCHIVED CONTENT (previous version, preserved for reference)
================================================================================

## Primary Statement of Which Maps Flew

- **Fergason et al. (LPSC 2020)**, "Mars 2020 Terrain Relative Navigation Flight Product Generation: Digital Terrain Model and Orthorectified Image Mosaics." `https://www.hou.usra.edu/meetings/lpsc2020/pdf/2020.pdf`
  - States the onboard TRN "truth dataset" was the **LVS appearance map from three CTX orthorectified images**.
  - States the **HiRISE orthomosaic** was the **hazard basemap**, and the derived **hazard map** was onboard for TRN hazard avoidance.

## EDL Camera Image Products (PDS)

**Bundle family**
- PDS Registry bundle identifiers seen for EDL camera products:
  - `urn:nasa:pds:mars2020_edlcam_ops_raw::*`
  - `urn:nasa:pds:mars2020_edlcam_ops_calibrated::*`

**Instrument streams discovered via PDS Registry (Sol 0 examples)**
- `LCAM` (Lander Vision System) — this repo's current focus (`data/m2020/lcam/*.IMG`).
- `EDL_DDCAM` (Descent Stage Downlook), `EDL_RDCAM` (Rover Downlook), `EDL_RUCAM` (Rover Uplook), `EDL_PUCAM1/2/3` (Parachute Uplook A/B/C).

**Which streams contain "TRN-ready" geometry inside the `.IMG`**
- `LCAM` `.IMG` files include:
  - `MODEL_TYPE=CAHVORE` (camera model) with `MODEL_COMPONENT_1..7`
  - `ORIGIN_OFFSET_VECTOR` + `ORIGIN_ROTATION_QUATERNION`
  - `REFERENCE_COORD_SYSTEM_NAME` (e.g., `MCMF_FRAME`)
- The probed `EDL_*` camera `.IMG` samples do **not** embed CAHVORE/pose fields in their PDS3 `IMAGE_HEADER` (you'd need geometry from elsewhere to use them for TRN).
- Probe script: `tools/edl_registry_probe.py`

**"Ancillary" products inside `mars2020_edlcam_ops_calibrated::1.1`**
- The only `Product_Ancillary` items in the calibrated EDL camera bundle are **collection inventory CSVs** (not calibration models):
  - `https://pds-imaging.jpl.nasa.gov/archive/m20/r8/mars2020_edlcam_ops_calibrated/data/r8_collection_data_inventory.csv`
  - `https://pds-imaging.jpl.nasa.gov/archive/m20/r9/mars2020_edlcam_ops_calibrated/data/r9_collection_data_inventory.csv`
  - `https://pds-imaging.jpl.nasa.gov/archive/m20/r8/mars2020_edlcam_ops_calibrated/browse/r8_collection_browse_inventory.csv`
  - `https://pds-imaging.jpl.nasa.gov/archive/m20/r9/mars2020_edlcam_ops_calibrated/browse/r9_collection_browse_inventory.csv`

## Camera Modeling References (EDL + LCAM)

**Mars 2020 Camera SIS (mission-wide)**
- PDS "mission" bundle: `urn:nasa:pds:mars2020_mission::4.0`
- Camera documentation collection: `urn:nasa:pds:mars2020_mission:document_camera::14.0`
  - `Mars2020_Camera_SIS.pdf` (includes EDL camera suite + LCAM characteristics tables, and a general CAHV/CAHVOR/CAHVORE model description):
    - `https://pds-geosciences.wustl.edu/m2020/urn-nasa-pds-mars2020_mission/document_camera/Mars2020_Camera_SIS.pdf`
  - `Mars2020_Camera_Characteristics.pdf` (tables of sensor properties and other characteristics):
    - `https://pds-geosciences.wustl.edu/m2020/urn-nasa-pds-mars2020_mission/document_camera/Mars2020_Camera_Characteristics.pdf`
  - Label keyword tables (useful for seeing what camera-model fields *can* appear in labels, and what they mean):
    - `Mars2020_Camera_SIS_Labels_sort_pds.html`:
      - `https://pds-geosciences.wustl.edu/m2020/urn-nasa-pds-mars2020_mission/document_camera/Mars2020_Camera_SIS_Labels_sort_pds.html`
    - `Mars2020_Camera_SIS_Operations_Label.pdf`:
      - `https://pds-geosciences.wustl.edu/m2020/urn-nasa-pds-mars2020_mission/document_camera/Mars2020_Camera_SIS_Operations_Label.pdf`

**Mars2020 mission calibration files (CAHV/CAHVOR/CAHVORE)**
- PDS calibration collection: `urn:nasa:pds:mars2020_mission:calibration_camera::1.1`
  - Directory listing (useful for manual discovery): `https://pds-geosciences.wustl.edu/m2020/urn-nasa-pds-mars2020_mission/calibration_camera/camera_models/`
  - Includes LCAM camera-model files (examples):
    - `lcam_FS_Cint_20191022.cahvore`
    - `M20_SN_012.cahvore` (LCAM serial `012` per `M20_camera_mapping.xmlx`)
  - Does **not** include EDL camera suite (`EDL_*`) camera-model files (no `PUC/RUC/RDC/DDC` mapping entries in `M20_camera_mapping.xmlx`, and no EDL-related filenames in `camera_models/`).

**EDLcam "raw video/audio" SIS**
- PDS bundle: `urn:nasa:pds:m2020_edlcam_raw::1.1`
  - `m2020_edlcam_raw_bundle_sis.pdf`:
    - `https://pds-imaging.jpl.nasa.gov/archive/m20/cumulative/m2020_edlcam_raw/document/m2020_edlcam_raw_bundle_sis.pdf`
  - PDS "data" landing page (directory listing):
    - `https://pds-imaging.jpl.nasa.gov/data/mars2020/m2020_edlcam_raw/`
  - Raw EDL videos (including RDC videos):
    - `https://pds-imaging.jpl.nasa.gov/data/mars2020/m2020_edlcam_raw/document_video/`

**Engineering cameras paper**
- Springer DOI: `10.1007/s11214-020-00765-9`
  - PDF: `https://link.springer.com/content/pdf/10.1007/s11214-020-00765-9.pdf`
  - JPL Dataverse mirror (PDF file `CL20_5537.pdf`): `https://doi.org/10.48577/jpl.JQOSAY`

**What we have (and don't have) for `EDL_RDCAM`**
- `EDL_RDCAM` PDS3 `.IMG` products do not include CAHV* vectors or per-frame pose in `IMAGE_HEADER` (unlike `LCAM`).
- The public NAIF Mars2020 SPICE archive contains **no instrument kernel (IK) for EDL cameras** (only `ik/m2020_struct_v00.ti`, which defines antenna FOVs).
- Practical path (until a per-camera CAHVORE is found publicly): use the EDL camera suite "first-order" intrinsics from `Mars2020_Camera_SIS.pdf` / the engineering cameras paper (FOV, focal length, pixel pitch), and treat lens distortion as unknown/unmodeled unless you fit it from data.
  - Best-available public intrinsics for `EDL_RDCAM` right now:
    - Image size: `1280×1024`
    - Focal length: `~9.5 mm`
    - Pixel pitch: `~4.8 µm`
    - Field of view: `~35° × 30°`
    - Pinhole approximation (pixels): `fx≈fy≈9.5/0.0048≈1979`, `cx≈(1280-1)/2=639.5`, `cy≈(1024-1)/2=511.5`
    - Distortion: not provided publicly in PDS/NAIF artifacts (treat as unknown unless you fit/calibrate)

**Meaning of EDL camera suite "raw" vs "calibrated" still images**
- Per `Mars2020_Camera_SIS.pdf` Section "5.6.7 EDL Camera Suite", EDLcam data were downlinked as JPEG frames, MPEG/H.264 video, and (later) lossless uncompressed frames; the pipeline then extracts still frames from those sources.
- In the EDL still-frame archives (`mars2020_edlcam_ops_raw` / `mars2020_edlcam_ops_calibrated`), this shows up as:
  - `ECM` / `ECV` (raw source frames, including Bayer-encoded lossless frames),
  - `EDR` / `EVD` ("partially processed"; for byte-space EDLcam it's effectively a copy of the corresponding raw frame),
  - `FDR` ("derived" / calibrated; debayered/demosaiced RGB frame, generally the most convenient format to start with).

## Coordinate Frames & Meanings (Needs Verification)

- `REFERENCE_COORD_SYSTEM_NAME=MCMF_FRAME` appears in `LCAM` products.
  - Treat as "Mars-centered, Mars-fixed" until verified against NAIF/SPICE naming; keep the original string in any exported benchmark.
- `ORIGIN_OFFSET_VECTOR` and CAHVORE `MODEL_COMPONENT_1` ("C") are both in meters and very close numerically for `LCAM` products (offset often differs by ~1 m).

## SPICE Kernel Recommendation (Frame/Time Conversions)

If you want to do frame conversions "the SPICE way" (rather than assuming formulas), use the official NAIF Mars2020 kernel set.

**Recommended starting point**
- NAIF Mars2020 meta-kernel (mission-wide): `https://naif.jpl.nasa.gov/pub/naif/pds/pds4/mars2020/mars2020_spice/spice_kernels/mk/m2020_v14.tm`
  - Archive root: `https://naif.jpl.nasa.gov/pub/naif/pds/pds4/mars2020/mars2020_spice/`
  - Note: the kernels are a **separate official PDS archive** (NAIF) from the PDS Imaging EDL camera products; they are not "packaged inside" each `.IMG`.

**Minimal subset for EDL + MCMF-style map projection**
- `lsk/naif0012.tls` (leap seconds)
- `pck/pck00010.tpc` (Mars orientation + radii; defines `IAU_MARS`)
- `fk/m2020_v04.tf` (Mars2020 frame definitions such as `M2020_TOPO`, `M2020_LOCAL_LEVEL`)
- `sclk/m2020_168_sclkscet_refit_v14.tsc` (SCLK↔ET, if using spacecraft clock)
- `spk/m2020_edl_v01.bsp` + `ck/m2020_edl_v01.bc` (EDL trajectory/orientation coverage)
- `spk/m2020_ls_ops210303_iau2000_v1.bsp` (landing site kernel used by the surface frame chain)

**Why this matters for `MCMF_FRAME`**
- LCAM PDS labels use `MCMF_FRAME` for camera model/pose coordinate space.
- In the trajectory reconstruction literature, "MCMF" is defined as "Mars-centered, Mars-fixed" with Z along Mars' spin axis and X through the prime meridian (i.e., the same concept SPICE represents with `IAU_MARS` + the PCK).
  - Example: `literature/karlgaard-medli2-trajectory-atmosphere-reconstruction.pdf`

**Local helper**
- `tools/spice_fetch_mars2020_edl.py` downloads a minimal EDL kernel set from the NAIF archive and writes `out/spice/mars2020_edl_naif/m2020_edl_min.tm` for `furnsh(...)`.

## Notes / Conventions for Benchmark Export

- Prefer preserving the native camera model as **CAHVORE** (do not collapse to OpenCV Brown unless you explicitly fit a distortion approximation).
- Keep raw provenance: include source URLs, bundle/collection IDs, and original filenames in any published benchmark.

-->
