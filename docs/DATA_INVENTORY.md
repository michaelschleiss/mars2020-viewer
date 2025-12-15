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

| Parameter | Value |
|-----------|-------|
| Frames | 87 |
| Resolution | 1024 × 1024, 8-bit grayscale |
| FOV | 90° × 90° |
| Camera model | CAHVORE (embedded in header) |
| Pose | 6-DOF in MCMF frame (embedded in header) |

**References:** Maki et al. (2020) "The Mars 2020 Engineering Cameras and Microphone" DOI: [10.1007/s11214-020-00765-9](https://doi.org/10.1007/s11214-020-00765-9) | Johnson et al. (2022) "Mars 2020 Lander Vision System Flight Performance" DOI: [10.2514/6.2022-1214](https://doi.org/10.2514/6.2022-1214)

### Products

| Type | Pixels | Pose |
|------|--------|------|
| EDR | Lossless | Onboard estimate |
| **FDR** | Lossless | Reconstructed (recommended) |

Image pixels identical — difference is pose accuracy only. FDR pose reconstructed post-landing from DIMU + known landing site.

**PDS:** `urn:nasa:pds:mars2020_edlcam_ops_calibrated`

**Download:** `python3 download.py lcam` → `data/m2020/lcam/`

**Details:** [LCAM_SEQUENCE.md](LCAM_SEQUENCE.md) — sensor specs, TRN processing, label fields, filename convention

**References:** Maki et al. (2020) "The Mars 2020 Engineering Cameras and Microphone" DOI: [10.1007/s11214-020-00765-9](https://doi.org/10.1007/s11214-020-00765-9) | Johnson et al. (2022) "Mars 2020 Lander Vision System Flight Performance" DOI: [10.2514/6.2022-1214](https://doi.org/10.2514/6.2022-1214)

---

## 2. RDCAM (Rover Downlook Camera)

Continuous video from powered descent through touchdown.

| Parameter | Value |
|-----------|-------|
| Frames | ~7000 |
| Resolution | 1280 × 1024, 8-bit color (Bayer) |
| FOV | ~35° × 30° |
| Frame rate | 30 fps |
| Camera model | Not in headers — use SPICE |
| Pose | Not in headers — use SPICE |

**PDS:** `urn:nasa:pds:mars2020_edlcam_ops_calibrated`

**Download:** `python3 download.py rdcam` → `data/m2020/rdcam/`

**Details:** [RDCAM_SEQUENCE.md](RDCAM_SEQUENCE.md) — sensor specs, products, pose recovery

---

## 3. Other EDL Cameras

Also in the EDL camera bundle (not primary focus of this repo):
- `EDL_DDCAM` — Descent Stage Downlook
- `EDL_RUCAM` — Rover Uplook
- `EDL_PUCAM1/2/3` — Parachute Uplook cameras

---

## 4. CTX Orbital Maps (TRN Reference)

The onboard TRN map — matched against LCAM images during descent.

| Product | Resolution | Grid | Download |
|---------|------------|------|----------|
| Orthomosaic | 6 m/pixel | 5040×5322, 8-bit | [GeoTIFF](https://planetarymaps.usgs.gov/mosaic/mars2020_trn/CTX/JEZ_ctx_B_soc_008_orthoMosaic_6m_Eqc_latTs0_lon0.tif) (35 MB) |
| DTM | 20 m/pixel | 1512×1596, 32-bit | [GeoTIFF](https://planetarymaps.usgs.gov/mosaic/mars2020_trn/CTX/JEZ_ctx_B_soc_008_DTM_MOLAtopography_DeltaGeoid_20m_Eqc_latTs0_lon0.tif) (9 MB) |

- Bounds: 18.21°–18.72°N, 77.16°–77.70°E
- Projection: Equirectangular, Mars radius 3,396,190 m
- DTM vertical accuracy: ±3.8 m
- Source: 3 CTX stereo pairs

**All files:** [USGS index](https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/mars2020_trn/CTX/index.html) (40 files incl. individual DTMs, labels, browse)

**References:** Cheng et al. (2021) [10.1029/2020EA001560](https://doi.org/10.1029/2020EA001560), Fergason et al. (2020) LPSC, Johnson et al. (2022) [10.2514/6.2022-1214](https://doi.org/10.2514/6.2022-1214)

---

## 5. HiRISE Orbital Maps

Hazard basemap for Safe Target Selection — not used for landmark matching.

| Product | Resolution | Grid | Download |
|---------|------------|------|----------|
| Orthomosaic | 25 cm/pixel | 85952×85600, 8-bit | [GeoTIFF](https://planetarymaps.usgs.gov/mosaic/mars2020_trn/HiRISE/JEZ_hirise_soc_006_orthoMosaic_25cm_Eqc_latTs0_lon0_first.tif) (6.9 GB) |
| DTM | 1 m/pixel | 21488×21400, 32-bit | [GeoTIFF](https://planetarymaps.usgs.gov/mosaic/mars2020_trn/HiRISE/JEZ_hirise_soc_006_DTM_MOLAtopography_DeltaGeoid_1m_Eqc_latTs0_lon0_blend40.tif) (1.8 GB) |

- Bounds: 18.31°–18.67°N, 77.22°–77.58°E
- Projection: Equirectangular, Mars radius 3,396,190 m
- DTM vertical accuracy: ~0.4 m median offset from CTX
- DTM horizontal accuracy: <1.1 m at 99th percentile

**All files:** [USGS index](https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/mars2020_trn/HiRISE/index.html)

**References:** Fergason et al. (2020) LPSC, Kirk et al. (2020) [10.5194/isprs-archives-XLIII-B3-2020-1129-2020](https://doi.org/10.5194/isprs-archives-XLIII-B3-2020-1129-2020)

---

## 6. SPICE Kernels & Coordinate System

EDL trajectory and attitude reconstruction. LCAM products use `MCMF_FRAME` (Mars-Centered, Mars-Fixed) — equivalent to SPICE `IAU_MARS`.

**Source:** [NAIF PDS Archive](https://naif.jpl.nasa.gov/pub/naif/pds/pds4/mars2020/mars2020_spice/)

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

**References:** Abilleira, F. et al. (2021). "Mars 2020 Perseverance trajectory reconstruction and performance from launch through landing", AAS 21-518, AAS/AIAA Astrodynamics Specialist Conference. [NTRS](https://ntrs.nasa.gov/citations/20220000775)

**Details:** [COORDINATES.md](COORDINATES.md) — MCMF→equirectangular projection, vertical datum

---
