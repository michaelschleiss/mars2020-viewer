# Data Inventory (MSL Curiosity)

## Overview

| Parameter | Value |
|-----------|-------|
| Landing site | Gale Crater / Bradbury Landing (4.5895°S, 137.4417°E) |
| Landing date | 2012-08-06 05:32 UTC |
| Elevation | -4501 m (MOLA datum) |
| TRN status | Not flown (DIMES velocity estimation only) |

---

## 1. MARDI (Mars Descent Imager)

The primary descent camera — documented entry through touchdown with color imagery.

| Parameter | Value |
|-----------|-------|
| Frames | 626 (heat shield separation to touchdown) |
| Resolution | 1600 × 1200, 8-bit RGB (Bayer mosaic) |
| FOV | 70° × 55° |
| Frame rate | ~4 fps |
| Focal length | 9.7 mm, f/3 |
| Camera model | CAHVOR (embedded in header) |
| Pose | Not in headers — use SPICE |

### Products

| Type | Format | Notes |
|------|--------|-------|
| EDR | Compressed `.DAT` | MSLMMM-COMPRESSED format, requires decompression |
| **RDR** | Uncompressed `.IMG` | Bayer mosaic, RGGB pattern (recommended) |

**PDS:** `MSL-M-MARDI-4-RDR-IMG-V1.0` (volumes `MSLMRD_0001`–`MSLMRD_0003`)

**Labels:** detached PDS3 `.LBL` paired with `.IMG` (local convention: co-located under `data/msl/rdr/`)

**Download:** `python3 download.py mardi` → `data/msl/rdr/`

**Details:** [MARDI_SENSOR.md](MARDI_SENSOR.md) — Bayer pattern, dark columns, processing pipeline

## 2. HiRISE Orbital Maps (EDL Reference)

EDL assessment basemap — reference map and DEM for descent-image validation at Gale.

| Product | Resolution | Download |
|---------|------------|----------|
| Orthomosaic | 25 cm/pixel | [GeoTIFF](https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/MSL_Gale_Orthophoto_Mosaic_25cm_v3.tif) |
| DEM | 1 m/pixel | [GeoTIFF](https://planetarymaps.usgs.gov/mosaic/Mars/MSL/MSL_Gale_DEM_Mosaic_1m_v3.tif) |

**Product pages:** [Orthomosaic](https://astrogeology.usgs.gov/search/map/mars_msl_gale_merged_orthophoto_mosaic_25cm) | [DEM](https://astrogeology.usgs.gov/search/map/mars_msl_gale_merged_dem_1m)

**References:** Calef III, F. J., & Parker, T. (2016). MSL Gale Merged Orthophoto Mosaic. PDS Annex, U.S. Geological Survey. | Parker, T. & Calef III, F. J. (2016). Mars MSL Gale Merged DEM. PDS Annex, U.S. Geological Survey.

---

## 3. SPICE Kernels & Coordinate System

EDL trajectory and attitude reconstruction.

**Source:** [NAIF PDS Archive](https://naif.jpl.nasa.gov/pub/naif/MSL/)

**Minimal EDL kernel set:**
| Kernel | Purpose |
|--------|---------|
| `spk/msl_edl_v01.bsp` | Trajectory |
| `ck/msl_edl_v01.bc` | Attitude |
| `fk/msl.tf` | Frame definitions |
| `ik/msl_mardi_20120731_c02.ti` | MARDI instrument |
| `sclk/msl.tsc` | Spacecraft clock |
| `pck/pck00010.tpc` | Mars orientation (IAU_MARS) |
| `lsk/naif0012.tls` | Leap seconds |

**Helper:** `tools/spice_fetch_msl_edl.py`

**References:** Abilleira, F. (2013). "2011 Mars Science Laboratory Trajectory Reconstruction and Performance from Launch Through Landing", AAS 04-113, AAS/AIAA Spaceflight Mechanics Meeting. [NTRS](https://ntrs.nasa.gov/search.jsp?R=20150007334)

---
