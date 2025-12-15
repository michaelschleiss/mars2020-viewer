# MSL (Curiosity) Official Artifacts (Tracking)

This file tracks the "official" upstream artifacts relevant to **MSL Entry/Descent/Landing (EDL)** and **MARDI descent imagery**.

## Landing Site & MARDI Overview

| Parameter | Value |
|-----------|-------|
| Landing site | Gale Crater / Bradbury Landing (4.5895°S, 137.4417°E) |
| Landing date | 2012-08-06 05:32 UTC |
| Elevation | -4501 m (MOLA datum) |
| TRN status | Not flown (DIMES velocity estimation only) |

**MARDI (Mars Descent Imager):**

| Parameter | Value |
|-----------|-------|
| Resolution | 1600 × 1200 pixels |
| Field of view | 70° × 55° |
| Color | RGB (Bayer CCD) |
| Bit depth | 8-bit (RDR), 12-bit native |
| Focal length | 9.7 mm, f/3 |
| Frame rate | ~4 fps |
| Camera model | CAHVOR |

**Descent imagery:** 626 in-flight frames (heat shield separation to touchdown), ~10 km to surface.

**Reference:** Malin et al. (2017) "The Mars Science Laboratory (MSL) Mast cameras and Descent imager" — https://pmc.ncbi.nlm.nih.gov/articles/PMC5652233/

## USGS EDL Reference Mosaics (Gale)

These mosaics were produced as **EDL assessment tools** for MSL and are the most citable “reference map/DEM” artifacts for descent-image validation at Gale.

### Orthophoto Mosaic (25 cm)

- Product page: `https://astrogeology.usgs.gov/search/map/mars_msl_gale_merged_orthophoto_mosaic_25cm`
- Direct download (GeoTIFF): `https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/MSL_Gale_Orthophoto_Mosaic_25cm_v3.tif`
- Product page label links (projection/datum metadata):
  - PDS3 label: `https://astrogeology.usgs.gov/ckan/dataset/06137610-fb79-4b45-a5ef-63c4869d48b2/resource/2ce00e4e-69a1-48d8-bdbf-62ef729e18ef/download/msl_gale_orthophoto_mosaic_25cm_v3_pds3.lbl`
  - ISIS3 label: `https://astrogeology.usgs.gov/ckan/dataset/06137610-fb79-4b45-a5ef-63c4869d48b2/resource/41e01c27-bc3e-44d6-a7af-f9c18f6f457d/download/msl_gale_orthophoto_mosaic_25cm_v3.lbl`
- Recommended citation (per product page): Calef III, F. J., & Parker, T. (2016). *MSL Gale Merged Orthophoto Mosaic*. PDS Annex, U.S. Geological Survey.

### DEM Mosaic (1 m)

- Product page: `https://astrogeology.usgs.gov/search/map/mars_msl_gale_merged_dem_1m`
- Direct download (GeoTIFF): `https://planetarymaps.usgs.gov/mosaic/Mars/MSL/MSL_Gale_DEM_Mosaic_1m_v3.tif`
- Product page label links (projection/datum metadata):
  - PDS3 label: `https://astrogeology.usgs.gov/ckan/dataset/b3d63877-8544-49bb-b0f3-532cf2c648d1/resource/cd140a94-1e66-46e0-8d5a-d8f0e4128b7e/download/msl_gale_dem_mosaic_1m_v3_pds3.lbl`
  - ISIS3 label: `https://astrogeology.usgs.gov/ckan/dataset/b3d63877-8544-49bb-b0f3-532cf2c648d1/resource/d1a8afb0-3a48-4dcb-92a8-6c5794bb662f/download/msl_gale_dem_mosaic_1m_v3.lbl`
- Recommended citation (per product page): Parker, T. & Calef III, F. J. (2016). *Mars MSL Gale Merged DEM*. PDS Annex, U.S. Geological Survey.

## MARDI Descent Images (PDS)

- Dataset ID seen in your local labels: `MSL-M-MARDI-4-RDR-IMG-V1.0`
- Local example labels/images:
  - `data/msl/labels/*.LBL` (detached PDS3 labels)
  - `data/msl/images/*.IMG` (detached image payloads)
  - Optional “official volume fetch” (EDR/RDR, EDL phase) via `tools/msl_mardi_fetch_edr_rdr.py` into `data/msl_mardi_volume0003/`

Notes:
- MARDI PDS3 labels include camera model (`MODEL_TYPE=CAHV` with `MODEL_COMPONENT_*`) and timing (`START_TIME`, `SPACECRAFT_CLOCK_START_COUNT`).
- They typically **do not** include a Mars-centered camera pose per frame; use NAIF SPICE for the reconstructed EDL trajectory.
- MARDI EDR/RDR images are stored as a **Bayer mosaic**. The MSL MMM Data Product SIS defines the sensor-space 2x2 tile as:
  - Row 0: `R G2 R G2 ...`
  - Row 1: `G1 B G1 B ...`
  - This corresponds to the common **RGGB** pattern (top-left is red). The viewer defaults to this via `--demosaic auto` in `tools/msl_mardi_view.py`.
- See [MARDI_SENSOR.md](MARDI_SENSOR.md) for Bayer pattern, processing pipeline, and dark columns.

## Local Conventions (This Repo)

- Existing local set (already downloaded): `data/msl/{labels,images}/`
  - Contains `*.LBL`/`*.IMG` pairs for `*_E01_DRCL` products (RDR) used in our current viewer/trajectory scripts.
- Official MSLMRD_0003 volume fetch (optional): `data/msl_mardi_volume0003/{edr,rdr}/{labels,images}/`
  - `tools/msl_mardi_fetch_edr_rdr.py` can pull EDL-phase products (commonly `*_C00_*` in this volume) directly from `https://planetarydata.jpl.nasa.gov/img/data/msl/MSLMRD_0003/`.
  - The **EDR** payloads are `*.DAT` with `ENCODING_TYPE="MSLMMM-COMPRESSED"` and must be decompressed to `*.IMG` before pixel viewing:
    - `python3 tools/msl_mardi_decompress_edr.py --skip-existing`
    - Outputs: `data/msl_mardi_volume0003/edr_uncompressed/{labels,images}/` (raw raster + detached labels produced by the volume’s `dat2img` tool).

## NAIF SPICE (Reconstructed EDL Trajectory + Attitude)

Primary EDL kernels (NAIF, MSL):
- Trajectory SPK: `https://naif.jpl.nasa.gov/pub/naif/MSL/kernels/spk/msl_edl_v01.bsp`
- Attitude CK: `https://naif.jpl.nasa.gov/pub/naif/MSL/kernels/ck/msl_edl_v01.bc`

Supporting kernels commonly required:
- Frames FK: `https://naif.jpl.nasa.gov/pub/naif/MSL/kernels/fk/msl.tf`
- MARDI IK: `https://naif.jpl.nasa.gov/pub/naif/MSL/kernels/ik/msl_mardi_20120731_c02.ti`
- SCLK: `https://naif.jpl.nasa.gov/pub/naif/MSL/kernels/sclk/msl.tsc`
- LSK: `https://naif.jpl.nasa.gov/pub/naif/MSL/kernels/lsk/naif0012.tls`
- Generic PCK (planet constants): `https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00010.tpc`

Local helper scripts in this repo:
- `tools/usgs_fetch_msl_labels.py` downloads the USGS PDS3/ISIS3 labels into `out/usgs_msl_labels/`.
- `tools/spice_fetch_msl_edl.py` downloads a minimal MSL EDL kernel set and writes a meta-kernel for `spiceypy`.
- `tools/msl_mardi_trajectory.py` uses those kernels + `data/msl/labels/*.LBL` to write a time-synced trajectory CSV.

