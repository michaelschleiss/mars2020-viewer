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
  - `data/msl/rdr/*.LBL` (detached PDS3 labels)
  - `data/msl/rdr/*.IMG` (detached image payloads)
  - Optional “official volume fetch” (EDR/RDR, EDL phase) via `tools/msl_mardi_fetch_edr_rdr.py` into `data/msl_mardi_volume0003/`

Notes:
- MARDI PDS3 labels include camera model (`MODEL_TYPE=CAHV` with `MODEL_COMPONENT_*`) and timing (`START_TIME`, `SPACECRAFT_CLOCK_START_COUNT`).
- They typically **do not** include a Mars-centered camera pose per frame; use NAIF SPICE for the reconstructed EDL trajectory.
- MARDI EDR/RDR images are stored as a **Bayer mosaic**. The MSL MMM Data Product SIS defines the sensor-space 2x2 tile as:
  - Row 0: `R G2 R G2 ...`
  - Row 1: `G1 B G1 B ...`
  - This corresponds to the common **RGGB** pattern (top-left is red). The viewer defaults to this via `--demosaic auto` in `tools/msl_mardi_view.py`.
- See [MARDI_SENSOR.md](MARDI_SENSOR.md) for Bayer pattern, processing pipeline, and dark columns.

## MARDI Sol 0000 Product Variants (“Quality” Matrix)

MSL MMM product IDs encode two different axes that often get conflated as “quality”:

- **`PGV`** (the 3 chars just before `_`): **product type** + **GOP index** + **version** (from the MMM Data Product SIS “PICNO” naming scheme).
  - `P` is the important part for MARDI: it tells you whether the camera returned a **lossy JPEG** (`E..`) or a **lossless** product (`C..`), or a **thumbnail** (`I..`).
- **Processing code** (the 4 chars after `_`): **ground processing** performed (`XXXX` for EDR, `DRXX/DRCX/DRLX/DRCL` for RDR; see [MARDI_SENSOR.md](MARDI_SENSOR.md)).

### How Many Images Exist In The Sol 0000 EDL Sequence?

Malin et al. (2017) describes the Sol 0000 EDL imaging sequence as **1504 commanded frames** (“commanded as 1504 frames at the maximum frame rate…”; see `out/docs_text/Malin2017_MSL_MARDI_Mastcam.txt:1057`).

The commonly used “descent” subset is **626 in-flight frames** (heat-shield separation → touchdown), which corresponds to `CCCCC=00001..00626` in the EDL PICNO naming used by MARDI.

### Why Does `C00` Look “Missing”?

Because it’s a **lossless re-downlink of only a subset** of frames, not a complete alternate archive of the sequence:

> “Most Mastcam and MARDI images are downlinked as color JPEGs; when downlink data volumes permit, a subset is downlinked a second time with lossless compression…”  
> — Malin et al. (2017), Section 7.7.3 (see `out/docs_text/Malin2017_MSL_MARDI_Mastcam.txt:1182` and `out/docs_text/Malin2017_MSL_MARDI_Mastcam.txt:1190`)

This also shows up in the PDS labels: the **same frame** can have an early `EARTH_RECEIVED_START_TIME` for the JPEG (`E01`) and a much later one for the lossless (`C00`), consistent with “re-downlinked later”.

### `PGV` Codes You’ll See For MARDI Sol 0000

The MMM Data Product SIS defines the `P` (product-type) letter (Table “Image ID or PICNO”; see `out/msl_mmm_docs/MSL_MMM_EDR_RDR_DPSIS.txt:983`):

| `PGV` example | `P` | Meaning (SIS) | What it is in practice |
|---|---:|---|---|
| `E01` | `E` | JPEG 4:2:2 image | Full-frame, lossy JPEG downlink (most common for MARDI) |
| `I01` | `I` | JPEG 4:4:4 thumbnail | Small thumbnail companion product |
| `C00` | `C` | Losslessly compressed raster 8-bit image | Full-frame, lossless subset (re-downlinked later when volume permits) |
| `C01` | `C` | Losslessly compressed raster 8-bit image | Same as `C00`, but a different “version” (`V=1`) |

### What Exists Where (PDS Volumes)

**`MSLMRD_0001` (RDR, Sol 0000, full 1504-frame sequence)**  
Directory: `https://planetarydata.jpl.nasa.gov/img/data/msl/MSLMRD_0001/DATA/RDR/SURFACE/0000/`

- `E01`: present across `CCCCC=00001..01504` (full sequence; multiple RDR processing codes).
- `I01`: present across `CCCCC=00001..01504` (thumbnail companion; multiple RDR processing codes).
- `C00`: present only for a **small subset** of frames in this directory (observed: `350, 354, 451, 461, 471, 481, 491, 501, 513, 525, 526, 532, 534, 543, 545, 1504`).

**`MSLMRD_0002` (EDL-phase, large lossless subset)**  
Index sources:
- `https://planetarydata.jpl.nasa.gov/img/data/msl/MSLMRD_0002/INDEX/EDRINDEX.TAB`
- `https://planetarydata.jpl.nasa.gov/img/data/msl/MSLMRD_0002/INDEX/RDRINDEX.TAB`

For `MISSION_PHASE_NAME="ENTRY, DESCENT, AND LANDING"` and `INSTRUMENT_ID="MD"`:
- EDR contains **432 frames total**: `C00=432` (`CCCCC` spans `00027..00600` with gaps).
- RDR contains those same 432 frames for each of the 4 processing codes (`DRXX/DRCX/DRLX/DRCL`) → **1728 RDR products** total.

**`MSLMRD_0003` (EDL-phase subset, lossless-focused)**  
Index sources:
- `https://planetarydata.jpl.nasa.gov/img/data/msl/MSLMRD_0003/INDEX/EDRINDEX.TAB`
- `https://planetarydata.jpl.nasa.gov/img/data/msl/MSLMRD_0003/INDEX/RDRINDEX.TAB`

For `MISSION_PHASE_NAME="ENTRY, DESCENT, AND LANDING"` and `INSTRUMENT_ID="MD"`:
- EDR contains **192 frames total**: `C00=187` and `C01=5`.
- RDR contains those same 192 frames for each of the 4 processing codes (`DRXX/DRCX/DRLX/DRCL`) → **768 RDR products** total.

This is the root cause of the “~180 images” surprise: if you download only `*_C00_DRCL` from `MSLMRD_0003`, you get **187 frames**. The more complete lossless coverage for the “descent” portion lives mostly in `MSLMRD_0002`, with additional frames filled in by `MSLMRD_0003` and a handful from `MSLMRD_0001`.

### Lossless Frame Coverage (From Local Inventory)

If you’ve fetched `data/msl_mardi_volume0003/rdr/*C00*DRCL*`, the `C00` frames present locally are:

- `CCCCC` ranges (15 ranges, 187 frames):  
  `(215,215), (296,297), (313,317), (319,337), (339,349), (351,353), (355,359), (361,379), (381,399), (402,410), (412,420), (422,430), (432,440), (442,448), (601,660)`

The remaining lossless frames in `MSLMRD_0003` are `C01` (5 frames): `350, 354, 513, 525, 526`.

Across `MSLMRD_0001`+`MSLMRD_0002`+`MSLMRD_0003`, the **union** of lossless MARDI frames for Sol 0000 covers:
- `CCCCC=00027..00660` (no gaps), plus `CCCCC=01504`.
- Within the common “descent imagery” subset `CCCCC=00001..00626`, **600 frames** are available losslessly (`00027..00626`); `00001..00026` do not appear as lossless products in the volume index tables.

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
- `tools/msl_mardi_trajectory.py` uses those kernels + `data/msl/rdr/*.LBL` to write a time-synced trajectory CSV.
