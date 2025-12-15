# Official Artifacts (Tracking)

This file tracks the "official" upstream artifacts relevant to **Mars 2020 Entry/Descent/Landing (EDL) TRN**, what they are, and where they come from.

## Landing Site & LCAM Overview

| Parameter | Value |
|-----------|-------|
| Landing site | Jezero Crater (18.4447°N, 77.4508°E) |
| Landing date | 2021-02-18 20:55 UTC |
| Elevation | -2.5 km (MOLA datum) |
| TRN status | Operational (first Mars mission with active TRN) |

**LCAM (Lander Vision System Camera):**

| Parameter | Value |
|-----------|-------|
| Resolution | 1024 × 1024 pixels |
| Field of view | 90° × 90° |
| Color | Grayscale |
| Bit depth | 8-bit |
| Camera model | CAHVORE |

**Descent imagery:** 87 calibrated frames from ~4 km altitude to touchdown.

## Primary Statement of Which Maps Flew

- **Fergason et al. (LPSC 2020)**, “Mars 2020 Terrain Relative Navigation Flight Product Generation: Digital Terrain Model and Orthorectified Image Mosaics.” `https://www.hou.usra.edu/meetings/lpsc2020/pdf/2020.pdf`
  - States the onboard TRN “truth dataset” was the **LVS appearance map from three CTX orthorectified images**.
  - States the **HiRISE orthomosaic** was the **hazard basemap**, and the derived **hazard map** was onboard for TRN hazard avoidance.

## Orbital Map / DTM Products (USGS)

**CTX (onboard-style appearance prior)**
- Product page: `https://astrogeology.usgs.gov/search/map/mars_2020_terrain_relative_navigation_context_camera_orthorectified_image_mosaic`
- What it is: CTX orthorectified mosaic used to generate the **LVS appearance map** (per Fergason et al., 2020).
- Local file(s) in this repo (examples):
  - `res/JEZ_ctx_B_soc_008_orthoMosaic_6m_Eqc_latTs0_lon0.tif` (GeoTIFF, equirectangular Mars 2000 sphere)

**CTX DTM (terrain prior aligned to the appearance map)**
- Product page: `https://astrogeology.usgs.gov/search/map/mars_2020_terrain_relative_navigation_ctx_dtm_mosaic`
- What it is: CTX DTM mosaic co-registered with the CTX orthomosaic.
- Local file(s) in this repo (examples):
  - `res/JEZ_ctx_B_soc_008_DTM_MOLAtopography_DeltaGeoid_20m_Eqc_latTs0_lon0.tif`

**HiRISE (hazard basemap / high-resolution reference)**
- Orthomosaic product page: `https://astrogeology.usgs.gov/search/map/mars_2020_terrain_relative_navigation_hirise_orthorectified_image_mosaic`
- DTM mosaic product page (DOI landing page): `https://doi.org/10.5066/P9REJ9JN`
- What it is: high-resolution mosaic used as a basemap for hazard mapping; not necessarily the direct “appearance map” used for onboard matching.
- Local file(s) in this repo (examples):
  - `res/JEZ_hirise_soc_007_orthoMosaic_25cm_Ortho_blend120.tif`

## MSL (Curiosity) Artifacts

See **[ARTIFACTS_MSL.md](ARTIFACTS_MSL.md)** for MSL/MARDI descent imagery, USGS Gale mosaics, and SPICE kernels.

## Coordinate Systems & Vertical Datum

See **[COORDINATES.md](COORDINATES.md)** for:
- MCMF_FRAME → equirectangular transform formulas
- Vertical datum considerations (MOLA areoid vs map sphere radius)
- Sanity/verification tooling (`tools/lcam_ctx_project.py`, `tools/lcam_ctx_verify_spice.py`)

## EDL Camera Image Products (PDS)

**Bundle family**
- PDS Registry bundle identifiers seen for EDL camera products:
  - `urn:nasa:pds:mars2020_edlcam_ops_raw::*`
  - `urn:nasa:pds:mars2020_edlcam_ops_calibrated::*`

**Instrument streams discovered via PDS Registry (Sol 0 examples)**
- `LCAM` (Lander Vision System) — this repo’s current focus (`res/raw/*.IMG`).
- `EDL_DDCAM` (Descent Stage Downlook), `EDL_RDCAM` (Rover Downlook), `EDL_RUCAM` (Rover Uplook), `EDL_PUCAM1/2/3` (Parachute Uplook A/B/C).

**Which streams contain “TRN-ready” geometry inside the `.IMG`**
- `LCAM` `.IMG` files include:
  - `MODEL_TYPE=CAHVORE` (camera model) with `MODEL_COMPONENT_1..7`
  - `ORIGIN_OFFSET_VECTOR` + `ORIGIN_ROTATION_QUATERNION`
  - `REFERENCE_COORD_SYSTEM_NAME` (e.g., `MCMF_FRAME`)
- The probed `EDL_*` camera `.IMG` samples do **not** embed CAHVORE/pose fields in their PDS3 `IMAGE_HEADER` (you’d need geometry from elsewhere to use them for TRN).
- Probe script: `tools/edl_registry_probe.py`

**“Ancillary” products inside `mars2020_edlcam_ops_calibrated::1.1`**
- The only `Product_Ancillary` items in the calibrated EDL camera bundle are **collection inventory CSVs** (not calibration models):
  - `https://pds-imaging.jpl.nasa.gov/archive/m20/r8/mars2020_edlcam_ops_calibrated/data/r8_collection_data_inventory.csv`
  - `https://pds-imaging.jpl.nasa.gov/archive/m20/r9/mars2020_edlcam_ops_calibrated/data/r9_collection_data_inventory.csv`
  - `https://pds-imaging.jpl.nasa.gov/archive/m20/r8/mars2020_edlcam_ops_calibrated/browse/r8_collection_browse_inventory.csv`
  - `https://pds-imaging.jpl.nasa.gov/archive/m20/r9/mars2020_edlcam_ops_calibrated/browse/r9_collection_browse_inventory.csv`

## Camera Modeling References (EDL + LCAM)

**Mars 2020 Camera SIS (mission-wide)**
- PDS “mission” bundle: `urn:nasa:pds:mars2020_mission::4.0`
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

**EDLcam “raw video/audio” SIS**
- PDS bundle: `urn:nasa:pds:m2020_edlcam_raw::1.1`
  - `m2020_edlcam_raw_bundle_sis.pdf`:
    - `https://pds-imaging.jpl.nasa.gov/archive/m20/cumulative/m2020_edlcam_raw/document/m2020_edlcam_raw_bundle_sis.pdf`
  - PDS “data” landing page (directory listing):
    - `https://pds-imaging.jpl.nasa.gov/data/mars2020/m2020_edlcam_raw/`
  - Raw EDL videos (including RDC videos):
    - `https://pds-imaging.jpl.nasa.gov/data/mars2020/m2020_edlcam_raw/document_video/`

**Engineering cameras paper**
- Springer DOI: `10.1007/s11214-020-00765-9`
  - PDF: `https://link.springer.com/content/pdf/10.1007/s11214-020-00765-9.pdf`
  - JPL Dataverse mirror (PDF file `CL20_5537.pdf`): `https://doi.org/10.48577/jpl.JQOSAY`

**What we have (and don’t have) for `EDL_RDCAM`**
- `EDL_RDCAM` PDS3 `.IMG` products do not include CAHV* vectors or per-frame pose in `IMAGE_HEADER` (unlike `LCAM`).
- The public NAIF Mars2020 SPICE archive contains **no instrument kernel (IK) for EDL cameras** (only `ik/m2020_struct_v00.ti`, which defines antenna FOVs).
- Practical path (until a per-camera CAHVORE is found publicly): use the EDL camera suite “first-order” intrinsics from `Mars2020_Camera_SIS.pdf` / the engineering cameras paper (FOV, focal length, pixel pitch), and treat lens distortion as unknown/unmodeled unless you fit it from data.
  - Best-available public intrinsics for `EDL_RDCAM` right now:
    - Image size: `1280×1024`
    - Focal length: `~9.5 mm`
    - Pixel pitch: `~4.8 µm`
    - Field of view: `~35° × 30°`
    - Pinhole approximation (pixels): `fx≈fy≈9.5/0.0048≈1979`, `cx≈(1280-1)/2=639.5`, `cy≈(1024-1)/2=511.5`
    - Distortion: not provided publicly in PDS/NAIF artifacts (treat as unknown unless you fit/calibrate)

**Meaning of EDL camera suite “raw” vs “calibrated” still images**
- Per `Mars2020_Camera_SIS.pdf` Section “5.6.7 EDL Camera Suite”, EDLcam data were downlinked as JPEG frames, MPEG/H.264 video, and (later) lossless uncompressed frames; the pipeline then extracts still frames from those sources.
- In the EDL still-frame archives (`mars2020_edlcam_ops_raw` / `mars2020_edlcam_ops_calibrated`), this shows up as:
  - `ECM` / `ECV` (raw source frames, including Bayer-encoded lossless frames),
  - `EDR` / `EVD` (“partially processed”; for byte-space EDLcam it’s effectively a copy of the corresponding raw frame),
  - `FDR` (“derived” / calibrated; debayered/demosaiced RGB frame, generally the most convenient format to start with).

## Coordinate Frames & Meanings (Needs Verification)

- `REFERENCE_COORD_SYSTEM_NAME=MCMF_FRAME` appears in `LCAM` products.
  - Treat as “Mars-centered, Mars-fixed” until verified against NAIF/SPICE naming; keep the original string in any exported benchmark.
- `ORIGIN_OFFSET_VECTOR` and CAHVORE `MODEL_COMPONENT_1` (“C”) are both in meters and very close numerically for `LCAM` products (offset often differs by ~1 m).

## SPICE Kernel Recommendation (Frame/Time Conversions)

If you want to do frame conversions “the SPICE way” (rather than assuming formulas), use the official NAIF Mars2020 kernel set.

**Recommended starting point**
- NAIF Mars2020 meta-kernel (mission-wide): `https://naif.jpl.nasa.gov/pub/naif/pds/pds4/mars2020/mars2020_spice/spice_kernels/mk/m2020_v14.tm`
  - Archive root: `https://naif.jpl.nasa.gov/pub/naif/pds/pds4/mars2020/mars2020_spice/`
  - Note: the kernels are a **separate official PDS archive** (NAIF) from the PDS Imaging EDL camera products; they are not “packaged inside” each `.IMG`.

**Minimal subset for EDL + MCMF-style map projection**
- `lsk/naif0012.tls` (leap seconds)
- `pck/pck00010.tpc` (Mars orientation + radii; defines `IAU_MARS`)
- `fk/m2020_v04.tf` (Mars2020 frame definitions such as `M2020_TOPO`, `M2020_LOCAL_LEVEL`)
- `sclk/m2020_168_sclkscet_refit_v14.tsc` (SCLK↔ET, if using spacecraft clock)
- `spk/m2020_edl_v01.bsp` + `ck/m2020_edl_v01.bc` (EDL trajectory/orientation coverage)
- `spk/m2020_ls_ops210303_iau2000_v1.bsp` (landing site kernel used by the surface frame chain)

**Why this matters for `MCMF_FRAME`**
- LCAM PDS labels use `MCMF_FRAME` for camera model/pose coordinate space.
- In the trajectory reconstruction literature, “MCMF” is defined as “Mars-centered, Mars-fixed” with Z along Mars’ spin axis and X through the prime meridian (i.e., the same concept SPICE represents with `IAU_MARS` + the PCK).
  - Example: `literature/karlgaard-medli2-trajectory-atmosphere-reconstruction.pdf`

**Local helper**
- `tools/spice_fetch_mars2020_edl.py` downloads a minimal EDL kernel set from the NAIF archive and writes `out/spice/mars2020_edl_naif/m2020_edl_min.tm` for `furnsh(...)`.

## Notes / Conventions for Benchmark Export

- Prefer preserving the native camera model as **CAHVORE** (do not collapse to OpenCV Brown unless you explicitly fit a distortion approximation).
- Keep raw provenance: include source URLs, bundle/collection IDs, and original filenames in any published benchmark.
