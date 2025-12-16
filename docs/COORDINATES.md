# Coordinate Conventions (Working Notes)

This repo is focused on **Mars 2020 EDL TRN** using `LCAM` (Lander Vision System) products.

## LCAM Pose Fields

In `res/raw/*.IMG`, the `IMAGE_HEADER` contains:
- `ORIGIN_OFFSET_VECTOR` (meters): camera center in `REFERENCE_COORD_SYSTEM_NAME` (commonly `MCMF_FRAME`).
- `ORIGIN_ROTATION_QUATERNION`: camera orientation in the same reference frame.
- `MODEL_TYPE=CAHVORE`: camera model parameters (`MODEL_COMPONENT_1..7`).

## CTX TRN Mosaic CRS

The CTX TRN mosaics in `res/` are GeoTIFFs in an equirectangular projection on a Mars reference sphere:
- Radius `R = 3396190m` (Mars 2000 sphere)
- `x = R * lon`, `y = R * lat` (radians), with `lon0=0`, `lat_ts=0`, no false easting/northing.

You can inspect this via `gdalinfo` (e.g., `gdalinfo res/JEZ_ctx_B_soc_008_orthoMosaic_6m_Eqc_latTs0_lon0.tif`).

## Working Transform: `MCMF_FRAME` → Equirectangular (meters)

Working assumption: `MCMF_FRAME` behaves like a Mars-centered, Mars-fixed Cartesian frame (IAU-style axis conventions).

This matches the common “MCMF” definition used in Mars 2020 EDL trajectory reconstruction literature (Z along Mars’ spin axis, X through the prime meridian).
See: `literature/karlgaard-medli2-trajectory-atmosphere-reconstruction.pdf`.

Given `p = (x, y, z)` in meters:
- `r = sqrt(x^2 + y^2 + z^2)`
- `lon = atan2(y, x)`
- `lat = asin(z / r)`
- `easting_m = R * lon`
- `northing_m = R * lat`

Then use the CTX GeoTIFF geotransform to convert `(easting_m, northing_m)` to pixel coordinates.

## Sanity Script

- `tools/lcam_ctx_project.py` projects every `LCAM` frame into the CTX mosaic bounds and writes:
  - `out/lcam_ctx/lcam_ctx_track.csv`
  - `out/lcam_ctx/lcam_ctx_track.geojson`
  - `out/lcam_ctx/summary.json`

If `in_ortho_bounds` is near 100%, the pose→map projection is at least self-consistent with the CTX mosaic extent.

## SPICE Verification

- `tools/lcam_ctx_verify_spice.py` repeats the lat/lon computation using SPICE (`reclat`) and compares against the direct formula.
- With `pck00010.tpc`, the maximum difference is effectively numerical noise (see `out/lcam_ctx/spice_verify.json`).

## Interactive 3D Scene (Ortho + DTM + Trajectory)

- `tools/trn_scene_mpl.py` opens an interactive 3D window that drapes the CTX orthomosaic over the CTX DTM and overlays the LCAM trajectory in the same equirectangular CRS.
- Requires `numpy`, `rasterio`, and `matplotlib`.
- For performance, use downsampling flags (e.g. `--dtm-downsample 6 --ortho-downsample 6`) and/or `--stride 2`.

### Vertical Datum Note (Why “Altitude Doesn’t Fit”)

The CTX DTM GeoTIFF stores **elevation in meters relative to the MOLA areoid**, while the LCAM pose stores a **Mars-centered Cartesian position** (`ORIGIN_OFFSET_VECTOR`) whose norm is a **Mars-centered radius**.

If you plot the surface as `R_sphere + DTM` (using `R_sphere=3396190m` from the map projection), you implicitly assume the map sphere is the same as the MOLA areoid. It is **not**. Around Jezero, the MOLA areoid radius is ~**3394507m**, which is ~**1683m smaller** than `3396190m`, so you’ll see an apparent ~1.7 km altitude offset.

Sources:
- USGS FGDC metadata for the CTX DTM lists `altdatum=MOLA`: `https://astrogeology.usgs.gov/search/map/mars_2020_terrain_relative_navigation_ctx_dtm_mosaic.xml`
- `out/mars2020_mission_docs/Mars2020_Camera_SIS.txt` (example `SITE,3` row) shows `areoid≈3394507.076m`, `elevation≈-2569.9m`, `radius≈3391937.166m` (radius = areoid + elevation).

`tools/trn_scene_mpl.py` defaults to plotting surface radius as:
- `surface_radius = areoid_radius + dtm_elevation`
and trajectory height mode as:
- `camera_elevation_mola = camera_radius - areoid_radius`

If you still see a small residual, use `--surface-bias-m` only if you have a known registration offset; don’t treat it as a generic “make it line up” knob.

## LCAM Pose vs SPICE `M2020` (Why the “Offset” Isn’t Constant)

It’s tempting to compare the LCAM header position (`ORIGIN_OFFSET_VECTOR`) directly to a SPICE query like:
`spkezr('M2020', et, 'IAU_MARS', 'NONE', 'MARS')` and expect them to match (or differ by a fixed camera lever-arm).
In practice, several things make this comparison non-trivial:

- `ORIGIN_OFFSET_VECTOR` is the **LCAM camera center** in `REFERENCE_COORD_SYSTEM_NAME` (typically `MCMF_FRAME`), updated in FDRs from reconstructed EDL solutions (SIS: `out/mars2020_mission_docs/Mars2020_Camera_SIS.txt`, Section 6.3.2).
- SPICE `M2020` in the EDL SPK is the rover navigation frame origin computed from reconstructed EDL data and a time-dependent modeling offset.
  The EDL SPK (`spice_kernels/m2020_edl_v01.bsp`) explicitly describes:
  - A segment providing `M2020_DIMU_A` (descent IMU) trajectory relative to Mars.
  - A segment giving the **offset between** `M2020_DIMU_A` and `M2020` using “mated” vs “suspended” configurations.
  - A segment making `M2020` coincident with `M2020_LANDING_SITE` (zero offset) after touchdown to bridge to surface SPKs.
- Around late EDL, `M2020` becomes **exactly** coincident with `M2020_LANDING_SITE` in SPICE (zero offset segment),
  so `spkezr('M2020', et, ..., 'M2020_LANDING_SITE')` returns a zero vector after touchdown. If you compare to a pose source that
  continues to move slightly (or uses a different reference/solution), you will see a step/change in the apparent “offset”.

If you want an apples-to-apples “camera lever-arm” comparison, you need the LCAM-to-rover (and possibly LCAM-to-descent) rigid
transform that the SIS says comes from **metrology** (measured mechanical alignment). Those LCAM metrology transforms are not
present in the minimal kernel set currently shipped in `spice_kernels/`, so a constant lever-arm cannot be reproduced from SPICE
alone here.

For M2020 SCLK parsing details (fractional ticks), see `docs/LCAM_METADATA.md`.
