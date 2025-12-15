# Mars 2020 EDL Imagery Viewer - Literature & Documentation

A curated collection of relevant documentation, papers, and resources for the Mars 2020 Perseverance rover EDL imagery viewer and analysis toolkit.

---

## NASA PDS & Data Archives

| Resource | Description |
|----------|-------------|
| [PDS3 Standards Reference](https://pds.nasa.gov/datastandards/pds3/standards/) | Official NASA PDS3 format documentation |
| [PDS3 Data Product Labels (Chapter 5)](https://pds.nasa.gov/datastandards/pds3/standards/sr/Chapter05.pdf) | Detailed specification for PDS label structure in ODL |
| [Mars 2020 Data Archive (PDS Imaging)](https://pds-imaging.jpl.nasa.gov/volumes/mars2020.html) | Raw and calibrated data products |
| [Mars 2020 Mission (Geosciences Node)](https://pds-geosciences.wustl.edu/missions/mars2020/) | Primary archive hub for all Mars 2020 data |
| [Mars Orbital Data Explorer](https://ode.rsl.wustl.edu/mars/) | Interactive data search and download |
| [PDS Main Portal](https://pds.nasa.gov/) | NASA Planetary Data System entry point |

---

## CAHVORE Camera Model

The CAHVORE (Camera, Axis, Horizontal, Vertical, Optical, Radial, Entrance pupil) model is JPL's camera calibration system used for Mars rover imagery.

### Foundational Papers

| Paper | Description |
|-------|-------------|
| [Di & Li (2004) - CAHVOR Photogrammetric Conversion](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2003je002199) | Primary reference for CAHVOR model and conversion to standard photogrammetric parameters |
| [Gennery (2006) - Generalized Camera Calibration Including Fish-Eye Lenses](https://link.springer.com/article/10.1007/s11263-006-5168-1) | Foundational paper for the CAHVORE extension ([JPL Tech Report](https://trs.jpl.nasa.gov/handle/2014/37251)) |
| [Mars 2020 Engineering Cameras Paper](https://link.springer.com/article/10.1007/s11214-020-00765-9) | EDL camera systems and image metadata for Perseverance |
| [Mars 2020 Mastcam-Z Pre-Flight Calibration](https://link.springer.com/article/10.1007/s11214-021-00795-x) | Detailed calibration procedures |
| [MER Engineering Cameras - Maki et al. (2003)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2003JE002077) | Mars Exploration Rover camera calibration |

### Technical Documentation & Tools

| Resource | Description |
|----------|-------------|
| [JPL MER Cameras PDF](https://www-robotics.jpl.nasa.gov/media/documents/MER_Cameras.pdf) | JPL documentation on Mars Exploration Rover camera models |
| [NASA-AMMOS CameraModelUtilsJS](https://github.com/NASA-AMMOS/CameraModelUtilsJS) | Official NASA tools for CAHVORE visualization and conversion |
| [mrcal library](https://mrcal.secretsauce.net/lensmodels.html) | JPL's modern camera calibration toolkit with CAHVORE support ([GitHub](https://github.com/dkogan/mrcal)) |
| [NASA Vision Workbench](https://github.com/visionworkbench/visionworkbench) | C++ implementation including CAHVModel |
| [Python CAHVOR Implementation](https://github.com/bvnayak/CAHVOR_camera_model) | Converting photogrammetric parameters to CAHVOR |
| [Observable CAHVORE Demo](https://observablehq.com/@andreasplesch/the-cahvore-camera-model) | Interactive educational implementation |

### Model Parameters

| Parameter | Description |
|-----------|-------------|
| **C** (Camera) | 3D position of the camera center |
| **A** (Axis) | 3D optical axis direction vector |
| **H** (Horizontal) | 3D vector defining horizontal image direction |
| **V** (Vertical) | 3D vector defining vertical image direction |
| **O** (Optical) | Radial distortion correction vector |
| **R** (Radial) | Radial distortion triplet (3 parameters) |
| **E** (Entrance pupil) | Entrance pupil offset parameters |

### CAHVORE vs OpenCV Brown Model

| Aspect | CAHVORE | OpenCV Brown |
|--------|---------|--------------|
| Projection Type | Noncentral (supports moving entrance pupil) | Central projection only |
| Distortion Model | O/R/E parameters, not directly comparable | Brown-Conrady polynomial model |
| Field of View | Supports very wide (>100°) fish-eye lenses | Limited to standard/wide angle |
| Calibration | Uses precise metrology surveys | Uses least-squares image reprojection |

**Note:** The O/R/E distortion terms don't map directly to OpenCV's Brown model. Use mrcal's `mrcal-to-cahvor` and `mrcal-from-cahvor` for conversion.

---

## SPICE Kernels

SPICE (Spacecraft Planet Instrument C-matrix Events) is NASA's toolkit for spacecraft geometry.

### Core Documentation

| Resource | Description |
|----------|-------------|
| [Mars 2020 SPICE Archive](https://naif.jpl.nasa.gov/pub/naif/pds/pds4/mars2020/mars2020_spice/document/spiceds_v001.html) | Complete Mars 2020 mission SPICE documentation |
| [NAIF SPICE Toolkit](https://naif.jpl.nasa.gov/naif/toolkit.html) | Official SPICE toolkit page |
| [NAIF SPICE Tutorials](https://naif.jpl.nasa.gov/naif/tutorials.html) | Collection of PDF tutorials |
| [SPICE Overview Tutorial](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/Tutorials/pdf/individual_docs/03_spice_overview.pdf) | High-level overview of SPICE capabilities |
| [NAIF Mars Data Page](https://naif.jpl.nasa.gov/naif/data_mars.html) | All Mars mission SPICE kernels |

### Python Interface

| Resource | Description |
|----------|-------------|
| [SpiceyPy Documentation](https://spiceypy.readthedocs.io/en/stable/) | Official Python wrapper documentation |
| [SpiceyPy GitHub](https://github.com/AndrewAnnex/SpiceyPy) | Source code and examples |
| [SpiceyPy on PyPI](https://pypi.org/project/spiceypy/) | Package installation |

### Kernel Types

| Type | Description |
|------|-------------|
| **SPK** | Spacecraft ephemerides (position and velocity) |
| **CK** | Spacecraft and instrument orientation (C-matrix kernels) |
| **PCK** | Planetary constants and target body information |
| **IK** | Instrument parameters and field-of-view definitions |
| **FK** | Reference frame definitions |
| **SCLK** | Spacecraft clock calibration data |
| **DSK** | Detailed shape models |

---

## HiRISE Orbital Imagery

High Resolution Imaging Science Experiment - Mars Reconnaissance Orbiter's high-resolution camera.

| Resource | Description |
|----------|-------------|
| [HiRISE Official Website](https://www.uahirise.org/) | Public data portal and image browser |
| [HiRISE Image Catalog](https://www.uahirise.org/catalog/) | Searchable image database |
| [HiRISE PDS Software Interface Spec](https://pds-imaging.jpl.nasa.gov/documentation/HiRISE_EDR_RDR_Vol_SIS_11_25_2009.pdf) | Official technical specification for EDR/RDR data formats |
| [USGS HiRISE DTMs/Orthoimages](https://stac.astrogeology.usgs.gov/docs/data/mars/hirise_dtms/) | Analysis-ready data products |
| [McEwen et al. (2007) - HiRISE Instrument](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2005JE002605) | Foundational paper |

### Specifications
- Resolution: ~25 cm/pixel (0.25 m/pixel)
- Red band: 20,048 pixels wide (6 km coverage at 300 km orbit)
- Blue-green and NIR: 4,048 pixels wide (1.2 km coverage)
- DTMs: 1-2 meter post spacing with orthorectified imagery

---

## CTX Orbital Imagery

Context Camera - Mars Reconnaissance Orbiter's wide-area imaging camera.

| Resource | Description |
|----------|-------------|
| [NASA Mars - CTX Overview](https://mars.nasa.gov/mro/mission/instruments/ctx/) | Instrument description |
| [CTX PDS Software Interface Spec](https://pds-imaging.jpl.nasa.gov/documentation/MRO_ctx_pds_sis.pdf) | Official technical specification |
| [USGS CTX DTMs/Orthoimages](https://stac.astrogeology.usgs.gov/docs/data/mars/ctxdtms/) | Analysis-ready data pipeline |
| [MSSS CTX Description](https://www.msss.com/mro/ctx/ctx_description.html) | Instrument operations |
| [Dickson et al. (2024) - Global CTX Mosaic](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024EA003555) | Global mosaic paper |

### Specifications
- Resolution: 6 meters/pixel
- Field of view: ~5.7 degrees, covers 30 km swath from 300 km altitude
- Linear CCD: 5000-element Kodak KLI-5001G
- Coverage: >100,000 images covering nearly entire Mars surface

---

## Mars Coordinate Systems & GeoTIFF

| Resource | Description |
|----------|-------------|
| [Coordinate System FAQ (ODE)](https://ode.rsl.wustl.edu/mars/pagehelp/Content/Frequently_Asked_Questions/Coordinate_System.htm) | Clear explanation of coordinate conventions |
| [USGS Map Projections Learning Guide](https://astrogeology.usgs.gov/docs/concepts/camera-geometry-and-projections/learning-about-map-projections/) | Map projection fundamentals |
| [Mars Geodesy/Cartography Working Group Standards](https://www.isprs.org/proceedings/xxxiv/part4/pdfpapers/521.pdf) | Official recommendations |
| [OGC GeoTIFF Standard](https://www.earthdata.nasa.gov/s3fs-public/imported/19-008r4.pdf) | Geographic TIFF specification |
| [OGC Testbed 19 Extraterrestrial GeoTIFF](https://docs.ogc.org/per/23-028.html) | Mars-specific GeoTIFF considerations |

### Mars Standards
- **Coordinate System:** Areocentric/planetocentric latitude with east-positive longitude (0-360°)
- **Map Projections:** Equirectangular for equatorial/mid-latitude; Polar Stereographic for poles
- **Mars Reference Ellipsoid:** Equatorial radius 3,396.19 km, Polar radius 3,376.2 km
- **Standards:** IAU2000 conventions with PositiveEast longitude direction

---

## Official USGS TRN Flight Products

These are the official products created by USGS Astrogeology Science Center for Mars 2020 Terrain Relative Navigation. **The CTX mosaic was the actual onboard reference map.**

### Product Summary

| Product | Resolution | Size | DOI |
|---------|------------|------|-----|
| **CTX Orthomosaic** (onboard map) | 6m/pixel | 35 MB | [10.5066/P9GV1ND3](https://doi.org/10.5066/P9GV1ND3) |
| CTX DTM | 20m/pixel | ~10 MB | — |
| HiRISE Orthomosaic | 25cm/pixel | 3.2 GB | [10.5066/P9QJDP48](https://doi.org/10.5066/P9QJDP48) |
| HiRISE DTM | 1m/pixel | 1.8 GB | [10.5066/P9REJ9JN](https://doi.org/10.5066/P9REJ9JN) |

See [ARTIFACTS.md](ARTIFACTS.md) for local file status.

### CTX Orthomosaic (The Onboard Map)

**This is the Lander Vision System (LVS) map that flew onboard Perseverance.**

**Processing chain:**
```
Individual CTX images (6 stereo pairs)
         ↓
  Uncropped 6m/pixel mosaic (not publicly available)
         ↓
  ┌──────┴──────┐
  ↓             ↓
Cropped 6m    20m/pixel
(onboard)     (derived)
```

| Specification | Value |
|---------------|-------|
| Product ID | `JEZ_ctx_B_soc_008_orthoMosaic_6m_Eqc_latTs0_lon0` |
| Resolution | 6 meters/pixel |
| Dimensions | 5,322 × 5,040 pixels |
| Bit Depth | 8-bit grayscale |
| Projection | Equirectangular (planetocentric) |
| Coverage | 18.21°–18.72°N, 77.16°–77.70°E |
| Horizontal Accuracy | 9.6m average displacement |

**Downloads:**
- [Cropped 6m GeoTIFF (onboard)](https://planetarymaps.usgs.gov/mosaic/mars2020_trn/CTX/JEZ_ctx_B_soc_008_orthoMosaic_6m_Eqc_latTs0_lon0.tif) (35 MB)
- [All CTX Products Index](https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/mars2020_trn/CTX/index.html)
- [USGS Product Page](https://astrogeology.usgs.gov/search/map/Mars/Mars2020/JEZ_ctx_B_soc_008_orthoMosaic_6m_Eqc_latTs0_lon0)

**Extended coverage (Science Investigation, not TRN):**
- [5m/pixel ortho - full Jezero crater](https://astrogeology.usgs.gov/search/map/mars_2020_science_investigation_ctx_dem_mosaic) (post-mission product)

### HiRISE Orthomosaic

**Two versions exist:**

| Version | File | Notes |
|---------|------|-------|
| `soc_006` | `..._first.tif` | Initial processing |
| `soc_007` | `..._blend120.tif` | **Refined version** - better seam blending, used for JPL surface ops |

| Specification | Value |
|---------------|-------|
| Product ID | `JEZ_hirise_soc_007_orthoMosaic_25cm_Ortho_blend120` |
| Resolution | 25 cm/pixel |
| Dimensions | 85,952 × 85,600 pixels |
| Bit Depth | 8-bit grayscale |
| File Size | 3.2 GB (LZW compressed) |
| Coverage | 18.31°–18.67°N, 77.22°–77.58°E |
| Horizontal Accuracy | 99th percentile < 3m |

**Downloads:**
- [soc_007 blend120 (recommended)](https://planetarymaps.usgs.gov/mosaic/mars2020_trn/HiRISE/JEZ_hirise_soc_007_orthoMosaic_25cm_Ortho_blend120.tif) (3.2 GB)
- [soc_006 first](https://planetarymaps.usgs.gov/mosaic/mars2020_trn/HiRISE/JEZ_hirise_soc_006_orthoMosaic_25cm_Eqc_latTs0_lon0_first.tif) (6.9 GB uncompressed)
- [USGS Product Page](https://astrogeology.usgs.gov/search/map/mars_2020_terrain_relative_navigation_hirise_orthorectified_image_mosaic)
- [All HiRISE Products Index](https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/mars2020_trn/HiRISE/index.html)

### HiRISE DTM

| Specification | Value |
|---------------|-------|
| Product ID | `JEZ_hirise_soc_006_DTM_MOLAtopography_DeltaGeoid_1m_Eqc_latTs0_lon0_blend40` |
| Resolution | 1 meter/pixel |
| Dimensions | 21,488 × 21,400 pixels |
| Bit Depth | 32-bit float |
| File Size | 1.8 GB |
| Vertical Reference | MOLA topography with Delta Geoid |

**Downloads:**
- [Full GeoTIFF](https://planetarymaps.usgs.gov/mosaic/mars2020_trn/HiRISE/JEZ_hirise_soc_006_DTM_MOLAtopography_DeltaGeoid_1m_Eqc_latTs0_lon0_blend40.tif) (1.8 GB)
- [USGS Product Page](https://astrogeology.usgs.gov/search/map/Mars/Mars2020/JEZ_hirise_soc_006_DTM_MOLAtopography_DeltaGeoid_1m_Eqc_latTs0_lon0_blend40)

### Key References for TRN Products

| Reference | Description |
|-----------|-------------|
| [Fergason et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020LPI....51.2020F) | "Mars 2020 TRN Flight Product Generation" - 51st LPSC. Primary reference for all USGS products. |
| [Cheng et al. (2021)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020EA001560) | "Making an Onboard Reference Map From MRO/CTX Imagery" - Earth & Space Science. Details CTX map creation. |
| [Tao et al. (2023)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023EA003045) | "High-Resolution DTM Mosaic of Mars 2020 Landing Site" - Post-mission HiRISE DTM refinement using MADNet. |

### Processing Pipeline

The USGS products were created using:
1. **SOCET SET** (BAE Systems) - stereo photogrammetry
2. **Ames Stereo Pipeline (ASP)** - DTM generation and alignment
3. **ISIS3** - radiometric calibration and map projection
4. **GDAL** - mosaic generation and format conversion

**Quality metrics:**
- HiRISE pairwise registration: 99th percentile < 3m, most overlaps < 1.1m
- HiRISE vs CTX alignment: 94.1% of features < 6m displacement
- CTX individual images: 9.6m average horizontal displacement

---

## Terrain Relative Navigation & Feature Matching

### JPL Terrain Relative Navigation (TRN)

**Directly relevant to the matching pipeline in this project.**

| Resource | Description |
|----------|-------------|
| [JPL TRN Page](https://www-robotics.jpl.nasa.gov/what-we-do/flight-projects/mars-2020-rover/terrain-relative-navigation/) | Official JPL documentation on the system used for Mars 2020 landing |

**How TRN worked during landing:**
1. **Coarse matching** (altitude >4200m): 5 large patches in each of 3 descent images matched to ~12m/pixel map
2. **Fine matching**: Up to 150 small patches per image matched to 6m/pixel CTX map (the onboard reference)
3. **Result**: <40m position error requirement met; actual landing within 5m of target

**Key insight:** The onboard map was **CTX at 6m/pixel**, not HiRISE. HiRISE was used for hazard mapping, not for TRN position fixes.

### OpenCV Feature Matching

| Resource | Description |
|----------|-------------|
| [OpenCV Feature Matching Tutorial](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html) | Official documentation |
| [OpenCV ORB Tutorial](https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html) | Oriented FAST and Rotated BRIEF |
| [OpenCV Feature Detection Overview](https://docs.opencv.org/4.x/db/d27/tutorial_py_table_of_contents_feature2d.html) | Complete feature2d module |
| [OpenCV FLANN Matcher](https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html) | Fast approximate nearest neighbors |

### Research Papers on Aerial/Satellite Registration

| Paper | Description |
|-------|-------------|
| [Ground-to-Satellite Matching (CMU)](https://www.ri.cmu.edu/pub_files/2015/9/anirudh_viswanathan_iros.pdf) | Vision-based robot localization |
| [NVIDIA Rapid Aerial/Orbital Registration](https://developer.nvidia.com/blog/rapid-registraion-aerial-orbital-imagery/) | GPU-accelerated registration |
| [UAV Image Registration (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10051850/) | Real-time registration algorithm |
| [Multiview Image Matching (MDPI)](https://www.mdpi.com/2072-4292/14/4/838) | Neural network-based matching |
| [MIT UAV-Satellite Registration](https://dspace.mit.edu/handle/1721.1/119920) | Automatic feature detection and matching |

---

## Mars 2020 EDL Mission Resources

| Resource | Description |
|----------|-------------|
| [NASA EDL Profile](https://science.nasa.gov/resource/perseverance-rovers-entry-descent-and-landing-profile/) | Official EDL sequence overview |
| [NASA Eyes EDL Interactive](https://eyes.nasa.gov/apps/mars2020/) | Interactive 3D visualization |
| [Descent and Touchdown Video](https://mars.nasa.gov/resources/25628/perseverance-rovers-descent-and-touchdown-on-mars-onboard-camera-views/) | Onboard camera footage |
| [NASA Mars EDL Technical Document](https://www.nasa.gov/wp-content/uploads/2025/02/iparch12-wp-mars-edl.pdf) | Technical overview of EDL systems |

---

## Data Access Points

| Repository | URL |
|------------|-----|
| PDS Imaging Node | https://pds-imaging.jpl.nasa.gov/ |
| PDS Geosciences Node | https://pds-geosciences.wustl.edu/ |
| USGS Astrogeology | https://www.usgs.gov/centers/astrogeology-science-center |
| AWS Open Data - CTX DTMs | https://registry.opendata.aws/nasa-usgs-controlled-mro-ctx-dtms/ |
| NAIF SPICE Archives | https://naif.jpl.nasa.gov/naif/data_archived.html |

---

### Software Dependencies Not in Requirements

| Package | Purpose | Install |
|---------|---------|---------|
| [mrcal](https://mrcal.secretsauce.net/) | Proper CAHVORE ↔ OpenCV conversion | `pip install mrcal` |
| [GDAL](https://gdal.org/) | GeoTIFF manipulation, reprojection | `mamba install gdal` |
| [pyproj](https://pyproj4.github.io/pyproj/) | Mars coordinate transformations | `mamba install pyproj` |

---

*Last updated: December 2024*
