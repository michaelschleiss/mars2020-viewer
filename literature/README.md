# Literature

References organized by relevance to this Mars EDL imagery viewer repository.

---

## Category 1: Specifications / System Design

Primary sources for Mars 2020 TRN/LCAM and MSL MARDI specifications.

**Maki et al. (2020)** — "The Mars 2020 Engineering Cameras and Microphone on the Perseverance Rover: A Next-Generation Imaging System for Mars Exploration"

- PDF: `Maki2020_Mars2020_Engineering_Cameras.pdf`
- DOI: https://doi.org/10.1007/s11214-020-00765-9
- Relevance to this repo:
  - **Definitive LCAM reference**: 90°×90° FOV, 1024×1024 pixels (2×2 summed from Python 5000 detector), 1.67 mrad/pixel, global shutter
  - Documents LCAM → LVS interface: images matched to CTX-derived onboard basemap during parachute descent (4200m to 500m altitude)
  - Specifies CAHVORE camera model usage for all engineering cameras (calibration in rover nav frame)
  - Describes PDS archival format for descent imagery
  - EDLCAM system: RDC (Rover Downlook Camera) captures 30 fps video through touchdown — relevant for structure-from-motion
  - Table 5 contains complete LCAM specs: f/2.7, 5.8mm focal length, On Semiconductor Python 5000

**Johnson et al. (2017)** — "The Lander Vision System for Mars 2020 Entry Descent and Landing"

- PDF: `Mars2020_Lander_Vision_System_Johnson2017.pdf`
- Relevance to this repo:
  - **LVS algorithm design**: Reduces 3.2km initial position error to 40m in 10 seconds using landmark matching
  - Coarse matching at 4200m AGL (5 large patches per image to 12m/pixel map), fine matching continues to 2000m AGL (150 small patches to 6m/pixel map)
  - **LCAM requirements**: 1024×1024 pixels, 90°×90° FOV, global shutter, ~1ms exposure, <100ms latency, 18% radial distortion
  - **Vision Compute Element (VCE)**: RAD750 processor + Virtex5QV FPGA for image processing, inherits MSL avionics heritage
  - Onboard map from CTX at 6m/pixel, cropped to 12km on a side; map position error <150m horizontal, <20m vertical
  - Simulation shows focal length calibration error (±1%) contributes up to 17m horizontal position error
  - Heat shield rejection: LVS matches terrain even with heat shield visible at 3m distance

**Malin et al. (2017)** — "The Mars Science Laboratory (MSL) Mast cameras and Descent imager: Investigation and instrument descriptions"

- PDF: `Malin2017_MSL_MARDI_Mastcam.pdf`
- DOI: https://doi.org/10.1002/2016EA000252
- Relevance to this repo:
  - **Definitive MSL MARDI reference**: f/3 lens, 9.7mm focal length, 2M pixel color camera (1600×1200 Bayer CFA)
  - FOV ~70°×52°, IFOV 0.76 mrad, focus 2m to infinity (out of focus at 66cm surface mount height)
  - Descent video at 4 fps full frame, 6 fps at 720p HD
  - MARDI mounted on bottom left front of rover at 66cm above surface
  - Primary objective: determine landing site location within 1 sol, bridge resolution gap between orbital and landed cameras
  - Shares electronics/focal plane design with Mastcams; common Bayer demosaic and JPEG/lossless compression

**Fergason et al. (2020)** — "Mars 2020 Terrain Relative Navigation Flight Product Generation: Digital Terrain Model and Orthorectified Image Mosaics"

- PDF: `Fergason2020_TRN_Flight_Products_LPSC.pdf`
- Conference: 51st Lunar and Planetary Science Conference (LPSC 2020)
- Relevance to this repo:
  - Explicitly states the **onboard TRN "truth" appearance map** was the **LVS map generated from three CTX orthorectified images**
  - Separately states a **HiRISE orthomosaic** was used as the **hazard basemap**, and the derived **hazard map** was onboard for safe landing site selection
  - CTX LVS mosaic: 6m/pixel ortho, 20m/pixel DTM; generated using SOCET SET + Ames Stereo Pipeline (ASP)
  - JPL-provided camera model and jitter corrections eliminated need for bundle adjustment (TRN sensitive to non-linear distortions)
  - CTX DTMs aligned to HRSC Level 5 DTM for absolute reference; vertical registration 3.8m, horizontal 9.6m average
  - HiRISE hazard basemap: 1m/pixel DTM, 14 images bundle-adjusted, manually edited to remove artifacts
  - USGS Astrogeology Science Center produced these flight products

---

## Category 2: Flight Analysis / Post-Processing

Post-landing analysis and trajectory reconstruction.

**Johnson et al. (2022)** — "Mars 2020 Lander Vision System Flight Performance"

- PDF: `Mars2020_LVS_Flight_Performance_Johnson2022.pdf`
- Relevance to this repo:
  - **Flight-proven results**: Final touchdown error was 5m (vs 60m requirement) — order of magnitude better than required
  - TRN composed of LVS (position estimation) + STS (Safe Target Selection) working together
  - LVS delivered during parachute descent between navigation solution acquisition and backshell separation
  - VCE architecture: stand-alone compute element with RAD750 + Virtex5 FPGA, processes ~150 landmarks per image
  - Full state estimation (position, velocity, attitude) internal to LVS allows compensation for initial attitude/velocity errors
  - "Bolt-on sensor" design minimized impact on heritage MSL EDL system
  - V&V "Trifecta" approach: simulation, testbed, and field test

**Abilleira et al. (2021)** — "Mars 2020 Perseverance Trajectory Reconstruction and Performance from Launch Through Landing"

- PDF: `Mars2020_Trajectory_Reconstruction_Abilleira2021.pdf`
- Relevance to this repo:
  - Perseverance landed ~1.7km southeast from intended target in Jezero Crater on Feb 18, 2021
  - Launch mass 4,061kg (220kg heavier than MSL), rover 1,026kg vs MSL's 899kg
  - Entry at 20:36:50 SCET UTC, landed at 20:43:49 SCET UTC
  - TRN system described as "never-used-before" capability for Mars landing
  - UHF relay via MRO/MAVEN provided full telemetry through landing
  - X-band DTE lost before touchdown due to antenna angle violations
  - Plasma blackout ~35 sec after entry for ~1 minute

**Mohan et al. (2021)** — "Assessment of M2020 Terrain Relative Landing Accuracy: Flight Performance vs. Predicts"

- PDF: `Mars2020_TRN_Landing_Accuracy_Mohan2021.pdf`
- Relevance to this repo:
  - **TRN error budget breakdown**: 60m requirement sub-allocated to targeting error, knowledge error, control error
  - Pre-landing best estimate: 33m; Post-landing analysis: 8.53m; Actual imagery: **5m** from target
  - LVS required <40m accuracy within 10s of initialization (99%tile) — this drove the 60m touchdown requirement
  - COARSE phase: 12m/pixel map, 5 landmarks per image × 3 images, reduces 3.2km error to ~200m
  - FINE phase: 6m/pixel map (1024×1024), up to 100 landmarks per image, EKF with DIMU propagation
  - LVS map from CTX imagery, validated against other products; 30×30km coverage at 6m/pixel
  - STS (Safe Target Selection) searches pre-generated Safe Targets Map using LVS position

**Cheng et al. (2021)** — "Making an Onboard Reference Map From MRO/CTX Imagery for Mars 2020 Lander Vision System"

- PDF: `Cheng2021_CTX_Onboard_Map.pdf`
- DOI: https://doi.org/10.1029/2020EA001560
- Relevance to this repo:
  - **LVS map generation process**: First-ever use of reference map during spacecraft EDL
  - Map requirements: 6m/pixel appearance, 12m/pixel elevation, co-registered to within 6m (99%)
  - Absolute horizontal error <300m, orientation error <1mrad
  - Without TRN, Jezero safe landing probability was 78%; with TRN, landing was successful
  - CTX dejittering and sensor model improvements described — critical for map fidelity
  - Hazard map co-registered to LVS reference map for STS integration
  - Final error between STS target and actual landing: 5m

---

## Category 3: General Background / Theory

CAHV/CAHVOR/CAHVORE camera model foundations.

**Di & Li (2004)** — "CAHVOR camera model and its photogrammetric conversion for planetary applications"

- PDF: `Di2004_CAHVOR_Photogrammetric_Conversion.pdf`
- DOI: https://doi.org/10.1029/2003JE002199
- Relevance to this repo:
  - **Canonical CAHVOR↔photogrammetry conversion reference**: Derives equations to convert between CAHVOR vectors and traditional IO/EO parameters
  - CAHV model: C (camera center), A (optical axis), H (horizontal info vector), V (vertical info vector)
  - CAHVOR adds O (optical axis for distortion) and R (radial distortion triplet) for lens distortion correction
  - `hs` and `vs` (horizontal/vertical scale factors) equal focal length in pixels — should be identical for square pixels
  - Principal point: `hc = A·H`, `vc = A·V`; scale factors: `hs = ||A×H||`, `vs = ||A×V||`
  - Used in Mars Pathfinder, MER, and subsequent missions; PDS archives store calibration in CAHVOR format
  - Essential for converting PDS CAHVORE parameters to OpenCV camera matrix (relates to `cahvore.py`)

**Gennery (2006)** — "Generalized Camera Calibration Including Fish-Eye Lenses"

- PDF: `Gennery2006_CAHVORE_Camera_Calibration.pdf`
- DOI: https://doi.org/10.1007/s11263-006-5168-1
- Relevance to this repo:
  - **Definitive CAHVORE reference**: Describes the complete CAHVORE model including E (entrance pupil) terms
  - Handles fish-eye lenses with FOV >180° — generalizes perspective projection and ideal fish-eye as special cases
  - O/R/E terms: O = optical axis (may differ from A), R = radial distortion triplet, E = entrance pupil movement vs off-axis angle
  - Calibration via least-squares adjustment using known points from calibration fixture
  - Includes decentering distortion model and outlier rejection
  - **Used to calibrate MER cameras** — same methodology applied to all subsequent Mars rover cameras
  - Provides projection equations and partial derivatives for both object→image and image→object mapping

---

## Category 4: Mars 2020 Extensions

MSR ELViS and future Mars TRN developments.

**Sackier et al. (2023)** — "Sample Retrieval Lander's Enhanced Lander Vision System (ELViS) Overview"

- PDF: `MSR_ELViS_Overview_Sackier2023.pdf`
- Relevance to this repo:
  - **ELViS = Enhanced LVS for Mars Sample Return**: Baselined June 2022 for SRL (Sample Retrieval Lander)
  - Extends M2020 LVS to replace radar — provides 3D position and velocity from hypersonic entry through powered descent to 50m altitude
  - M2020 LVS achieved 5m landing accuracy (vs 40m requirement, 60m touchdown requirement)
  - ELViS uses same LCAM + VCE architecture as M2020
  - Key difference: ELViS operates through more flight phases (heading alignment, hypersonic entry, parachute, powered descent)
  - Map products: appearance map + elevation map used for landmark matching

**Setterfield et al. (2023)** — "Enhanced Lander Vision System (ELViS) Algorithms for Pinpoint Landing of the Mars Sample Retrieval Lander"

- PDF: `MSR_ELViS_Algorithms_Setterfield2023.pdf`
- Relevance to this repo:
  - **60m pinpoint landing without Doppler radar**: ELViS must land within 60m of pre-selected target
  - Three phases: TRN0 (heading alignment, 8-15km AGL), TRN1 (parachute + powered approach to 550m), VO (visual odometry, 175m to 50m)
  - **Stereo altitude estimation**: New algorithm for scale initialization without radar — uses image-to-image feature matching + triangulation
  - Legacy M2020 LVS relied on radar for scale (altitude) — ELViS uses laser altimeter + stereo backup
  - COARSE mode: 5 patches × 3 images to 12m/pixel map via 2D FFT correlation
  - FINE mode: up to 150 patches (21×21 pixels) to 6m/pixel map via EKF
  - VO phase uses wide-beam laser altimeter for multiple ranges; cuts off at 50m due to thruster dust

---

## Category 5: Other TRN Systems

Lunar, Titan, and other destination TRN systems.

**Johnson & Montgomery (2007)** — "Overview of Terrain Relative Navigation Approaches for Precise Lunar Landing"

- PDF: `TRN_Lunar_Landing_Overview_Johnson2007.pdf`
- Relevance to this repo:
  - **TRN survey for ALHAT** (Autonomous Landing and Hazard Avoidance Technology): Requirement to land within 100m of predetermined location
  - Three TRN functions: global position estimation, local position estimation, velocity estimation
  - Two algorithm types: **correlation** (patch matching to map) vs **pattern matching** (landmark detection + matching)
  - Active (LIDAR) vs passive (imaging) sensors — lunar TRN favors active due to lighting variability
  - Correlation approaches: place patch at every map location, measure similarity, highest score = best match
  - Pattern matching: detect landmarks (e.g., craters), match by diameter and relative positions
  - Documents lunar reference map availability from various missions

**Matthies et al. (2020)** — "Terrain Relative Navigation for Guided Descent on Titan"

- PDF: `TRN_Guided_Descent_Titan_Matthies2020.pdf`
- Relevance to this repo:
  - **Titan TRN challenges**: 90+ minute descent, 110×110km landing ellipses, hazy atmosphere limits visibility
  - Titan map resolution ~300m–3km/pixel (vs <1m for Mars) — orders of magnitude worse
  - Two altitude regimes: >20km (map matching feasible), <20km (feature tracking preferred)
  - Dual camera concept: VNIR (0.5–1μm) for lower regime, SWIR (2.0–2.1μm) for upper regime
  - Contrast requirements derived from radiative transfer modeling through Titan atmosphere
  - M2020 LVS cited as state-of-the-art: 12m/pixel coarse map, 6m/pixel fine map, NCC template matching in FPGA
  - Novel approach: terrain type discrimination (lake vs ground, dune vs interdune) as navigation aid
  - Simulation shows 3σ position error ~2km at touchdown — promising but needs validation
