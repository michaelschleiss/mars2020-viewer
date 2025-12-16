# LCAM Complete Metadata Reference

Complete metadata fields from LCAM PDS3 products. Example file: `ELM_0000_0666952834_300FDR_N0000019LVS_04000_0000LUJ01.IMG`

## File Structure

| Field | Value | Description |
|-------|-------|-------------|
| ODL_VERSION_ID | ODL3 | PDS label format version |
| RECORD_TYPE | FIXED_LENGTH | Record format |
| RECORD_BYTES | 1024 | Bytes per record |
| FILE_RECORDS | 1037 | Total records in file |
| LABEL_RECORDS | 8 | PDS3 label size (records) |
| ^IMAGE_HEADER | 9 | VICAR header starts at record 9 |
| ^IMAGE | 14 | Image data starts at record 14 |

## Mission Identification

| Field | Value |
|-------|-------|
| MISSION_NAME | MARS 2020 |
| MISSION_PHASE_NAME | PRIMARY SURFACE MISSION |
| INSTRUMENT_HOST_ID | M20 |
| INSTRUMENT_HOST_NAME | MARS 2020 |
| TARGET_NAME | MARS |
| TARGET_TYPE | PLANET |
| SPACECRAFT_CLOCK_CNT_PARTITION | 1 |

## Instrument Identification

| Field | Value |
|-------|-------|
| INSTRUMENT_ID | LCAM |
| INSTRUMENT_NAME | LANDER VISION SYSTEM |
| INSTRUMENT_TYPE | IMAGING CAMERA |
| INSTRUMENT_VERSION_ID | FM |
| INSTRUMENT_SERIAL_NUMBER | 12 |
| PRODUCER_INSTITUTION_NAME | MULTIMISSION INSTRUMENT PROCESSING LAB, JET PROPULSION LAB |

## Product Identification (per-frame varying)

| Field | Example Value | Description |
|-------|---------------|-------------|
| PRODUCT_ID | ELM_0000_0666952834_300FDR_N0000019LVS_04000_0000LUJ01 | Unique product ID |
| SOURCE_PRODUCT_ID | (same as PRODUCT_ID) | Source product |
| SEQUENCE_ID | LVS_04000 | Sequence identifier |
| IMAGE_ID | 19 | Frame number (1-87) |
| ROVER_MOTION_COUNTER | (0, 19) | (SET, INSTANCE) |
| ROVER_MOTION_COUNTER_NAME | ('SET', 'INSTANCE') | Counter names |

## Timing (per-frame varying)

| Field | Example Value | Description |
|-------|---------------|-------------|
| SPACECRAFT_CLOCK_START_COUNT | 666952834.305263901 | SCLK at exposure start (9 decimal precision) |
| SPACECRAFT_CLOCK_MID_COUNT | 666952834.3 | SCLK at exposure midpoint |
| SPACECRAFT_CLOCK_STOP_COUNT | 666952834.4 | SCLK at exposure end |
| START_TIME | 2021-02-18T20:42:04.748391 | UTC at exposure start |
| STOP_TIME | 2021-02-18T20:42:04.843119 | UTC at exposure end |
| LOCAL_MEAN_SOLAR_TIME | Sol-00000M15:51:39.21383 | Mars local mean solar time |
| LOCAL_MEAN_SOLAR_TIME_STOP | Sol-00000M15:51:39.30602 | LMST at stop |
| LOCAL_TRUE_SOLAR_TIME | 15:13:40 | Mars local true solar time |
| LOCAL_TRUE_SOLAR_TIME_STOP | 15:13:40 | LTST at stop |
| LOCAL_TRUE_SOLAR_TIME_SOL | 0 | Sol number for LTST |
| LOCAL_TRUE_SOLAR_TIME_SOL_STOP | 0 | Sol number for LTST stop |
| PLANET_DAY_NUMBER | 0 | Sol number |
| PRODUCT_CREATION_TIME | 2022-02-03T20:25:46 | When product was created |

## Image Properties

| Field | Value | Description |
|-------|-------|-------------|
| GEOMETRY_PROJECTION_TYPE | RAW | No geometric correction |
| IMAGE_TYPE | REGULAR | Standard image |
| IMAGE_ACQUIRE_MODE | IMAGE | Acquisition mode |
| FRAME_ID | MONO | Monochrome |
| FRAME_TYPE | MONO | Frame type |

## IMAGE Object

| Field | Value | Description |
|-------|-------|-------------|
| INTERCHANGE_FORMAT | BINARY | Binary data |
| LINES | 1024 | Image height (pixels) |
| LINE_SAMPLES | 1024 | Image width (pixels) |
| SAMPLE_TYPE | UNSIGNED_INTEGER | Pixel data type |
| SAMPLE_BITS | 8 | Bits per pixel |
| SAMPLE_BIT_MASK | 255 (0xFF) | Valid bit mask |
| BANDS | 1 | Number of bands (grayscale) |
| BAND_STORAGE_TYPE | BAND_SEQUENTIAL | BSQ format |
| FIRST_LINE | 1 | Starting line |
| FIRST_LINE_SAMPLE | 1 | Starting sample |
| INVALID_CONSTANT | 0.0 | Invalid pixel value |
| MISSING_CONSTANT | 0.0 | Missing pixel value |

## Detector State

| Field | Value | Description |
|-------|-------|-------------|
| DETECTOR_LINES | 1024 | Detector height |
| DETECTOR_LINE_SAMPLES | 1024 | Detector width |
| DETECTOR_FIRST_LINE | 1 | First detector line used |
| DETECTOR_FIRST_LINE_SAMPLE | 1 | First detector sample used |
| DETECTOR_TO_IMAGE_ROTATION | 0.0 | Rotation (degrees) |
| CFA_TYPE | NONE | No color filter array (grayscale) |
| PIXEL_AVERAGING_HEIGHT | 1 | Vertical binning |
| PIXEL_AVERAGING_WIDTH | 1 | Horizontal binning |
| DOWNSAMPLE_METHOD | NONE | No downsampling |
| SAMPLE_BIT_METHOD | NONE | No bit manipulation |
| SAMPLE_BIT_MODE_ID | NONE | Bit mode |

## Exposure (per-frame varying)

| Field | Value | Description |
|-------|-------|-------------|
| EXPOSURE_DURATION | 0.094736099 s | Label exposure time (~95 ms) |
| EXPOSURE_TYPE | AUTO | Label value (see note below) |

**Note on EXPOSURE_DURATION:** The label reports ~95 ms, but Johnson et al. (2022) specifies actual sensor integration time of **150 µs**. The ~95 ms value corresponds to the LCAM frame latency requirement: "less than 100 ms between the camera image trigger and the last pixel output of the image" (Maki et al. 2020, Section 2.3.2, p. 15). For motion blur calculations, use 150 µs. Observed values: 94.720 ms (4 frames at frames 34, 44, 55, 66) and 94.736 ms (83 frames).

**Note on EXPOSURE_TYPE:** Despite the label value "AUTO", LCAM does **not** have automatic exposure control. The exposure was pre-computed before flight:

> "The LCAM does not have automatic exposure control, so great care must be taken to set the exposure correctly. This process combines detailed testing of the LCAM optical transmission combined with analysis of the Mars surface reflectance, atmospheric scattering and solar illumination."
> — Johnson et al. (2017), Section "Sensitivity Studies: Exposure Time", p. 14

**Timing observation:** Empirically, STOP_SCLK values are always exact 0.1s boundaries (e.g., .1, .2, .3), while START_SCLK has irregular fractional parts. EXPOSURE_DURATION = STOP_SCLK - START_SCLK exactly. This suggests LVS triggers frame capture at 0.1s SCLK boundaries, with START computed as STOP minus exposure duration.

### SPICE SCLK conversion (fractional ticks)

Mars 2020 SCLK is a two-field clock: **seconds** + **fractional ticks**. The fractional field is in **ticks of 1/65536 second**.

Evidence (from the shipped NAIF SCLK kernel):

- `spice_kernels/M2020_168_SCLKSCET.00007.tsc` (“SCLK Format” comment): the clock is `SSSSSSSSSS-FFFFF` and `FFFFF` is “count of fractions of a second with one fraction being 1/65536 of a second”.
- `spice_kernels/M2020_168_SCLKSCET.00007.tsc` (“Kernel DATA”): `SCLK01_MODULI_168 = ( 4294967296 65536 )`, i.e. the fractional modulus is `65536`.

Practical implication:

- LCAM labels provide `SPACECRAFT_CLOCK_*_COUNT` as a **decimal seconds** value (e.g. `666952834.305263901`), where the fractional part is seconds.
- Passing that string directly to `spice.scs2e` (e.g. `"666952834.305263901"`) makes SPICE interpret `305263901` as **ticks**, producing a wildly wrong ET.
- Convert the decimal fraction to ticks before calling SPICE:

  ```python
  sclk = 666952834.305263901  # from label
  coarse = int(sclk)
  fine = int(round((sclk - coarse) * 65536.0))  # ticks
  sclk_str = f"1/{coarse}.{fine:05d}"  # partition 1 (LCAM labels set SPACECRAFT_CLOCK_CNT_PARTITION=1)
  et = spice.scs2e(-168, sclk_str)
  ```

This is required for any code path that turns LCAM `SPACECRAFT_CLOCK_*` values into ET using SPICE.

## Compression

| Field | Value | Description |
|-------|-------|-------------|
| INST_CMPRS_NAME | Lossless | Compression method |

## Telemetry

| Field | Example Value | Description |
|-------|---------------|-------------|
| TELEMETRY_SOURCE_TYPE | TEAM-GENERATED IMAGE | Source type |
| TELEMETRY_SOURCE_NAME | IMAGE_VcemgrScidata_... | Source filename |
| TELEMETRY_SOURCE_CHECKSUM | 69569804 | Checksum (per-frame varying) |

## Processing History

| Field | Value |
|-------|-------|
| SOFTWARE_NAME | LVS_EDRGEN |
| SOFTWARE_VERSION_ID | 1.0 |
| PROCESSING_HISTORY_TEXT | EDR created from LVS team data via JPL/MIPL lvs_edrgen script |

## Coordinate System (CINT_FRAME) (per-frame varying)

Camera Internal frame referenced to MCMF (Mars-Centered Mars-Fixed).

| Field | Example Value | Description |
|-------|---------------|-------------|
| COORDINATE_SYSTEM_NAME | CINT_FRAME | Camera internal frame |
| SOLUTION_ID | TELEMETRY | Solution source |
| COORDINATE_SYSTEM_INDEX | (0, 19) | (SET, INSTANCE) |
| COORDINATE_SYSTEM_INDEX_NAME | ('SET', 'INSTANCE') | Index names |
| ORIGIN_OFFSET_VECTOR | (701403.20, 3145680.62, 1075894.36) | Camera position (MCMF, meters) |
| ORIGIN_ROTATION_QUATERNION | (0.5412, 0.7842, 0.1876, -0.2386) | Camera orientation (MCMF) |
| POSITIVE_AZIMUTH_DIRECTION | CLOCKWISE | Azimuth convention |
| POSITIVE_ELEVATION_DIRECTION | UP | Elevation convention |
| REFERENCE_COORD_SYSTEM_NAME | MCMF_FRAME | Reference frame |
| REFERENCE_COORD_SYSTEM_INDEX | 0 | Reference index |

## Geometric Camera Model (CAHVORE) (per-frame varying)

Full CAHVORE model embedded in each product, referenced to MCMF_FRAME.

| Field | Value | Description |
|-------|-------|-------------|
| CALIBRATION_SOURCE_ID | SYNTHETIC | Calibration source |
| MODEL_TYPE | CAHVORE | Camera model type |
| REFERENCE_COORD_SYSTEM_NAME | MCMF_FRAME | Coordinate system |
| REFERENCE_COORD_SYSTEM_INDEX | 0 | Index |
| INTERPOLATION_METHOD | NONE | No interpolation |

### CAHVORE Components

| Component | ID | Name | Unit | Example Value |
|-----------|-----|------|------|---------------|
| MODEL_COMPONENT_1 | C | CENTER | METER | (701403.0, 3145680.0, 1075890.0) |
| MODEL_COMPONENT_2 | A | AXIS | N/A | (-0.171207, -0.938368, -0.300257) |
| MODEL_COMPONENT_3 | H | HORIZONTAL | PIXEL | (402.013, -460.182, -500.756) |
| MODEL_COMPONENT_4 | V | VERTICAL | PIXEL | (241.732, -699.869, 298.075) |
| MODEL_COMPONENT_5 | O | OPTICAL | N/A | (-0.170765, -0.938306, -0.300702) |
| MODEL_COMPONENT_6 | R | RADIAL | N/A | (-1.165e-06, 0.120304, 0.033668) |
| MODEL_COMPONENT_7 | E | ENTRANCE | N/A | (-0.0793475, 0.182685, -0.11582) |
| MODEL_COMPONENT_8 | - | Linearity type | - | 2.0 |
| MODEL_COMPONENT_9 | - | (reserved) | - | 0.0 |

## VICAR Processing History

| Field | Value |
|-------|-------|
| TASK | LABEL |
| USER | rgd |
| DAT_TIM | Thu Feb 3 12:25:46 2022 |
| TASK | MARSRELA |
| USER | rgd |
| DAT_TIM | Thu Feb 3 12:25:48 2022 |
| INP | ELM_..._LUJ01.VIC.x |
| OUT | ELM_..._LUJ01.VIC |
| CM | CM |
| CHANGE_CM_CS | CHANGE_CM_CS |
| COORD | MCMF |

## IMAGE_HEADER Object

| Field | Value | Description |
|-------|-------|-------------|
| HEADER_TYPE | VICAR2 | VICAR label format |
| INTERCHANGE_FORMAT | ASCII | Text format |
| BYTES | 5120 | Header size |
| ^DESCRIPTION | VICAR2.TXT | Description file |

## VICAR System Fields

Low-level VICAR image format descriptors in the IMAGE_HEADER.

| Field | Value | Description |
|-------|-------|-------------|
| LBLSIZE | 5120 | Label size in bytes |
| FORMAT | BYTE | Pixel data format |
| TYPE | IMAGE | Data type |
| BUFSIZ | 1024 | Buffer size |
| DIM | 3 | Number of dimensions |
| EOL | 0 | End-of-line marker |
| RECSIZE | 1024 | Record size |
| ORG | BSQ | Band organization (Band Sequential) |
| NL | 1024 | Number of lines |
| NS | 1024 | Number of samples |
| NB | 1 | Number of bands |
| N1 | 1024 | Dimension 1 size |
| N2 | 1024 | Dimension 2 size |
| N3 | 1 | Dimension 3 size |
| N4 | 0 | Dimension 4 size |
| NBB | 0 | Number of binary prefix bytes |
| NLB | 0 | Number of binary header lines |
| HOST | JAVA | Host system |
| INTFMT | HIGH | Integer format (big-endian) |
| REALFMT | RIEEE | Real format (IEEE) |
| BHOST | VAX-VMS | Binary host |
| BINTFMT | LOW | Binary integer format |
| BREALFMT | VAX | Binary real format |
| BLTYPE | (empty) | Binary label type |
| COMPRESS | NONE | Compression in VICAR |
| EOCI1 | 0 | End-of-channel indicator 1 |
| EOCI2 | 0 | End-of-channel indicator 2 |
| PROPERTY | (various) | Property group marker |

## CAHVORE Component Metadata

| Field | Value |
|-------|-------|
| MODEL_COMPONENT_ID | ('C', 'A', 'H', 'V', 'O', 'R', 'E') |
| MODEL_COMPONENT_NAME | ('CENTER', 'AXIS', 'HORIZONTAL', 'VERTICAL', 'OPTICAL', 'RADIAL', 'ENTRANCE') |
| MODEL_COMPONENT_UNIT | ('METER', 'N/A', 'PIXEL', 'PIXEL', 'N/A', 'N/A', 'N/A') |

## Exposure Unit

| Field | Value |
|-------|-------|
| EXPOSURE_DURATION__UNIT | s |

## Precision Notes

### CAHVORE C vs ORIGIN_OFFSET_VECTOR

The PDS3 label stores MODEL_COMPONENT_1 (C) with reduced precision due to scientific notation:

```
ORIGIN_OFFSET_VECTOR (VICAR):  701403.201675757, 3145680.62090063, 1075894.363558644
MODEL_COMPONENT_1 C (PDS3):    701403.0,         3.14568e+06,      1.07589e+06
```

**Verified across all 87 frames**: These are the **same value**, but stored at different precisions.

The VICAR header contains **two separate fields** with the camera position:
- `MODEL_COMPONENT_1` (GEOMETRIC_CAMERA_MODEL group): 6 significant figures
- `ORIGIN_OFFSET_VECTOR` (CINT_COORDINATE_SYSTEM group): 15 significant figures

The PDS3 label only exposes `MODEL_COMPONENT_1`. Both are ASCII text. The source of the precision difference is not documented.

For trajectory work requiring sub-meter precision, use `ORIGIN_OFFSET_VECTOR` from VICAR header (not available in PDS3 label).

### Position Accuracy by Product Type

| Product | Position Source | Approx. Accuracy |
|---------|-----------------|------------------|
| ECM/EDR | TRN-corrected onboard | ~30-40m vs ground truth |
| FDR | Ground reconstruction | Best available (reference) |

The FDR ground reconstruction used DIMU backward-integration from landing site, initialized with HiRISE-derived landing position and landing radar measurements.

### Altitude Reference

Positions are in MCMF (Mars-Centered Mars-Fixed) Cartesian coordinates. To compute altitude above ground level (AGL), subtract the landing site radius (~3,391,938 m from Mars center for Jezero). The descent spans ~11 km AGL (frame 1) to touchdown (frame 87).

## EDR vs FDR Products

LCAM products come in two versions (Camera SIS Section 6.3.2):

| Type | Source | Description |
|------|--------|-------------|
| **ECM/EDR** | TRN-corrected | LVS output after coarse+fine matching |
| **FDR** | Ground reconstruction | Post-landing using DIMU + radar + HiRISE |

Our products are **FDR** (filename contains `FDR`). The CAHVORE values are ground-reconstructed.

**Important**: The PDS EDR/ECM products do **not** contain the raw spacecraft GNC estimate. They already include TRN corrections applied during descent. The initial ~137m position error (before TRN coarse matching at 4200m altitude) is not archived in PDS. Johnson et al. (2023) reports the actual TRN performance:
- Initial error at LVS start: ~137m (mostly in X/east-west)
- After coarse matching: ~200m
- After fine matching: ~40m requirement, achieved in ~6 seconds
- Final landing accuracy: 5m

## Summary: Per-Frame Varying vs Static Fields

### Varying per frame:
- PRODUCT_ID, SOURCE_PRODUCT_ID, IMAGE_ID
- ROVER_MOTION_COUNTER, COORDINATE_SYSTEM_INDEX
- All SPACECRAFT_CLOCK_* fields
- All *_TIME fields (START, STOP, LOCAL_*)
- ORIGIN_OFFSET_VECTOR, ORIGIN_ROTATION_QUATERNION
- All MODEL_COMPONENT_* (CAHVORE)
- EXPOSURE_DURATION (minor variation: 94.720-94.736 ms)
- TELEMETRY_SOURCE_NAME, TELEMETRY_SOURCE_CHECKSUM

### Static (same for all frames):
- All mission/instrument identification
- IMAGE dimensions and format
- DETECTOR configuration
- COMPRESSION settings
- SEQUENCE_ID, MODEL_TYPE, CALIBRATION_SOURCE_ID
