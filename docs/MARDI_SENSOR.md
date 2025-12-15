# MARDI Sensor Notes

## References

- Malin et al. (2017) "The Mars Science Laboratory Mast Cameras and Descent Imager" DOI: [10.1016/j.epsl.2016.12.038](https://doi.org/10.1016/j.epsl.2016.12.038) | PMC: [PMC5652233](https://pmc.ncbi.nlm.nih.gov/articles/PMC5652233/)

## Bayer Pattern

Source: MSL MMM EDR/RDR Data Product SIS (`out/msl_mmm_docs/MSL_MMM_EDR_RDR_DPSIS.PDF`)

The SIS states that, when assembling decompressed pixels into sensor space, rows interleave as:
- Row 0: `R G2 R G2 …`
- Row 1: `G1 B G1 B …`

This means the top-left pixel `(row=0, col=0)` is **Red**, i.e. the common CFA order **RGGB**.

In this repo, `tools/msl_mardi_view.py` uses this as the default (`--demosaic auto` → `rggb`).

## Radiometric Processing

Source: MSL MMM EDR/RDR Data Product SIS (`out/msl_mmm_docs/MSL_MMM_EDR_RDR_DPSIS.PDF`)

**RDR product codes** (in product ID, e.g., `DRCL`):
- `DRXX`: radiometric
- `DRCX`: radiometric + color corrected (8-bit)
- `DRLX`: radiometric + geometrically linearized
- `DRCL`: radiometric + color corrected + geometrically linearized (8-bit)

**Processing flow** (Section 5.1.2):
1. Extraction from EDR
2. Decompression
3. Decompanding (8→12/16 bit)
4. Dark current compensation
5. Shutter smear mitigation (important for MARDI's short exposures)
6. Flat field correction
7. Bad pixel mitigation
8. Radiometric calibration
9. Color correction
10. Geometric linearization

## Onboard Encoding (`PGV`) vs Ground Processing Code

MMM image products encode two independent “flavors” directly in the `PRODUCT_ID` / filename:

- `...CCCCC PGV _PROC.EXT`
  - `PGV` = product type (`P`) + GOP index (`G`) + version (`V`)
  - `PROC` = ground processing code (`XXXX` for EDR; `DRXX/DRCX/DRLX/DRCL` for RDR)

For example, `...0313E01_DRCL` and `...0313C00_DRCL` refer to the same underlying frame number (`CCCCC=00313`) but different onboard/downlinked encodings (`E..` is JPEG, `C..` is lossless). The MMM Data Product SIS defines these fields in the “PICNO” naming scheme (see `out/msl_mmm_docs/MSL_MMM_EDR_RDR_DPSIS.txt:983`).

Note: if you ever see something like `R01`, the `P=R` product type is defined by the SIS as a **“JPEG 444 focus merge image”** (see `out/msl_mmm_docs/MSL_MMM_EDR_RDR_DPSIS.txt:1005`; unrelated to MARDI).

## Dark Columns ("Black Border")

Full-frame MMM rows include masked (non-photoactive) columns:
- Total width: 1648 samples
- Photoactive: 1608 samples (columns 24–1631)
- Dark columns: 1–23 and 1632–1648

Black vertical strips in raw products are expected unless cropped to the photoactive region.
