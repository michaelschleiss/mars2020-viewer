# MSL MARDI Technical Notes

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

## Dark Columns ("Black Border")

Full-frame MMM rows include masked (non-photoactive) columns:
- Total width: 1648 samples
- Photoactive: 1608 samples (columns 24–1631)
- Dark columns: 1–23 and 1632–1648

Black vertical strips in raw products are expected unless cropped to the photoactive region.
