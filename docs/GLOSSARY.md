# Glossary

Terms and acronyms used in Mars EDL imagery and this repository.

---

## S

### SCLK (Spacecraft Clock)

The onboard clock counter that runs on the spacecraft, independent of Earth time. Counts seconds (plus fractional ticks) since a mission-specific epoch.

**Example:** `666952774.005263901`

- Does not require ground communication to operate
- Used to timestamp all onboard events (images, telemetry, commands)
- Appears in PDS filenames as a 10-digit number: `ELM_0000_0666952774_...`
- Conversion to UTC requires a SPICE SCLK kernel (`*.tsc`), but PDS products include pre-computed UTC in the `START_TIME` label field

**Related fields in PDS labels:**
- `SPACECRAFT_CLOCK_START_COUNT` — raw SCLK value
- `START_TIME` — pre-computed UTC timestamp

**See also:** [DATA_INVENTORY.md](DATA_INVENTORY.md) → LCAM → Filename Convention
