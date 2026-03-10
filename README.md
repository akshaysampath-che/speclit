# Spectroscopy CSV/XLSX Multi-Series Analyzer

Streamlit app for spectroscopy analysis with one wavelength x-axis column and multiple independent y-series.

## Overview
This app is built for practical spectroscopy workflows where one file contains one wavelength column and multiple spectra (for example absorbance, reflectance, excitation, or PL traces). Each selected y-series is kept independent through plotting, analysis, and export.

## Features
- Multi-format input with table preview before analysis.
- Automatic x/y candidate detection with manual x-axis override.
- Multi-series processing: choose one or many y-columns and analyze each selected series independently.
- Raw-data-first behavior with explicit, optional preprocessing (no hidden smoothing/normalization/baseline correction).
- Measurement workflows: Absorbance, Reflectance, Excitation, PL Emission.
- Per-series plots, per-series analysis results, and CSV exports.
- Publication-template plotting and multi-format figure export.

## Supported File Types
- Delimited text: `.csv`, `.txt`, `.dat` (common delimiter handling via loader).
- Excel: `.xlsx`, `.xls`, `.xlsm`.
- Excel worksheet picker is shown when multiple sheets are available.

## Input Data Expectations
- One x-column (wavelength) and one or more y-columns.
- Typical structures:
  - `Wavelength, Abs1, Abs2, Abs3`
  - `Wavelength, Emission_A, Emission_B`
  - `Wavelength, Excitation_1, Excitation_2, ...`
  - Excel sheets like `Wavelength | Sample_A | Sample_B | ...`
- If headers are not descriptive, the app can still detect candidates and allows manual x-column override.
- Selected x-column wavelength cleanup (analysis-only): cast to string, trim spaces, remove `nm` (case-insensitive), remove extra spaces, convert to numeric, and drop unparsable rows with warnings.

## Analysis Modes
### Absorbance
- Treats each selected y-series as absorbance/OD.
- Optional absorbance-based Tauc analysis per series with:
  - photon energy conversion from wavelength
  - transition-type choice (direct/indirect; allowed/forbidden)
  - manual fit-region selection
  - optional linear-region auto-suggestion (R2-based heuristic)
  - fit equation, R2, fit range, and conditional Eg reporting

### Reflectance
- Separate reflectance workflow (not treated as direct absorbance).
- Optional reflectance-based Tauc analysis per series using Kubelka-Munk transform:
  - `F(R) = (1 - R)^2 / (2R)`
  - supports reflectance scale handling: fraction (0-1), percent (0-100), or auto-detect only when unambiguous

### Excitation
- Peak detection per selected series with configurable prominence, distance, width, and optional height threshold.

### PL Emission
- Peak detection per selected series.
- Optional FWHM computation for selected detected peaks.

## Plotting and Publication Templates
### Publication templates (sidebar)
The app includes publication-inspired presets (not official journal templates):
- Default
- Nature-inspired single column
- Nature-inspired double column
- Science-inspired single column
- Science-inspired double column
- Presentation
- Custom

### Template controls
Under **Publication Templates** in the sidebar:
- template selection
- output size mode (`Template default` or `Manual override`)
- export format
- raster DPI selection (for raster formats)
- transparent background toggle
- optional font-family override
- custom template parameters when `Custom` is selected
- final export-size preview in mm and inches

### What templates/style settings control
- figure width/height (mm)
- font family and font sizes
- line width and marker size
- axis spine width, tick width, tick length
- top/right spine visibility
- grid visibility
- background color and transparency default
- palette
- margins/padding

### Title behavior
Plot titles are fully user-controlled:
- blank title input -> no title rendered
- non-blank title input -> exact text is used
- no automatic/default title prefix is added

### Additional styling controls
Under **Advanced Plot Settings**:
- manual x/y limits
- manual major tick increments
- major/minor tick toggles
- grid mode
- tick direction
- axis title overrides
- text color
- per-series line color and line width overrides
- optional Tauc fit/Eg line styling overrides

Template-aware defaults are used for line color/width where custom per-series overrides are not set.

## Export Options
### Figure exports
Per-plot download supports:
- PNG
- JPG / JPEG
- TIFF
- SVG
- PDF

Behavior:
- export uses selected physical dimensions (mm)
- vector formats (`SVG`, `PDF`) preserve vector output
- raster formats use selectable DPI in UI
- export filenames include a descriptive pattern with base name, template token, physical size, and raster DPI token when applicable
- JPEG transparency is not supported; the app warns and exports with opaque background

### Data exports
- per-column analysis summary CSV
- processed/analysis data CSV
- peak table CSV and FWHM CSV (where applicable)
- loaded-table preview CSV

## Backward Compatibility
Original scientific workflows remain intact:
- Absorbance
- Reflectance
- Excitation
- PL Emission
- Tauc analysis (absorbance and reflectance pathways)
- peak analysis and optional FWHM where applicable

## Installation
```powershell
git clone https://github.com/akshaysampath-che/speclit.git
cd speclit
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## How to Use
1. Upload a file (`.csv/.txt/.dat/.xlsx/.xls/.xlsm`) and choose worksheet if needed.
2. Review loaded table preview.
3. Select/override x-axis and choose one or more y-series.
4. Choose measurement mode (Absorbance, Reflectance, Excitation, PL Emission).
5. Optionally configure preprocessing and advanced styling/template settings.
6. Run per-series analysis and export plots/data.

## Scientific Note
- Absorbance Tauc in this app uses absorbance as a practical input for manual band-gap estimation.
- Reflectance Tauc uses Kubelka-Munk-transformed reflectance `F(R)` as an absorbance-like quantity.
- Both are approximation/model-based workflows; reported Eg values depend on transition assumption and selected fit region.

## Notes / Limitations
- Tauc fit quality is sensitive to selected fit region and transition model.
- Auto-suggested Tauc fit regions use a heuristic (R2-based sliding-window logic), not a universally validated fitting model.
- Reflectance auto-scale detection intentionally rejects ambiguous cases instead of silently assuming scale.
- Invalid rows/series are skipped with warnings; missing values are not fabricated.
- Actual font rendering depends on font availability in the runtime environment.
- The UI warns when very small font sizes are selected and when many traces may reduce legend readability at small export widths.

