# Spectroscopy CSV/XLSX Multi-Series Analyzer

Streamlit app for spectroscopy analysis with one wavelength x-axis column and multiple independent y-series.

## Overview
This app is designed for practical spectroscopy workflows where a single file contains one wavelength column and multiple spectra (e.g., multiple absorbance, reflectance, excitation, or PL traces). It keeps each selected y-series independent throughout plotting, analysis, and export.

## Features
- File input:
  - Delimited text: `.csv`, `.txt`, `.dat` (automatic common delimiter detection)
  - Excel: `.xlsx`, `.xls`, `.xlsm` (loaded with `openpyxl`)
  - Excel worksheet picker when multiple sheets are present
- Table handling:
  - Auto-detect x-axis candidate and y-series candidates
  - Manual x-axis override
  - Multi-select one or more y-series for analysis
  - Raw loaded table preview before analysis
- Wavelength cleanup (selected x-column only):
  - Cast to string, trim whitespace
  - Remove `nm` case-insensitively
  - Remove extra spaces and convert to numeric
  - Drop unparsable wavelength rows with warning
- Data integrity:
  - Raw values are always preserved and displayed
  - No hidden smoothing, normalization, baseline correction, or value fabrication
  - Invalid/non-numeric rows are dropped with warnings; problematic series are skipped instead of crashing
- Multi-series plotting:
  - Combined multi-series plot
  - Optional separate overview plot for each selected series
- Per-series analysis loop:
  - Raw plot for each series
  - Optional processed-signal plot when preprocessing is enabled
  - Mode-specific analysis per series (Tauc or peak analysis)

## Supported Analyses
### Absorbance mode
- Treats each selected y-series as absorbance/OD.
- Optional absorbance-based Tauc analysis per series:
  - Photon energy conversion from wavelength (eV)
  - Transition options: direct allowed, direct forbidden, indirect allowed, indirect forbidden
  - Manual fit region selection
  - Optional auto-suggested linear region (R2-based sliding window heuristic)
  - Reports fit equation, R2, fit range, and estimated Eg when criteria are met
  - If linearity/fit conditions are not met, band gap is explicitly reported as not reported

### Reflectance mode
- Distinct reflectance workflow (not treated as absorbance directly).
- Reflectance plotting for each selected series.
- Optional reflectance-based Tauc analysis per series via Kubelka-Munk transform:
  - Uses \( F(R) = (1-R)^2 / (2R) \) after reflectance normalization
  - Uses \(F(R)\) in the Tauc transform pathway
  - Same transition options and fit workflow style as absorbance Tauc
- Reflectance scale handling:
  - `Fraction (0 to 1)`
  - `Percent (0 to 100)`
  - `Auto-detect (only if unambiguous)`
  - Ambiguous auto-detection is rejected with a clear message (no silent assumption)

### Excitation mode
- Treats each selected y-series as excitation intensity.
- Peak detection per series with user controls:
  - prominence
  - minimum distance (points)
  - minimum width (points)
  - optional minimum height
- Reports peak positions/intensities and allows peak-table export.

### PL Emission mode
- Treats each selected y-series as emission intensity.
- Peak detection per series (same controls as excitation).
- Optional FWHM computation for selected detected peaks.

## Reflectance and Tauc Analysis
Reflectance Tauc in this app is model-based and uses Kubelka-Munk transformed reflectance \(F(R)\) as an absorbance-like quantity for diffuse reflectance workflows. The app does not claim this is direct absorbance.

## Plot Customization
Under **Advanced Plot Settings**:

### Axis and ticks
- Optional manual x-range and y-range
- Optional manual x/y major tick increments
- Major tick enable/disable
- Minor tick enable/disable
- Tick direction: outside, inside, both
- Grid mode:
  - automatic (current style)
  - no grid
  - major grid only
  - major + minor grid

### Figure heading and export size
- Optional figure heading prepended to plot titles
- Export image size controls:
  - width (px)
  - height (px)
  - DPI

### Typography and style
- Default requested font: **Times New Roman**
- User-selectable font family
- Font size controls:
  - global font size
  - or detailed title/axis/tick/legend sizes
- Custom x-axis and y-axis title overrides
- Custom text/font color
- Per-series custom line color and line width
- Optional custom Tauc fit/Eg line color and width
- Font preview shows requested font and matplotlib-resolved font

Note: actual font rendering depends on available fonts in the runtime environment.

## Input Data Expectations
- Expected structure: one x-column (wavelength) + one or more y-columns
- Typical patterns:
  - `Wavelength, Abs1, Abs2, Abs3`
  - `Wavelength, Emission_A, Emission_B`
  - `Wavelength, Excitation_1, Excitation_2, ...`
  - Excel sheets with `Wavelength | Sample_A | Sample_B | ...`
- If headers are non-descriptive, the app can default to first column as x-axis, with manual override available.

## Export Outputs
- PNG export for plots via the plot download button:
  - combined plot
  - separate overview plots
  - per-series raw/processed plots
  - per-series peak plots (where applicable)
  - per-series Tauc plots (where applicable)
- CSV exports:
  - per-column analysis summary
  - processed/analysis data table
  - peak/FWHM tables where applicable
  - loaded table preview CSV

Exported figures use the active plot customization/style settings applied to the figure.

## Installation
```powershell
cd C:\Users\ASUS ZENBOOK\Documents\Playground
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage
```powershell
streamlit run app.py
```

## Example Workflow
1. Upload a file (`.csv/.txt/.dat/.xlsx/.xls/.xlsm`) and choose worksheet if needed.
2. Confirm table preview, select/override x-axis, and choose one or more y-series.
3. Optionally enable preprocessing (baseline/smoothing/normalization) and choose analysis signal source (raw or processed).
4. Pick measurement mode (Absorbance, Reflectance, Excitation, PL Emission).
5. Configure plot styling/customization and optional mode-specific analysis settings (Tauc or peak/FWHM controls).
6. Review per-series results and export plots, summary CSV, and processed/analysis CSV.

## Scientific Note
- **Absorbance Tauc** in this app uses absorbance as a practical input for manual band-gap estimation.
- **Reflectance Tauc** uses Kubelka-Munk-transformed reflectance \(F(R)\) as an absorbance-like quantity.
- Both are approximation-based workflows; reported band gaps depend on transition assumption and selected fitting region.

## Limitations / Assumptions
- Tauc fit quality depends strongly on the selected fit region and chosen transition model.
- Auto-suggested Tauc fit regions use a heuristic (R2-based sliding-window search), not a universally validated model.
- Reflectance auto-scale detection intentionally fails when ambiguous (e.g., all non-negative values within 0-1) and requires explicit user choice.
- The app does not infer or fabricate missing data.
- Actual font rendering is environment-dependent; requesting Times New Roman does not guarantee it is installed.
