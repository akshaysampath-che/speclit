# Spectroscopy CSV/XLSX Multi-Series Analyzer

Streamlit app for spectroscopy analysis with one x-axis column (wavelength) and multiple independent y-series.

## Supported file formats
- `.csv`, `.txt`, `.dat` (automatic common-delimiter handling)
- `.xlsx`, `.xls`, `.xlsm` (Excel read with `openpyxl`)

If an Excel file has multiple worksheets, you can choose the sheet in the sidebar.

## Supported table patterns
- `Wavelength, Abs1, Abs2, Abs3`
- `Wavelength, Emission_A, Emission_B`
- `Wavelength, Excitation_1, Excitation_2, Excitation_3, Excitation_4`
- Excel sheet with columns like `Wavelength | Sample_A | Sample_B | Sample_C`
- No column tags: first column is auto-used as x-axis by default (manual override available)

## Key behavior
- Shows loaded table preview before analysis.
- Auto-detects x-axis and y-candidate columns.
- Cleans selected x-axis wavelength values for analysis by stripping "nm" (case-insensitive), trimming spaces, and converting to numeric; rows with unparsable wavelength values are dropped with a warning.
- Lets you manually override x-axis and select one or more y-columns.
- Treats every selected y-column as a separate spectrum (no hidden merge/average).
- Optional processing only when explicitly enabled:
  - baseline correction
  - Savitzky-Golay smoothing
  - normalization
- Combined multi-series plotting and optional separate overview plots.
- Per-series analysis loop:
  - raw plot
  - peak analysis (Excitation/PL)
  - optional FWHM (PL)
  - optional Tauc + band-gap estimation (Absorbance)
- Export:
  - combined plot
  - individual plots
  - per-column analysis summary CSV
  - processed/analysis data CSV

## Install
```powershell
cd C:\Users\ASUS ZENBOOK\Documents\Playground
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run
```powershell
streamlit run app.py
```