# Spectroscopy CSV Analyzer (Streamlit)

A Streamlit app for spectroscopy CSV analysis with three measurement modes:
- Absorbance (with absorbance-based Tauc analysis)
- Excitation (automatic peak detection)
- PL Emission (automatic peak detection + optional FWHM)

## CSV format
- Two columns only
- x: wavelength (nm)
- y: measured signal

## Features
- Automatic parsing and detection of two numeric columns
- Raw data plot shown first (always)
- Optional and explicit preprocessing only:
  - Baseline correction
  - Savitzky-Golay smoothing
  - Normalization
- Mode-specific analysis:
  - **Absorbance**
    - Absorbance/OD vs wavelength plot
    - Correct wavelength-to-eV conversion
    - Transition assumptions: direct allowed, direct forbidden, indirect allowed, indirect forbidden
    - Manual fitting-range selection for Tauc analysis
    - Optional transparent auto-suggestion of linear range
    - Fitted line, R^2, intercept, and estimated band gap (with linearity gating)
  - **Excitation**
    - Intensity vs wavelength
    - Automatic peak detection with explicit parameters
  - **PL Emission**
    - Intensity vs wavelength
    - Automatic peak detection with explicit parameters
    - Optional FWHM for selected peaks
- Export plots (PNG), peak tables (CSV), and analysis summaries (CSV/TXT)

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

## Notes on Absorbance/Tauc output
- The reported band gap is an **absorbance-based estimate**.
- It depends on the selected transition type and manually selected fit region.
- If the selected region does not meet the linearity threshold, band gap is intentionally not reported.