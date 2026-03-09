
"""Streamlit app for spectroscopy CSV/XLSX analysis with multi-series support."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from analysis_utils import (
    TAUC_TRANSITIONS,
    PreprocessOptions,
    apply_preprocessing,
    compute_fwhm_table,
    compute_tauc_curve,
    clean_wavelength_column,
    detect_peaks,
    detect_x_column,
    detect_y_columns,
    extract_series_data,
    figure_to_png_bytes,
    linear_fit_subset,
    list_excel_sheets,
    load_tabular_file,
    suggest_linear_region,
)


plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.grid"] = True

MEASUREMENT_MODES = ["Absorbance", "Excitation", "PL Emission"]
Y_LABELS = {
    "Absorbance": "Absorbance / OD",
    "Excitation": "Intensity (a.u.)",
    "PL Emission": "Intensity (a.u.)",
}

BASELINE_LABEL_TO_KEY = {
    "None": "none",
    "Subtract minimum": "subtract_min",
    "Endpoint linear (first-last)": "endpoint_linear",
}
NORMALIZATION_LABEL_TO_KEY = {
    "None": "none",
    "Divide by max absolute intensity": "divide_by_max",
}


@dataclass
class SeriesBundle:
    name: str
    x: np.ndarray
    y_raw: np.ndarray
    y_processed: np.ndarray
    y_analysis: np.ndarray
    notes: List[str]


@dataclass
class PeakSettings:
    prominence: float
    distance_points: int
    min_width_points: float
    min_height: Optional[float]


def key_token(value: str) -> str:
    base = re.sub(r"[^a-zA-Z0-9_]+", "_", value)
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()[:8]
    return f"{base}_{digest}"


def make_line_figure(
    x: np.ndarray,
    y: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    color: str = "#1f77b4",
):
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(x, y, linewidth=1.4, color=color)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    return fig, ax


def render_plot_with_download(fig, download_name: str, key: str) -> None:
    st.pyplot(fig, use_container_width=True)
    st.download_button(
        "Download plot (PNG)",
        data=figure_to_png_bytes(fig),
        file_name=download_name,
        mime="image/png",
        key=key,
    )
    plt.close(fig)


def format_float_list(values: List[float], decimals: int = 3, max_items: int = 8) -> str:
    if not values:
        return ""
    clipped = values[:max_items]
    text = "; ".join([f"{value:.{decimals}f}" for value in clipped])
    if len(values) > max_items:
        text += "; ..."
    return text


def plot_multi_series(
    series_list: List[SeriesBundle],
    mode: str,
    source: str,
):
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for series in series_list:
        y = series.y_analysis if source == "analysis" else series.y_raw
        ax.plot(series.x, y, linewidth=1.2, label=series.name)

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(Y_LABELS[mode])
    source_label = "Analysis Signal" if source == "analysis" else "Raw Signal"
    ax.set_title(f"{mode}: Combined Multi-Series Plot ({source_label})")
    ax.legend(loc="best", fontsize=8)
    return fig


def run_tauc_for_series(
    series_name: str,
    x_nm: np.ndarray,
    y_absorbance: np.ndarray,
    min_r2_default: float,
) -> Dict[str, object]:
    series_key = key_token(series_name)
    result: Dict[str, object] = {
        "Transition": "",
        "Tauc exponent": np.nan,
        "Fit range min (eV)": np.nan,
        "Fit range max (eV)": np.nan,
        "Fit equation": "",
        "Tauc R2": np.nan,
        "Band gap Eg (eV)": "Not reported",
        "Band gap note": "Tauc not run",
    }

    st.markdown(f"#### Tauc Analysis - {series_name}")
    st.caption(
        "Absorbance-based estimate: results depend on transition assumption and manually selected fit range."
    )

    transition = st.selectbox(
        "Transition type",
        options=list(TAUC_TRANSITIONS.keys()),
        key=f"transition_{series_key}",
    )
    exponent = float(TAUC_TRANSITIONS[transition]["exponent"])
    transition_n = TAUC_TRANSITIONS[transition]["n"]

    result["Transition"] = transition
    result["Transition n"] = transition_n
    result["Tauc exponent"] = exponent

    st.caption(
        f"Transform: (A*hnu)^m with m = {exponent:.6g}. Reference n = {transition_n} in (alpha*hnu)^(1/n)."
    )

    try:
        energy_ev, tauc_y, excluded_count = compute_tauc_curve(x_nm, y_absorbance, exponent)
    except ValueError as exc:
        st.warning(f"{series_name}: {exc}")
        result["Band gap note"] = f"Tauc error: {exc}"
        return result

    if excluded_count > 0:
        st.info(
            f"{series_name}: excluded {excluded_count} point(s) where A*hnu <= 0 for Tauc transform."
        )

    energy_min = float(np.min(energy_ev))
    energy_max = float(np.max(energy_ev))
    default_min = float(np.percentile(energy_ev, 25))
    default_max = float(np.percentile(energy_ev, 60))

    suggestion = None
    with st.expander(f"Auto-suggest linear region ({series_name})", expanded=False):
        auto_enable = st.checkbox(
            "Enable auto-suggestion",
            value=False,
            key=f"auto_enable_{series_key}",
        )
        if auto_enable:
            window_fraction = st.slider(
                "Window fraction",
                min_value=0.10,
                max_value=0.60,
                value=0.20,
                step=0.05,
                key=f"window_fraction_{series_key}",
            )
            min_points_floor = 5
            max_points = int(len(energy_ev))
            min_points_default = min(20, max_points)
            if min_points_default < min_points_floor:
                min_points_default = min_points_floor

            auto_min_points = st.slider(
                "Min points per window",
                min_value=min_points_floor,
                max_value=max(min_points_floor, max_points),
                value=min_points_default,
                step=1,
                key=f"auto_min_points_{series_key}",
            )
            suggestion = suggest_linear_region(
                energy_ev,
                tauc_y,
                window_fraction=window_fraction,
                min_points=auto_min_points,
                require_positive_slope=True,
            )
            if suggestion is None:
                st.info("No robust auto-suggested linear region found.")
            else:
                st.write(
                    f"Suggested region: {suggestion.x_min:.4f} to {suggestion.x_max:.4f} eV; "
                    f"R2 = {suggestion.r2:.5f}."
                )

    if suggestion is not None:
        use_suggestion = st.checkbox(
            "Use suggested region as default",
            value=True,
            key=f"use_suggestion_{series_key}",
        )
        if use_suggestion:
            default_min = suggestion.x_min
            default_max = suggestion.x_max

    if default_min >= default_max:
        default_min = energy_min
        default_max = float(np.percentile(energy_ev, 70))

    fit_range = st.slider(
        f"Manual fit region for {series_name} (eV)",
        min_value=energy_min,
        max_value=energy_max,
        value=(float(default_min), float(default_max)),
        step=(energy_max - energy_min) / 500.0,
        key=f"fit_range_{series_key}",
    )

    min_r2 = st.slider(
        f"Minimum R2 for band-gap reporting ({series_name})",
        min_value=0.85,
        max_value=0.999,
        value=min_r2_default,
        step=0.001,
        key=f"min_r2_{series_key}",
    )

    result["Fit range min (eV)"] = float(fit_range[0])
    result["Fit range max (eV)"] = float(fit_range[1])
    result["Min R2 requirement"] = min_r2

    fig_tauc, ax_tauc = plt.subplots(figsize=(8.6, 4.8))
    ax_tauc.plot(energy_ev, tauc_y, color="#1d3557", linewidth=1.2, label="Tauc data")
    ax_tauc.axvspan(
        fit_range[0],
        fit_range[1],
        color="#f4a261",
        alpha=0.20,
        label="Manual fit region",
    )

    if suggestion is not None:
        ax_tauc.axvspan(
            suggestion.x_min,
            suggestion.x_max,
            color="#2a9d8f",
            alpha=0.15,
            label="Auto-suggested region",
        )

    fit: Optional[object]
    try:
        fit = linear_fit_subset(
            energy_ev,
            tauc_y,
            x_min=float(fit_range[0]),
            x_max=float(fit_range[1]),
            min_points=max(5, min(8, len(energy_ev))),
        )
    except ValueError as exc:
        fit = None
        st.warning(f"{series_name}: {exc}")
        result["Band gap note"] = str(exc)

    if fit is not None:
        fit_x = np.linspace(fit_range[0], fit_range[1], 200)
        fit_y = fit.slope * fit_x + fit.intercept
        ax_tauc.plot(fit_x, fit_y, color="#e63946", linewidth=2.0, label="Linear fit")

        result["Fit equation"] = f"y = {fit.slope:.6g}x + {fit.intercept:.6g}"
        result["Tauc slope"] = fit.slope
        result["Tauc intercept"] = fit.intercept
        result["Tauc R2"] = fit.r2
        result["Tauc x-intercept (eV)"] = fit.x_intercept

        if not np.isfinite(fit.r2) or fit.r2 < min_r2:
            result["Band gap note"] = (
                f"Not reported: selected region not sufficiently linear (R2={fit.r2:.5f} < {min_r2:.3f})."
            )
        elif fit.slope <= 0:
            result["Band gap note"] = "Not reported: fitted slope is non-positive."
        elif not np.isfinite(fit.x_intercept) or fit.x_intercept <= 0:
            result["Band gap note"] = "Not reported: fitted x-intercept is not physically meaningful."
        else:
            eg = float(fit.x_intercept)
            result["Band gap Eg (eV)"] = eg
            result["Band gap note"] = "Absorbance-based estimated band gap."
            ax_tauc.axvline(
                eg,
                color="#2a9d8f",
                linestyle="--",
                linewidth=1.4,
                label=f"Eg = {eg:.4f} eV",
            )

        st.write(f"Fitting range: {fit_range[0]:.4f} to {fit_range[1]:.4f} eV")
        st.write(f"Fit equation: {result['Fit equation']}")
        st.write(f"R2: {fit.r2:.5f}")
        if isinstance(result["Band gap Eg (eV)"], float):
            st.success(f"Estimated Eg: {result['Band gap Eg (eV)']:.4f} eV")
        else:
            st.warning(str(result["Band gap note"]))

    ax_tauc.set_xlabel("Photon Energy (eV)")
    ax_tauc.set_ylabel(f"(A*hnu)^{exponent:.4g} (a.u.)")
    ax_tauc.set_title(f"Tauc Plot - {series_name}")
    ax_tauc.legend(loc="best", fontsize=8)
    render_plot_with_download(
        ax_tauc.figure,
        download_name=f"{series_key}_tauc_plot.png",
        key=f"download_tauc_{series_key}",
    )

    return result


def analyze_single_series(
    series: SeriesBundle,
    mode: str,
    data_source_label: str,
    peak_settings: Optional[PeakSettings],
    pl_enable_fwhm: bool,
    absorbance_enable_tauc: bool,
    absorbance_min_r2_default: float,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    st.markdown(f"### Series: {series.name}")
    if series.notes:
        st.caption("Series notes: " + " | ".join(series.notes))

    raw_fig, _ = make_line_figure(
        series.x,
        series.y_raw,
        x_label="Wavelength (nm)",
        y_label=Y_LABELS[mode],
        title=f"Raw Plot - {series.name}",
        color="#264653",
    )
    render_plot_with_download(
        raw_fig,
        download_name=f"{key_token(series.name)}_raw_plot.png",
        key=f"download_raw_{key_token(series.name)}",
    )

    if data_source_label == "Processed data":
        processed_fig, _ = make_line_figure(
            series.x,
            series.y_analysis,
            x_label="Wavelength (nm)",
            y_label=Y_LABELS[mode],
            title=f"Processed Signal Used for Analysis - {series.name}",
            color="#2a9d8f",
        )
        render_plot_with_download(
            processed_fig,
            download_name=f"{key_token(series.name)}_processed_signal_plot.png",
            key=f"download_processed_{key_token(series.name)}",
        )

    summary: Dict[str, object] = {
        "Timestamp": datetime.now().isoformat(timespec="seconds"),
        "Mode": mode,
        "Series": series.name,
        "Valid points": int(len(series.x)),
        "Data source": data_source_label,
        "Processing notes": " | ".join(series.notes) if series.notes else "None",
        "Peak count": np.nan,
        "Peak maxima (nm)": "",
        "Transition": "",
        "Fit range min (eV)": np.nan,
        "Fit range max (eV)": np.nan,
        "Fit equation": "",
        "Tauc R2": np.nan,
        "Band gap Eg (eV)": "",
        "Band gap note": "",
    }

    if mode == "Absorbance":
        if absorbance_enable_tauc:
            tauc_result = run_tauc_for_series(
                series.name,
                series.x,
                series.y_analysis,
                min_r2_default=absorbance_min_r2_default,
            )
            summary.update(tauc_result)
        else:
            summary["Band gap note"] = "Tauc disabled by user"

    if mode in {"Excitation", "PL Emission"}:
        if peak_settings is None:
            summary["Band gap note"] = "Peak settings missing"
        else:
            try:
                peaks = detect_peaks(
                    wavelength_nm=series.x,
                    signal=series.y_analysis,
                    prominence=peak_settings.prominence,
                    distance_points=peak_settings.distance_points,
                    min_width_points=peak_settings.min_width_points,
                    min_height=peak_settings.min_height,
                )
            except ValueError as exc:
                st.warning(f"{series.name}: peak analysis skipped ({exc})")
                summary["Band gap note"] = f"Peak analysis skipped: {exc}"
                peaks = None

            if peaks is not None:
                peak_fig, ax_peak = plt.subplots(figsize=(8.6, 4.8))
                ax_peak.plot(series.x, series.y_analysis, color="#1d3557", linewidth=1.2, label="Signal")
                if len(peaks.indices) > 0:
                    ax_peak.scatter(
                        series.x[peaks.indices],
                        series.y_analysis[peaks.indices],
                        color="#e63946",
                        s=36,
                        zorder=3,
                        label="Detected peaks",
                    )
                    for i, idx in enumerate(peaks.indices, start=1):
                        ax_peak.annotate(
                            str(i),
                            (series.x[idx], series.y_analysis[idx]),
                            textcoords="offset points",
                            xytext=(0, 7),
                            ha="center",
                            fontsize=8,
                        )

                ax_peak.set_xlabel("Wavelength (nm)")
                ax_peak.set_ylabel("Intensity (a.u.)")
                ax_peak.set_title(f"{mode} Peaks - {series.name}")
                ax_peak.legend(loc="best", fontsize=8)
                render_plot_with_download(
                    peak_fig,
                    download_name=f"{key_token(series.name)}_peaks_plot.png",
                    key=f"download_peaks_{key_token(series.name)}",
                )

                if peaks.table.empty:
                    st.warning(f"{series.name}: no peaks found with current settings.")
                    summary["Peak count"] = 0
                else:
                    st.success(f"{series.name}: detected {len(peaks.table)} peak(s).")
                    st.dataframe(peaks.table, use_container_width=True, height=240)
                    st.download_button(
                        "Download peak table (CSV)",
                        data=peaks.table.to_csv(index=False).encode("utf-8"),
                        file_name=f"{key_token(series.name)}_peaks.csv",
                        mime="text/csv",
                        key=f"download_peak_table_{key_token(series.name)}",
                    )

                    maxima_nm = peaks.table["Wavelength (nm)"].tolist()
                    summary["Peak count"] = int(len(peaks.table))
                    summary["Peak maxima (nm)"] = format_float_list(maxima_nm, decimals=2, max_items=12)

                    if mode == "PL Emission" and pl_enable_fwhm:
                        st.markdown(f"**Optional FWHM - {series.name}**")
                        option_map = {
                            f"Peak {int(row['Peak #'])} ({row['Wavelength (nm)']:.2f} nm)": int(row["Peak #"])
                            for _, row in peaks.table.iterrows()
                        }
                        selected_labels = st.multiselect(
                            "Select peak(s) for FWHM",
                            options=list(option_map.keys()),
                            default=list(option_map.keys()),
                            key=f"fwhm_select_{key_token(series.name)}",
                        )
                        selected_peak_numbers = [option_map[label] for label in selected_labels]

                        fwhm_all = compute_fwhm_table(series.x, series.y_analysis, peaks.indices)
                        fwhm_selected = fwhm_all[
                            fwhm_all["Peak #"].isin(selected_peak_numbers)
                        ].copy()

                        if fwhm_selected.empty:
                            st.info("No FWHM values available for selected peaks.")
                        else:
                            merged = peaks.table.merge(fwhm_selected, on="Peak #", how="left")
                            st.dataframe(merged, use_container_width=True, height=240)
                            st.download_button(
                                "Download FWHM table (CSV)",
                                data=merged.to_csv(index=False).encode("utf-8"),
                                file_name=f"{key_token(series.name)}_fwhm.csv",
                                mime="text/csv",
                                key=f"download_fwhm_{key_token(series.name)}",
                            )
                            summary["FWHM mean (nm)"] = float(np.nanmean(merged["FWHM (nm)"].to_numpy()))

    export_df = pd.DataFrame(
        {
            "Series": series.name,
            "Wavelength (nm)": series.x,
            "Raw signal": series.y_raw,
            "Processed signal": series.y_processed,
            "Analysis signal": series.y_analysis,
            "Signal source": data_source_label,
        }
    )

    return summary, export_df


def summarize_results(rows: List[Dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    summary_df = pd.DataFrame(rows)
    preferred_order = [
        "Timestamp",
        "Mode",
        "Series",
        "Valid points",
        "Data source",
        "Processing notes",
        "Peak count",
        "Peak maxima (nm)",
        "FWHM mean (nm)",
        "Transition",
        "Transition n",
        "Tauc exponent",
        "Fit range min (eV)",
        "Fit range max (eV)",
        "Fit equation",
        "Tauc slope",
        "Tauc intercept",
        "Tauc R2",
        "Band gap Eg (eV)",
        "Band gap note",
    ]
    existing = [column for column in preferred_order if column in summary_df.columns]
    remaining = [column for column in summary_df.columns if column not in existing]
    return summary_df[existing + remaining]


def main() -> None:
    st.set_page_config(page_title="Spectroscopy Multi-Series Analyzer", layout="wide")
    st.title("Spectroscopy CSV/XLSX Multi-Series Analyzer")
    st.caption(
        "Upload a CSV or Excel file, pick wavelength x-axis and one or more y-series, then analyze each y-column independently."
    )

    with st.sidebar:
        st.header("Inputs")
        mode = st.selectbox("Measurement type", MEASUREMENT_MODES, index=0)
        uploaded_file = st.file_uploader("Upload table file", type=["csv", "txt", "dat", "xlsx", "xls", "xlsm"])

    if uploaded_file is None:
        st.info("Select measurement type and upload a file to begin.")
        return

    file_bytes = uploaded_file.getvalue()
    extension = Path(uploaded_file.name).suffix.lower()

    selected_sheet: Optional[str] = None
    if extension in {".xlsx", ".xls", ".xlsm"}:
        try:
            sheet_names = list_excel_sheets(file_bytes)
        except ValueError as exc:
            st.error(str(exc))
            return

        with st.sidebar:
            if len(sheet_names) == 1:
                selected_sheet = sheet_names[0]
                st.caption(f"Worksheet: {selected_sheet}")
            else:
                selected_sheet = st.selectbox("Worksheet", options=sheet_names, index=0)

    try:
        table_df, load_notes = load_tabular_file(
            file_bytes=file_bytes,
            filename=uploaded_file.name,
            sheet_name=selected_sheet,
        )
    except ValueError as exc:
        st.error(f"File load error: {exc}")
        return
    except Exception as exc:
        st.error(f"Unexpected loading error: {exc}")
        return

    if load_notes:
        st.info("Load notes:\n- " + "\n- ".join(load_notes))

    st.subheader("Loaded Table Preview")
    st.write(f"Rows: {len(table_df)}, Columns: {table_df.shape[1]}")
    st.dataframe(table_df.head(200), use_container_width=True, height=280)
    st.download_button(
        "Download loaded table preview (CSV)",
        data=table_df.to_csv(index=False).encode("utf-8"),
        file_name="loaded_table.csv",
        mime="text/csv",
        key="download_loaded_table",
    )

    try:
        auto_x_col, x_notes = detect_x_column(table_df)
    except ValueError as exc:
        st.error(f"x-axis detection error: {exc}")
        return

    all_columns = list(table_df.columns)
    default_x_index = all_columns.index(auto_x_col) if auto_x_col in all_columns else 0

    st.subheader("Column Selection")
    x_column = st.selectbox(
        "X-axis column (wavelength in nm)",
        options=all_columns,
        index=default_x_index,
    )

    if x_notes:
        st.caption("x-axis detection notes: " + " | ".join(x_notes))

    cleaned_x_numeric, wavelength_rows_removed = clean_wavelength_column(table_df[x_column])
    parseable_wavelength_points = int(cleaned_x_numeric.notna().sum())

    st.caption(
        "Selected x-axis cleaning (analysis only): cast to string, trim spaces, strip 'nm' case-insensitively, remove extra spaces, then convert to numeric."
    )
    if wavelength_rows_removed > 0:
        st.warning(
            f"Wavelength cleaning removed {wavelength_rows_removed} row(s) from '{x_column}' because they could not be parsed as numeric values."
        )

    st.caption(
        "Raw table preview above remains unchanged; cleaning is applied only to the selected x-axis column for plotting and analysis."
    )

    if parseable_wavelength_points < 5:
        st.error(
            f"Selected x-axis column '{x_column}' has only {parseable_wavelength_points} parseable wavelength row(s) after cleaning. Need at least 5."
        )
        return

    try:
        default_y_candidates, y_notes = detect_y_columns(table_df, x_column=x_column)
    except ValueError as exc:
        st.error(f"y-column detection error: {exc}")
        return

    if y_notes:
        st.caption("y-column detection notes: " + " | ".join(y_notes))

    if not default_y_candidates:
        st.error("No numeric y-columns detected after excluding the selected x-column.")
        return

    selected_y_columns = st.multiselect(
        "Select one or more y-columns",
        options=default_y_candidates,
        default=default_y_candidates,
    )

    if not selected_y_columns:
        st.info("Select at least one y-column to continue.")
        return

    st.subheader("Optional Processing (Explicit, User-Controlled)")
    st.caption(
        "No hidden smoothing, normalization, or baseline correction is applied. Raw values are preserved."
    )

    with st.expander("Processing controls", expanded=False):
        baseline_label = st.selectbox("Baseline", list(BASELINE_LABEL_TO_KEY.keys()), index=0)
        smoothing_enabled = st.checkbox("Apply Savitzky-Golay smoothing", value=False)

        max_window_candidates = [
            max(5, len(table_df) if len(table_df) % 2 == 1 else len(table_df) - 1),
            5,
        ]
        max_window = max(max_window_candidates)
        default_window = min(11, max_window)
        if default_window % 2 == 0:
            default_window -= 1
        if default_window < 5:
            default_window = 5

        smoothing_window = st.slider(
            "Smoothing window length (odd points)",
            min_value=5,
            max_value=max_window,
            value=default_window,
            step=2,
            disabled=not smoothing_enabled,
        )
        poly_max = max(1, min(6, smoothing_window - 1))
        smoothing_poly = st.slider(
            "Smoothing polynomial order",
            min_value=1,
            max_value=poly_max,
            value=min(2, poly_max),
            step=1,
            disabled=not smoothing_enabled,
        )
        normalization_label = st.selectbox(
            "Normalization",
            list(NORMALIZATION_LABEL_TO_KEY.keys()),
            index=0,
        )

    preprocess_options = PreprocessOptions(
        baseline_method=BASELINE_LABEL_TO_KEY[baseline_label],
        smoothing_enabled=smoothing_enabled,
        smoothing_window=smoothing_window,
        smoothing_polyorder=smoothing_poly,
        normalization=NORMALIZATION_LABEL_TO_KEY[normalization_label],
    )

    processing_enabled = (
        preprocess_options.baseline_method != "none"
        or preprocess_options.smoothing_enabled
        or preprocess_options.normalization != "none"
    )

    if processing_enabled:
        data_source_label = st.radio(
            "Signal source for combined plot + analysis",
            options=["Raw data", "Processed data"],
            index=0,
            horizontal=True,
        )
    else:
        data_source_label = "Raw data"

    series_list: List[SeriesBundle] = []
    for y_column in selected_y_columns:
        try:
            x_vals, y_raw, extraction_notes = extract_series_data(
                table_df,
                x_column=x_column,
                y_column=y_column,
                min_points=5,
                cleaned_x_numeric=cleaned_x_numeric,
            )
        except ValueError as exc:
            st.warning(f"Skipping '{y_column}': {exc}")
            continue

        y_processed = y_raw.copy()
        processing_notes: List[str] = []
        if processing_enabled:
            try:
                y_processed, processing_notes = apply_preprocessing(x_vals, y_raw, preprocess_options)
            except ValueError as exc:
                st.warning(f"{y_column}: processing failed ({exc}). Falling back to raw signal.")
                y_processed = y_raw.copy()
                processing_notes = [f"Processing failed: {exc}"]

        y_analysis = y_processed if data_source_label == "Processed data" else y_raw
        merged_notes = extraction_notes + processing_notes

        series_list.append(
            SeriesBundle(
                name=y_column,
                x=x_vals,
                y_raw=y_raw,
                y_processed=y_processed,
                y_analysis=y_analysis,
                notes=merged_notes,
            )
        )

    if not series_list:
        st.error("No valid y-series available after validation. Please adjust selected columns or input data.")
        return

    st.subheader("Plotting Behavior")
    show_combined_plot = st.checkbox("Show combined plot of selected y-series", value=True)
    show_separate_overview = st.checkbox(
        "Show separate overview plots for each selected y-series",
        value=False,
    )

    combined_source_key = "analysis" if data_source_label == "Processed data" else "raw"
    if show_combined_plot:
        combined_fig = plot_multi_series(series_list, mode=mode, source=combined_source_key)
        render_plot_with_download(
            combined_fig,
            download_name=f"{mode.lower().replace(' ', '_')}_combined_plot.png",
            key="download_combined_plot",
        )

    if show_separate_overview:
        st.markdown("### Separate Overview Plots")
        for series in series_list:
            y_display = series.y_analysis if combined_source_key == "analysis" else series.y_raw
            fig_series, _ = make_line_figure(
                series.x,
                y_display,
                x_label="Wavelength (nm)",
                y_label=Y_LABELS[mode],
                title=f"{mode} - {series.name}",
                color="#1d3557",
            )
            render_plot_with_download(
                fig_series,
                download_name=f"{key_token(series.name)}_overview_plot.png",
                key=f"download_overview_{key_token(series.name)}",
            )

    peak_settings: Optional[PeakSettings] = None
    enable_fwhm_for_pl = False
    enable_tauc_for_absorbance = False
    absorbance_min_r2_default = 0.97

    if mode in {"Excitation", "PL Emission"}:
        st.subheader("Peak Detection Settings")
        y_spans = [float(np.nanmax(series.y_analysis) - np.nanmin(series.y_analysis)) for series in series_list]
        representative_span = float(np.nanmedian(y_spans)) if y_spans else 1.0

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            prominence = st.number_input(
                "Min prominence",
                min_value=0.0,
                value=max(0.0, 0.05 * representative_span),
                format="%.6g",
            )
        with col2:
            distance_points = st.number_input(
                "Min distance (points)",
                min_value=1,
                value=max(1, int(np.median([len(series.x) for series in series_list]) / 80)),
                step=1,
            )
        with col3:
            min_width = st.number_input(
                "Min width (points)",
                min_value=0.0,
                value=0.0,
                step=1.0,
            )
        with col4:
            use_min_height = st.checkbox("Use minimum height", value=False)
            min_height = None
            if use_min_height:
                median_values = [float(np.nanmedian(series.y_analysis)) for series in series_list]
                min_height = st.number_input(
                    "Minimum height",
                    value=float(np.nanmedian(median_values)),
                    format="%.6g",
                )

        peak_settings = PeakSettings(
            prominence=float(prominence),
            distance_points=int(distance_points),
            min_width_points=float(min_width),
            min_height=float(min_height) if min_height is not None else None,
        )

        if mode == "PL Emission":
            enable_fwhm_for_pl = st.checkbox(
                "Enable optional FWHM computation for selected peaks",
                value=False,
            )

    if mode == "Absorbance":
        st.subheader("Absorbance / Tauc Settings")
        st.warning(
            "Band-gap values are absorbance-based estimates and depend on transition assumption and chosen fitting range."
        )
        enable_tauc_for_absorbance = st.checkbox(
            "Run Tauc analysis for each selected absorbance column",
            value=True,
        )
        absorbance_min_r2_default = st.slider(
            "Default minimum R2 threshold for band-gap reporting",
            min_value=0.85,
            max_value=0.999,
            value=0.97,
            step=0.001,
        )

    st.markdown("---")
    st.subheader("Per-Series Analysis")

    summary_rows: List[Dict[str, object]] = []
    processed_export_tables: List[pd.DataFrame] = []

    for series in series_list:
        summary_row, export_table = analyze_single_series(
            series=series,
            mode=mode,
            data_source_label=data_source_label,
            peak_settings=peak_settings,
            pl_enable_fwhm=enable_fwhm_for_pl,
            absorbance_enable_tauc=enable_tauc_for_absorbance,
            absorbance_min_r2_default=absorbance_min_r2_default,
        )
        summary_rows.append(summary_row)
        processed_export_tables.append(export_table)

    summary_df = summarize_results(summary_rows)
    if not summary_df.empty:
        st.subheader("Per-Column Analysis Summary")
        st.dataframe(summary_df, use_container_width=True, height=280)
        st.download_button(
            "Download analysis summary (CSV)",
            data=summary_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{mode.lower().replace(' ', '_')}_analysis_summary.csv",
            mime="text/csv",
            key="download_summary_csv",
        )

    if processed_export_tables:
        processed_export_df = pd.concat(processed_export_tables, ignore_index=True)
        st.download_button(
            "Download processed/analysis data (CSV)",
            data=processed_export_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{mode.lower().replace(' ', '_')}_processed_data.csv",
            mime="text/csv",
            key="download_processed_data",
        )


if __name__ == "__main__":
    main()
