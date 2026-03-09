"""Streamlit app for spectroscopy CSV analysis."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List

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
    detect_peaks,
    figure_to_png_bytes,
    linear_fit_subset,
    parse_spectroscopy_csv,
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


def make_line_figure(
    x: np.ndarray,
    y: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    color: str = "#1f77b4",
):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(x, y, color=color, linewidth=1.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    return fig, ax


def render_plot_with_download(fig, download_name: str, key: str) -> None:
    st.pyplot(fig, use_container_width=True)
    png_bytes = figure_to_png_bytes(fig)
    st.download_button(
        "Download plot (PNG)",
        data=png_bytes,
        file_name=download_name,
        mime="image/png",
        key=key,
    )
    plt.close(fig)


def render_summary_downloads(summary: Dict[str, object], file_prefix: str, key_prefix: str) -> None:
    summary_df = pd.DataFrame([summary])
    summary_csv = summary_df.to_csv(index=False).encode("utf-8")
    summary_txt = "\n".join([f"{k}: {v}" for k, v in summary.items()]).encode("utf-8")

    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "Download summary (CSV)",
            data=summary_csv,
            file_name=f"{file_prefix}_summary.csv",
            mime="text/csv",
            key=f"{key_prefix}_summary_csv",
        )
    with col_b:
        st.download_button(
            "Download summary (TXT)",
            data=summary_txt,
            file_name=f"{file_prefix}_summary.txt",
            mime="text/plain",
            key=f"{key_prefix}_summary_txt",
        )


def render_raw_data_section(mode: str, raw_df: pd.DataFrame) -> None:
    x_nm = raw_df["wavelength_nm"].to_numpy()
    y_raw = raw_df["signal"].to_numpy()

    st.subheader("Raw Data (always shown first)")
    fig_raw, _ = make_line_figure(
        x_nm,
        y_raw,
        x_label="Wavelength (nm)",
        y_label=Y_LABELS[mode],
        title=f"Raw {mode} Signal vs Wavelength",
        color="#1f77b4",
    )
    render_plot_with_download(
        fig_raw,
        download_name=f"{mode.lower().replace(' ', '_')}_raw_plot.png",
        key=f"raw_plot_{mode}",
    )

    st.dataframe(
        raw_df.rename(
            columns={
                "wavelength_nm": "Wavelength (nm)",
                "signal": "Signal",
            }
        ),
        use_container_width=True,
        height=260,
    )
    st.download_button(
        "Download parsed raw data (CSV)",
        data=raw_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{mode.lower().replace(' ', '_')}_parsed_raw_data.csv",
        mime="text/csv",
        key=f"raw_data_csv_{mode}",
    )


def render_processing_section(
    mode: str,
    x_nm: np.ndarray,
    y_raw: np.ndarray,
) -> tuple[np.ndarray, List[str], str]:
    st.subheader("Optional Processing (explicit and user-controlled)")
    st.caption(
        "Raw data remains unchanged. Any baseline correction, smoothing, or normalization below is optional and disclosed."
    )

    with st.expander("Processing controls", expanded=False):
        baseline_label = st.selectbox(
            "Baseline correction",
            list(BASELINE_LABEL_TO_KEY.keys()),
            index=0,
            key=f"baseline_{mode}",
        )

        smoothing_enabled = st.checkbox(
            "Apply Savitzky-Golay smoothing",
            value=False,
            key=f"smooth_enable_{mode}",
        )

        max_window = len(y_raw) if len(y_raw) % 2 == 1 else len(y_raw) - 1
        default_window = max(5, min(11, max_window))
        if default_window % 2 == 0:
            default_window -= 1
        if default_window < 5:
            default_window = 5

        if smoothing_enabled and max_window < 5:
            st.warning("Dataset is too short for stable Savitzky-Golay smoothing. Smoothing will be skipped.")
            smoothing_enabled = False

        smoothing_window = st.slider(
            "Smoothing window length (odd points)",
            min_value=5,
            max_value=max(5, max_window),
            value=min(default_window, max(5, max_window)),
            step=2,
            key=f"smooth_window_{mode}",
            disabled=not smoothing_enabled,
        )

        poly_max = max(1, min(6, smoothing_window - 1))
        smoothing_poly = st.slider(
            "Smoothing polynomial order",
            min_value=1,
            max_value=poly_max,
            value=min(2, poly_max),
            step=1,
            key=f"smooth_poly_{mode}",
            disabled=not smoothing_enabled,
        )

        normalization_label = st.selectbox(
            "Normalization",
            list(NORMALIZATION_LABEL_TO_KEY.keys()),
            index=0,
            key=f"norm_{mode}",
        )

    options = PreprocessOptions(
        baseline_method=BASELINE_LABEL_TO_KEY[baseline_label],
        smoothing_enabled=smoothing_enabled,
        smoothing_window=smoothing_window,
        smoothing_polyorder=smoothing_poly,
        normalization=NORMALIZATION_LABEL_TO_KEY[normalization_label],
    )

    try:
        y_processed, processing_notes = apply_preprocessing(x_nm, y_raw, options)
    except ValueError as exc:
        st.error(f"Processing error: {exc}")
        y_processed = y_raw.copy()
        processing_notes = []

    if processing_notes:
        st.info("Applied processing steps:\n- " + "\n- ".join(processing_notes))
        fig_proc, _ = make_line_figure(
            x_nm,
            y_processed,
            x_label="Wavelength (nm)",
            y_label=Y_LABELS[mode],
            title=f"Processed {mode} Signal vs Wavelength",
            color="#2a9d8f",
        )
        render_plot_with_download(
            fig_proc,
            download_name=f"{mode.lower().replace(' ', '_')}_processed_plot.png",
            key=f"processed_plot_{mode}",
        )

        data_source = st.radio(
            "Data source for downstream analysis",
            ["Raw data", "Processed data"],
            index=0,
            horizontal=True,
            key=f"source_{mode}",
        )
    else:
        st.caption("No optional processing selected. Downstream analysis uses raw data.")
        data_source = "Raw data"

    y_for_analysis = y_processed if data_source == "Processed data" else y_raw
    return y_for_analysis, processing_notes, data_source


def render_absorbance_mode(
    x_nm: np.ndarray,
    y_absorbance: np.ndarray,
    data_source: str,
    processing_notes: List[str],
) -> None:
    st.subheader("Absorbance Analysis")

    fig_abs, _ = make_line_figure(
        x_nm,
        y_absorbance,
        x_label="Wavelength (nm)",
        y_label="Absorbance / OD",
        title=f"Absorbance / OD vs Wavelength ({data_source})",
        color="#264653",
    )
    render_plot_with_download(
        fig_abs,
        download_name="absorbance_signal_plot.png",
        key="absorbance_signal_plot",
    )

    st.warning(
        "Band-gap results from this section are absorbance-based estimates. They depend on the transition assumption and your manually selected fitting range."
    )

    transition_name = st.selectbox(
        "Transition type assumption",
        list(TAUC_TRANSITIONS.keys()),
        index=0,
        key="transition_type",
    )
    exponent = float(TAUC_TRANSITIONS[transition_name]["exponent"])
    symbolic_n = TAUC_TRANSITIONS[transition_name]["n"]
    st.caption(
        f"Tauc transform used: (A*hnu)^m with m = {exponent:.6g}. "
        f"Reference relation: (alpha*hnu)^(1/n) ~ (hnu - Eg), n = {symbolic_n}."
    )

    try:
        energy_ev, tauc_y, excluded_count = compute_tauc_curve(x_nm, y_absorbance, exponent)
    except ValueError as exc:
        st.error(f"Tauc analysis error: {exc}")
        return

    if excluded_count > 0:
        st.info(
            f"Excluded {excluded_count} point(s) with non-positive A*hnu from the Tauc transform."
        )

    suggestion = None
    with st.expander("Optional auto-suggestion for fitting region", expanded=False):
        st.caption(
            "Transparent method: sliding-window linear fits; the region with highest R^2 and positive slope is suggested."
        )
        auto_enable = st.checkbox("Compute auto-suggestion", value=False, key="auto_suggest_enable")
        if auto_enable:
            window_fraction = st.slider(
                "Sliding window fraction",
                min_value=0.10,
                max_value=0.60,
                value=0.20,
                step=0.05,
                key="auto_window_fraction",
            )
            max_points = int(len(energy_ev))
            min_points_floor = 5
            min_points_default = min(20, max_points)
            if min_points_default < min_points_floor:
                min_points_default = min_points_floor

            min_points = st.slider(
                "Minimum points per window",
                min_value=min_points_floor,
                max_value=max(min_points_floor, max_points),
                value=min_points_default,
                step=1,
                key="auto_min_points",
            )
            suggestion = suggest_linear_region(
                energy_ev,
                tauc_y,
                window_fraction=window_fraction,
                min_points=min_points,
                require_positive_slope=True,
            )
            if suggestion is None:
                st.info("No robust linear window found with the current auto-suggestion settings.")
            else:
                st.write(
                    f"Suggested region: {suggestion.x_min:.4f} to {suggestion.x_max:.4f} eV, "
                    f"R^2 = {suggestion.r2:.5f}, points = {suggestion.points_used}."
                )

    energy_min = float(np.min(energy_ev))
    energy_max = float(np.max(energy_ev))
    default_min = float(np.percentile(energy_ev, 25))
    default_max = float(np.percentile(energy_ev, 60))

    if suggestion is not None:
        use_suggestion_default = st.checkbox(
            "Use suggested region as initial manual range",
            value=True,
            key="use_suggestion_default",
        )
        if use_suggestion_default:
            default_min = suggestion.x_min
            default_max = suggestion.x_max

    if default_min >= default_max:
        default_min = energy_min
        default_max = float(np.percentile(energy_ev, 70))

    fit_range = st.slider(
        "Manual fitting region on Tauc plot (eV)",
        min_value=energy_min,
        max_value=energy_max,
        value=(float(default_min), float(default_max)),
        step=(energy_max - energy_min) / 500.0,
        key="manual_fit_range",
    )

    fig_tauc, ax_tauc = plt.subplots(figsize=(8.5, 4.8))
    ax_tauc.plot(energy_ev, tauc_y, color="#1d3557", linewidth=1.2, label="Tauc data")
    ax_tauc.axvspan(
        fit_range[0],
        fit_range[1],
        color="#f4a261",
        alpha=0.20,
        label="Manual fitting region",
    )

    if suggestion is not None:
        ax_tauc.axvspan(
            suggestion.x_min,
            suggestion.x_max,
            color="#2a9d8f",
            alpha=0.15,
            label="Auto-suggested region",
        )

    ax_tauc.set_xlabel("Photon Energy (eV)")
    ax_tauc.set_ylabel(f"(A*hnu)^{exponent:.4g} (a.u.)")
    ax_tauc.set_title("Tauc Plot (Absorbance-Based)")
    ax_tauc.legend(loc="best")
    render_plot_with_download(fig_tauc, "absorbance_tauc_plot.png", "absorbance_tauc_plot")

    min_r2_required = st.slider(
        "Minimum R^2 required to report band gap",
        min_value=0.85,
        max_value=0.999,
        value=0.97,
        step=0.001,
        key="min_r2_required",
    )

    if st.button("Fit selected region", key="fit_selected_region"):
        try:
            fit = linear_fit_subset(
                energy_ev,
                tauc_y,
                x_min=float(fit_range[0]),
                x_max=float(fit_range[1]),
                min_points=max(5, min(8, len(energy_ev))),
            )
        except ValueError as exc:
            st.error(f"Fit error: {exc}")
            return

        fit_x = np.linspace(fit_range[0], fit_range[1], 200)
        fit_y = fit.slope * fit_x + fit.intercept

        fig_fit, ax_fit = plt.subplots(figsize=(8.5, 4.8))
        ax_fit.plot(energy_ev, tauc_y, color="#1d3557", linewidth=1.2, label="Tauc data")
        ax_fit.plot(fit_x, fit_y, color="#e63946", linewidth=2.0, label="Linear fit")
        ax_fit.axvspan(
            fit_range[0],
            fit_range[1],
            color="#f4a261",
            alpha=0.20,
            label="Manual fitting region",
        )

        band_gap_estimate = None
        band_gap_note = ""

        if not np.isfinite(fit.r2) or fit.r2 < min_r2_required:
            band_gap_note = (
                f"Band gap not reported because selected region is not sufficiently linear (R^2={fit.r2:.5f} < {min_r2_required:.3f})."
            )
        elif fit.slope <= 0:
            band_gap_note = "Band gap not reported because fitted slope is non-positive."
        elif not np.isfinite(fit.x_intercept) or fit.x_intercept <= 0:
            band_gap_note = "Band gap not reported because fitted x-intercept is not physically meaningful."
        else:
            band_gap_estimate = float(fit.x_intercept)
            ax_fit.axvline(
                band_gap_estimate,
                color="#2a9d8f",
                linestyle="--",
                linewidth=1.5,
                label=f"Estimated Eg = {band_gap_estimate:.4f} eV",
            )

        ax_fit.set_xlabel("Photon Energy (eV)")
        ax_fit.set_ylabel(f"(A*hnu)^{exponent:.4g} (a.u.)")
        ax_fit.set_title("Tauc Plot with Linear Fit")
        ax_fit.legend(loc="best")
        render_plot_with_download(fig_fit, "absorbance_tauc_fit_plot.png", "absorbance_tauc_fit_plot")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Slope", f"{fit.slope:.5g}")
        col2.metric("Intercept", f"{fit.intercept:.5g}")
        col3.metric("R^2", f"{fit.r2:.5f}")
        col4.metric("x-intercept (eV)", f"{fit.x_intercept:.5f}")

        if band_gap_estimate is None:
            st.error(band_gap_note)
        else:
            st.success(
                f"Estimated absorbance-based band gap: {band_gap_estimate:.4f} eV"
            )
            if band_gap_estimate < energy_min or band_gap_estimate > energy_max:
                st.warning(
                    "Estimated intercept lies outside measured photon-energy range; interpret with extra caution."
                )

        summary = {
            "Timestamp": datetime.now().isoformat(timespec="seconds"),
            "Mode": "Absorbance",
            "Data source": data_source,
            "Transition assumption": transition_name,
            "Transition n": symbolic_n,
            "Tauc exponent used": exponent,
            "Manual fit min (eV)": fit_range[0],
            "Manual fit max (eV)": fit_range[1],
            "Slope": fit.slope,
            "Intercept": fit.intercept,
            "R^2": fit.r2,
            "x-intercept (eV)": fit.x_intercept,
            "Min R^2 requirement": min_r2_required,
            "Band gap estimate (eV)": (
                f"{band_gap_estimate:.6g}" if band_gap_estimate is not None else "Not reported"
            ),
            "Band gap note": band_gap_note if band_gap_estimate is None else "Absorbance-based estimate",
            "Processing steps": " | ".join(processing_notes) if processing_notes else "None",
        }
        render_summary_downloads(summary, "absorbance", "absorbance_fit")


def render_peak_mode(
    mode: str,
    x_nm: np.ndarray,
    y_signal: np.ndarray,
    data_source: str,
    processing_notes: List[str],
    include_fwhm: bool,
) -> None:
    st.subheader(f"{mode} Analysis")

    fig_signal, _ = make_line_figure(
        x_nm,
        y_signal,
        x_label="Wavelength (nm)",
        y_label="Intensity (a.u.)",
        title=f"{mode} Intensity vs Wavelength ({data_source})",
        color="#264653",
    )
    render_plot_with_download(
        fig_signal,
        download_name=f"{mode.lower().replace(' ', '_')}_signal_plot.png",
        key=f"{mode}_signal_plot",
    )

    st.markdown("**Automatic Peak Detection**")
    st.caption("All peak-detection parameters are explicit and user-controlled.")

    y_range = float(np.nanmax(y_signal) - np.nanmin(y_signal))
    prominence_default = max(0.0, 0.05 * y_range)
    distance_default = max(1, int(len(y_signal) / 80))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        prominence = st.number_input(
            "Min prominence",
            min_value=0.0,
            value=float(prominence_default),
            step=max(1e-6, prominence_default / 5 if prominence_default > 0 else 0.01),
            format="%.6g",
            key=f"prominence_{mode}",
        )
    with col2:
        min_distance = st.number_input(
            "Min distance (points)",
            min_value=1,
            max_value=max(1, len(y_signal) - 1),
            value=int(distance_default),
            step=1,
            key=f"distance_{mode}",
        )
    with col3:
        min_width = st.number_input(
            "Min width (points)",
            min_value=0.0,
            value=0.0,
            step=1.0,
            key=f"width_{mode}",
        )
    with col4:
        use_min_height = st.checkbox("Use min height", value=False, key=f"use_height_{mode}")
        min_height = None
        if use_min_height:
            min_height = st.number_input(
                "Min height",
                value=float(np.nanmedian(y_signal)),
                format="%.6g",
                key=f"height_{mode}",
            )

    try:
        peak_result = detect_peaks(
            wavelength_nm=x_nm,
            signal=y_signal,
            prominence=float(prominence),
            distance_points=int(min_distance),
            min_width_points=float(min_width),
            min_height=float(min_height) if min_height is not None else None,
        )
    except ValueError as exc:
        st.error(f"Peak detection error: {exc}")
        return

    fig_peaks, ax_peaks = plt.subplots(figsize=(8.5, 4.8))
    ax_peaks.plot(x_nm, y_signal, color="#1d3557", linewidth=1.2, label="Signal")

    if len(peak_result.indices) > 0:
        ax_peaks.scatter(
            x_nm[peak_result.indices],
            y_signal[peak_result.indices],
            color="#e63946",
            s=40,
            label="Detected peaks",
            zorder=3,
        )
        for i, idx in enumerate(peak_result.indices, start=1):
            ax_peaks.annotate(
                str(i),
                (x_nm[idx], y_signal[idx]),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
            )

    ax_peaks.set_xlabel("Wavelength (nm)")
    ax_peaks.set_ylabel("Intensity (a.u.)")
    ax_peaks.set_title(f"{mode} with Detected Peaks")
    ax_peaks.legend(loc="best")
    render_plot_with_download(
        fig_peaks,
        download_name=f"{mode.lower().replace(' ', '_')}_peaks_plot.png",
        key=f"{mode}_peaks_plot",
    )

    if peak_result.table.empty:
        st.warning("No peaks found with the current parameter settings.")
    else:
        st.success(f"Detected {len(peak_result.table)} peak(s).")
        st.dataframe(peak_result.table, use_container_width=True, height=260)
        st.download_button(
            "Download peak table (CSV)",
            data=peak_result.table.to_csv(index=False).encode("utf-8"),
            file_name=f"{mode.lower().replace(' ', '_')}_peaks.csv",
            mime="text/csv",
            key=f"{mode}_peak_table_csv",
        )

    if include_fwhm and not peak_result.table.empty:
        st.markdown("**Optional FWHM Calculation**")
        enable_fwhm = st.checkbox(
            "Compute FWHM for selected peaks",
            value=False,
            key="pl_fwhm_toggle",
        )

        if enable_fwhm:
            option_map = {
                f"Peak {int(row['Peak #'])} ({row['Wavelength (nm)']:.2f} nm)": int(row["Peak #"])
                for _, row in peak_result.table.iterrows()
            }
            selected_options = st.multiselect(
                "Select peak(s)",
                options=list(option_map.keys()),
                default=list(option_map.keys()),
                key="pl_fwhm_selection",
            )
            selected_peak_numbers = [option_map[label] for label in selected_options]

            fwhm_all = compute_fwhm_table(x_nm, y_signal, peak_result.indices)
            fwhm_selected = fwhm_all[fwhm_all["Peak #"].isin(selected_peak_numbers)].copy()

            if fwhm_selected.empty:
                st.info("No FWHM values available for the selected peaks.")
            else:
                merged = peak_result.table.merge(fwhm_selected, on="Peak #", how="left")
                st.dataframe(merged, use_container_width=True, height=260)
                st.download_button(
                    "Download FWHM table (CSV)",
                    data=merged.to_csv(index=False).encode("utf-8"),
                    file_name="pl_emission_fwhm.csv",
                    mime="text/csv",
                    key="pl_fwhm_csv",
                )

    summary = {
        "Timestamp": datetime.now().isoformat(timespec="seconds"),
        "Mode": mode,
        "Data source": data_source,
        "Detected peak count": int(len(peak_result.table)),
        "Min prominence": float(prominence),
        "Min distance (points)": int(min_distance),
        "Min width (points)": float(min_width),
        "Min height": "None" if min_height is None else float(min_height),
        "Processing steps": " | ".join(processing_notes) if processing_notes else "None",
    }
    render_summary_downloads(
        summary,
        file_prefix=mode.lower().replace(" ", "_"),
        key_prefix=f"summary_{mode}",
    )


def main() -> None:
    st.set_page_config(page_title="Spectroscopy CSV Analyzer", layout="wide")
    st.title("Spectroscopy CSV Analyzer")
    st.caption(
        "Upload a two-column CSV (wavelength and signal), inspect raw data first, then run mode-specific analysis."
    )

    with st.sidebar:
        st.header("Inputs")
        mode = st.selectbox("Measurement type", MEASUREMENT_MODES, index=0)
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is None:
        st.info("Select measurement type and upload a CSV file to begin.")
        return

    try:
        parsed = parse_spectroscopy_csv(uploaded_file)
    except ValueError as exc:
        st.error(f"CSV parse error: {exc}")
        return
    except Exception as exc:  # defensive guard for unexpected parser issues
        st.error(f"Unexpected error while parsing CSV: {exc}")
        return

    raw_df = parsed.dataframe
    if parsed.notes:
        st.warning("CSV parsing notes:\n- " + "\n- ".join(parsed.notes))

    x_nm = raw_df["wavelength_nm"].to_numpy()
    y_raw = raw_df["signal"].to_numpy()

    render_raw_data_section(mode, raw_df)
    y_for_analysis, processing_notes, data_source = render_processing_section(mode, x_nm, y_raw)

    st.markdown("---")
    if mode == "Absorbance":
        render_absorbance_mode(x_nm, y_for_analysis, data_source, processing_notes)
    elif mode == "Excitation":
        render_peak_mode(mode, x_nm, y_for_analysis, data_source, processing_notes, include_fwhm=False)
    elif mode == "PL Emission":
        render_peak_mode(mode, x_nm, y_for_analysis, data_source, processing_notes, include_fwhm=True)


if __name__ == "__main__":
    main()