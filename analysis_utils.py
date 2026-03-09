"""Utility functions for spectroscopy CSV analysis."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import constants
from scipy.signal import find_peaks, peak_widths, savgol_filter


# h*c/e converted to eV*nm, used for photon-energy conversion.
EV_NM_CONSTANT = constants.h * constants.c / constants.e * 1e9


TAUC_TRANSITIONS: Dict[str, Dict[str, float | str]] = {
    "Direct allowed": {"n": "1/2", "exponent": 2.0},
    "Direct forbidden": {"n": "3/2", "exponent": 2.0 / 3.0},
    "Indirect allowed": {"n": "2", "exponent": 0.5},
    "Indirect forbidden": {"n": "3", "exponent": 1.0 / 3.0},
}


@dataclass
class ParsedCSV:
    dataframe: pd.DataFrame
    notes: List[str]


@dataclass
class PreprocessOptions:
    baseline_method: str = "none"  # none, subtract_min, endpoint_linear
    smoothing_enabled: bool = False
    smoothing_window: int = 11
    smoothing_polyorder: int = 2
    normalization: str = "none"  # none, divide_by_max


@dataclass
class FitResult:
    slope: float
    intercept: float
    r2: float
    x_intercept: float
    points_used: int
    x_min: float
    x_max: float


@dataclass
class PeakDetectionResult:
    indices: np.ndarray
    table: pd.DataFrame
    properties: Dict[str, np.ndarray]


def _read_csv_candidate(csv_bytes: bytes, header: Optional[int | str]) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(
            BytesIO(csv_bytes),
            sep=None,
            engine="python",
            header=header,
            comment="#",
        )
    except Exception:
        return None


def _extract_numeric_pair(candidate: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], List[str]]:
    notes: List[str] = []
    if candidate is None or candidate.empty:
        return None, notes

    numeric_map: Dict[str, pd.Series] = {}
    counts: Dict[str, int] = {}
    for column in candidate.columns:
        coerced = pd.to_numeric(candidate[column], errors="coerce")
        numeric_map[str(column)] = coerced
        counts[str(column)] = int(coerced.notna().sum())

    sorted_columns = sorted(counts.keys(), key=lambda name: counts[name], reverse=True)
    usable_columns = [name for name in sorted_columns if counts[name] >= 3]
    if len(usable_columns) < 2:
        return None, notes

    x_col, y_col = usable_columns[0], usable_columns[1]
    if len(usable_columns) > 2:
        notes.append(
            f"Found more than two numeric columns; using '{x_col}' as x and '{y_col}' as y."
        )

    parsed = pd.DataFrame(
        {
            "wavelength_nm": numeric_map[x_col],
            "signal": numeric_map[y_col],
        }
    ).dropna()

    parsed = parsed[np.isfinite(parsed["wavelength_nm"]) & np.isfinite(parsed["signal"])]
    if len(parsed) < 5:
        return None, notes

    nonpositive_mask = parsed["wavelength_nm"] <= 0
    if nonpositive_mask.any():
        removed = int(nonpositive_mask.sum())
        parsed = parsed[~nonpositive_mask]
        notes.append(f"Removed {removed} row(s) with non-positive wavelength.")

    if len(parsed) < 5:
        return None, notes

    parsed = parsed.sort_values("wavelength_nm", kind="mergesort").reset_index(drop=True)
    return parsed, notes


def parse_spectroscopy_csv(uploaded_file) -> ParsedCSV:
    csv_bytes = uploaded_file.getvalue()
    if not csv_bytes:
        raise ValueError("Uploaded CSV is empty.")

    candidates: List[Tuple[pd.DataFrame, List[str]]] = []
    for header in ("infer", None):
        candidate = _read_csv_candidate(csv_bytes, header=header)
        parsed, notes = _extract_numeric_pair(candidate)
        if parsed is not None:
            candidates.append((parsed, notes))

    if not candidates:
        raise ValueError(
            "Could not detect two numeric columns. Ensure the CSV contains wavelength and signal columns."
        )

    parsed_df, notes = max(candidates, key=lambda item: len(item[0]))
    if parsed_df["wavelength_nm"].nunique() < 3:
        raise ValueError("Wavelength values appear degenerate. Please upload a valid spectrum CSV.")

    return ParsedCSV(dataframe=parsed_df, notes=notes)


def apply_preprocessing(
    wavelength_nm: np.ndarray, signal: np.ndarray, options: PreprocessOptions
) -> Tuple[np.ndarray, List[str]]:
    x = np.asarray(wavelength_nm, dtype=float)
    y = np.asarray(signal, dtype=float).copy()
    if x.shape != y.shape:
        raise ValueError("Wavelength and signal arrays must have the same length.")

    notes: List[str] = []

    if options.baseline_method == "subtract_min":
        baseline = float(np.nanmin(y))
        y = y - baseline
        notes.append(f"Baseline correction: subtracted minimum ({baseline:.6g}).")
    elif options.baseline_method == "endpoint_linear":
        if len(y) < 2:
            raise ValueError("Need at least two points for endpoint linear baseline correction.")
        x0, x1 = float(x[0]), float(x[-1])
        if x1 == x0:
            raise ValueError("Cannot apply endpoint linear baseline when wavelength span is zero.")
        slope = (y[-1] - y[0]) / (x1 - x0)
        baseline = y[0] + slope * (x - x0)
        y = y - baseline
        notes.append("Baseline correction: subtracted line connecting first and last points.")
    elif options.baseline_method != "none":
        raise ValueError(f"Unknown baseline method: {options.baseline_method}")

    if options.smoothing_enabled:
        max_window = len(y) if len(y) % 2 == 1 else len(y) - 1
        if max_window < 3:
            raise ValueError("Not enough points for Savitzky-Golay smoothing.")

        window = int(options.smoothing_window)
        if window % 2 == 0:
            window += 1
        window = min(window, max_window)

        polyorder = int(options.smoothing_polyorder)
        if polyorder < 1:
            raise ValueError("Smoothing polynomial order must be at least 1.")
        if window <= polyorder:
            window = polyorder + 2
            if window % 2 == 0:
                window += 1
            if window > max_window:
                raise ValueError(
                    "Smoothing settings invalid for this dataset. Reduce polyorder or disable smoothing."
                )

        y = savgol_filter(y, window_length=window, polyorder=polyorder, mode="interp")
        notes.append(f"Smoothing: Savitzky-Golay (window={window}, polyorder={polyorder}).")

    if options.normalization == "divide_by_max":
        max_value = float(np.nanmax(np.abs(y)))
        if max_value <= 0:
            raise ValueError("Cannot normalize because the processed signal has zero magnitude.")
        y = y / max_value
        notes.append("Normalization: divided by max absolute intensity.")
    elif options.normalization != "none":
        raise ValueError(f"Unknown normalization option: {options.normalization}")

    return y, notes


def photon_energy_from_wavelength_nm(wavelength_nm: np.ndarray) -> np.ndarray:
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    if np.any(wavelength_nm <= 0):
        raise ValueError("Wavelength values must be strictly positive to compute photon energy.")
    return EV_NM_CONSTANT / wavelength_nm


def compute_tauc_curve(
    wavelength_nm: np.ndarray, absorbance: np.ndarray, exponent: float
) -> Tuple[np.ndarray, np.ndarray, int]:
    energy_ev = photon_energy_from_wavelength_nm(wavelength_nm)
    absorbance = np.asarray(absorbance, dtype=float)
    if energy_ev.shape != absorbance.shape:
        raise ValueError("Wavelength and absorbance arrays must have the same length.")

    valid_mask = np.isfinite(energy_ev) & np.isfinite(absorbance)
    energy_ev = energy_ev[valid_mask]
    absorbance = absorbance[valid_mask]

    tauc_base = absorbance * energy_ev
    positive_mask = tauc_base > 0
    excluded_count = int((~positive_mask).sum())

    energy_valid = energy_ev[positive_mask]
    tauc_y = np.power(tauc_base[positive_mask], exponent)
    if len(energy_valid) < 5:
        raise ValueError("Too few positive (A*hnu) points for Tauc analysis.")

    sort_idx = np.argsort(energy_valid)
    return energy_valid[sort_idx], tauc_y[sort_idx], excluded_count


def _linear_fit(x: np.ndarray, y: np.ndarray) -> FitResult:
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = np.nan if ss_tot <= 0 else 1.0 - (ss_res / ss_tot)
    x_intercept = np.nan if slope == 0 else -intercept / slope
    return FitResult(
        slope=float(slope),
        intercept=float(intercept),
        r2=float(r2),
        x_intercept=float(x_intercept),
        points_used=int(len(x)),
        x_min=float(np.min(x)),
        x_max=float(np.max(x)),
    )


def linear_fit_subset(
    x: np.ndarray, y: np.ndarray, x_min: float, x_max: float, min_points: int = 8
) -> FitResult:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    fit_mask = (x >= x_min) & (x <= x_max) & np.isfinite(x) & np.isfinite(y)
    if int(fit_mask.sum()) < min_points:
        raise ValueError(
            f"Selected fitting region has {int(fit_mask.sum())} point(s). Need at least {min_points}."
        )
    return _linear_fit(x[fit_mask], y[fit_mask])


def suggest_linear_region(
    x: np.ndarray,
    y: np.ndarray,
    window_fraction: float = 0.2,
    min_points: int = 12,
    require_positive_slope: bool = True,
) -> Optional[FitResult]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite_mask = np.isfinite(x) & np.isfinite(y)
    x = x[finite_mask]
    y = y[finite_mask]

    if len(x) < min_points:
        return None

    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]
    full_range = float(np.nanmax(y) - np.nanmin(y))

    window_points = max(min_points, int(round(window_fraction * len(x))))
    window_points = min(window_points, len(x))
    if window_points < min_points:
        return None

    best_fit: Optional[FitResult] = None
    best_r2 = -np.inf

    for start in range(0, len(x) - window_points + 1):
        x_window = x[start : start + window_points]
        y_window = y[start : start + window_points]
        local_range = float(np.nanmax(y_window) - np.nanmin(y_window))

        if full_range > 0 and local_range < 0.05 * full_range:
            continue

        fit = _linear_fit(x_window, y_window)
        if require_positive_slope and fit.slope <= 0:
            continue
        if not np.isfinite(fit.r2):
            continue
        if fit.r2 > best_r2:
            best_r2 = fit.r2
            best_fit = fit

    return best_fit


def detect_peaks(
    wavelength_nm: np.ndarray,
    signal: np.ndarray,
    prominence: float,
    distance_points: int,
    min_width_points: float = 0.0,
    min_height: Optional[float] = None,
) -> PeakDetectionResult:
    x = np.asarray(wavelength_nm, dtype=float)
    y = np.asarray(signal, dtype=float)
    finite_mask = np.isfinite(x) & np.isfinite(y)

    x_clean = x[finite_mask]
    y_clean = y[finite_mask]
    if len(x_clean) < 3:
        raise ValueError("Need at least three finite data points for peak detection.")

    index_map = np.where(finite_mask)[0]
    width_arg = min_width_points if min_width_points > 0 else None
    height_arg = min_height if min_height is not None else None

    peaks_clean, properties = find_peaks(
        y_clean,
        prominence=max(float(prominence), 0.0),
        distance=max(int(distance_points), 1),
        width=width_arg,
        height=height_arg,
    )
    peaks_original = index_map[peaks_clean]

    if len(peaks_original) == 0:
        empty_table = pd.DataFrame(
            columns=["Peak #", "Wavelength (nm)", "Intensity", "Prominence"]
        )
        return PeakDetectionResult(indices=peaks_original, table=empty_table, properties=properties)

    prominences = properties.get("prominences", np.full(len(peaks_original), np.nan))
    table = pd.DataFrame(
        {
            "Peak #": np.arange(1, len(peaks_original) + 1),
            "Wavelength (nm)": x[peaks_original],
            "Intensity": y[peaks_original],
            "Prominence": prominences,
        }
    )
    return PeakDetectionResult(indices=peaks_original, table=table, properties=properties)


def compute_fwhm_table(
    wavelength_nm: np.ndarray, signal: np.ndarray, peak_indices: np.ndarray
) -> pd.DataFrame:
    x = np.asarray(wavelength_nm, dtype=float)
    y = np.asarray(signal, dtype=float)
    selected_indices = np.asarray(peak_indices, dtype=int)
    if len(selected_indices) == 0:
        return pd.DataFrame(
            columns=["Peak #", "Left wavelength (nm)", "Right wavelength (nm)", "FWHM (nm)"]
        )

    finite_mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[finite_mask]
    y_clean = y[finite_mask]
    index_map = np.where(finite_mask)[0]

    valid_selection_mask = np.isin(index_map, selected_indices)
    peak_indices_clean = np.where(valid_selection_mask)[0]
    if len(peak_indices_clean) == 0:
        return pd.DataFrame(
            columns=["Peak #", "Left wavelength (nm)", "Right wavelength (nm)", "FWHM (nm)"]
        )

    _, _, left_ips, right_ips = peak_widths(y_clean, peak_indices_clean, rel_height=0.5)
    left_nm = np.interp(left_ips, np.arange(len(x_clean), dtype=float), x_clean)
    right_nm = np.interp(right_ips, np.arange(len(x_clean), dtype=float), x_clean)

    return pd.DataFrame(
        {
            "Peak #": np.arange(1, len(peak_indices_clean) + 1),
            "Left wavelength (nm)": left_nm,
            "Right wavelength (nm)": right_nm,
            "FWHM (nm)": right_nm - left_nm,
        }
    )


def figure_to_png_bytes(fig) -> bytes:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    return buffer.getvalue()