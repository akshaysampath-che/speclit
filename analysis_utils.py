"""Utility functions for spectroscopy table loading and spectral analysis."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
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

CSV_DELIMITERS = [",", ";", "\t", "|"]
WAVELENGTH_KEYWORDS = ["wavelength", "lambda", "wl", "nm"]


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


@dataclass
class ReflectanceTaucPreparation:
    wavelength_nm: np.ndarray
    reflectance_fraction: np.ndarray
    kubelka_munk: np.ndarray
    energy_ev: np.ndarray
    tauc_y: np.ndarray
    scale_used: str
    excluded_count: int
    notes: List[str]


def _standardize_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        raise ValueError("Loaded table is empty.")

    table = df.copy()
    table = table.dropna(how="all")
    table = table.dropna(axis=1, how="all")
    if table.empty:
        raise ValueError("Loaded table has no usable rows or columns.")

    normalized_columns: List[str] = []
    for i, column in enumerate(table.columns):
        col_name = str(column).strip()
        if not col_name or col_name.lower().startswith("unnamed"):
            col_name = f"Column_{i + 1}"
        normalized_columns.append(col_name)
    table.columns = normalized_columns

    return table


def _numeric_score(df: pd.DataFrame) -> Tuple[int, int]:
    numeric_like_cols = 0
    for column in df.columns:
        valid_count = int(pd.to_numeric(df[column], errors="coerce").notna().sum())
        if valid_count >= 3:
            numeric_like_cols += 1
    return numeric_like_cols, len(df)


def list_excel_sheets(file_bytes: bytes) -> List[str]:
    try:
        with pd.ExcelFile(BytesIO(file_bytes), engine="openpyxl") as excel:
            return list(excel.sheet_names)
    except Exception as exc:
        raise ValueError(f"Failed to inspect Excel sheets: {exc}") from exc


def load_tabular_file(
    file_bytes: bytes,
    filename: str,
    sheet_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    extension = Path(filename).suffix.lower()
    notes: List[str] = []

    if extension in {".xlsx", ".xlsm", ".xls"}:
        selected_sheet = sheet_name if sheet_name is not None else 0
        candidates: List[pd.DataFrame] = []
        for header in (0, None):
            try:
                candidate = pd.read_excel(
                    BytesIO(file_bytes),
                    sheet_name=selected_sheet,
                    header=header,
                    engine="openpyxl",
                )
                candidates.append(_standardize_table(candidate))
            except Exception:
                continue

        if not candidates:
            raise ValueError("Could not read worksheet data from Excel file.")

        best = max(candidates, key=_numeric_score)
        if sheet_name is not None:
            notes.append(f"Loaded worksheet: {sheet_name}")
        else:
            notes.append("Loaded default worksheet.")
        return best.reset_index(drop=True), notes

    if extension in {".csv", ".txt", ".dat"}:
        candidates: List[pd.DataFrame] = []

        for header in ("infer", None):
            for delimiter in [None] + CSV_DELIMITERS:
                try:
                    if delimiter is None:
                        candidate = pd.read_csv(
                            BytesIO(file_bytes),
                            sep=None,
                            engine="python",
                            header=header,
                            comment="#",
                        )
                    else:
                        candidate = pd.read_csv(
                            BytesIO(file_bytes),
                            sep=delimiter,
                            header=header,
                            comment="#",
                        )
                    candidates.append(_standardize_table(candidate))
                except Exception:
                    continue

        if not candidates:
            raise ValueError("Could not parse CSV/TXT file with common delimiters.")

        best = max(candidates, key=_numeric_score)
        notes.append("Loaded delimited text table with automatic delimiter detection.")
        return best.reset_index(drop=True), notes

    raise ValueError("Unsupported file format. Please upload .csv, .txt, .dat, .xlsx, .xls, or .xlsm.")


def clean_wavelength_column(column: pd.Series) -> Tuple[pd.Series, int]:
    """Clean a wavelength column and return numeric values + removed-row count.

    Cleaning logic:
    1) cast to string
    2) trim whitespace
    3) remove 'nm' case-insensitively
    4) remove extra spaces
    5) convert to numeric (errors='coerce')
    """
    raw_as_str = column.astype(str)
    cleaned = raw_as_str.str.strip()
    cleaned = cleaned.str.replace('"', "", regex=False).str.replace("'", "", regex=False)
    cleaned = cleaned.str.replace(r"(?i)nm", "", regex=True)
    cleaned = cleaned.str.replace(r"\s+", "", regex=True)

    numeric = pd.to_numeric(cleaned, errors="coerce").astype(float)

    nonempty_original = column.notna() & raw_as_str.str.strip().ne("")
    removed_rows = int((numeric.isna() & nonempty_original).sum())
    return numeric, removed_rows


def detect_x_column(df: pd.DataFrame, min_valid_points: int = 3) -> Tuple[str, List[str]]:
    if df is None or df.empty:
        raise ValueError("Input table is empty.")

    columns = list(df.columns)
    if not columns:
        raise ValueError("Input table has no columns.")

    raw_numeric_counts: Dict[str, int] = {}
    for column in columns:
        raw_numeric_counts[column] = int(pd.to_numeric(df[column], errors="coerce").notna().sum())

    notes: List[str] = []

    keyword_candidates = [
        column
        for column in columns
        if any(keyword in str(column).strip().lower() for keyword in WAVELENGTH_KEYWORDS)
    ]

    if keyword_candidates:
        x_column = max(keyword_candidates, key=lambda c: raw_numeric_counts[c])
        notes.append(f"Auto-detected x-axis column '{x_column}' using wavelength-like column name.")
        return x_column, notes

    first_column = columns[0]
    notes.append("No wavelength-like header detected; using the first column as x-axis by default.")

    if raw_numeric_counts[first_column] >= min_valid_points:
        return first_column, notes

    numeric_columns = [column for column in columns if raw_numeric_counts[column] >= min_valid_points]
    if numeric_columns:
        notes.append(
            f"First column has limited direct numeric values; you can override x-axis manually if needed."
        )
        return first_column, notes

    notes.append(
        "No directly numeric columns detected before cleaning; selected first column as x-axis for wavelength cleanup."
    )
    return first_column, notes


def detect_y_columns(
    df: pd.DataFrame,
    x_column: str,
    min_valid_points: int = 3,
) -> Tuple[List[str], List[str]]:
    if x_column not in df.columns:
        raise ValueError(f"Selected x-axis column '{x_column}' is not in the table.")

    notes: List[str] = []
    y_columns: List[str] = []

    for column in df.columns:
        if column == x_column:
            continue
        numeric_count = int(pd.to_numeric(df[column], errors="coerce").notna().sum())
        if numeric_count >= min_valid_points:
            y_columns.append(column)
        else:
            notes.append(
                f"Column '{column}' skipped from default y-candidates (numeric points: {numeric_count})."
            )

    return y_columns, notes


def extract_series_data(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    min_points: int = 5,
    cleaned_x_numeric: Optional[pd.Series] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if x_column not in df.columns:
        raise ValueError(f"x-axis column '{x_column}' not found.")
    if y_column not in df.columns:
        raise ValueError(f"y-axis column '{y_column}' not found.")

    notes: List[str] = []

    if cleaned_x_numeric is None:
        x_numeric, removed_rows = clean_wavelength_column(df[x_column])
        if removed_rows > 0:
            notes.append(
                f"Wavelength cleanup removed {removed_rows} row(s) with unparsable x-values in '{x_column}'."
            )
    else:
        x_numeric = pd.to_numeric(cleaned_x_numeric, errors="coerce")
        if len(x_numeric) != len(df):
            raise ValueError("Pre-cleaned x-axis series length does not match table length.")

    y_numeric = pd.to_numeric(df[y_column], errors="coerce")

    x_valid_mask = x_numeric.notna()
    y_valid_mask = y_numeric.notna()
    finite_mask = x_valid_mask & y_valid_mask

    y_only_dropped = int((x_valid_mask & ~y_valid_mask).sum())
    if y_only_dropped > 0:
        notes.append(f"Dropped {y_only_dropped} row(s) with non-numeric y-values in '{y_column}'.")

    x_values = x_numeric[finite_mask].astype(float).to_numpy(dtype=float)
    y_values = y_numeric[finite_mask].astype(float).to_numpy(dtype=float)

    finite_array_mask = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[finite_array_mask]
    y_values = y_values[finite_array_mask]

    nonpositive_mask = x_values <= 0
    if nonpositive_mask.any():
        removed_nonpositive = int(nonpositive_mask.sum())
        x_values = x_values[~nonpositive_mask]
        y_values = y_values[~nonpositive_mask]
        notes.append(
            f"Removed {removed_nonpositive} row(s) with non-positive wavelength values for '{y_column}'."
        )

    if len(x_values) < min_points:
        raise ValueError(
            f"Insufficient valid points for '{y_column}' after cleanup ({len(x_values)} points, need {min_points})."
        )

    order = np.argsort(x_values, kind="mergesort")
    x_sorted = x_values[order]
    y_sorted = y_values[order]

    if np.unique(x_sorted).size < 3:
        raise ValueError(
            f"Wavelength values for '{y_column}' are too limited after cleanup; cannot analyze reliably."
        )

    return x_sorted, y_sorted, notes


def apply_preprocessing(
    wavelength_nm: np.ndarray,
    signal: np.ndarray,
    options: PreprocessOptions,
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
    wavelength_nm: np.ndarray,
    absorbance: np.ndarray,
    exponent: float,
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


def normalize_reflectance_scale(
    reflectance_values: np.ndarray,
    scale_mode: str,
) -> Tuple[np.ndarray, str, List[str]]:
    reflectance_values = np.asarray(reflectance_values, dtype=float)
    finite_mask = np.isfinite(reflectance_values)

    if int(finite_mask.sum()) == 0:
        raise ValueError("No finite reflectance values available.")

    notes: List[str] = []
    mode = scale_mode.lower().strip()
    if mode not in {"auto", "fraction", "percent"}:
        raise ValueError(f"Unknown reflectance scale mode: {scale_mode}")

    scale_used = mode
    nonnegative_finite = reflectance_values[finite_mask & (reflectance_values >= 0)]

    if mode == "auto":
        if nonnegative_finite.size == 0:
            raise ValueError("Auto scale detection failed because no non-negative reflectance values were found.")

        has_gt_one = bool(np.any(nonnegative_finite > 1.0))

        if has_gt_one:
            scale_used = "percent"
            notes.append(
                "Reflectance scale auto-detected as percent (0-100) because values greater than 1 were found."
            )
        elif float(np.nanmax(nonnegative_finite)) <= 1.0:
            raise ValueError(
                "Reflectance scale auto-detect is ambiguous because all non-negative values are within 0-1. Please choose fraction or percent explicitly."
            )
        else:
            raise ValueError(
                "Reflectance scale auto-detect failed due to out-of-range values. Please choose fraction or percent explicitly."
            )

    reflectance_fraction = np.full_like(reflectance_values, np.nan, dtype=float)
    if scale_used == "fraction":
        valid_range_mask = finite_mask & (reflectance_values >= 0.0) & (reflectance_values <= 1.0)
        reflectance_fraction[valid_range_mask] = reflectance_values[valid_range_mask]

        invalid_range_count = int((finite_mask & ~valid_range_mask).sum())
        if invalid_range_count > 0:
            notes.append(
                f"Skipped {invalid_range_count} reflectance row(s) outside fraction range [0, 1]."
            )
    else:
        valid_range_mask = finite_mask & (reflectance_values >= 0.0) & (reflectance_values <= 100.0)
        reflectance_fraction[valid_range_mask] = reflectance_values[valid_range_mask] / 100.0

        invalid_range_count = int((finite_mask & ~valid_range_mask).sum())
        if invalid_range_count > 0:
            notes.append(
                f"Skipped {invalid_range_count} reflectance row(s) outside percent range [0, 100]."
            )

    nonfinite_count = int((~finite_mask).sum())
    if nonfinite_count > 0:
        notes.append(f"Skipped {nonfinite_count} reflectance row(s) with non-finite values.")

    return reflectance_fraction, scale_used, notes


def kubelka_munk_transform(reflectance_fraction: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    reflectance_fraction = np.asarray(reflectance_fraction, dtype=float)
    kubelka_munk = np.full_like(reflectance_fraction, np.nan, dtype=float)

    valid_mask = np.isfinite(reflectance_fraction) & (reflectance_fraction > 0.0) & (reflectance_fraction <= 1.0)
    kubelka_munk[valid_mask] = ((1.0 - reflectance_fraction[valid_mask]) ** 2) / (
        2.0 * reflectance_fraction[valid_mask]
    )

    notes: List[str] = []
    nonpositive_count = int((np.isfinite(reflectance_fraction) & (reflectance_fraction <= 0.0)).sum())
    if nonpositive_count > 0:
        notes.append(
            f"Skipped {nonpositive_count} row(s) with reflectance <= 0 because Kubelka-Munk F(R) is undefined there."
        )

    above_one_count = int((np.isfinite(reflectance_fraction) & (reflectance_fraction > 1.0)).sum())
    if above_one_count > 0:
        notes.append(
            f"Skipped {above_one_count} row(s) with reflectance > 1 after normalization."
        )

    return kubelka_munk, notes


def prepare_reflectance_tauc_data(
    wavelength_nm: np.ndarray,
    reflectance_values: np.ndarray,
    exponent: float,
    scale_mode: str,
) -> ReflectanceTaucPreparation:
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    reflectance_values = np.asarray(reflectance_values, dtype=float)
    if wavelength_nm.shape != reflectance_values.shape:
        raise ValueError("Wavelength and reflectance arrays must have the same length.")

    energy_ev = photon_energy_from_wavelength_nm(wavelength_nm)

    reflectance_fraction, scale_used, scale_notes = normalize_reflectance_scale(
        reflectance_values=reflectance_values,
        scale_mode=scale_mode,
    )
    kubelka_munk, km_notes = kubelka_munk_transform(reflectance_fraction)

    valid_mask = np.isfinite(energy_ev) & np.isfinite(kubelka_munk)
    excluded_count = int((~valid_mask).sum())

    wavelength_valid = wavelength_nm[valid_mask]
    reflectance_valid = reflectance_fraction[valid_mask]
    km_valid = kubelka_munk[valid_mask]
    energy_valid = energy_ev[valid_mask]

    tauc_base = km_valid * energy_valid
    positive_mask = tauc_base > 0
    excluded_count += int((~positive_mask).sum())

    wavelength_valid = wavelength_valid[positive_mask]
    reflectance_valid = reflectance_valid[positive_mask]
    km_valid = km_valid[positive_mask]
    energy_valid = energy_valid[positive_mask]
    tauc_y = np.power(tauc_base[positive_mask], exponent)

    if len(energy_valid) < 5:
        raise ValueError("Too few valid Kubelka-Munk points for reflectance Tauc analysis.")

    sort_idx = np.argsort(energy_valid)
    notes = scale_notes + km_notes
    if excluded_count > 0:
        notes.append(f"Excluded {excluded_count} row(s) from reflectance Tauc preparation.")

    return ReflectanceTaucPreparation(
        wavelength_nm=wavelength_valid[sort_idx],
        reflectance_fraction=reflectance_valid[sort_idx],
        kubelka_munk=km_valid[sort_idx],
        energy_ev=energy_valid[sort_idx],
        tauc_y=tauc_y[sort_idx],
        scale_used=scale_used,
        excluded_count=excluded_count,
        notes=notes,
    )


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
    x: np.ndarray,
    y: np.ndarray,
    x_min: float,
    x_max: float,
    min_points: int = 8,
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
    wavelength_nm: np.ndarray,
    signal: np.ndarray,
    peak_indices: np.ndarray,
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


def figure_to_png_bytes(
    fig,
    width_px: Optional[int] = None,
    height_px: Optional[int] = None,
    dpi: int = 300,
    tight_bbox: bool = True,
) -> bytes:
    if dpi <= 0:
        raise ValueError("Export DPI must be positive.")

    if (width_px is None) != (height_px is None):
        raise ValueError("Both width_px and height_px must be provided together.")

    if width_px is not None and width_px <= 0:
        raise ValueError("Export width must be positive.")
    if height_px is not None and height_px <= 0:
        raise ValueError("Export height must be positive.")

    buffer = BytesIO()
    original_size_inches = fig.get_size_inches().copy()

    try:
        if width_px is not None and height_px is not None:
            fig.set_size_inches(float(width_px) / float(dpi), float(height_px) / float(dpi), forward=True)

        bbox_setting = "tight" if tight_bbox else None
        fig.savefig(buffer, format="png", dpi=dpi, bbox_inches=bbox_setting)
    finally:
        fig.set_size_inches(original_size_inches, forward=True)

    return buffer.getvalue()