
"""Speclit app for spectroscopy CSV/XLSX analysis with multi-series support."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator, NullLocator
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
    linear_fit_subset,
    list_excel_sheets,
    load_tabular_file,
    prepare_reflectance_tauc_data,
    suggest_linear_region,
)


plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.grid"] = True
plt.rcParams["font.family"] = ["Times New Roman"]

MEASUREMENT_MODES = ["Absorbance", "Reflectance", "Excitation", "PL Emission"]
Y_LABELS = {
    "Absorbance": "Absorbance / OD",
    "Excitation": "Intensity (a.u.)",
    "PL Emission": "Intensity (a.u.)",
    "Reflectance": "Reflectance (as provided)",
}

DEFAULT_REQUESTED_FONT_FAMILY = "Times New Roman"
GENERIC_FONT_FAMILIES = {
    "serif",
    "sans-serif",
    "sans serif",
    "cursive",
    "fantasy",
    "monospace",
}
SERIES_STYLE_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
PUBLICATION_TEMPLATE_NAMES = [
    "Default",
    "Nature-inspired single column",
    "Nature-inspired double column",
    "Science-inspired single column",
    "Science-inspired double column",
    "Presentation",
    "Custom",
]
EXPORT_FORMAT_OPTIONS = ["png", "jpg", "jpeg", "tiff", "svg", "pdf"]
RASTER_EXPORT_FORMATS = {"png", "jpg", "jpeg", "tiff"}
EXPORT_MIME_MAP = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "tiff": "image/tiff",
    "svg": "image/svg+xml",
    "pdf": "application/pdf",
}

DEFAULT_PUBLICATION_COLOR_PALETTE = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#E69F00",
    "#000000",
]
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


@dataclass
class PublicationTemplate:
    name: str
    width_mm: float
    height_mm: float
    dpi: int
    font_family: str
    title_font_size: float
    axis_label_font_size: float
    tick_label_font_size: float
    legend_font_size: float
    line_width: float
    marker_size: float
    axis_spine_width: float
    tick_width: float
    tick_length: float
    show_top_right_spines: bool
    grid_on: bool
    background_color: str
    transparent_background: bool
    color_palette: List[str]
    margins: Dict[str, float]


@dataclass
class PlotCustomizationSettings:
    use_manual_x_range: bool = False
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    use_manual_y_range: bool = False
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    use_custom_x_tick_interval: bool = False
    x_major_tick_interval: Optional[float] = None
    use_custom_y_tick_interval: bool = False
    y_major_tick_interval: Optional[float] = None
    show_major_ticks: bool = True
    show_minor_ticks: bool = False
    grid_mode: str = "default"  # default, off, major, major_minor
    tick_direction: str = "out"  # out, in, inout
    user_title: str = ""
    font_family: str = DEFAULT_REQUESTED_FONT_FAMILY
    font_size_global: Optional[float] = None
    title_font_size: Optional[float] = None
    axis_label_font_size: Optional[float] = None
    tick_label_font_size: Optional[float] = None
    legend_font_size: Optional[float] = None
    text_color: str = "#000000"
    x_axis_title_override: str = ""
    y_axis_title_override: str = ""
    series_line_colors: Dict[str, str] = field(default_factory=dict)
    series_line_widths: Dict[str, float] = field(default_factory=dict)
    tauc_fit_line_color: Optional[str] = None
    tauc_fit_line_width: Optional[float] = None
    tauc_eg_line_color: Optional[str] = None
    tauc_eg_line_width: Optional[float] = None
    publication_template: Optional[PublicationTemplate] = None


@dataclass
class ExportImageSettings:
    export_format: str = "png"
    width_mm: float = 90.0
    height_mm: float = 60.0
    raster_dpi: int = 300
    transparent_background: bool = False
    template_name: str = "Default"

def key_token(value: str) -> str:
    base = re.sub(r"[^a-zA-Z0-9_]+", "_", value)
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()[:8]
    return f"{base}_{digest}"


def get_available_font_names() -> List[str]:
    font_names = {font.name for font in font_manager.fontManager.ttflist if font.name}
    return sorted(font_names)


def normalize_font_family(font_family: str) -> str:
    cleaned = font_family.strip()
    return cleaned if cleaned else DEFAULT_REQUESTED_FONT_FAMILY


def is_generic_font_family(font_family: str) -> bool:
    return normalize_font_family(font_family).strip().lower() in GENERIC_FONT_FAMILIES


def is_font_available(font_family: str) -> bool:
    requested = normalize_font_family(font_family).strip().lower()
    if not requested:
        return False
    if requested in GENERIC_FONT_FAMILIES:
        return True
    return any(name.strip().lower() == requested for name in get_available_font_names())


def resolve_font_preview_name(font_family: str) -> str:
    try:
        font_path = font_manager.findfont(
            font_manager.FontProperties(family=[font_family]),
            fallback_to_default=True,
        )
        return font_manager.FontProperties(fname=font_path).get_name()
    except Exception:
        return "Unknown"


def is_valid_matplotlib_color(color_value: str) -> bool:
    try:
        return bool(mcolors.is_color_like(color_value))
    except Exception:
        return False


def should_warn_on_font_fallback(requested_font_family: str, resolved_font_family: str) -> bool:
    requested_normalized = normalize_font_family(requested_font_family)
    if is_generic_font_family(requested_normalized):
        return False
    if is_font_available(requested_normalized):
        return False
    return requested_normalized.strip().lower() != resolved_font_family.strip().lower()


def default_series_color_for_name(series_name: str, palette: Optional[List[str]] = None) -> str:
    colors = palette if palette else SERIES_STYLE_PALETTE
    if not colors:
        return "#1f77b4"

    digest = hashlib.md5(series_name.encode("utf-8")).hexdigest()
    index = int(digest[:8], 16) % len(colors)
    return colors[index]

def mm_to_inches(value_mm: float) -> float:
    return float(value_mm) / 25.4


def slugify_filename_token(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    return cleaned.strip("_") or "plot"


def get_plot_template(template_name: str) -> PublicationTemplate:
    name = template_name.strip()

    base_margins = {"left": 0.16, "right": 0.98, "top": 0.96, "bottom": 0.16}
    presets: Dict[str, PublicationTemplate] = {
        "Default": PublicationTemplate(
            name="Default",
            width_mm=120.0,
            height_mm=80.0,
            dpi=300,
            font_family=DEFAULT_REQUESTED_FONT_FAMILY,
            title_font_size=10.0,
            axis_label_font_size=9.0,
            tick_label_font_size=8.0,
            legend_font_size=8.0,
            line_width=1.2,
            marker_size=4.5,
            axis_spine_width=0.8,
            tick_width=0.8,
            tick_length=3.5,
            show_top_right_spines=False,
            grid_on=False,
            background_color="#ffffff",
            transparent_background=False,
            color_palette=DEFAULT_PUBLICATION_COLOR_PALETTE.copy(),
            margins=base_margins.copy(),
        ),
        "Nature-inspired single column": PublicationTemplate(
            name="Nature-inspired single column",
            width_mm=90.0,
            height_mm=62.0,
            dpi=600,
            font_family="DejaVu Sans",
            title_font_size=8.5,
            axis_label_font_size=8.0,
            tick_label_font_size=7.0,
            legend_font_size=7.0,
            line_width=1.0,
            marker_size=4.0,
            axis_spine_width=0.7,
            tick_width=0.7,
            tick_length=3.0,
            show_top_right_spines=False,
            grid_on=False,
            background_color="#ffffff",
            transparent_background=False,
            color_palette=["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#333333"],
            margins={"left": 0.18, "right": 0.98, "top": 0.95, "bottom": 0.18},
        ),
        "Nature-inspired double column": PublicationTemplate(
            name="Nature-inspired double column",
            width_mm=180.0,
            height_mm=115.0,
            dpi=600,
            font_family="DejaVu Sans",
            title_font_size=9.0,
            axis_label_font_size=8.5,
            tick_label_font_size=7.5,
            legend_font_size=7.5,
            line_width=1.1,
            marker_size=4.5,
            axis_spine_width=0.7,
            tick_width=0.7,
            tick_length=3.0,
            show_top_right_spines=False,
            grid_on=False,
            background_color="#ffffff",
            transparent_background=False,
            color_palette=["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#333333"],
            margins={"left": 0.12, "right": 0.99, "top": 0.95, "bottom": 0.15},
        ),
        "Science-inspired single column": PublicationTemplate(
            name="Science-inspired single column",
            width_mm=90.0,
            height_mm=66.0,
            dpi=600,
            font_family="DejaVu Sans",
            title_font_size=8.5,
            axis_label_font_size=8.0,
            tick_label_font_size=7.0,
            legend_font_size=7.0,
            line_width=1.1,
            marker_size=4.2,
            axis_spine_width=0.75,
            tick_width=0.75,
            tick_length=3.2,
            show_top_right_spines=False,
            grid_on=False,
            background_color="#ffffff",
            transparent_background=False,
            color_palette=["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#1f1f1f"],
            margins={"left": 0.18, "right": 0.98, "top": 0.95, "bottom": 0.18},
        ),
        "Science-inspired double column": PublicationTemplate(
            name="Science-inspired double column",
            width_mm=180.0,
            height_mm=118.0,
            dpi=600,
            font_family="DejaVu Sans",
            title_font_size=9.0,
            axis_label_font_size=8.5,
            tick_label_font_size=7.5,
            legend_font_size=7.5,
            line_width=1.15,
            marker_size=4.6,
            axis_spine_width=0.75,
            tick_width=0.75,
            tick_length=3.2,
            show_top_right_spines=False,
            grid_on=False,
            background_color="#ffffff",
            transparent_background=False,
            color_palette=["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#1f1f1f"],
            margins={"left": 0.12, "right": 0.99, "top": 0.95, "bottom": 0.15},
        ),
        "Presentation": PublicationTemplate(
            name="Presentation",
            width_mm=190.0,
            height_mm=110.0,
            dpi=300,
            font_family="DejaVu Sans",
            title_font_size=16.0,
            axis_label_font_size=13.0,
            tick_label_font_size=11.0,
            legend_font_size=11.0,
            line_width=2.0,
            marker_size=7.0,
            axis_spine_width=1.2,
            tick_width=1.2,
            tick_length=5.0,
            show_top_right_spines=False,
            grid_on=False,
            background_color="#ffffff",
            transparent_background=False,
            color_palette=["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9"],
            margins={"left": 0.10, "right": 0.98, "top": 0.92, "bottom": 0.14},
        ),
    }

    if name == "Custom":
        default_template = presets["Default"]
        return PublicationTemplate(
            name="Custom",
            width_mm=default_template.width_mm,
            height_mm=default_template.height_mm,
            dpi=default_template.dpi,
            font_family=default_template.font_family,
            title_font_size=default_template.title_font_size,
            axis_label_font_size=default_template.axis_label_font_size,
            tick_label_font_size=default_template.tick_label_font_size,
            legend_font_size=default_template.legend_font_size,
            line_width=default_template.line_width,
            marker_size=default_template.marker_size,
            axis_spine_width=default_template.axis_spine_width,
            tick_width=default_template.tick_width,
            tick_length=default_template.tick_length,
            show_top_right_spines=default_template.show_top_right_spines,
            grid_on=default_template.grid_on,
            background_color=default_template.background_color,
            transparent_background=default_template.transparent_background,
            color_palette=default_template.color_palette.copy(),
            margins=default_template.margins.copy(),
        )

    return presets.get(name, presets["Default"])


def apply_plot_template(fig, ax, template: PublicationTemplate) -> None:
    fig.set_dpi(template.dpi)
    fig.patch.set_facecolor(template.background_color)
    ax.set_facecolor(template.background_color)

    for spine_name, spine in ax.spines.items():
        if spine_name in {"top", "right"}:
            spine.set_visible(template.show_top_right_spines)
        else:
            spine.set_visible(True)
        spine.set_linewidth(float(template.axis_spine_width))

    ax.tick_params(
        axis="both",
        which="major",
        width=float(template.tick_width),
        length=float(template.tick_length),
    )
    ax.tick_params(
        axis="both",
        which="minor",
        width=max(0.1, float(template.tick_width) * 0.8),
        length=max(1.0, float(template.tick_length) * 0.6),
    )

    if template.grid_on:
        ax.grid(True, which="major", alpha=0.25)
    else:
        ax.grid(False, which="both")

    if template.margins:
        fig.subplots_adjust(
            left=template.margins.get("left", 0.12),
            right=template.margins.get("right", 0.98),
            top=template.margins.get("top", 0.95),
            bottom=template.margins.get("bottom", 0.14),
        )


def resolve_template_line_width(settings: Optional[PlotCustomizationSettings], fallback: float) -> float:
    if settings is None or settings.publication_template is None:
        return float(fallback)
    candidate = float(settings.publication_template.line_width)
    return candidate if candidate > 0 else float(fallback)


def resolve_template_marker_size(settings: Optional[PlotCustomizationSettings], fallback: float) -> float:
    if settings is None or settings.publication_template is None:
        return float(fallback)
    candidate = float(settings.publication_template.marker_size)
    return candidate if candidate > 0 else float(fallback)


def export_figure(
    fig,
    format: str,
    dpi: int,
    width_mm: float,
    height_mm: float,
    transparent: bool = False,
) -> bytes:
    fmt = format.strip().lower()
    if fmt not in EXPORT_FORMAT_OPTIONS:
        raise ValueError(f"Unsupported export format: {format}")

    if width_mm <= 0 or height_mm <= 0:
        raise ValueError("Export width and height must be positive in mm.")

    if fmt in RASTER_EXPORT_FORMATS and dpi <= 0:
        raise ValueError("Raster DPI must be positive.")

    save_format = "jpeg" if fmt in {"jpg", "jpeg"} else fmt
    buffer = BytesIO()

    original_size_inches = fig.get_size_inches().copy()
    original_facecolor = fig.get_facecolor()
    axis_facecolors = [axis.get_facecolor() for axis in fig.axes]

    try:
        fig.set_size_inches(mm_to_inches(width_mm), mm_to_inches(height_mm), forward=True)

        if transparent:
            fig.patch.set_alpha(0.0)
            for axis in fig.axes:
                axis.set_facecolor((1.0, 1.0, 1.0, 0.0))

        save_kwargs: Dict[str, object] = {
            "format": save_format,
            "bbox_inches": "tight",
            "transparent": bool(transparent),
        }

        if save_format in RASTER_EXPORT_FORMATS:
            save_kwargs["dpi"] = int(dpi)

        if save_format == "jpeg":
            save_kwargs["transparent"] = False
            save_kwargs["facecolor"] = "white"

        fig.savefig(buffer, **save_kwargs)
    finally:
        fig.set_size_inches(original_size_inches, forward=True)
        fig.patch.set_facecolor(original_facecolor)
        for axis, facecolor in zip(fig.axes, axis_facecolors):
            axis.set_facecolor(facecolor)

    return buffer.getvalue()


def build_export_filename(base_name: str, export_settings: ExportImageSettings) -> str:
    stem = slugify_filename_token(Path(base_name).stem)
    template_token = slugify_filename_token(export_settings.template_name)
    width_token = f"{export_settings.width_mm:.1f}".replace(".", "p")
    height_token = f"{export_settings.height_mm:.1f}".replace(".", "p")

    ext = export_settings.export_format.lower()
    if ext == "jpeg":
        ext = "jpg"

    dpi_token = ""
    if ext in RASTER_EXPORT_FORMATS:
        dpi_token = f"_{int(export_settings.raster_dpi)}dpi"

    return f"{stem}_{template_token}_{width_token}x{height_token}mm{dpi_token}.{ext}"


def sanitize_series_style_overrides(
    color_overrides: Dict[str, str],
    width_overrides: Dict[str, float],
) -> Tuple[Dict[str, str], Dict[str, float], List[str]]:
    sanitized_colors: Dict[str, str] = {}
    sanitized_widths: Dict[str, float] = {}
    messages: List[str] = []

    for series_name, color_value in color_overrides.items():
        if is_valid_matplotlib_color(color_value):
            sanitized_colors[series_name] = color_value
        else:
            messages.append(
                f"Invalid line color '{color_value}' for series '{series_name}'. Custom series color was not applied."
            )

    for series_name, width_value in width_overrides.items():
        if width_value > 0:
            sanitized_widths[series_name] = float(width_value)
        else:
            messages.append(
                f"Invalid line width ({width_value}) for series '{series_name}'. Custom series width was not applied."
            )

    return sanitized_colors, sanitized_widths, messages


def resolve_optional_line_style(
    fallback_color: str,
    fallback_width: float,
    custom_color: Optional[str],
    custom_width: Optional[float],
) -> Tuple[str, float]:
    color = custom_color if custom_color is not None and is_valid_matplotlib_color(custom_color) else fallback_color
    width = float(custom_width) if custom_width is not None and custom_width > 0 else float(fallback_width)
    return color, width


def resolve_plot_title(base_title: str, settings: Optional[PlotCustomizationSettings]) -> str:
    _ = base_title
    if settings is None:
        return ""
    return settings.user_title.strip()

def resolve_series_line_color(
    series_name: Optional[str],
    fallback_color: Optional[str],
    settings: Optional[PlotCustomizationSettings],
) -> Optional[str]:
    if settings is None or series_name is None:
        return fallback_color

    custom_candidate = settings.series_line_colors.get(series_name)
    if custom_candidate is not None:
        if is_valid_matplotlib_color(custom_candidate):
            return custom_candidate
        return fallback_color

    if fallback_color is not None:
        return fallback_color

    if settings.publication_template is not None and settings.publication_template.color_palette:
        return default_series_color_for_name(series_name, settings.publication_template.color_palette)

    return fallback_color
def resolve_series_line_width(
    series_name: Optional[str],
    fallback_width: float,
    settings: Optional[PlotCustomizationSettings],
) -> float:
    if settings is None or series_name is None:
        return float(fallback_width)

    candidate = settings.series_line_widths.get(series_name, fallback_width)
    try:
        numeric = float(candidate)
    except (TypeError, ValueError):
        return float(fallback_width)

    return numeric if numeric > 0 else float(fallback_width)


def run_style_helper_self_checks() -> List[str]:
    """Lightweight helper-level sanity checks for style resolution paths."""
    issues: List[str] = []

    probe_settings = PlotCustomizationSettings(
        series_line_colors={"SeriesA": "#112233"},
        series_line_widths={"SeriesA": 2.0},
    )

    if resolve_series_line_color("SeriesA", "#000000", probe_settings) != "#112233":
        issues.append("Series color override resolution failed for SeriesA.")

    if abs(resolve_series_line_width("SeriesA", 1.0, probe_settings) - 2.0) > 1e-9:
        issues.append("Series line-width override resolution failed for SeriesA.")

    if not is_valid_matplotlib_color("#112233"):
        issues.append("Matplotlib color validation unexpectedly rejected a valid hex color.")

    if is_valid_matplotlib_color("not_a_real_color"):
        issues.append("Matplotlib color validation unexpectedly accepted an invalid color string.")

    return issues


def _resolve_font_sizes(settings: PlotCustomizationSettings) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    template = settings.publication_template

    title_size = settings.title_font_size
    if title_size is None:
        title_size = settings.font_size_global if settings.font_size_global is not None else (template.title_font_size if template else None)

    axis_label_size = settings.axis_label_font_size
    if axis_label_size is None:
        axis_label_size = settings.font_size_global if settings.font_size_global is not None else (template.axis_label_font_size if template else None)

    tick_size = settings.tick_label_font_size
    if tick_size is None:
        tick_size = settings.font_size_global if settings.font_size_global is not None else (template.tick_label_font_size if template else None)

    legend_size = settings.legend_font_size
    if legend_size is None:
        legend_size = settings.font_size_global if settings.font_size_global is not None else (template.legend_font_size if template else None)

    return title_size, axis_label_size, tick_size, legend_size
def format_font_size_for_preview(value: Optional[float]) -> str:
    return "auto" if value is None else f"{float(value):g}"


def apply_title(ax, base_title: str, settings: Optional[PlotCustomizationSettings]) -> None:
    title_text = resolve_plot_title(base_title, settings)
    if title_text:
        ax.set_title(title_text)
    else:
        ax.set_title("")


def apply_label_settings(ax, settings: Optional[PlotCustomizationSettings]) -> None:
    if settings is None:
        return

    if settings.x_axis_title_override.strip():
        ax.set_xlabel(settings.x_axis_title_override.strip())
    if settings.y_axis_title_override.strip():
        ax.set_ylabel(settings.y_axis_title_override.strip())


def apply_axis_limits(ax, settings: Optional[PlotCustomizationSettings]) -> None:
    if settings is None:
        return

    if settings.use_manual_x_range and settings.x_min is not None and settings.x_max is not None:
        ax.set_xlim(float(settings.x_min), float(settings.x_max))

    if settings.use_manual_y_range and settings.y_min is not None and settings.y_max is not None:
        ax.set_ylim(float(settings.y_min), float(settings.y_max))


def apply_tick_settings(ax, settings: Optional[PlotCustomizationSettings]) -> None:
    if settings is None:
        return

    direction = settings.tick_direction if settings.tick_direction in {"in", "out", "inout"} else "out"
    tick_color = settings.text_color if is_valid_matplotlib_color(settings.text_color) else None
    if tick_color is not None:
        ax.tick_params(axis="both", which="both", direction=direction, colors=tick_color)
    else:
        ax.tick_params(axis="both", which="both", direction=direction)

    if settings.show_major_ticks:
        if settings.use_custom_x_tick_interval and settings.x_major_tick_interval is not None:
            ax.xaxis.set_major_locator(MultipleLocator(float(settings.x_major_tick_interval)))
        if settings.use_custom_y_tick_interval and settings.y_major_tick_interval is not None:
            ax.yaxis.set_major_locator(MultipleLocator(float(settings.y_major_tick_interval)))
    else:
        ax.xaxis.set_major_locator(NullLocator())
        ax.yaxis.set_major_locator(NullLocator())
        ax.tick_params(axis="both", which="major", bottom=False, top=False, left=False, right=False)
        ax.tick_params(axis="both", labelbottom=False, labelleft=False)

    if settings.show_minor_ticks:
        ax.minorticks_on()
        ax.tick_params(axis="both", which="minor", bottom=True, top=True, left=True, right=True)
    else:
        ax.minorticks_off()
        ax.tick_params(axis="both", which="minor", bottom=False, top=False, left=False, right=False)


def apply_grid_settings(ax, settings: Optional[PlotCustomizationSettings]) -> None:
    if settings is None:
        return

    if settings.grid_mode == "default":
        return

    if settings.grid_mode == "off":
        ax.grid(False, which="both")
        return

    if settings.grid_mode == "major":
        ax.grid(True, which="major", alpha=0.35)
        ax.grid(False, which="minor")
        return

    if settings.grid_mode == "major_minor":
        ax.grid(True, which="major", alpha=0.35)
        if settings.show_minor_ticks:
            ax.grid(True, which="minor", alpha=0.20)
        else:
            ax.grid(False, which="minor")


def apply_font_settings(ax, settings: Optional[PlotCustomizationSettings]) -> None:
    if settings is None:
        return

    font_family = normalize_font_family(settings.font_family)
    text_color = settings.text_color if is_valid_matplotlib_color(settings.text_color) else None
    title_size, axis_label_size, tick_size, _ = _resolve_font_sizes(settings)

    title_obj = ax.title
    title_obj.set_fontfamily(font_family)
    if title_size is not None:
        title_obj.set_fontsize(float(title_size))
    if text_color is not None:
        title_obj.set_color(text_color)

    for axis_label in (ax.xaxis.label, ax.yaxis.label):
        axis_label.set_fontfamily(font_family)
        if axis_label_size is not None:
            axis_label.set_fontsize(float(axis_label_size))
        if text_color is not None:
            axis_label.set_color(text_color)

    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontfamily(font_family)
        if tick_size is not None:
            tick_label.set_fontsize(float(tick_size))
        if text_color is not None:
            tick_label.set_color(text_color)


def apply_legend_settings(ax, settings: Optional[PlotCustomizationSettings]) -> None:
    if settings is None:
        return

    legend = ax.get_legend()
    if legend is None:
        return

    font_family = normalize_font_family(settings.font_family)
    text_color = settings.text_color if is_valid_matplotlib_color(settings.text_color) else None
    _, _, _, legend_size = _resolve_font_sizes(settings)

    for legend_text in legend.get_texts():
        legend_text.set_fontfamily(font_family)
        if legend_size is not None:
            legend_text.set_fontsize(float(legend_size))
        if text_color is not None:
            legend_text.set_color(text_color)

    legend_title = legend.get_title()
    if legend_title is not None:
        legend_title.set_fontfamily(font_family)
        if legend_size is not None:
            legend_title.set_fontsize(float(legend_size))
        if text_color is not None:
            legend_title.set_color(text_color)


def apply_plot_customization(ax, base_title: str, settings: Optional[PlotCustomizationSettings]) -> None:
    if settings is not None and settings.publication_template is not None:
        apply_plot_template(ax.figure, ax, settings.publication_template)

    apply_title(ax, base_title, settings)
    apply_label_settings(ax, settings)
    apply_axis_limits(ax, settings)
    apply_tick_settings(ax, settings)
    apply_grid_settings(ax, settings)
    apply_font_settings(ax, settings)

def make_line_figure(
    x: np.ndarray,
    y: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    color: Optional[str] = None,
    line_width: float = 1.4,
    series_name: Optional[str] = None,
    plot_settings: Optional[PlotCustomizationSettings] = None,
):
    fig, ax = plt.subplots(figsize=(8.6, 4.8))

    resolved_color = resolve_series_line_color(series_name, color, plot_settings)
    default_line_width = resolve_template_line_width(plot_settings, line_width)
    resolved_width = resolve_series_line_width(series_name, default_line_width, plot_settings)
    plot_kwargs: Dict[str, object] = {"linewidth": float(resolved_width)}
    if resolved_color is not None:
        plot_kwargs["color"] = resolved_color

    ax.plot(x, y, **plot_kwargs)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    apply_plot_customization(ax, title, plot_settings)
    return fig, ax
def render_plot_with_download(
    fig,
    download_name: str,
    key: str,
    export_settings: Optional[ExportImageSettings] = None,
    plot_settings: Optional[PlotCustomizationSettings] = None,
) -> None:
    for axis in fig.axes:
        apply_font_settings(axis, plot_settings)
        apply_legend_settings(axis, plot_settings)

    st.pyplot(fig, use_container_width=True)

    effective_export = export_settings if export_settings is not None else ExportImageSettings()
    export_format = effective_export.export_format.lower()
    export_dpi = int(effective_export.raster_dpi)

    if export_format in {"jpg", "jpeg"} and effective_export.transparent_background:
        st.warning("JPEG does not support transparency. Export will use an opaque background.")

    try:
        export_bytes = export_figure(
            fig,
            format=export_format,
            dpi=export_dpi,
            width_mm=float(effective_export.width_mm),
            height_mm=float(effective_export.height_mm),
            transparent=bool(effective_export.transparent_background and export_format not in {"jpg", "jpeg"}),
        )
    except ValueError as exc:
        st.error(f"Export failed: {exc}")
        plt.close(fig)
        return

    final_export_settings = ExportImageSettings(
        export_format=export_format,
        width_mm=float(effective_export.width_mm),
        height_mm=float(effective_export.height_mm),
        raster_dpi=export_dpi,
        transparent_background=bool(effective_export.transparent_background),
        template_name=effective_export.template_name,
    )

    final_filename = build_export_filename(download_name, final_export_settings)
    mime_type = EXPORT_MIME_MAP.get(export_format, "application/octet-stream")

    st.download_button(
        f"Download plot ({export_format.upper()})",
        data=export_bytes,
        file_name=final_filename,
        mime=mime_type,
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
    plot_settings: Optional[PlotCustomizationSettings] = None,
):
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for series in series_list:
        y = series.y_analysis if source == "analysis" else series.y_raw
        series_color = resolve_series_line_color(series.name, None, plot_settings)
        default_width = resolve_template_line_width(plot_settings, 1.2)
        series_width = resolve_series_line_width(series.name, default_width, plot_settings)
        plot_kwargs: Dict[str, object] = {
            "linewidth": float(series_width),
            "label": series.name,
        }
        if series_color is not None:
            plot_kwargs["color"] = series_color
        ax.plot(series.x, y, **plot_kwargs)

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(Y_LABELS[mode])
    source_label = "Analysis Signal" if source == "analysis" else "Raw Signal"
    apply_plot_customization(ax, f"{mode}: Combined Multi-Series Plot ({source_label})", plot_settings)
    ax.legend(loc="best", fontsize=8)
    apply_legend_settings(ax, plot_settings)
    return fig


def run_tauc_fit_workflow(
    series_name: str,
    series_key_prefix: str,
    energy_ev: np.ndarray,
    tauc_y: np.ndarray,
    exponent: float,
    min_r2_default: float,
    y_axis_symbol: str,
    plot_title_prefix: str,
    estimate_note: str,
    plot_settings: Optional[PlotCustomizationSettings],
    export_settings: Optional[ExportImageSettings],
) -> Dict[str, object]:
    series_key = f"{series_key_prefix}_{key_token(series_name)}"
    result: Dict[str, object] = {
        "Fit range min (eV)": np.nan,
        "Fit range max (eV)": np.nan,
        "Fit equation": "",
        "Tauc R2": np.nan,
        "Band gap Eg (eV)": "Not reported",
        "Band gap note": "Tauc fit not available",
    }

    energy_min = float(np.min(energy_ev))
    energy_max = float(np.max(energy_ev))
    default_min = float(np.percentile(energy_ev, 25))
    default_max = float(np.percentile(energy_ev, 60))

    suggestion = None
    with st.expander(f"Auto-suggest linear region ({series_name})", expanded=False):
        st.caption(
            "Auto-suggested region uses the same sliding-window linearity heuristic (R2-based) used in absorbance mode."
        )
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
    tauc_data_color = resolve_series_line_color(series_name, None, plot_settings)
    default_tauc_width = resolve_template_line_width(plot_settings, 1.2)
    tauc_data_width = resolve_series_line_width(series_name, default_tauc_width, plot_settings)
    ax_tauc.plot(
        energy_ev,
        tauc_y,
        color=tauc_data_color,
        linewidth=float(tauc_data_width),
        label="Tauc data",
    )
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
        fit_line_color, fit_line_width = resolve_optional_line_style(
            fallback_color="#e63946",
            fallback_width=resolve_template_line_width(plot_settings, 2.0),
            custom_color=(plot_settings.tauc_fit_line_color if plot_settings is not None else None),
            custom_width=(plot_settings.tauc_fit_line_width if plot_settings is not None else None),
        )
        ax_tauc.plot(
            fit_x,
            fit_y,
            color=fit_line_color,
            linewidth=float(fit_line_width),
            label="Linear fit",
        )

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
            result["Band gap note"] = estimate_note
            eg_line_color, eg_line_width = resolve_optional_line_style(
                fallback_color="#2a9d8f",
                fallback_width=resolve_template_line_width(plot_settings, 1.4),
                custom_color=(plot_settings.tauc_eg_line_color if plot_settings is not None else None),
                custom_width=(plot_settings.tauc_eg_line_width if plot_settings is not None else None),
            )
            ax_tauc.axvline(
                eg,
                color=eg_line_color,
                linestyle="--",
                linewidth=float(eg_line_width),
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
    ax_tauc.set_ylabel(f"({y_axis_symbol}*hnu)^{exponent:.4g} (a.u.)")
    apply_plot_customization(ax_tauc, f"{plot_title_prefix} - {series_name}", plot_settings)
    ax_tauc.legend(loc="best", fontsize=8)
    apply_legend_settings(ax_tauc, plot_settings)
    render_plot_with_download(
        ax_tauc.figure,
        download_name=f"{series_key}_tauc_plot.png",
        key=f"download_tauc_{series_key}",
        export_settings=export_settings,
        plot_settings=plot_settings,
    )

    return result


def run_tauc_for_series(
    series_name: str,
    x_nm: np.ndarray,
    y_absorbance: np.ndarray,
    min_r2_default: float,
    plot_settings: Optional[PlotCustomizationSettings],
    export_settings: Optional[ExportImageSettings],
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

    st.markdown(f"#### Absorbance Tauc Analysis - {series_name}")
    st.caption(
        "Absorbance-based estimate: results depend on transition assumption and manually selected fit range."
    )

    transition = st.selectbox(
        "Transition type",
        options=list(TAUC_TRANSITIONS.keys()),
        key=f"transition_abs_{series_key}",
    )
    exponent = float(TAUC_TRANSITIONS[transition]["exponent"])
    transition_n = TAUC_TRANSITIONS[transition]["n"]

    result["Transition"] = transition
    result["Transition n"] = transition_n
    result["Tauc exponent"] = exponent
    result["Tauc input quantity"] = "Absorbance (A)"

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
    result["Excluded Tauc points"] = excluded_count

    fit_result = run_tauc_fit_workflow(
        series_name=series_name,
        series_key_prefix="abs_tauc",
        energy_ev=energy_ev,
        tauc_y=tauc_y,
        exponent=exponent,
        min_r2_default=min_r2_default,
        y_axis_symbol="A",
        plot_title_prefix="Absorbance Tauc Plot",
        estimate_note="Absorbance-based estimated band gap.",
        plot_settings=plot_settings,
        export_settings=export_settings,
    )
    result.update(fit_result)
    return result


def run_reflectance_tauc_for_series(
    series_name: str,
    x_nm: np.ndarray,
    reflectance_signal: np.ndarray,
    reflectance_scale_mode: str,
    min_r2_default: float,
    plot_settings: Optional[PlotCustomizationSettings],
    export_settings: Optional[ExportImageSettings],
) -> Tuple[Dict[str, object], Optional[pd.DataFrame]]:
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
        "Reflectance scale mode": reflectance_scale_mode,
        "Reflectance scale used": "",
    }

    st.markdown(f"#### Reflectance Tauc Analysis - {series_name}")
    st.caption(
        "Reflectance-based Tauc analysis in this app uses the Kubelka-Munk transformed reflectance F(R) as an absorbance-like quantity, which is commonly used for diffuse reflectance data. This is an approximation/model-based approach, not direct absorbance."
    )

    transition = st.selectbox(
        "Transition type",
        options=list(TAUC_TRANSITIONS.keys()),
        key=f"transition_reflectance_{series_key}",
    )
    exponent = float(TAUC_TRANSITIONS[transition]["exponent"])
    transition_n = TAUC_TRANSITIONS[transition]["n"]

    result["Transition"] = transition
    result["Transition n"] = transition_n
    result["Tauc exponent"] = exponent
    result["Tauc input quantity"] = "Kubelka-Munk F(R)"

    st.caption(
        f"Transform: (F(R)*hnu)^m with m = {exponent:.6g}. Reference n = {transition_n} in (alpha*hnu)^(1/n)."
    )

    try:
        prep = prepare_reflectance_tauc_data(
            wavelength_nm=x_nm,
            reflectance_values=reflectance_signal,
            exponent=exponent,
            scale_mode=reflectance_scale_mode,
        )
    except ValueError as exc:
        st.warning(f"{series_name}: {exc}")
        result["Band gap note"] = f"Reflectance Tauc error: {exc}"
        return result, None

    for note in prep.notes:
        st.info(f"{series_name}: {note}")

    result["Reflectance scale used"] = prep.scale_used
    result["Excluded Tauc points"] = prep.excluded_count

    fit_result = run_tauc_fit_workflow(
        series_name=series_name,
        series_key_prefix="reflectance_tauc",
        energy_ev=prep.energy_ev,
        tauc_y=prep.tauc_y,
        exponent=exponent,
        min_r2_default=min_r2_default,
        y_axis_symbol="F(R)",
        plot_title_prefix="Reflectance Tauc Plot (Kubelka-Munk)",
        estimate_note="Kubelka-Munk reflectance-based estimated band gap.",
        plot_settings=plot_settings,
        export_settings=export_settings,
    )
    result.update(fit_result)

    export_df = pd.DataFrame(
        {
            "Record type": "reflectance_tauc",
            "Series": series_name,
            "Wavelength (nm)": prep.wavelength_nm,
            "Reflectance normalized (0-1)": prep.reflectance_fraction,
            "Kubelka-Munk F(R)": prep.kubelka_munk,
            "Photon Energy (eV)": prep.energy_ev,
            "Tauc transformed y": prep.tauc_y,
            "Reflectance scale used": prep.scale_used,
            "Transition": transition,
        }
    )

    return result, export_df

def analyze_single_series(
    series: SeriesBundle,
    mode: str,
    data_source_label: str,
    peak_settings: Optional[PeakSettings],
    pl_enable_fwhm: bool,
    absorbance_enable_tauc: bool,
    absorbance_min_r2_default: float,
    reflectance_enable_tauc: bool,
    reflectance_scale_mode: str,
    reflectance_min_r2_default: float,
    plot_settings: Optional[PlotCustomizationSettings],
    export_settings: Optional[ExportImageSettings],
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
        color=None,
        series_name=series.name,
        plot_settings=plot_settings,
    )
    render_plot_with_download(
        raw_fig,
        download_name=f"{key_token(series.name)}_raw_plot.png",
        key=f"download_raw_{key_token(series.name)}",
        export_settings=export_settings,
        plot_settings=plot_settings,
    )

    if data_source_label == "Processed data":
        processed_fig, _ = make_line_figure(
            series.x,
            series.y_analysis,
            x_label="Wavelength (nm)",
            y_label=Y_LABELS[mode],
            title=f"Processed Signal Used for Analysis - {series.name}",
            color=None,
            series_name=series.name,
            plot_settings=plot_settings,
        )
        render_plot_with_download(
            processed_fig,
            download_name=f"{key_token(series.name)}_processed_signal_plot.png",
            key=f"download_processed_{key_token(series.name)}",
            export_settings=export_settings,
            plot_settings=plot_settings,
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
        "Reflectance scale mode": "",
        "Reflectance scale used": "",
    }

    export_frames: List[pd.DataFrame] = [
        pd.DataFrame(
            {
                "Record type": "signal",
                "Series": series.name,
                "Wavelength (nm)": series.x,
                "Raw signal": series.y_raw,
                "Processed signal": series.y_processed,
                "Analysis signal": series.y_analysis,
                "Signal source": data_source_label,
            }
        )
    ]

    if mode == "Absorbance":
        if absorbance_enable_tauc:
            tauc_result = run_tauc_for_series(
                series.name,
                series.x,
                series.y_analysis,
                min_r2_default=absorbance_min_r2_default,
                plot_settings=plot_settings,
                export_settings=export_settings,
            )
            summary.update(tauc_result)
        else:
            summary["Band gap note"] = "Tauc disabled by user"

    if mode == "Reflectance":
        summary["Reflectance scale mode"] = reflectance_scale_mode
        if reflectance_enable_tauc:
            reflectance_result, reflectance_export_df = run_reflectance_tauc_for_series(
                series_name=series.name,
                x_nm=series.x,
                reflectance_signal=series.y_analysis,
                reflectance_scale_mode=reflectance_scale_mode,
                min_r2_default=reflectance_min_r2_default,
                plot_settings=plot_settings,
                export_settings=export_settings,
            )
            summary.update(reflectance_result)
            if reflectance_export_df is not None and not reflectance_export_df.empty:
                export_frames.append(reflectance_export_df)
        else:
            summary["Band gap note"] = "Reflectance Tauc disabled by user"

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
                signal_color = resolve_series_line_color(series.name, None, plot_settings)
                default_signal_width = resolve_template_line_width(plot_settings, 1.2)
                signal_width = resolve_series_line_width(series.name, default_signal_width, plot_settings)
                ax_peak.plot(
                    series.x,
                    series.y_analysis,
                    color=signal_color,
                    linewidth=float(signal_width),
                    label="Signal",
                )
                if len(peaks.indices) > 0:
                    ax_peak.scatter(
                        series.x[peaks.indices],
                        series.y_analysis[peaks.indices],
                        color="#e63946",
                        s=resolve_template_marker_size(plot_settings, 6.0) ** 2,
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
                apply_plot_customization(ax_peak, f"{mode} Peaks - {series.name}", plot_settings)
                ax_peak.legend(loc="best", fontsize=8)
                apply_legend_settings(ax_peak, plot_settings)
                render_plot_with_download(
                    peak_fig,
                    download_name=f"{key_token(series.name)}_peaks_plot.png",
                    key=f"download_peaks_{key_token(series.name)}",
                    export_settings=export_settings,
                    plot_settings=plot_settings,
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

    export_df = pd.concat(export_frames, ignore_index=True, sort=False)
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
        "Tauc input quantity",
        "Tauc exponent",
        "Reflectance scale mode",
        "Reflectance scale used",
        "Excluded Tauc points",
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

    x_all = np.concatenate([series.x for series in series_list])
    y_pool = [series.y_analysis for series in series_list] if data_source_label == "Processed data" else [series.y_raw for series in series_list]
    y_all = np.concatenate(y_pool)

    x_auto_min = float(np.nanmin(x_all))
    x_auto_max = float(np.nanmax(x_all))
    y_auto_min = float(np.nanmin(y_all))
    y_auto_max = float(np.nanmax(y_all))

    x_span = x_auto_max - x_auto_min
    y_span = y_auto_max - y_auto_min
    default_x_tick_interval = x_span / 10.0 if x_span > 0 else 1.0
    default_y_tick_interval = y_span / 10.0 if y_span > 0 else 1.0
    with st.sidebar:
        st.subheader("Publication Templates")
        template_name = st.selectbox(
            "Template",
            options=PUBLICATION_TEMPLATE_NAMES,
            index=PUBLICATION_TEMPLATE_NAMES.index("Default"),
            help="Publication-inspired presets (not official journal templates).",
        )
        publication_template = get_plot_template(template_name)

        if template_name == "Custom":
            with st.expander("Custom Template Parameters", expanded=False):
                publication_template.width_mm = float(st.number_input("Template width (mm)", min_value=1.0, value=float(publication_template.width_mm), step=1.0))
                publication_template.height_mm = float(st.number_input("Template height (mm)", min_value=1.0, value=float(publication_template.height_mm), step=1.0))
                publication_template.dpi = int(st.number_input("Template DPI", min_value=72, value=int(publication_template.dpi), step=10))
                publication_template.font_family = st.text_input("Template font family", value=publication_template.font_family)
                publication_template.title_font_size = float(st.number_input("Title font size", min_value=1.0, value=float(publication_template.title_font_size), step=0.5))
                publication_template.axis_label_font_size = float(st.number_input("Axis label font size", min_value=1.0, value=float(publication_template.axis_label_font_size), step=0.5))
                publication_template.tick_label_font_size = float(st.number_input("Tick label font size", min_value=1.0, value=float(publication_template.tick_label_font_size), step=0.5))
                publication_template.legend_font_size = float(st.number_input("Legend font size", min_value=1.0, value=float(publication_template.legend_font_size), step=0.5))
                publication_template.line_width = float(st.number_input("Line width", min_value=0.1, value=float(publication_template.line_width), step=0.1))
                publication_template.marker_size = float(st.number_input("Marker size", min_value=0.1, value=float(publication_template.marker_size), step=0.1))
                publication_template.axis_spine_width = float(st.number_input("Axis spine width", min_value=0.1, value=float(publication_template.axis_spine_width), step=0.1))
                publication_template.tick_width = float(st.number_input("Tick width", min_value=0.1, value=float(publication_template.tick_width), step=0.1))
                publication_template.tick_length = float(st.number_input("Tick length", min_value=0.1, value=float(publication_template.tick_length), step=0.1))
                publication_template.show_top_right_spines = st.checkbox("Show top/right spines", value=publication_template.show_top_right_spines)
                publication_template.grid_on = st.checkbox("Template grid enabled", value=publication_template.grid_on)
                publication_template.background_color = st.color_picker("Template background color", value=publication_template.background_color)
                publication_template.transparent_background = st.checkbox("Template default transparent background", value=publication_template.transparent_background)
                custom_palette_text = st.text_area(
                    "Template color palette (comma-separated matplotlib colors)",
                    value=", ".join(publication_template.color_palette),
                )
                custom_palette_values = [token.strip() for token in custom_palette_text.split(",") if token.strip()]
                if custom_palette_values:
                    invalid_palette = [value for value in custom_palette_values if not is_valid_matplotlib_color(value)]
                    if invalid_palette:
                        st.warning("Invalid custom template palette entries: " + ", ".join(invalid_palette))
                    else:
                        publication_template.color_palette = custom_palette_values

                publication_template.margins["left"] = float(st.number_input("Template margin left", min_value=0.0, max_value=1.0, value=float(publication_template.margins.get("left", 0.16)), step=0.01))
                publication_template.margins["right"] = float(st.number_input("Template margin right", min_value=0.0, max_value=1.0, value=float(publication_template.margins.get("right", 0.98)), step=0.01))
                publication_template.margins["top"] = float(st.number_input("Template margin top", min_value=0.0, max_value=1.0, value=float(publication_template.margins.get("top", 0.96)), step=0.01))
                publication_template.margins["bottom"] = float(st.number_input("Template margin bottom", min_value=0.0, max_value=1.0, value=float(publication_template.margins.get("bottom", 0.16)), step=0.01))

        output_size_mode = st.radio("Output size mode", ["Template default", "Manual override"], index=0, horizontal=True)
        export_width_mm = float(publication_template.width_mm)
        export_height_mm = float(publication_template.height_mm)
        if output_size_mode == "Manual override":
            export_width_mm = float(st.number_input("Export width (mm)", min_value=1.0, value=float(publication_template.width_mm), step=1.0))
            export_height_mm = float(st.number_input("Export height (mm)", min_value=1.0, value=float(publication_template.height_mm), step=1.0))

        export_format = st.selectbox("Export format", options=EXPORT_FORMAT_OPTIONS, index=EXPORT_FORMAT_OPTIONS.index("png"))
        raster_dpi = int(publication_template.dpi)
        if export_format in RASTER_EXPORT_FORMATS:
            raster_dpi = int(st.selectbox("Raster DPI", options=[300, 600, 1200], index=0))
        else:
            st.caption("Vector format selected; raster DPI is ignored.")

        transparent_background = st.checkbox("Transparent background", value=bool(publication_template.transparent_background))
        template_font_override_enabled = st.checkbox("Override template font family", value=False)
        template_font_override_value = publication_template.font_family
        if template_font_override_enabled:
            template_font_override_value = st.text_input("Template font override", value=publication_template.font_family)

        st.caption(
            f"Final export size: {export_width_mm:.2f} mm x {export_height_mm:.2f} mm "
            f"({mm_to_inches(export_width_mm):.2f} in x {mm_to_inches(export_height_mm):.2f} in)."
        )



    st.subheader("Advanced Plot Settings")
    with st.expander("Plot Customization", expanded=False):
        st.markdown("**Typography**")
        font_family_input = st.text_input(
            "Font family",
            value=publication_template.font_family,
            disabled=template_font_override_enabled,
            key=f"font_family_input_{template_name}",
            help="Times New Roman is requested by default; matplotlib may fall back if unavailable.",
        )
        if template_font_override_enabled:
            st.caption("Template font override is active in the sidebar and takes precedence for this run.")
        font_size_mode = st.selectbox(
            "Font size control",
            options=["Automatic (current style)", "Global font size", "Detailed font sizes"],
            index=0,
        )

        font_size_global_input: Optional[float] = None
        title_font_size_input: Optional[float] = None
        axis_label_font_size_input: Optional[float] = None
        tick_label_font_size_input: Optional[float] = None
        legend_font_size_input: Optional[float] = None

        if font_size_mode == "Global font size":
            font_size_global_input = float(st.number_input("Global font size", min_value=1.0, value=12.0, step=0.5))
        elif font_size_mode == "Detailed font sizes":
            size_col1, size_col2 = st.columns(2)
            with size_col1:
                title_font_size_input = float(st.number_input("Title font size", min_value=1.0, value=14.0, step=0.5))
                axis_label_font_size_input = float(
                    st.number_input("Axis label font size", min_value=1.0, value=12.0, step=0.5)
                )
            with size_col2:
                tick_label_font_size_input = float(
                    st.number_input("Tick label font size", min_value=1.0, value=10.0, step=0.5)
                )
                legend_font_size_input = float(
                    st.number_input("Legend font size", min_value=1.0, value=9.0, step=0.5)
                )

        text_color_input = st.color_picker("Text/font color", value="#000000")

        st.markdown("**Axis Title Overrides**")
        x_axis_title_override_input = st.text_input(
            "Custom x-axis title (optional)",
            value="",
            help="If blank, each plot keeps its scientific x-axis label.",
        )
        y_axis_title_override_input = st.text_input(
            "Custom y-axis title (optional)",
            value="",
            help="If blank, each plot keeps its scientific y-axis label.",
        )

        user_title_input = st.text_input("Plot title / heading (optional)", value="", help="If blank, no title is rendered. If provided, only this exact text is used.")

        use_manual_x_range = st.checkbox(
            "Set x-axis range manually",
            value=False,
            help="Applies to all plots. Tauc plots use photon energy (eV) on x-axis.",
        )
        x_min_input = x_auto_min
        x_max_input = x_auto_max
        if use_manual_x_range:
            x_col1, x_col2 = st.columns(2)
            with x_col1:
                x_min_input = st.number_input("x-axis min", value=float(x_auto_min), format="%.6g")
            with x_col2:
                x_max_input = st.number_input("x-axis max", value=float(x_auto_max), format="%.6g")

        use_manual_y_range = st.checkbox(
            "Set y-axis range manually",
            value=False,
            help="Applies to all plots in each plot's native y-units.",
        )
        y_min_input = y_auto_min
        y_max_input = y_auto_max
        if use_manual_y_range:
            y_col1, y_col2 = st.columns(2)
            with y_col1:
                y_min_input = st.number_input("y-axis min", value=float(y_auto_min), format="%.6g")
            with y_col2:
                y_max_input = st.number_input("y-axis max", value=float(y_auto_max), format="%.6g")

        st.markdown("**Tick Controls**")
        show_major_ticks = st.checkbox("Show major ticks", value=True)
        show_minor_ticks = st.checkbox("Show minor ticks", value=False)

        tick_direction_label = st.selectbox(
            "Tick direction",
            options=["Outside", "Inside", "Both"],
            index=0,
            help="Applies to both x and y axes.",
        )

        use_custom_x_tick_interval = st.checkbox("Set x-axis major tick increment", value=False)
        x_major_tick_input = default_x_tick_interval
        if use_custom_x_tick_interval:
            x_major_tick_input = st.number_input(
                "x-axis major tick increment",
                value=float(default_x_tick_interval),
                format="%.6g",
            )

        use_custom_y_tick_interval = st.checkbox("Set y-axis major tick increment", value=False)
        y_major_tick_input = default_y_tick_interval
        if use_custom_y_tick_interval:
            y_major_tick_input = st.number_input(
                "y-axis major tick increment",
                value=float(default_y_tick_interval),
                format="%.6g",
            )

        grid_mode_label = st.selectbox(
            "Grid mode",
            options=[
                "Automatic (current style)",
                "No grid",
                "Major grid only",
                "Major + minor grid",
            ],
            index=0,
        )

        st.markdown("**Series Line Styles**")
        st.caption("Per-series styles are keyed by selected column name and remain stable across preview/export.")

        series_line_color_overrides: Dict[str, str] = {}
        series_line_width_overrides: Dict[str, float] = {}
        for series_name in selected_y_columns:
            series_token = key_token(series_name)
            with st.expander(f"Series: {series_name}", expanded=False):
                color_col, width_col = st.columns(2)
                with color_col:
                    use_custom_series_color = st.checkbox(
                        "Custom line color",
                        value=False,
                        key=f"series_color_enable_{series_token}",
                    )
                    selected_series_color = st.color_picker(
                        "Line color",
                        value=default_series_color_for_name(series_name, publication_template.color_palette),
                        key=f"series_color_value_{series_token}",
                        disabled=not use_custom_series_color,
                    )
                with width_col:
                    use_custom_series_width = st.checkbox(
                        "Custom line width",
                        value=False,
                        key=f"series_width_enable_{series_token}",
                    )
                    selected_series_width = st.number_input(
                        "Line width",
                        min_value=0.1,
                        value=float(publication_template.line_width),
                        step=0.1,
                        key=f"series_width_value_{series_token}",
                        disabled=not use_custom_series_width,
                    )

                if use_custom_series_color:
                    series_line_color_overrides[series_name] = selected_series_color
                if use_custom_series_width:
                    series_line_width_overrides[series_name] = float(selected_series_width)

        st.markdown("**Tauc Line Styles (Optional)**")
        tauc_fit_color_enable = st.checkbox("Custom Tauc fit line color", value=False)
        tauc_fit_line_color_input = st.color_picker(
            "Tauc fit line color",
            value="#e63946",
            disabled=not tauc_fit_color_enable,
        )
        tauc_fit_width_enable = st.checkbox("Custom Tauc fit line width", value=False)
        tauc_fit_line_width_input = st.number_input(
            "Tauc fit line width",
            min_value=0.1,
            value=max(1.0, float(publication_template.line_width) * 1.5),
            step=0.1,
            disabled=not tauc_fit_width_enable,
        )

        tauc_eg_color_enable = st.checkbox("Custom Eg marker line color", value=False)
        tauc_eg_line_color_input = st.color_picker(
            "Eg marker line color",
            value="#2a9d8f",
            disabled=not tauc_eg_color_enable,
        )
        tauc_eg_width_enable = st.checkbox("Custom Eg marker line width", value=False)
        tauc_eg_line_width_input = st.number_input(
            "Eg marker line width",
            min_value=0.1,
            value=max(0.8, float(publication_template.line_width)),
            step=0.1,
            disabled=not tauc_eg_width_enable,
        )

        requested_preview_font = normalize_font_family(template_font_override_value if template_font_override_enabled else font_family_input)
        resolved_preview_font = resolve_font_preview_name(requested_preview_font)
        st.caption(
            f"Font preview (matplotlib resolution): requested '{requested_preview_font}', resolved '{resolved_preview_font}'."
        )

    plot_validation_messages: List[str] = []
    plot_info_messages: List[str] = []

    style_helper_issues = run_style_helper_self_checks()
    for issue in style_helper_issues:
        plot_validation_messages.append(f"Internal style helper check: {issue}")

    requested_font_family = normalize_font_family(template_font_override_value if template_font_override_enabled else font_family_input)
    resolved_font_name = resolve_font_preview_name(requested_font_family)
    font_notice_key = "_font_fallback_notice"
    font_notice_value = f"{requested_font_family.strip().lower()}::{resolved_font_name.strip().lower()}"
    if should_warn_on_font_fallback(requested_font_family, resolved_font_name):
        fallback_message = (
            f"Requested font '{requested_font_family}' was not found in matplotlib's font list. "
            f"Matplotlib will fall back (resolved preview: '{resolved_font_name}')."
        )
        if st.session_state.get(font_notice_key) != font_notice_value:
            if requested_font_family.strip().lower() == DEFAULT_REQUESTED_FONT_FAMILY.lower():
                plot_info_messages.append(fallback_message)
            else:
                plot_validation_messages.append(fallback_message)
        st.session_state[font_notice_key] = font_notice_value
    else:
        st.session_state.pop(font_notice_key, None)

    if not is_valid_matplotlib_color(text_color_input):
        plot_validation_messages.append(
            f"Invalid text color '{text_color_input}'. Default text color '#000000' was used."
        )
        text_color_input = "#000000"

    series_line_color_overrides, series_line_width_overrides, series_style_messages = sanitize_series_style_overrides(
        series_line_color_overrides,
        series_line_width_overrides,
    )
    plot_validation_messages.extend(series_style_messages)

    if tauc_fit_color_enable and not is_valid_matplotlib_color(tauc_fit_line_color_input):
        plot_validation_messages.append(
            f"Invalid Tauc fit line color '{tauc_fit_line_color_input}'. Custom Tauc fit color was not applied."
        )
        tauc_fit_color_enable = False

    if tauc_eg_color_enable and not is_valid_matplotlib_color(tauc_eg_line_color_input):
        plot_validation_messages.append(
            f"Invalid Eg marker line color '{tauc_eg_line_color_input}'. Custom Eg line color was not applied."
        )
        tauc_eg_color_enable = False

    if tauc_fit_width_enable and tauc_fit_line_width_input <= 0:
        plot_validation_messages.append("Invalid Tauc fit line width: value must be positive. Custom fit width was not applied.")
        tauc_fit_width_enable = False

    if tauc_eg_width_enable and tauc_eg_line_width_input <= 0:
        plot_validation_messages.append("Invalid Eg marker line width: value must be positive. Custom Eg width was not applied.")
        tauc_eg_width_enable = False

    if use_manual_x_range and x_min_input >= x_max_input:
        plot_validation_messages.append(
            "Invalid x-axis range: x-axis min must be smaller than x-axis max. Manual x-range was not applied."
        )
        use_manual_x_range = False

    if use_manual_y_range and y_min_input >= y_max_input:
        plot_validation_messages.append(
            "Invalid y-axis range: y-axis min must be smaller than y-axis max. Manual y-range was not applied."
        )
        use_manual_y_range = False

    if use_custom_x_tick_interval and x_major_tick_input <= 0:
        plot_validation_messages.append(
            "Invalid x-axis major tick increment: value must be positive. Custom x tick spacing was not applied."
        )
        use_custom_x_tick_interval = False

    if use_custom_y_tick_interval and y_major_tick_input <= 0:
        plot_validation_messages.append(
            "Invalid y-axis major tick increment: value must be positive. Custom y tick spacing was not applied."
        )
        use_custom_y_tick_interval = False

    if not show_major_ticks and (use_custom_x_tick_interval or use_custom_y_tick_interval):
        plot_validation_messages.append(
            "Major tick increments are ignored while major ticks are disabled."
        )

    if grid_mode_label == "Major + minor grid" and not show_minor_ticks:
        plot_validation_messages.append(
            "Major + minor grid selected, but minor ticks are disabled. Minor grid lines will not be shown."
        )

    if export_width_mm <= 0 or export_height_mm <= 0:
        plot_validation_messages.append(
            "Invalid export size settings: width and height in mm must be positive."
        )

    if export_format in RASTER_EXPORT_FORMATS and raster_dpi <= 0:
        plot_validation_messages.append(
            "Invalid export DPI settings: raster DPI must be positive."
        )

    if publication_template.margins.get("left", 0.0) >= publication_template.margins.get("right", 1.0):
        plot_validation_messages.append("Invalid template margins: left must be smaller than right.")
    if publication_template.margins.get("bottom", 0.0) >= publication_template.margins.get("top", 1.0):
        plot_validation_messages.append("Invalid template margins: bottom must be smaller than top.")

    if export_format in {"jpg", "jpeg"} and transparent_background:
        plot_info_messages.append("JPEG does not support transparency. Export will use an opaque background.")
    for validation_message in plot_validation_messages:
        st.warning(validation_message)

    for info_message in plot_info_messages:
        st.info(info_message)

    tick_direction = {
        "Outside": "out",
        "Inside": "in",
        "Both": "inout",
    }[tick_direction_label]
    grid_mode = {
        "Automatic (current style)": "default",
        "No grid": "off",
        "Major grid only": "major",
        "Major + minor grid": "major_minor",
    }[grid_mode_label]

    plot_settings = PlotCustomizationSettings(
        use_manual_x_range=use_manual_x_range,
        x_min=float(x_min_input) if use_manual_x_range else None,
        x_max=float(x_max_input) if use_manual_x_range else None,
        use_manual_y_range=use_manual_y_range,
        y_min=float(y_min_input) if use_manual_y_range else None,
        y_max=float(y_max_input) if use_manual_y_range else None,
        use_custom_x_tick_interval=use_custom_x_tick_interval,
        x_major_tick_interval=float(x_major_tick_input) if use_custom_x_tick_interval else None,
        use_custom_y_tick_interval=use_custom_y_tick_interval,
        y_major_tick_interval=float(y_major_tick_input) if use_custom_y_tick_interval else None,
        show_major_ticks=show_major_ticks,
        show_minor_ticks=show_minor_ticks,
        grid_mode=grid_mode,
        tick_direction=tick_direction,
        user_title=user_title_input,
        font_family=requested_font_family,
        font_size_global=font_size_global_input,
        title_font_size=title_font_size_input,
        axis_label_font_size=axis_label_font_size_input,
        tick_label_font_size=tick_label_font_size_input,
        legend_font_size=legend_font_size_input,
        text_color=text_color_input,
        x_axis_title_override=x_axis_title_override_input,
        y_axis_title_override=y_axis_title_override_input,
        series_line_colors=series_line_color_overrides,
        series_line_widths=series_line_width_overrides,
        tauc_fit_line_color=tauc_fit_line_color_input if tauc_fit_color_enable else None,
        tauc_fit_line_width=float(tauc_fit_line_width_input) if tauc_fit_width_enable else None,
        tauc_eg_line_color=tauc_eg_line_color_input if tauc_eg_color_enable else None,
        tauc_eg_line_width=float(tauc_eg_line_width_input) if tauc_eg_width_enable else None,
        publication_template=publication_template,
    )

    active_font_resolution = resolve_font_preview_name(requested_font_family)
    preview_title_size, preview_axis_size, preview_tick_size, preview_legend_size = _resolve_font_sizes(plot_settings)

    effective_font_sizes = [value for value in [preview_title_size, preview_axis_size, preview_tick_size, preview_legend_size] if value is not None]
    if effective_font_sizes and min(effective_font_sizes) < 6.0:
        st.warning("Some selected font sizes are below 6 pt, which may be hard to read in publication figures.")

    if len(series_list) > 8 and export_width_mm < 120.0:
        st.warning(f"{len(series_list)} traces selected; legend readability may be limited at {export_width_mm:.1f} mm export width.")
    st.caption(
        "Active style preview: "
        + f"template='{template_name}', "
        + f"export={export_width_mm:.1f}x{export_height_mm:.1f} mm, "
        + f"format={export_format.upper()}, "
        + f"font='{requested_font_family}' (resolved '{active_font_resolution}'), "
        + f"text color={text_color_input}, "
        + f"title={user_title_input.strip() if user_title_input.strip() else '[none]'}, "
        + f"x-title={'[default]' if not x_axis_title_override_input.strip() else x_axis_title_override_input}, "
        + f"y-title={'[default]' if not y_axis_title_override_input.strip() else y_axis_title_override_input}, "
        + f"title size={format_font_size_for_preview(preview_title_size)}, "
        + f"axis-label size={format_font_size_for_preview(preview_axis_size)}, "
        + f"tick-label size={format_font_size_for_preview(preview_tick_size)}, "
        + f"legend size={format_font_size_for_preview(preview_legend_size)}."
    )

    if selected_y_columns:
        available_series_names = {series.name for series in series_list}
        style_preview_rows = []
        for series_name in selected_y_columns:
            resolved_color = resolve_series_line_color(series_name, None, plot_settings)
            resolved_width = resolve_series_line_width(series_name, resolve_template_line_width(plot_settings, 1.2), plot_settings)
            style_preview_rows.append(
                {
                    "Series": series_name,
                    "Active in plots": "Yes" if series_name in available_series_names else "No (skipped)",
                    "Line color": resolved_color if resolved_color is not None else "[matplotlib default cycle]",
                    "Line width": float(resolved_width),
                }
            )
        st.dataframe(pd.DataFrame(style_preview_rows), use_container_width=True, height=220)

    export_settings = ExportImageSettings(
        export_format=export_format,
        width_mm=float(export_width_mm),
        height_mm=float(export_height_mm),
        raster_dpi=int(raster_dpi),
        transparent_background=bool(transparent_background),
        template_name=template_name,
    )

    st.subheader("Plotting Behavior")
    show_combined_plot = st.checkbox("Show combined plot of selected y-series", value=True)
    show_separate_overview = st.checkbox(
        "Show separate overview plots for each selected y-series",
        value=False,
    )

    combined_source_key = "analysis" if data_source_label == "Processed data" else "raw"
    if show_combined_plot:
        combined_fig = plot_multi_series(
            series_list,
            mode=mode,
            source=combined_source_key,
            plot_settings=plot_settings,
        )
        render_plot_with_download(
            combined_fig,
            download_name=f"{mode.lower().replace(' ', '_')}_combined_plot.png",
            key="download_combined_plot",
            export_settings=export_settings,
            plot_settings=plot_settings,
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
                color=None,
                series_name=series.name,
                plot_settings=plot_settings,
            )
            render_plot_with_download(
                fig_series,
                download_name=f"{key_token(series.name)}_overview_plot.png",
                key=f"download_overview_{key_token(series.name)}",
                export_settings=export_settings,
                plot_settings=plot_settings,
            )

    peak_settings: Optional[PeakSettings] = None
    enable_fwhm_for_pl = False
    enable_tauc_for_absorbance = False
    absorbance_min_r2_default = 0.97
    enable_tauc_for_reflectance = False
    reflectance_scale_mode = "auto"
    reflectance_min_r2_default = 0.97

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

    if mode == "Reflectance":
        st.subheader("Reflectance / Tauc Settings")
        st.warning(
            "Reflectance-based Tauc analysis in this app uses the Kubelka-Munk transformed reflectance F(R) as an absorbance-like quantity, which is commonly used for diffuse reflectance data. This is an approximation/model-based approach, not direct absorbance."
        )
        scale_option = st.selectbox(
            "Reflectance scale",
            options=[
                "Auto-detect (only if unambiguous)",
                "Fraction (0 to 1)",
                "Percent (0 to 100)",
            ],
            index=0,
        )
        reflectance_scale_mode = {
            "Auto-detect (only if unambiguous)": "auto",
            "Fraction (0 to 1)": "fraction",
            "Percent (0 to 100)": "percent",
        }[scale_option]

        enable_tauc_for_reflectance = st.checkbox(
            "Run Kubelka-Munk Tauc analysis for each selected reflectance column",
            value=True,
        )
        reflectance_min_r2_default = st.slider(
            "Default minimum R2 threshold for reflectance band-gap reporting",
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
            reflectance_enable_tauc=enable_tauc_for_reflectance,
            reflectance_scale_mode=reflectance_scale_mode,
            reflectance_min_r2_default=reflectance_min_r2_default,
            plot_settings=plot_settings,
            export_settings=export_settings,
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









































































