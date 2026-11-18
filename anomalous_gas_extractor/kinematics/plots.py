"""Figures of the tilted-ring models.

:func:`plot_model_profiles` shows the radial profiles of a fitted disc in
stacked panels that share the radius axis: surface brightness, rotation
velocity, inclination, position angle and velocity dispersion. The approaching
and the receding side are drawn separately where they differ, and the radius of
the disc and r_max are marked by vertical lines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from matplotlib import font_manager, rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

from anomalous_gas_extractor.kinematics.fat import ARCSEC_TO_RADIAN, EMPTY_RING_BRIGHTNESS

# face-on column density (10^20 cm^-2) of a surface brightness of 1 mJy km/s arcsec^-2
COLUMN_DENSITY_PER_SBR = 12.509
# the surface brightness is drawn in mJy km/s arcsec^-2
SBR_SCALE = 1e3
LEGEND_ORDER = ("approaching side", "receding side", "both sides")
COLUMN_DENSITY_LABEL = r"$N_{HI}$ (10$^{20}$ cm$^{-2}$)"
APPROACHING = dict(color="black", marker="o", markersize=5, linestyle="-", linewidth=1.3,
                   capsize=2.5, elinewidth=0.9, zorder=3)
RECEDING = dict(color="0.55", marker="o", markersize=5, markerfacecolor="white", linestyle="--",
                linewidth=1.3, capsize=2.5, elinewidth=0.9, zorder=2)


@dataclass(frozen=True)
class Panel:
    """One radial profile: the column of the ring table and the labels of its axis."""

    column: str
    label: str
    unit: str


PANELS = (Panel("sbr", "SBR", r"mJy km s$^{-1}$ arcsec$^{-2}$"),
          Panel("vrot_kms", "VROT", r"km s$^{-1}$"),
          Panel("inclination_deg", "INCL", "degree"),
          Panel("position_angle_deg", "PA", "degree"),
          Panel("dispersion_kms", "SDIS", r"km s$^{-1}$"))


def figure_style(font_file: Optional[str] = None) -> Dict[str, object]:
    """Matplotlib settings of the figures; ``font_file`` selects the font."""
    style = {"xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
             "ytick.right": True, "xtick.major.size": 5.5, "ytick.major.size": 5.5,
             "xtick.minor.size": 3.0, "ytick.minor.size": 3.0, "xtick.labelsize": 10,
             "ytick.labelsize": 10, "axes.labelsize": 11, "legend.fontsize": 9.5,
             "mathtext.default": "regular"}
    if font_file:
        font_manager.fontManager.addfont(font_file)
        style["font.family"] = font_manager.FontProperties(fname=font_file).get_name()
    return style


def _column(rings: List[Dict[str, float]], name: str) -> np.ndarray:
    return np.array([ring[name] for ring in rings], dtype=float)


def _draw_side(axis, rings, panel: Panel, side: str, style: dict, label: Optional[str]) -> None:
    """Profile of one side of the disc; rings without emission show their brightness only."""
    radius = _column(rings, "radius_arcsec")
    values = _column(rings, f"{panel.column}_{side}")
    errors = _column(rings, f"{panel.column}_{side}_error")
    if panel.column == "sbr":
        values, errors = values * SBR_SCALE, errors * SBR_SCALE
    else:
        filled = _column(rings, f"sbr_{side}") > EMPTY_RING_BRIGHTNESS
        radius, values, errors = radius[filled], values[filled], errors[filled]
    axis.errorbar(radius, values, yerr=np.where(np.isfinite(errors), errors, 0.0), label=label,
                  **style)


@dataclass
class ModelProfiles:
    """A tilted-ring model to draw: its ring table and the radii to mark."""

    title: str
    rings: List[Dict[str, float]]
    disc_radius_arcsec: Optional[float] = None
    r_max_arcsec: Optional[float] = None


def _draw_model(axes, model: ModelProfiles, distance_mpc: Optional[float],
                label_axes: bool, quote_radii: bool) -> Dict[str, object]:
    """Draw the profiles of one model in a column of axes; returns its legend entries."""
    rings = model.rings
    radius = _column(rings, "radius_arcsec")
    limit = 1.08 * np.nanmax([radius.max(), model.disc_radius_arcsec or 0.0,
                              model.r_max_arcsec or 0.0])
    for axis, panel in zip(axes, PANELS):
        both = np.allclose(_column(rings, f"{panel.column}_approaching"),
                           _column(rings, f"{panel.column}_receding"), equal_nan=True)
        if both:
            _draw_side(axis, rings, panel, "approaching", APPROACHING, "both sides")
        else:
            _draw_side(axis, rings, panel, "receding", RECEDING, "receding side")
            _draw_side(axis, rings, panel, "approaching", APPROACHING, "approaching side")
        if model.disc_radius_arcsec is not None:
            label = "radius of the rotating disc"
            if quote_radii:
                label += f" = {model.disc_radius_arcsec:.0f} arcsec"
            axis.axvline(model.disc_radius_arcsec, color="black", linestyle="--", linewidth=1.2,
                         zorder=1, label=label)
        if model.r_max_arcsec is not None:
            label = r"$r_{max}$ (Kinemetry)"
            if quote_radii:
                label += f" = {model.r_max_arcsec:.0f} arcsec"
            axis.axvline(model.r_max_arcsec, color="0.4", linestyle=":", linewidth=1.4, zorder=1,
                         label=label)
        if label_axes:
            axis.set_ylabel(f"{panel.label}\n({panel.unit})")
        axis.set_xlim(-0.03 * limit, limit)
        axis.margins(y=0.18)
        # the panels touch: no tick label at the edges of the lower panels
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5, prune=None if axis is axes[0] else "both"))
        axis.xaxis.set_minor_locator(AutoMinorLocator(5))
        axis.yaxis.set_minor_locator(AutoMinorLocator(5))
        axis.tick_params(which="both", top=True, right=True)

    brightness = axes[0]
    # zero a little above the edge of the panel, clear of the tick labels of the next one
    brightness.set_ylim(bottom=-0.06 * brightness.get_ylim()[1])
    density = brightness.secondary_yaxis(
        "right", functions=(lambda sbr: sbr * COLUMN_DENSITY_PER_SBR,
                            lambda column: column / COLUMN_DENSITY_PER_SBR))
    density.yaxis.set_minor_locator(AutoMinorLocator(5))
    brightness.tick_params(which="both", right=False)
    axes[-1].set_xlabel("R (arcsec)")
    if distance_mpc is not None:
        scale = ARCSEC_TO_RADIAN * distance_mpc * 1e3
        physical = brightness.secondary_xaxis(
            "top", functions=(lambda arcsec: arcsec * scale, lambda kpc: kpc / scale))
        physical.set_xlabel("R (kpc)")
        physical.xaxis.set_minor_locator(AutoMinorLocator(5))
        brightness.tick_params(which="both", top=False)
    entries = {}
    for axis in axes:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            entries.setdefault(label, handle)
    return {"entries": entries, "density_axis": density}


def _legend(figure, entries: Dict[str, object], anchor_y: float, columns: int) -> None:
    """One legend for all panels: every line style that occurs, the two sides first."""
    labels = sorted(entries, key=lambda label: LEGEND_ORDER.index(label)
                    if label in LEGEND_ORDER else len(LEGEND_ORDER))
    figure.legend([entries[label] for label in labels], labels, loc="upper center",
                  bbox_to_anchor=(0.5, anchor_y), ncol=columns, frameon=True, edgecolor="0.6")


def plot_model_profiles(rings: List[Dict[str, float]], file_name: str, title: Optional[str] = None,
                        disc_radius_arcsec: Optional[float] = None,
                        r_max_arcsec: Optional[float] = None,
                        distance_mpc: Optional[float] = None,
                        font_file: Optional[str] = None) -> None:
    """Plot the radial profiles of a tilted-ring model.

    Parameters
    ----------
    rings : list of dict
        Ring table of the model (:func:`anomalous_gas_extractor.kinematics.fat.model_rings`).
    file_name : str
        Output file; its extension sets the format.
    title : str, optional
        Title of the figure.
    disc_radius_arcsec, r_max_arcsec : float, optional
        Radius of the model disc and r_max from Kinemetry, marked by a dashed and
        a dotted vertical line.
    distance_mpc : float, optional
        Adds a radius axis in kpc at the top.
    font_file : str, optional
        Font of the figure.
    """
    model = ModelProfiles(title or "", rings, disc_radius_arcsec, r_max_arcsec)
    with rc_context(figure_style(font_file)):
        figure = Figure(figsize=(5.6, 1.75 * len(PANELS) + 1.4))
        FigureCanvasAgg(figure)
        axes = figure.subplots(len(PANELS), 1, sharex=True, gridspec_kw=dict(hspace=0.0))
        drawn = _draw_model(axes, model, distance_mpc, label_axes=True, quote_radii=True)
        drawn["density_axis"].set_ylabel(COLUMN_DENSITY_LABEL)
        _legend(figure, drawn["entries"], anchor_y=0.065, columns=2)
        if title:
            figure.suptitle(title, fontsize=12, y=0.945)
        figure.align_ylabels(axes)
        figure.savefig(file_name, bbox_inches="tight", dpi=300)


def plot_model_grid(models: List[ModelProfiles], file_name: str,
                    distance_mpc: Optional[float] = None,
                    font_file: Optional[str] = None) -> None:
    """Plot the radial profiles of several models side by side, one column per model.

    The rows are the panels of :func:`plot_model_profiles`; every column has its
    own axes because the discs differ in size and rotation velocity. The name of
    each model is written above its column.
    """
    with rc_context(figure_style(font_file)):
        figure = Figure(figsize=(3.3 * len(models) + 1.0, 1.75 * len(PANELS) + 1.4))
        FigureCanvasAgg(figure)
        axes = figure.subplots(len(PANELS), len(models), sharex="col", squeeze=False,
                               gridspec_kw=dict(hspace=0.0, wspace=0.42))
        entries = {}
        for column, model in enumerate(models):
            drawn = _draw_model(axes[:, column], model, distance_mpc, label_axes=column == 0,
                                quote_radii=False)
            if column == len(models) - 1:
                drawn["density_axis"].set_ylabel(COLUMN_DENSITY_LABEL)
            for label, handle in drawn["entries"].items():
                entries.setdefault(label, handle)
            axes[0, column].set_title(model.title, fontsize=11, pad=34 if distance_mpc else 8)
        _legend(figure, entries, anchor_y=0.06, columns=len(entries))
        figure.align_ylabels(axes[:, 0])
        figure.savefig(file_name, bbox_inches="tight", dpi=300)
