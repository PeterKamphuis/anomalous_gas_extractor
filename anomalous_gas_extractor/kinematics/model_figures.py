"""Figures that compare the rotating discs with their tilted-ring models.

The model cube is placed on the grid of the data cube and the voxels of the
component select data and model alike
(:func:`anomalous_gas_extractor.kinematics.residuals.residual_moments`). The
maps are shown on the pixels where the integrated intensity of both exceeds the
signal-to-noise limit; gas of the component outside the model is not shown.

:func:`plot_model_on_data` draws the iso-velocity contours of the model in
yellow on the velocity field of the data with its own contours in black.

:func:`plot_residual_moments` shows data minus model for the surface density,
the velocity and the velocity dispersion, one row per disc, with the contours
of the surface density of the data and the median and rms of each residual.
The colour scales depend on the data only, so that the figures of two sets of
models can be compared.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from matplotlib import colormaps, patheffects
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

from anomalous_gas_extractor.kinematics import hi_mass, sky_axes
from anomalous_gas_extractor.kinematics.datacube import DataCube
from anomalous_gas_extractor.kinematics.figure_data import Component, inside_rmax
from anomalous_gas_extractor.kinematics.residuals import residual_moments
from anomalous_gas_extractor.kinematics.sky_axes import VELOCITY_CMAP, View

# empty margin around the compared pixels (beams)
FRAME_MARGIN_BEAMS = 1.5
# contours of the surface density of the data (solar masses per square parsec)
SURFACE_DENSITY_LEVELS = (1.0, 2.0, 4.0, 8.0, 16.0)
# colour ranges of the residuals: fraction of the brightest surface density, fraction of the
# velocity spread, and the dispersion (km/s)
SURFACE_DENSITY_RANGE = 0.6
VELOCITY_RANGE = 0.4
DISPERSION_RANGE_KMS = 40.0
# smallest half colour range of a velocity or a velocity residual (km/s)
MINIMUM_HALF_RANGE_KMS = 30.0
CONTOURS_ACROSS = 7
DATA_COLOUR, MODEL_COLOUR = "black", "#FFD92F"
DATA_CASING = [patheffects.Stroke(linewidth=2.6, foreground="white"), patheffects.Normal()]
MODEL_CASING = [patheffects.Stroke(linewidth=3.0, foreground="black"), patheffects.Normal()]


@dataclass(frozen=True)
class Quantity:
    """A residual map: its name in the table of statistics, colour map and label."""

    key: str
    cmap: str
    label: str


QUANTITIES = (
    Quantity("sigma", "BrBG", r"$\Sigma_{HI}$ data $-$ model (M$_{\odot}$ pc$^{-2}$)"),
    Quantity("velocity", "RdBu_r", r"velocity, data $-$ model (km s$^{-1}$)"),
    Quantity("dispersion", "PuOr_r", r"dispersion, data $-$ model (km s$^{-1}$)"))


@dataclass
class Comparison:
    """Maps of a rotating disc and of its model on the pixels where they are compared.

    Attributes
    ----------
    component : Component
        The component.
    residuals : dict
        Data minus model by quantity: sigma (solar masses per square parsec),
        velocity and dispersion (km/s); NaN outside ``compared``.
    limits : dict
        Half the colour range of every residual.
    statistics : dict
        Median and rms of every residual.
    compared : 2-D boolean array
        Pixels where data and model are compared.
    surface_density, data_velocity, model_velocity : 2-D arrays
        Surface density of the data and the velocity fields of data and model.
    """

    component: Component
    residuals: Dict[str, np.ndarray]
    limits: Dict[str, float]
    statistics: Dict[str, Tuple[float, float]]
    compared: np.ndarray
    surface_density: np.ndarray
    data_velocity: np.ndarray
    model_velocity: np.ndarray

    def view(self, cube: DataCube) -> View:
        """A square around the compared pixels."""
        margin = FRAME_MARGIN_BEAMS * cube.beam_fwhm_arcsec / cube.pixel_arcsec
        return sky_axes.footprint_view(self.compared, margin)


def compare_with_model(cube: DataCube, component: Component, snr: float = 3.0) -> Comparison:
    """Residual maps of a component with a model cube, their colour ranges and statistics.

    The colour ranges follow from the data inside r_max around the kinematic
    centre, so they do not depend on the model.
    """
    maps = residual_moments(cube, component.label, component.model_cube, snr)
    compared = maps.compared
    density = hi_mass.surface_density(1.0, cube.beam_major_arcsec, cube.beam_minor_arcsec)
    residuals = {"sigma": np.where(compared, density * maps.intensity, np.nan),
                 "velocity": maps.velocity, "dispersion": maps.dispersion}

    inside = inside_rmax(cube, component)
    offsets = np.abs(component.maps.velocity[inside] - component.kinematic_vsys_kms)
    spread = np.nanpercentile(offsets, 98.0)
    brightest = np.percentile(density * maps.data_intensity[inside & maps.footprint], 99.0)
    limits = {"sigma": SURFACE_DENSITY_RANGE * brightest,
              "velocity": max(VELOCITY_RANGE * spread, MINIMUM_HALF_RANGE_KMS),
              "dispersion": DISPERSION_RANGE_KMS}
    statistics = {key: (float(np.median(residual[compared])), float(np.std(residual[compared])))
                  for key, residual in residuals.items()}
    return Comparison(component, residuals, limits, statistics, compared,
                      np.where(compared, density * maps.data_intensity, np.nan),
                      maps.data_velocity, maps.model_velocity)


def _mark_centre(axis, cube: DataCube, component: Component) -> None:
    sky_axes.mark_position(axis, cube, component.geometry.ra_deg, component.geometry.dec_deg,
                           size=110)


def _model_on_data_panel(figure: Figure, rectangle: Sequence[float], cube: DataCube,
                         comparison: Comparison) -> None:
    """Velocity field of the data with the iso-velocity contours of data and model."""
    component = comparison.component
    data, model, vsys = comparison.data_velocity, comparison.model_velocity, component.vsys_kms
    half_range = max(float(np.nanpercentile(np.abs(data - vsys), 98.0)), MINIMUM_HALF_RANGE_KMS)
    norm = Normalize(vmin=vsys - half_range, vmax=vsys + half_range)
    axis = sky_axes.sky_axes(figure, rectangle, cube)
    axis.imshow(data, origin="lower", cmap=VELOCITY_CMAP, norm=norm, interpolation="nearest")

    step = sky_axes.contour_step(2.0 * half_range, CONTOURS_ACROSS)
    levels = sky_axes.velocity_levels(np.stack([data, model]), vsys, step)
    smooth = sky_axes.smooth_nan(data)
    # black on white for the data, yellow on black for the model: both stand out on blue and red
    for image, colour, casing, widths, zorder in (
            (smooth, DATA_COLOUR, "white", (1.1, 2.6, 2.2, 3.8), 4),
            (model, MODEL_COLOUR, "black", (1.5, 3.0, 2.6, 4.2), 6)):
        sky_axes.cased_contours(axis, image, levels, colour, widths[0], casing, widths[1], zorder)
        sky_axes.cased_contours(axis, image, [vsys], colour, widths[2], casing, widths[3], zorder)
    _mark_centre(axis, cube, component)

    sky_axes.velocity_bar(axis, cube, VELOCITY_CMAP, norm, vsys)
    handles = [Line2D([], [], color=DATA_COLOUR, linewidth=1.1, label="data",
                      path_effects=DATA_CASING),
               Line2D([], [], color=MODEL_COLOUR, linewidth=1.5, label="model",
                      path_effects=MODEL_CASING)]
    axis.legend(handles=handles, loc="upper left", fontsize=10, frameon=True, edgecolor="0.6",
                facecolor="0.82", framealpha=1.0, handlelength=2.0, borderpad=0.4).set_zorder(12)
    sky_axes.panel_note(axis, f"$v_{{sys}}$ = {vsys:.0f}\niso step {step:.0f}")
    sky_axes.panel_name(axis, component.title)
    sky_axes.frame_panel(axis, cube, comparison.view(cube), component.distance_mpc)


def plot_model_on_data(cube: DataCube, comparisons: List[Comparison], file_name: str) -> None:
    """Plot the velocity field of every model on that of the data, one panel per disc.

    Parameters
    ----------
    cube : DataCube
        Data cube with the mask of the components.
    comparisons : list of Comparison
        Maps of the discs and their models (:func:`compare_with_model`).
    file_name : str
        Output file; its extension sets the format.
    """
    figure, rectangles = sky_axes.panel_figure(len(comparisons), 2, panel=(3.8, 3.8),
                                               gap=(1.2, 1.5))
    for rectangle, comparison in zip(rectangles, comparisons):
        _model_on_data_panel(figure, rectangle, cube, comparison)
    sky_axes.save(figure, file_name)


def _residual_panel(figure: Figure, rectangle: Sequence[float], cube: DataCube,
                    comparison: Comparison, quantity: Quantity, show_name: bool) -> None:
    """One residual map of one disc."""
    limit = comparison.limits[quantity.key]
    norm = Normalize(vmin=-limit, vmax=limit)
    axis = sky_axes.sky_axes(figure, rectangle, cube)
    axis.imshow(comparison.residuals[quantity.key], origin="lower", cmap=colormaps[quantity.cmap],
                norm=norm, interpolation="nearest")
    # black on a white casing: readable on the light and on the dark ends of the colour maps
    sky_axes.cased_contours(axis, comparison.surface_density, SURFACE_DENSITY_LEVELS, "black", 0.9,
                            "white", 2.4, zorder=4)
    _mark_centre(axis, cube, comparison.component)
    bar_axis = sky_axes.colour_bar(axis, [0.0, 1.045, 1.0, 0.04], colormaps[quantity.cmap], norm,
                                   quantity.label, mark=0.0)
    bar_axis.xaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True))
    bar_axis.xaxis.set_minor_locator(AutoMinorLocator(5))
    median, rms = comparison.statistics[quantity.key]
    sky_axes.panel_note(axis, f"median {median:+.1f}\nrms {rms:.1f}")
    if show_name:
        axis.text(0.04, 0.96, comparison.component.title, transform=axis.transAxes, ha="left",
                  va="top", fontsize=13, fontweight="bold", zorder=13, bbox=sky_axes.LABEL_BOX)
    sky_axes.frame_panel(axis, cube, comparison.view(cube), comparison.component.distance_mpc)


def plot_residual_moments(cube: DataCube, comparisons: List[Comparison], file_name: str) -> None:
    """Plot the residual moment maps, data minus model, one row per disc.

    Parameters
    ----------
    cube : DataCube
        Data cube with the mask of the components.
    comparisons : list of Comparison
        Maps of the discs and their models (:func:`compare_with_model`).
    file_name : str
        Output file; its extension sets the format.
    """
    figure, rectangles = sky_axes.panel_figure(len(comparisons) * len(QUANTITIES),
                                               len(QUANTITIES), panel=(3.1, 3.1), gap=(1.1, 1.3))
    for row, comparison in enumerate(comparisons):
        for column, quantity in enumerate(QUANTITIES):
            _residual_panel(figure, rectangles[row * len(QUANTITIES) + column], cube, comparison,
                            quantity, show_name=column == 0)
    sky_axes.save(figure, file_name)


def residual_statistics(comparisons: List[Comparison]) -> List[Dict[str, object]]:
    """Median and rms of the residuals of every disc, one row per disc."""
    rows = []
    for comparison in comparisons:
        row = {"component": comparison.component.name}
        for quantity in QUANTITIES:
            median, rms = comparison.statistics[quantity.key]
            row[f"{quantity.key}_median"], row[f"{quantity.key}_rms"] = median, rms
        rows.append(row)
    return rows
