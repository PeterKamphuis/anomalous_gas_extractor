"""Position-velocity diagrams of the components along their kinematic major axis.

Only the gas of the component itself is shown, sampled along its slice
(:mod:`anomalous_gas_extractor.kinematics.slices`):

blue            emission within the radius of the rotating disc along the slice,
orange          emission beyond it, the anomalous gas; both on one logarithmic scale,
grey contour    outline of the emission of the component at three times the noise,
black contours  tilted-ring model, sampled in the same way, at 3 sigma x (1, 2, 4, ...),
circles, line   rotation curve of the model projected on the sky, ring by ring:
                v_sys -/+ V_rot sin(i), approaching side at negative offsets,
dashed, dotted  radius of the model disc and r_max; horizontal line: systemic velocity,
cross           beam along the slice and channel width.

The compass direction of each end of the slice is written at the bottom of a
panel. :func:`plot_all_diagrams` shows all components and an overview with the
slices, :func:`plot_diagram` one component with an axis in kpc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib import patheffects
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator

from anomalous_gas_extractor.kinematics import disc_figures, sky_axes
from anomalous_gas_extractor.kinematics.datacube import DataCube, spectral_axis
from anomalous_gas_extractor.kinematics.fat import ARCSEC_TO_RADIAN, EMPTY_RING_BRIGHTNESS
from anomalous_gas_extractor.kinematics.figure_data import Component
from anomalous_gas_extractor.kinematics.pv import position_velocity
from anomalous_gas_extractor.kinematics.sky_axes import (ANOMALOUS_CMAP, ANOMALOUS_COLOUR,
                                                         DISC_CMAP, DISC_COLOUR, LABEL_BOX)
from anomalous_gas_extractor.kinematics.slices import Slice

# faintest emission drawn and first contour, in units of the noise; the contours double
SHOW_ABOVE_SIGMA = 1.0
CONTOUR_SIGMA = 3.0
N_CONTOURS = 12
# the colour scale reaches at least this many times the noise
MINIMUM_PEAK_SIGMA = 8.0
# velocity range (km/s) added below and above the emission and the rotation curve
VELOCITY_MARGIN = 35.0
CURVE_MARGIN = 15.0
INTENSITY_UNIT = r" (mJy beam$^{-1}$)"
CURVE_STYLE = dict(color="black", linewidth=1.3, marker="o", markersize=5.5,
                   markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.2)


@dataclass
class Diagram:
    """Position-velocity diagram of one component.

    Attributes
    ----------
    component : Component
        The component.
    strip : Slice
        Where the diagram was taken.
    offsets, velocity : 1-D arrays
        Offsets along the slice (arcsec) and velocities (km/s, increasing).
    data : 2-D array (velocity, offset)
        Emission of the component in mJy/beam.
    model : 2-D array, optional
        The model cube sampled like the data.
    curve : 2-D array (ring, 2)
        Offsets and sky velocities of the rings of the model.
    rms : float
        Noise of the cube in mJy/beam.
    """

    component: Component
    strip: Slice
    offsets: np.ndarray
    velocity: np.ndarray
    data: np.ndarray
    model: Optional[np.ndarray]
    curve: np.ndarray
    rms: float

    @property
    def channel_kms(self) -> float:
        return float(np.median(np.abs(np.diff(self.velocity))))


def _model_diagram(model_file: str, strip: Slice, offsets: np.ndarray,
                   velocity: np.ndarray) -> np.ndarray:
    """The model cube sampled along a slice, on the offsets and velocities of the data."""
    with fits.open(model_file) as hdul:
        header, model = hdul[0].header, np.squeeze(hdul[0].data)
    model_velocity, _ = spectral_axis(header, model.shape[0])
    diagram = position_velocity(model, WCS(header, naxis=2), strip.ra_deg, strip.dec_deg,
                                strip.position_angle_deg, offsets, strip.width_arcsec,
                                abs(header["CDELT2"]) * 3600.0)
    order = np.argsort(model_velocity)
    return 1e3 * np.array([np.interp(velocity, model_velocity[order], column, left=0.0, right=0.0)
                           for column in diagram[order].T]).T


def rotation_curve(rings: List[Dict[str, float]], vsys_kms: float) -> np.ndarray:
    """Offsets and sky velocities of the rings of a model along its major axis.

    The approaching side is at negative offsets, the receding side at positive
    offsets and the central ring, which does not rotate, at the systemic
    velocity. Returns an array (ring, 2) sorted by offset.
    """
    points = [(0.0, vsys_kms)] if rings else []
    for side, sign in (("approaching", -1.0), ("receding", 1.0)):
        for ring in rings:
            if ring["radius_arcsec"] > 0 and ring[f"sbr_{side}"] > EMPTY_RING_BRIGHTNESS:
                projected = ring[f"vrot_kms_{side}"] * np.sin(
                    np.radians(ring[f"inclination_deg_{side}"]))
                points.append((sign * ring["radius_arcsec"], vsys_kms + sign * projected))
    return np.array(sorted(points)).reshape(-1, 2)


def extract_diagram(cube: DataCube, component: Component, strip: Slice,
                    rings: Sequence[Dict[str, float]] = ()) -> Diagram:
    """Position-velocity diagram of a component and of its model along a slice.

    ``rings`` is the ring table of the tilted-ring model, for its rotation curve.
    """
    pixel = cube.pixel_arcsec
    offsets = np.arange(strip.first_arcsec, strip.last_arcsec + 0.5 * pixel, pixel)
    channels = cube.bounding_box(component.label)[0]
    own = np.where(cube.labels[channels] == component.label, np.nan_to_num(cube.data[channels]),
                   0.0)
    data = position_velocity(own, cube.celestial_wcs, strip.ra_deg, strip.dec_deg,
                             strip.position_angle_deg, offsets, strip.width_arcsec, pixel)
    order = np.argsort(cube.velocity_kms[channels])
    velocity = cube.velocity_kms[channels][order]
    model, curve = None, np.empty((0, 2))
    if component.model_cube is not None:
        model = _model_diagram(component.model_cube, strip, offsets, velocity)
        curve = rotation_curve(list(rings), component.vsys_kms)
    return Diagram(component, strip, offsets, velocity, 1e3 * data[order], model, curve,
                   1e3 * cube.rms)


def _draw_emission(axis, diagram: Diagram) -> LogNorm:
    """Emission, contours of data and model and the rotation curve; returns the colour scale."""
    component, offsets, velocity = diagram.component, diagram.offsets, diagram.velocity
    data, rms = diagram.data, diagram.rms
    seen = data > SHOW_ABOVE_SIGMA * rms
    peak = np.percentile(data[seen], 99.8) if seen.any() else 0.0
    norm = LogNorm(vmin=SHOW_ABOVE_SIGMA * rms, vmax=max(peak, MINIMUM_PEAK_SIGMA * rms),
                   clip=True)
    inside = np.zeros_like(seen)
    if component.has_disc:
        extent = component.geometry.radius_arcsec + component.margin_arcsec
        inside = np.broadcast_to(np.abs(offsets) <= extent, data.shape)
    image = np.zeros(data.shape + (4,))
    image[seen & inside] = DISC_CMAP(norm(data[seen & inside]))
    image[seen & ~inside] = ANOMALOUS_CMAP(norm(data[seen & ~inside]))
    half_step, half_channel = 0.5 * (offsets[1] - offsets[0]), 0.5 * diagram.channel_kms
    axis.imshow(image, origin="lower", aspect="auto", interpolation="nearest", zorder=2,
                extent=[offsets[0] - half_step, offsets[-1] + half_step,
                        velocity[0] - half_channel, velocity[-1] + half_channel])
    levels = CONTOUR_SIGMA * rms * 2.0 ** np.arange(N_CONTOURS)
    if data.max() > levels[0]:
        axis.contour(offsets, velocity, data, levels=levels[:1], colors="0.4", linewidths=0.6,
                     zorder=3)
    if diagram.model is not None and diagram.model.max() > levels[0]:
        axis.contour(offsets, velocity, diagram.model, levels=levels[levels < diagram.model.max()],
                     colors="black", linewidths=1.0, zorder=5)
    axis.axhline(component.vsys_kms, color="0.35", linewidth=0.9, linestyle=(0, (6, 3)), zorder=4)
    if component.has_disc:
        for sign in (-1.0, 1.0):
            axis.axvline(sign * component.geometry.radius_arcsec, color="black", linewidth=1.2,
                         linestyle="--", zorder=4)
            axis.axvline(sign * component.row["r_max_arcsec"], color="0.4", linewidth=1.3,
                         linestyle=":", zorder=4)
    if len(diagram.curve):
        axis.plot(diagram.curve[:, 0], diagram.curve[:, 1], zorder=6, **CURVE_STYLE,
                  path_effects=[patheffects.withStroke(linewidth=3.0, foreground="white")])
    axis.set_xlim(offsets[0] - half_step, offsets[-1] + half_step)
    emission = velocity[seen.any(axis=1)] if seen.any() else velocity
    low, high = emission[0] - VELOCITY_MARGIN, emission[-1] + VELOCITY_MARGIN
    if len(diagram.curve):
        low = min(low, diagram.curve[:, 1].min() - CURVE_MARGIN)
        high = max(high, diagram.curve[:, 1].max() + CURVE_MARGIN)
    axis.set_ylim(low, high)
    return norm


def draw_diagram(axis, cube: DataCube, diagram: Diagram, kpc_axis: bool = False) -> LogNorm:
    """Draw a position-velocity diagram with its annotations; returns the colour scale."""
    norm = _draw_emission(axis, diagram)
    strip, component = diagram.strip, diagram.component
    (left, right), (low, high) = axis.get_xlim(), axis.get_ylim()
    # resolution: beam along the slice and channel width
    axis.errorbar(left + 0.05 * (right - left) + 0.5 * cube.beam_fwhm_arcsec,
                  high - 0.17 * (high - low), xerr=0.5 * cube.beam_fwhm_arcsec,
                  yerr=0.5 * diagram.channel_kms, color="black", linewidth=1.3, capsize=2.5,
                  zorder=7)
    for x, direction, align in zip((0.03, 0.97), strip.directions, ("left", "right")):
        axis.text(x, 0.035, direction, transform=axis.transAxes, ha=align, va="bottom",
                  fontsize=11, fontweight="bold", color="0.25", zorder=8, bbox=LABEL_BOX)
    axis.text(0.97, 0.965, component.title, transform=axis.transAxes, ha="right", va="top",
              fontsize=13, fontweight="bold", zorder=9, bbox=LABEL_BOX)
    axis.text(0.03, 0.965, f"PA = {strip.position_angle_deg:.0f}$^\\circ$ ({strip.source})",
              transform=axis.transAxes, ha="left", va="top", fontsize=10, zorder=9, bbox=LABEL_BOX)
    axis.tick_params(which="both", top=not kpc_axis, right=True, width=1.1, pad=6)
    axis.xaxis.set_minor_locator(AutoMinorLocator(5))
    axis.yaxis.set_minor_locator(AutoMinorLocator(5))
    axis.set_xlabel("Offset along the major axis (arcsec)")
    axis.set_ylabel(sky_axes.velocity_label(cube))
    if kpc_axis:
        scale = ARCSEC_TO_RADIAN * component.distance_mpc * 1e3
        top = axis.secondary_xaxis("top", functions=(lambda arcsec: arcsec * scale,
                                                     lambda kpc: kpc / scale))
        top.set_xlabel("Offset (kpc)")
        top.tick_params(which="both", width=1.1)
        top.xaxis.set_minor_locator(AutoMinorLocator(5))
    return norm


def legend_handles(diagrams: Sequence[Diagram]) -> List[object]:
    """Legend entries of what is drawn in some position-velocity diagrams."""
    handles = [Patch(facecolor=DISC_COLOUR, edgecolor="0.3", linewidth=0.6,
                     label="within the radius of the rotating disc"),
               Patch(facecolor=ANOMALOUS_COLOUR, edgecolor="0.3", linewidth=0.6,
                     label="beyond it: anomalous gas"),
               Line2D([], [], color="0.4", linewidth=0.8,
                      label=rf"data, {CONTOUR_SIGMA:.0f}$\sigma$")]
    if any(diagram.model is not None for diagram in diagrams):
        handles += [Line2D([], [], color="black", linewidth=1.0,
                           label=rf"model, {CONTOUR_SIGMA:.0f}$\sigma$ $\times$ 2$^n$"),
                    Line2D([], [], label=r"model rotation curve, $v_{sys} \pm V_{rot} \sin i$",
                           **CURVE_STYLE)]
    if any(diagram.component.has_disc for diagram in diagrams):
        handles += [Line2D([], [], color="black", linewidth=1.2, linestyle="--",
                           label="radius of the rotating disc"),
                    Line2D([], [], color="0.4", linewidth=1.3, linestyle=":",
                           label=r"$r_{max}$ (Kinemetry)")]
    handles.append(Line2D([], [], color="0.35", linewidth=0.9, linestyle=(0, (6, 3)),
                          label=r"$v_{sys}$"))
    return handles


def plot_all_diagrams(cube: DataCube, diagrams: List[Diagram], file_name: str) -> None:
    """Plot the position-velocity diagrams of all components and an overview of the slices.

    Parameters
    ----------
    cube : DataCube
        Data cube with the mask of the components.
    diagrams : list of Diagram
        The diagrams, one panel each.
    file_name : str
        Output file; its extension sets the format.
    """
    overview = len(diagrams) > 1
    figure, rectangles = sky_axes.panel_figure(len(diagrams) + overview, 3, panel=(3.75, 3.6))
    for rectangle, diagram in zip(rectangles, diagrams):
        axis = figure.add_axes(rectangle)
        norm = draw_diagram(axis, cube, diagram)
        sky_axes.split_bars(axis, norm, diagram.component.has_disc, INTENSITY_UNIT, short=True,
                            height_above=1.035)
    if overview:
        components = [diagram.component for diagram in diagrams]
        axis = disc_figures.split_overview(figure, rectangles[-1], cube, components, DISC_COLOUR,
                                            ANOMALOUS_COLOUR, "position-velocity slices (green)")
        for diagram in diagrams:
            disc_figures.draw_slice(axis, cube, diagram.strip, edges=False)
    lowest = min(rectangle[1] for rectangle in rectangles)
    figure.legend(handles=legend_handles(diagrams),
                  loc="upper center", bbox_to_anchor=(0.5, lowest - 0.7 / figure.get_figheight()),
                  ncol=4, fontsize=10, frameon=True, edgecolor="0.6")
    sky_axes.save(figure, file_name)


def plot_diagram(cube: DataCube, diagram: Diagram, file_name: str) -> None:
    """Plot the position-velocity diagram of one component, with the offsets in kpc as well."""
    figure = Figure(figsize=(7.6, 6.4))
    FigureCanvasAgg(figure)
    axis = figure.add_axes([0.13, 0.25, 0.80, 0.50])
    norm = draw_diagram(axis, cube, diagram, kpc_axis=True)
    sky_axes.split_bars(axis, norm, diagram.component.has_disc, INTENSITY_UNIT,
                        height_above=1.19)
    figure.legend(handles=legend_handles([diagram]), loc="lower center",
                  bbox_to_anchor=(0.53, 0.0), ncol=2, fontsize=9.5, frameon=True, edgecolor="0.6")
    sky_axes.save(figure, file_name)
