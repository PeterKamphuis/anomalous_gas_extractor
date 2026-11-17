"""Figures of the step rmax: velocity fields with r_max and the Kinemetry profiles.

:func:`plot_velocity_fields` shows the velocity field of every component in a
panel of its own, each with its own colour map, and on it the kinematic centre
(+), the starting centre if one was given (x), the kinematic major axis (dashed
line) and a circle with the radius r_max (dashed if r_max is a lower limit).

:func:`plot_kinemetry_profiles` shows what the ring tests see: the rotation term
k1, the ratio k5/k1, the position angle and the drift of the systemic velocity
against the semi-major axis of the rings, with the limits of the tests in red
and the rotating disc in green. Rings with a coverage below the minimum are
drawn as open symbols.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator

from anomalous_gas_extractor.kinematics import sky_axes
from anomalous_gas_extractor.kinematics.datacube import DataCube
from anomalous_gas_extractor.kinematics.figure_data import Component, inside_rmax
from anomalous_gas_extractor.kinematics.overview import overview_panel
from anomalous_gas_extractor.kinematics.sky_axes import ARCSEC, truncated

RMAX_COLOUR = "#32CD32"
LIMIT_COLOUR = "#D62728"
# colour maps of the components, used in turn: neighbours differ in colour
COMPONENT_CMAPS = (truncated("Oranges", 0.30, 1.00), truncated("Greys", 0.25, 0.85),
                   colormaps["viridis"], colormaps["coolwarm"], truncated("Greens", 0.30, 1.00),
                   truncated("Purples", 0.30, 1.00), colormaps["cividis"])
VELOCITY_PERCENTILES = (2.0, 98.0)
CONTOURS_ACROSS = 7
# shortest half length of the major axis drawn on a velocity field (arcsec)
MINIMUM_AXIS_ARCSEC = 20.0
PROFILE_LABELS = (r"$k_1$ (km s$^{-1}$)", r"$k_5/k_1$", r"PA$_{kin}$ (deg)",
                  r"$\Delta v_{sys}$ (km s$^{-1}$)")
# limits of the axes of k5/k1 and of the drift of the systemic velocity (km/s)
K5K1_AXIS = (0.0, 0.6)
DRIFT_AXIS_KMS = 60.0


@dataclass
class RingProfiles:
    """The Kinemetry rings of a component and how its r_max was found."""

    title: str
    rings: List[Dict[str, object]]
    r_max_arcsec: Optional[float]
    is_lower_limit: bool

    def column(self, name: str) -> np.ndarray:
        return np.array([ring[name] for ring in self.rings], dtype=float)


def _component_cmap(index: int):
    return COMPONENT_CMAPS[index % len(COMPONENT_CMAPS)]


def rmax_label(r_max_arcsec: Optional[float], is_lower_limit: bool) -> str:
    """r_max as written in the figures."""
    if r_max_arcsec is None:
        return r"$r_{max}$: none"
    relation = r"$\geq$" if is_lower_limit else "="
    return rf"$r_{{max}}$ {relation} {r_max_arcsec:.0f}{ARCSEC}"


def _draw_rmax(axis, cube: DataCube, component: Component,
               start_centre: Optional[Tuple[float, float]]) -> None:
    """Centres, major axis and r_max of a component."""
    row = component.row
    centre = row["centre_ra_deg"], row["centre_dec_deg"]
    r_max = row["r_max_arcsec"]
    if row["position_angle_deg"] is not None:
        half_length = max(r_max or 0.0, MINIMUM_AXIS_ARCSEC)
        x, y = sky_axes.sky_line(cube, *centre, row["position_angle_deg"],
                                 [-half_length, half_length])
        axis.plot(x, y, color="black", linewidth=1.6, linestyle=(0, (4, 2)), zorder=7,
                  path_effects=sky_axes.HALO)
    sky_axes.mark_position(axis, cube, *centre, size=160)
    if start_centre is not None:
        sky_axes.mark_position(axis, cube, *start_centre, marker="x", size=70, colour="0.25")
    if r_max is None:
        return
    linestyle = "--" if row["r_max_is_lower_limit"] else "-"
    sky_axes.draw_ellipse(axis, cube, centre, (r_max, r_max), 0.0, RMAX_COLOUR, linestyle,
                          width=2.6)
    handle = Line2D([], [], color=RMAX_COLOUR, linestyle=linestyle, linewidth=2.6,
                    label=rmax_label(r_max, row["r_max_is_lower_limit"]))
    axis.legend(handles=[handle], loc="upper left", fontsize=10, handlelength=2.0,
                **sky_axes.LEGEND_STYLE).set_zorder(12)


def _velocity_panel(figure: Figure, rectangle: Sequence[float], cube: DataCube,
                    component: Component, cmap,
                    start_centre: Optional[Tuple[float, float]]) -> None:
    """Velocity field of one component with its r_max."""
    velocity, vsys = component.maps.velocity, component.kinematic_vsys_kms
    low, high = np.nanpercentile(velocity, VELOCITY_PERCENTILES)
    axis = sky_axes.sky_axes(figure, rectangle, cube)
    image = axis.imshow(velocity, origin="lower", cmap=cmap, vmin=low, vmax=high,
                        interpolation="nearest")
    step = sky_axes.contour_step(high - low, CONTOURS_ACROSS)
    smooth = sky_axes.smooth_nan(velocity)
    levels = [level for level in sky_axes.velocity_levels(velocity, vsys, step)
              if low <= level <= high]
    if levels:
        axis.contour(smooth, levels=levels, colors="0.25", linewidths=0.6, zorder=4)
    axis.contour(smooth, levels=[vsys], colors="0.05", linewidths=1.4, linestyles="--", zorder=4)
    sky_axes.draw_outline(axis, component.footprint, width=1.7)
    _draw_rmax(axis, cube, component, start_centre)
    sky_axes.velocity_bar(axis, cube, cmap, image.norm, vsys)
    sky_axes.panel_note(axis, f"$v_{{sys}}$ = {vsys:.0f}\niso step {step:.0f}")
    sky_axes.panel_name(axis, component.title)
    view = sky_axes.footprint_view(component.footprint | inside_rmax(cube, component),
                                   cube.beam_major_arcsec / cube.pixel_arcsec)
    sky_axes.frame_panel(axis, cube, view, component.distance_mpc)


def plot_velocity_fields(cube: DataCube, components: List[Component], file_name: str,
                         start_centres: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
    """Plot the velocity field of every component with its r_max.

    Parameters
    ----------
    cube : DataCube
        Data cube with the mask of the components.
    components : list of Component
        The components to draw, one panel each.
    file_name : str
        Output file; its extension sets the format.
    start_centres : dict, optional
        Starting centre (RA, Dec in degrees) of the centre fit by component name.
    """
    overview = len(components) > 1
    figure, rectangles = sky_axes.panel_figure(len(components) + overview, 3)
    for index, component in enumerate(components):
        _velocity_panel(figure, rectangles[index], cube, component, _component_cmap(index),
                        (start_centres or {}).get(component.name))
    if overview:
        # the largest component first, so that the small ones stay visible
        order = sorted(range(len(components)), key=lambda i: -components[i].footprint.sum())
        fills = [(components[i].footprint, _component_cmap(i)(0.6)) for i in order]
        overview_panel(figure, rectangles[-1], cube,
                       [component.footprint for component in components],
                       [component.title for component in components], fills,
                       "deblended components")
    sky_axes.save(figure, file_name)


def _draw_profiles(axes, profiles: RingProfiles, k5k1_max: float, beam_arcsec: float) -> None:
    """The four profiles of one component in a column of axes."""
    radius = profiles.column("radius_arcsec")
    sampled = profiles.column("pass_coverage").astype(bool)
    regular = profiles.column("regular").astype(bool)
    # the position angle of an ellipse is defined modulo 180 degrees
    angle = np.degrees(np.unwrap(np.radians(2.0 * profiles.column("position_angle_deg")))) / 2.0
    k1 = profiles.column("k1_kms")
    series = (k1, profiles.column("k5_over_k1"), angle, profiles.column("vsys_drift_kms"))
    r_max = profiles.r_max_arcsec
    for axis, values in zip(axes, series):
        axis.plot(radius[sampled], values[sampled], "-o", color="black", markersize=4,
                  linewidth=1.2, zorder=3)
        axis.plot(radius[~sampled], values[~sampled], "o", markerfacecolor="white",
                  markeredgecolor="0.5", markersize=4, zorder=2)
        if r_max is not None:
            first = int(np.argmax(regular))
            axis.axvspan(radius[first] if first else 0.0, r_max, color=RMAX_COLOUR, alpha=0.16,
                         linewidth=0, zorder=0)
            axis.axvline(r_max, color=RMAX_COLOUR, linewidth=2.0, zorder=1,
                         linestyle="--" if profiles.is_lower_limit else "-")
        axis.axvline(beam_arcsec, color="0.6", linewidth=0.9, linestyle=":", zorder=1)
        axis.xaxis.set_minor_locator(AutoMinorLocator(5))
        axis.yaxis.set_minor_locator(AutoMinorLocator(5))
        axis.tick_params(which="both", top=True, right=True)
    axes[1].axhline(k5k1_max, color=LIMIT_COLOUR, linewidth=1.0, linestyle="--")
    axes[1].set_ylim(*K5K1_AXIS)
    tolerance = profiles.column("vsys_tolerance_kms")
    for sign in (-1.0, 1.0):
        axes[3].plot(radius, sign * tolerance, color=LIMIT_COLOUR, linewidth=1.0, linestyle="--")
    axes[3].set_ylim(-DRIFT_AXIS_KMS, DRIFT_AXIS_KMS)
    # the scale of k1 follows the disc and the well-sampled rings, not badly fitted outliers
    disc = regular & (radius <= (r_max or 0.0) + 1e-6)
    reference = k1[sampled] if sampled.any() else k1
    top = max(k1[disc].max() if disc.any() else 0.0, np.nanpercentile(reference, 90.0) / 1.13)
    axes[0].set_ylim(0.0, 1.3 * top)
    axes[3].set_xlim(0.0, min(radius.max(), max(3.2 * (r_max or 60.0), 150.0)))
    axes[3].set_xlabel("semi-major axis (arcsec)", fontsize=12)
    axes[0].set_title(f"{profiles.title}\n{rmax_label(r_max, profiles.is_lower_limit)}",
                      fontsize=12, fontweight="bold")


def plot_kinemetry_profiles(profiles: List[RingProfiles], file_name: str, k5k1_max: float,
                            beam_arcsec: float) -> None:
    """Plot the Kinemetry profiles of several components, one column per component.

    Parameters
    ----------
    profiles : list of RingProfiles
        Ring tables of the step rmax with r_max.
    file_name : str
        Output file; its extension sets the format.
    k5k1_max : float
        Limit of the test on k5/k1, drawn as a horizontal line.
    beam_arcsec : float
        Size of the beam, marked by a dotted vertical line.
    """
    figure = Figure(figsize=(3.3 * len(profiles), 9.6))
    FigureCanvasAgg(figure)
    axes = figure.subplots(len(PROFILE_LABELS), len(profiles), sharex="col", squeeze=False,
                           gridspec_kw=dict(hspace=0.07, wspace=0.34))
    for column, component in enumerate(profiles):
        _draw_profiles(axes[:, column], component, k5k1_max, beam_arcsec)
    for axis, label in zip(axes[:, 0], PROFILE_LABELS):
        axis.set_ylabel(label, fontsize=12)
    figure.align_ylabels(axes[:, 0])
    sky_axes.save(figure, file_name)
