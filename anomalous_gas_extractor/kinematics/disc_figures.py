"""Maps of every component separated into its rotating disc and its anomalous gas.

The rotating disc is the gas of a component inside the ellipse of its
tilted-ring model (dashed, with the centre as +), the anomalous gas the rest.

:func:`plot_column_density` shows the HI column density of the two parts in
blue and in orange on one logarithmic scale, with their HI masses.

:func:`plot_velocity_fields` shows the velocity field, zoomed on the rotating
disc: the disc in full colour with black iso-velocity contours, the anomalous
gas faded with grey contours. The colour scale is centred on the systemic
velocity and spans the velocities of the disc. With slices, the
position-velocity slices are drawn as well and every panel holds its whole
slice.

In both figures the last panel is an overview of all components.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from matplotlib import patheffects
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from anomalous_gas_extractor.kinematics import hi_mass, sky_axes
from anomalous_gas_extractor.kinematics.datacube import DataCube
from anomalous_gas_extractor.kinematics.figure_data import Component
from anomalous_gas_extractor.kinematics.overview import overview_panel
from anomalous_gas_extractor.kinematics.sky_axes import (ANOMALOUS_CMAP, ANOMALOUS_COLOUR, ARCSEC,
                                                         DISC_CMAP, DISC_COLOUR, SLICE_COLOUR,
                                                         VELOCITY_CMAP, View)
from anomalous_gas_extractor.kinematics.slices import Slice

# column density (10^20 cm^-2) of a surface density of 1 solar mass per square parsec
COLUMN_DENSITY_PER_SURFACE_DENSITY = 1.249
COLUMN_DENSITY_UNIT = r"$N_{HI}$ (10$^{20}$ cm$^{-2}$)"
COLUMN_DENSITY_PERCENTILES = (3.0, 99.8)
# colour strength of the anomalous gas in the velocity fields
FADE = 0.55
# greys of the rotating discs and of the anomalous gas in the overview of the velocity fields
DISC_GREY, ANOMALOUS_GREY = "0.30", "0.80"
# half width of a panel in units of the radius of the disc, and its lower limit in beams
ZOOM_RADII = 2.4
MINIMUM_HALF_WIDTH_BEAMS = 2.5
# panel width in units of the length of the position-velocity slice
SLICE_MARGIN = 1.12
CONTOURS_ACROSS = 11
MINIMUM_HALF_RANGE_KMS = 30.0
# percentile of the velocity offsets that sets the colour range: with and without disc
RANGE_PERCENTILES = (98.0, 95.0)


def mass_label(mass_msun: float) -> str:
    """An HI mass as mantissa and power of ten."""
    exponent = int(np.floor(np.log10(mass_msun)))
    return rf"{mass_msun / 10 ** exponent:.1f}$\times$10$^{{{exponent}}}$ M$_{{\odot}}$"


def _disc_handle(component: Component) -> Line2D:
    """Legend entry of the ellipse of the rotating disc."""
    radius = component.geometry.radius_arcsec
    return Line2D([], [], color="black", linestyle="--", linewidth=1.8,
                  label=f"{component.disc_source} disc, $R$ = {radius:.0f}{ARCSEC}")


def _draw_disc(axis, cube: DataCube, component: Component) -> None:
    """Ellipse and centre of the rotating disc of a component."""
    geometry, margin = component.geometry, component.margin_arcsec
    minor = geometry.radius_arcsec * np.cos(np.radians(geometry.inclination_deg))
    centre = geometry.ra_deg, geometry.dec_deg
    sky_axes.draw_ellipse(axis, cube, centre, (geometry.radius_arcsec + margin, minor + margin),
                          geometry.position_angle_deg)
    sky_axes.mark_position(axis, cube, *centre)


def split_overview(figure: Figure, rectangle: Sequence[float], cube: DataCube,
              components: List[Component], disc_colour, anomalous_colour, title: str):
    """All components in the two colours of rotating disc and anomalous gas."""
    fills = [(component.footprint, anomalous_colour) for component in components]
    fills += [(component.disc_pixels, disc_colour) for component in components]
    handles = [Patch(facecolor=disc_colour, edgecolor="0.3", linewidth=0.6, label="rotating disc"),
               Patch(facecolor=anomalous_colour, edgecolor="0.3", linewidth=0.6,
                     label="anomalous gas")]
    return overview_panel(figure, rectangle, cube,
                          [component.footprint for component in components],
                          [component.title for component in components], fills, title, handles)


def _component_view(cube: DataCube, component: Component) -> View:
    """The whole component with a strip for the legend."""
    padding = cube.beam_major_arcsec / cube.pixel_arcsec
    return sky_axes.footprint_view(component.footprint, padding).with_legend_strip()


def _column_density_panel(figure: Figure, rectangle: Sequence[float], cube: DataCube,
                          component: Component) -> None:
    """Column density of one component: rotating disc in blue, anomalous gas in orange."""
    density = COLUMN_DENSITY_PER_SURFACE_DENSITY * hi_mass.surface_density(
        component.maps.intensity, cube.beam_major_arcsec, cube.beam_minor_arcsec)
    disc, anomalous = component.disc_pixels, component.anomalous_pixels
    low, high = np.percentile(density[disc | anomalous], COLUMN_DENSITY_PERCENTILES)
    norm = LogNorm(vmin=low, vmax=high, clip=True)
    image = np.zeros(density.shape + (4,))
    image[disc] = DISC_CMAP(norm(density[disc]))
    image[anomalous] = ANOMALOUS_CMAP(norm(density[anomalous]))
    axis = sky_axes.sky_axes(figure, rectangle, cube)
    axis.imshow(image, origin="lower", interpolation="nearest", zorder=2)
    sky_axes.draw_outline(axis, component.footprint)

    region = component.region if component.has_disc else None
    budget = hi_mass.mass_budget(cube, component.label, component.distance_mpc, region)
    handles = []
    if component.has_disc:
        _draw_disc(axis, cube, component)
        handles.append(Patch(facecolor=DISC_COLOUR, edgecolor="0.3", linewidth=0.6,
                             label=f"rotating disc  {mass_label(budget.disc_mass)}"
                                   f"  ({100 * budget.disc_fraction:.0f} %)"))
    handles.append(Patch(facecolor=ANOMALOUS_COLOUR, edgecolor="0.3", linewidth=0.6,
                         label=f"anomalous gas  {mass_label(budget.anomalous_mass)}"
                               f"  ({100 * (1 - budget.disc_fraction):.0f} %)"))
    if component.has_disc:
        handles.append(_disc_handle(component))
    axis.legend(handles=handles, loc="upper left", fontsize=9, handlelength=1.8,
                labelspacing=0.35, **sky_axes.LEGEND_STYLE).set_zorder(12)
    sky_axes.split_bars(axis, norm, component.has_disc)
    sky_axes.panel_note(axis, COLUMN_DENSITY_UNIT, size=10)
    sky_axes.panel_name(axis, component.title)
    sky_axes.frame_panel(axis, cube, _component_view(cube, component), component.distance_mpc)


def plot_column_density(cube: DataCube, components: List[Component], file_name: str) -> None:
    """Plot the column density of the rotating disc and the anomalous gas of every component.

    Parameters
    ----------
    cube : DataCube
        Data cube with the mask of the components.
    components : list of Component
        The components to draw, one panel each.
    file_name : str
        Output file; its extension sets the format.
    """
    overview = len(components) > 1
    figure, rectangles = sky_axes.panel_figure(len(components) + overview, 3)
    for rectangle, component in zip(rectangles, components):
        _column_density_panel(figure, rectangle, cube, component)
    if overview:
        split_overview(figure, rectangles[-1], cube, components, DISC_COLOUR, ANOMALOUS_COLOUR,
                       "rotating discs and anomalous gas")
    sky_axes.save(figure, file_name)


def draw_slice(axis, cube: DataCube, strip: Slice, edges: bool = True) -> None:
    """A position-velocity slice on a sky panel: its axis and, if ``edges``, its two long sides."""
    ends = [strip.first_arcsec, strip.last_arcsec]
    x, y = sky_axes.sky_line(cube, strip.ra_deg, strip.dec_deg, strip.position_angle_deg, ends)
    if not edges:
        axis.plot(x, y, color=SLICE_COLOUR, linewidth=1.6, zorder=10)
        return
    axis.plot(x, y, color=SLICE_COLOUR, linewidth=1.8, zorder=10, solid_capstyle="butt",
              path_effects=[patheffects.withStroke(linewidth=3.4, foreground="white")])
    for side in (-0.5, 0.5):
        x, y = sky_axes.sky_line(cube, strip.ra_deg, strip.dec_deg, strip.position_angle_deg,
                                 ends, side * strip.width_arcsec)
        axis.plot(x, y, color=SLICE_COLOUR, linewidth=0.7, linestyle=(0, (3, 2)), zorder=10)


def _velocity_view(cube: DataCube, component: Component, strip: Optional[Slice]) -> View:
    """Panel zoomed on the rotating disc, or on the whole slice or the whole component."""
    pixel = cube.pixel_arcsec
    if strip is not None:
        middle = 0.5 * (strip.first_arcsec + strip.last_arcsec)
        x, y = sky_axes.sky_line(cube, strip.ra_deg, strip.dec_deg, strip.position_angle_deg,
                                 [middle])
        half_width = 0.5 * SLICE_MARGIN * (strip.last_arcsec - strip.first_arcsec) / pixel
        return View(float(x[0]), float(y[0]), half_width).with_legend_strip()
    if component.has_disc:
        geometry = component.geometry
        half_width = max(ZOOM_RADII * geometry.radius_arcsec,
                         MINIMUM_HALF_WIDTH_BEAMS * cube.beam_fwhm_arcsec) / pixel
        return View(*cube.world_to_pixel(geometry.ra_deg, geometry.dec_deg), half_width)
    return _component_view(cube, component)


def _velocity_panel(figure: Figure, rectangle: Sequence[float], cube: DataCube,
                    component: Component, strip: Optional[Slice]) -> None:
    """Velocity field of one component: rotating disc in full colour, anomalous gas faded."""
    velocity, region, vsys = component.maps.velocity, component.region, component.vsys_kms
    seen = np.isfinite(velocity)
    reference = seen & region if (seen & region).any() else seen
    percentile = RANGE_PERCENTILES[0 if component.has_disc else 1]
    half_range = max(float(np.percentile(np.abs(velocity[reference] - vsys), percentile)),
                     MINIMUM_HALF_RANGE_KMS)
    norm = Normalize(vmin=vsys - half_range, vmax=vsys + half_range, clip=True)
    image = np.zeros(velocity.shape + (4,))
    image[seen] = VELOCITY_CMAP(norm(velocity[seen]))
    faded = seen & ~region
    image[faded, :3] = 1.0 - FADE * (1.0 - image[faded, :3])
    axis = sky_axes.sky_axes(figure, rectangle, cube)
    axis.imshow(image, origin="lower", interpolation="nearest", zorder=2)

    step = sky_axes.contour_step(2.0 * half_range, CONTOURS_ACROSS)
    smooth = sky_axes.smooth_nan(velocity)
    levels = sky_axes.velocity_levels(smooth, vsys, step)
    for part, colour, widths in ((np.where(region, np.nan, smooth), "0.45", (0.6, 1.1)),
                                 (np.where(region, smooth, np.nan), "black", (1.0, 2.0))):
        if np.isfinite(part).any():
            axis.contour(part, levels=levels, colors=colour, linewidths=widths[0], zorder=4)
            axis.contour(part, levels=[vsys], colors=colour, linewidths=widths[1],
                         linestyles="--", zorder=4)
    sky_axes.draw_outline(axis, component.footprint)

    if component.has_disc:
        _draw_disc(axis, cube, component)
        handles = [_disc_handle(component)]
    else:
        handles = [Line2D([], [], color="none", label="no rotating disc")]
    if strip is not None:
        draw_slice(axis, cube, strip)
        handles.append(Line2D([], [], color=SLICE_COLOUR, linewidth=1.8,
                              label=f"slice, PA = {strip.position_angle_deg:.0f}$^\\circ$"
                                    f" ({strip.source})"))
    axis.legend(handles=handles, loc="upper left", fontsize=10 if strip is None else 9,
                handlelength=2.0 if component.has_disc or strip is not None else 0.0,
                **sky_axes.LEGEND_STYLE).set_zorder(12)
    sky_axes.velocity_bar(axis, cube, VELOCITY_CMAP, norm, vsys)
    sky_axes.panel_note(axis, f"$v_{{sys}}$ = {vsys:.0f}\niso step {step:.0f}")
    sky_axes.panel_name(axis, component.title)
    sky_axes.frame_panel(axis, cube, _velocity_view(cube, component, strip),
                         component.distance_mpc)


def plot_velocity_fields(cube: DataCube, components: List[Component], file_name: str,
                         slices: Optional[Dict[str, Slice]] = None) -> None:
    """Plot the velocity field of the rotating disc and the anomalous gas of every component.

    Parameters
    ----------
    cube : DataCube
        Data cube with the mask of the components.
    components : list of Component
        The components to draw, one panel each.
    file_name : str
        Output file; its extension sets the format.
    slices : dict, optional
        Position-velocity slice of every component by name. They are drawn and
        every panel is widened to hold its slice.
    """
    slices = slices or {}
    overview = len(components) > 1
    figure, rectangles = sky_axes.panel_figure(len(components) + overview, 3)
    for rectangle, component in zip(rectangles, components):
        _velocity_panel(figure, rectangle, cube, component, slices.get(component.name))
    if overview:
        axis = split_overview(figure, rectangles[-1], cube, components, DISC_GREY,
                              ANOMALOUS_GREY, "rotating discs and anomalous gas")
        for strip in slices.values():
            draw_slice(axis, cube, strip, edges=False)
    sky_axes.save(figure, file_name)
