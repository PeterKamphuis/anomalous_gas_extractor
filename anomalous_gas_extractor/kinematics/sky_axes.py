"""Drawing helpers shared by the figures: sky panels in the house style and their annotations.

A sky panel is a matplotlib axes with the celestial coordinates of the cube
(``projection=wcs``), so everything is drawn in pixels of the cube: north is up
and east to the left for the usual cube with a negative ``CDELT1``. The helpers
draw with the matplotlib settings in force
(:func:`anomalous_gas_extractor.kinematics.plots.figure_style`).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib import colormaps, font_manager, patheffects
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox, TextArea, VPacker
from matplotlib.patches import Ellipse
from matplotlib.ticker import AutoMinorLocator
from scipy.ndimage import gaussian_filter

from anomalous_gas_extractor.kinematics.datacube import DataCube
from anomalous_gas_extractor.kinematics.fat import ARCSEC_TO_RADIAN
from anomalous_gas_extractor.kinematics.plots import figure_style
from anomalous_gas_extractor.kinematics.pv import slice_positions

logger = logging.getLogger(__name__)

# the fonts of the figures have no double prime
ARCSEC = '"'
HALO = [patheffects.withStroke(linewidth=2.4, foreground="white")]
LABEL_BOX = dict(facecolor="white", edgecolor="none", alpha=0.8, boxstyle="round,pad=0.2")
LEGEND_STYLE = dict(frameon=True, edgecolor="0.6", framealpha=0.9, borderpad=0.4)
# fraction of a panel that is kept free above the emission for the legend and the name
LEGEND_STRIP = 0.24
# spacings (km/s) of iso-velocity contours and of the ticks of a velocity colour bar
CONTOUR_STEPS = (5, 10, 15, 20, 25, 30, 40, 50, 75, 100)
TICK_STEPS = (10, 20, 25, 50, 100, 200)
# ticks of the logarithmic colour bars
LOG_TICKS = (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000)
# a scale bar is at most this fraction of the width of a panel
SCALE_BAR_FRACTION = 0.3
FIGURE_DPI = 300
FONT_EXTENSIONS = (".otf", ".ttf")


def truncated(name: str, low: float, high: float) -> Colormap:
    """Part of a colour map, for example a single-hue map without its lightest colours."""
    return LinearSegmentedColormap.from_list(f"{name}_{low}_{high}",
                                             colormaps[name](np.linspace(low, high, 256)))


VELOCITY_CMAP = colormaps["RdBu_r"]
DISC_CMAP = truncated("Blues", 0.30, 1.00)
ANOMALOUS_CMAP = truncated("Oranges", 0.30, 1.00)
DISC_COLOUR, ANOMALOUS_COLOUR = DISC_CMAP(0.5), ANOMALOUS_CMAP(0.5)
SLICE_COLOUR = "#1B9E3A"


def house_style(font_file: Optional[str] = None) -> Dict[str, object]:
    """Matplotlib settings of the figures, with all faces of the font.

    :func:`anomalous_gas_extractor.kinematics.plots.figure_style` registers the
    given font file only. Names are written in bold, so the other font files of
    the same family in the directory of ``font_file`` are registered as well.
    """
    style = figure_style(font_file)
    if font_file:
        directory = os.path.dirname(font_file)
        for name in sorted(os.listdir(directory)):
            face = os.path.join(directory, name)
            if os.path.splitext(name)[1].lower() in FONT_EXTENSIONS \
                    and font_manager.get_font(face).family_name == style["font.family"]:
                font_manager.fontManager.addfont(face)
    return style


@dataclass(frozen=True)
class View:
    """A square part of the sky: its centre and half its width in pixels of the cube."""

    x: float
    y: float
    half_width: float

    def with_legend_strip(self) -> View:
        """The view extended upwards by the strip for the legend."""
        return View(self.x, self.y + LEGEND_STRIP * self.half_width,
                    (1.0 + LEGEND_STRIP) * self.half_width)


def footprint_view(footprint: np.ndarray, padding_pixels: float) -> View:
    """The smallest square around the pixels of a footprint, plus a padding."""
    rows, columns = np.nonzero(footprint)
    extent = max(columns.max() - columns.min(), rows.max() - rows.min())
    return View(0.5 * (columns.min() + columns.max()), 0.5 * (rows.min() + rows.max()),
                0.5 * extent + padding_pixels)


def panel_figure(n_panels: int, n_columns: int, panel: Tuple[float, float] = (3.75, 3.75),
                 gap: Tuple[float, float] = (1.2, 1.4)) -> Tuple[Figure, List[List[float]]]:
    """A figure for a grid of panels and the rectangle of every panel.

    ``panel`` is the size of a panel and ``gap`` the space between two panels for
    their labels and colour bars, in inches. The rows are filled from the top.
    """
    n_columns = max(min(n_columns, n_panels), 1)
    n_rows = int(np.ceil(n_panels / n_columns))
    width, height = n_columns * (panel[0] + gap[0]), n_rows * (panel[1] + gap[1])
    figure = Figure(figsize=(width, height))
    FigureCanvasAgg(figure)
    rectangles = []
    for index in range(n_panels):
        row, column = divmod(index, n_columns)
        left = column * (panel[0] + gap[0]) + 0.7 * gap[0]
        bottom = height - (row + 1) * (panel[1] + gap[1]) + 0.45 * gap[1]
        rectangles.append([left / width, bottom / height, panel[0] / width, panel[1] / height])
    return figure, rectangles


def save(figure: Figure, file_name: str) -> None:
    """Write a figure; the extension of the file name sets the format."""
    figure.savefig(file_name, bbox_inches="tight", dpi=FIGURE_DPI)
    logger.info("figure written to %s", file_name)


def sky_axes(figure: Figure, rectangle: Sequence[float], cube: DataCube):
    """Axes with the celestial coordinates of the cube: ticks inside on all sides, minor ticks."""
    axis = figure.add_axes(rectangle, projection=cube.celestial_wcs)
    for coordinate, label in zip(axis.coords, ("RA (J2000)", "Dec (J2000)")):
        coordinate.set_ticks(number=5, direction="in", size=6, width=1.1, color="black")
        coordinate.display_minor_ticks(True)
        coordinate.set_minor_frequency(5)
        coordinate.set_ticklabel(size=10, exclude_overlapping=True)
        coordinate.set_axislabel(label, size=11)
    return axis


def show_view(axis, view: View) -> None:
    """Limit a sky panel to a view."""
    axis.set_xlim(view.x - view.half_width, view.x + view.half_width)
    axis.set_ylim(view.y - view.half_width, view.y + view.half_width)


def frame_panel(axis, cube: DataCube, view: View, distance_mpc: Optional[float] = None) -> None:
    """Show a view in a sky panel, with the beam and, if the distance is known, a scale bar."""
    show_view(axis, view)
    pixel = cube.pixel_arcsec
    beam = AuxTransformBox(axis.transData)
    beam.add_artist(Ellipse((0.0, 0.0), cube.beam_minor_arcsec / pixel,
                            cube.beam_major_arcsec / pixel, angle=cube.header.get("BPA", 0.0),
                            facecolor="0.5", edgecolor="black"))
    axis.add_artist(AnchoredOffsetbox(loc="lower left", child=beam, frameon=False, pad=0.2,
                                      borderpad=0.5, zorder=11))
    if distance_mpc is not None:
        _scale_bar(axis, view, pixel * ARCSEC_TO_RADIAN * distance_mpc * 1e3)


def _scale_bar(axis, view: View, kpc_per_pixel: float) -> None:
    """A bar of 1, 2 or 5 times a power of ten kpc at the bottom of a panel."""
    longest = SCALE_BAR_FRACTION * 2.0 * view.half_width * kpc_per_pixel
    decade = 10.0 ** np.floor(np.log10(longest))
    length = max(factor for factor in (1.0, 2.0, 5.0) if factor * decade <= longest) * decade
    bar = AuxTransformBox(axis.transData)
    bar.add_artist(Line2D([0.0, length / kpc_per_pixel], [0.0, 0.0], color="black", linewidth=1.6))
    label = TextArea(f"{length:g} kpc", textprops=dict(size=9))
    box = AnchoredOffsetbox(loc="lower center", frameon=True, pad=0.25, borderpad=0.5, zorder=11,
                            child=VPacker(children=[label, bar], align="center", pad=0, sep=3))
    box.patch.set(facecolor="white", edgecolor="none", alpha=0.8)
    axis.add_artist(box)


def smooth_nan(image: np.ndarray, sigma_pixels: float = 1.0) -> np.ndarray:
    """A map smoothed for its contours; blank pixels stay blank and do not spread."""
    valid = np.isfinite(image)
    weight = gaussian_filter(valid.astype(float), sigma_pixels)
    return gaussian_filter(np.nan_to_num(image), sigma_pixels) / np.where(valid, weight, np.nan)


def contour_step(span_kms: float, n_contours: int) -> float:
    """Spacing (km/s) that gives about ``n_contours`` iso-velocity contours over a span."""
    return next((step for step in CONTOUR_STEPS if step >= span_kms / n_contours),
                CONTOUR_STEPS[-1])


def velocity_levels(velocity: np.ndarray, vsys_kms: float, step_kms: float) -> List[float]:
    """Contour levels every ``step_kms`` from the systemic velocity, which is left out."""
    span = np.ceil(np.nanmax(np.abs(velocity - vsys_kms)) / step_kms)
    return [float(vsys_kms + k * step_kms) for k in np.arange(-span, span + 1) if k != 0]


def velocity_ticks(low: float, high: float) -> np.ndarray:
    """At most five round tick values between two velocities."""
    step = next((step for step in TICK_STEPS if (high - low) / step <= 4), TICK_STEPS[-1])
    return np.arange(np.ceil(low / step) * step, high + 1e-6, step)


def velocity_label(cube: DataCube) -> str:
    """Label of a velocity axis: radio or optical definition, as in the cube."""
    kind = "rad" if cube.header["CTYPE3"].upper().startswith("VRAD") else "opt"
    return rf"$v_{{{kind}}}$ (km s$^{{-1}}$)"


def cased_contours(axis, image: np.ndarray, levels: Sequence[float], colour: str, width: float,
                   casing: str, casing_width: float, zorder: float, **kwargs) -> None:
    """Contours with a casing: a line of one colour on a wider line of another.

    A cased line stays readable on any colour of the map below it.
    """
    drawn = len(axis.collections)
    axis.contour(image, levels=levels, colors=colour, linewidths=width, zorder=zorder, **kwargs)
    for collection in axis.collections[drawn:]:
        collection.set_path_effects([patheffects.Stroke(linewidth=casing_width, foreground=casing),
                                     patheffects.Normal()])


def draw_outline(axis, footprint: np.ndarray, width: float = 1.3) -> None:
    """Edge of the mask of a component: black on white."""
    cased_contours(axis, footprint.astype(float), [0.5], "black", width, "white", width + 1.7,
                   zorder=6)


def sky_line(cube: DataCube, ra_deg: float, dec_deg: float, position_angle_deg: float,
             offsets_arcsec: Sequence[float], across_arcsec: float = 0.0
             ) -> Tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates of points on a line through a sky position.

    The offsets are positive towards the position angle (north through east);
    ``across_arcsec`` shifts the line sideways.
    """
    ra, dec = slice_positions(ra_deg, dec_deg, position_angle_deg, offsets_arcsec,
                              [across_arcsec])
    return cube.celestial_wcs.world_to_pixel_values(ra[0], dec[0])


def draw_ellipse(axis, cube: DataCube, centre: Tuple[float, float],
                 semi_axes_arcsec: Tuple[float, float], position_angle_deg: float,
                 colour: str = "black", linestyle: str = "--", width: float = 1.8) -> None:
    """An ellipse on the sky, on a white casing.

    ``centre`` is (RA, Dec) in degrees, ``semi_axes_arcsec`` the semi-major and
    the semi-minor axis; the major axis lies along the position angle.
    """
    angle = np.linspace(0.0, 2.0 * np.pi, 181)
    along = semi_axes_arcsec[0] * np.cos(angle)
    across = semi_axes_arcsec[1] * np.sin(angle)
    pa = np.radians(position_angle_deg)
    east = along * np.sin(pa) + across * np.cos(pa)
    north = along * np.cos(pa) - across * np.sin(pa)
    x, y = cube.celestial_wcs.world_to_pixel_values(
        centre[0] + east / 3600.0 / np.cos(np.radians(centre[1])), centre[1] + north / 3600.0)
    axis.plot(x, y, color="white", linewidth=width + 1.8, zorder=8)
    axis.plot(x, y, color=colour, linestyle=linestyle, linewidth=width, zorder=8.1)


def mark_position(axis, cube: DataCube, ra_deg: float, dec_deg: float, marker: str = "+",
                  size: float = 130.0, colour: str = "black") -> None:
    """A marker with a white halo at a sky position."""
    x, y = cube.world_to_pixel(ra_deg, dec_deg)
    axis.scatter([x], [y], marker=marker, s=size, linewidths=1.8, color=colour, zorder=9,
                 path_effects=HALO)


def panel_name(axis, name: str) -> None:
    """Name of the component in the upper right corner of a panel."""
    axis.text(0.97, 0.97, name, transform=axis.transAxes, ha="right", va="top", fontsize=13,
              fontweight="bold", zorder=13, bbox=LABEL_BOX)


def panel_note(axis, text: str, size: float = 9) -> None:
    """A note in the lower right corner of a panel."""
    axis.text(0.97, 0.03, text, transform=axis.transAxes, ha="right", va="bottom", fontsize=size,
              zorder=12, bbox=LABEL_BOX)


def colour_bar(axis, bounds: Sequence[float], cmap: Colormap, norm: Normalize, label: str,
               ticks: Optional[Sequence[float]] = None, mark: Optional[float] = None,
               label_size: float = 10):
    """A horizontal colour bar above a panel; ``bounds`` in units of the panel.

    ``ticks`` replaces the automatic ticks and ``mark`` draws a dashed line at
    one value, e.g. the systemic velocity. Returns the axes of the bar.
    """
    bar_axis = axis.inset_axes(bounds)
    bar = axis.figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=bar_axis,
                               orientation="horizontal")
    if ticks is not None:
        bar.set_ticks(list(ticks))
        bar.set_ticklabels([f"{tick:g}" for tick in ticks])
    bar_axis.xaxis.set_ticks_position("top")
    bar_axis.xaxis.set_label_position("top")
    bar_axis.tick_params(which="both", direction="in", bottom=False, pad=2, labelsize=9)
    bar_axis.tick_params(which="major", length=3.5)
    bar_axis.tick_params(which="minor", length=2)
    if mark is not None:
        bar_axis.axvline(mark, color="black", linewidth=1.4, linestyle="--")
    bar_axis.set_xlabel(label, fontsize=label_size, fontweight="bold", labelpad=4)
    return bar_axis


def velocity_bar(axis, cube: DataCube, cmap: Colormap, norm: Normalize,
                 vsys_kms: Optional[float] = None) -> None:
    """Colour bar of a velocity field over the full width of a panel."""
    if vsys_kms is not None and not norm.vmin < vsys_kms < norm.vmax:
        vsys_kms = None
    bar_axis = colour_bar(axis, [0.0, 1.045, 1.0, 0.04], cmap, norm, velocity_label(cube),
                          ticks=velocity_ticks(norm.vmin, norm.vmax), mark=vsys_kms, label_size=11)
    bar_axis.xaxis.set_minor_locator(AutoMinorLocator(5))


def split_bars(axis, norm: Normalize, has_disc: bool, unit: str = "", short: bool = False,
               height_above: float = 1.045) -> None:
    """Logarithmic colour bars of the rotating disc and of the anomalous gas above a panel.

    A component without rotating disc has the bar of the anomalous gas only.
    ``unit`` is added to the labels; ``short`` shortens them for narrow panels.
    """
    ticks = [tick for tick in LOG_TICKS if norm.vmin <= tick <= norm.vmax] or None
    parts = [(ANOMALOUS_CMAP, "anomalous gas", [0.0, height_above, 1.0, 0.04])]
    if has_disc:
        names = ("disc", "anomalous") if short else ("rotating disc", "anomalous gas")
        parts = [(DISC_CMAP, names[0], [0.0, height_above, 0.47, 0.04]),
                 (ANOMALOUS_CMAP, names[1], [0.53, height_above, 0.47, 0.04])]
    for cmap, name, bounds in parts:
        colour_bar(axis, bounds, cmap, norm, name + unit, ticks=ticks).minorticks_off()
