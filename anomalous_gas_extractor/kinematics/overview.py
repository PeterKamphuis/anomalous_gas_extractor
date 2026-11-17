"""Overview of all components: a map with the outline of every component and a legend."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from anomalous_gas_extractor.kinematics.datacube import DataCube
from anomalous_gas_extractor.kinematics.sky_axes import (LEGEND_STYLE, View, footprint_view,
                                                         show_view, sky_axes)

# outlines of the components, used in turn
COMPONENT_COLOURS = ("#0072B2", "#CC00CC", "#000000", "#6A3D9A", "#E6AB02", "#D55E00",
                     "#009E73", "#56B4E9")
# width of the overview in units of the extent of the emission
OVERVIEW_MARGIN = 1.12


def _component_colour(index: int) -> str:
    """Colour of the outline of the component with this number."""
    return COMPONENT_COLOURS[index % len(COMPONENT_COLOURS)]


def _emptiest_corner(covered: np.ndarray, view: View) -> str:
    """The corner of a view with the least emission, as the location of a legend."""
    rows, columns = np.nonzero(covered)
    upper, left = rows > view.y, columns < view.x
    corners = {"upper left": upper & left, "upper right": upper & ~left,
               "lower left": ~upper & left, "lower right": ~upper & ~left}
    return min(corners, key=lambda corner: corners[corner].sum())


def overview_panel(figure: Figure, rectangle: Sequence[float], cube: DataCube,
                   footprints: Sequence[np.ndarray], names: Sequence[str],
                   fills: Sequence[Tuple[np.ndarray, tuple]], title: str,
                   handles: Sequence[object] = ()):
    """Map of all components: coloured areas, the outline of every component and a legend.

    ``fills`` lists masks with the colour of each; they are painted in the given
    order. ``handles`` are legend entries placed before the names of the
    components. Returns the axes.
    """
    axis = sky_axes(figure, rectangle, cube)
    image = np.zeros(footprints[0].shape + (4,))
    for mask, colour in fills:
        image[mask] = to_rgba(colour)
    axis.imshow(image, origin="lower", interpolation="nearest", zorder=2)
    for index, footprint in enumerate(footprints):
        axis.contour(footprint.astype(float), levels=[0.5], colors=[_component_colour(index)],
                     linewidths=1.6, zorder=5)
    covered = np.any(footprints, axis=0)
    view = footprint_view(covered, 0.0)
    view = View(view.x, view.y, OVERVIEW_MARGIN * view.half_width)
    show_view(axis, view)
    handles = list(handles) + [Line2D([], [], color=_component_colour(index), linewidth=2.4,
                                      label=name) for index, name in enumerate(names)]
    axis.legend(handles=handles, loc=_emptiest_corner(covered, view), fontsize=9.5,
                **LEGEND_STYLE).set_zorder(12)
    axis.set_title(title, fontsize=11, fontweight="bold", pad=8)
    return axis
