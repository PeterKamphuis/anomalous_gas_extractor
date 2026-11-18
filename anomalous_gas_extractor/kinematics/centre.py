"""Kinematic centre of the rotating disc of a component.

Kinemetry is run with a free centre on the rings of the rotating disc; a wrong
centre shows up as even harmonic terms, which the ellipse fit minimises
(Krajnovic et al. 2006, appendix B). The kinematic centre is the median of the
ring centres.

Where the rotation curve still rises linearly a shift of the centre is absorbed
by the systemic velocity, so the kinematics constrain the centre only where the
rotation curve bends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from anomalous_gas_extractor.kinematics.datacube import MAD_TO_SIGMA
from anomalous_gas_extractor.kinematics.kinemetry_rings import SKIPPED_INNER_RINGS, fit_rings

# the innermost rings are too poorly resolved to constrain the centre
INNERMOST_RING_BEAMS = 0.7


@dataclass
class CentreEstimate:
    """Kinematic centre in pixel coordinates and the scatter of the ring centres."""

    x: float
    y: float
    scatter_pixels: float
    n_rings: int


def kinematic_centre(velocity: np.ndarray, x_start: float, y_start: float,
                     outer_radius_pixels: float, beam_major_pixels: float,
                     ring_step_pixels: float) -> Optional[CentreEstimate]:
    """Median centre of the Kinemetry rings between 0.7 beam and the outer radius.

    Parameters
    ----------
    velocity : 2-D array
        Velocity field (km/s), NaN outside the field.
    x_start, y_start : float
        First guess of the centre (pixels).
    outer_radius_pixels : float
        Only rings of the rotating disc should be used: the radius of the last
        regular ring found with the first guess of the centre plus a margin.

    Returns
    -------
    CentreEstimate or None
        None if no ring could be fitted.
    """
    first = np.ceil(INNERMOST_RING_BEAMS * beam_major_pixels / ring_step_pixels) * ring_step_pixels
    radii = np.arange(first, outer_radius_pixels + 1e-6, ring_step_pixels)
    fits = fit_rings(velocity, x_start, y_start, radii, fix_centre=False,
                     skip_inner_rings=SKIPPED_INNER_RINGS)
    if len(fits) == 0:
        return None
    x, y = float(np.median(fits.x_centre)), float(np.median(fits.y_centre))
    scatter = MAD_TO_SIGMA * np.median(np.hypot(fits.x_centre - x, fits.y_centre - y))
    return CentreEstimate(x=x, y=y, scatter_pixels=float(scatter), n_rings=len(fits))
