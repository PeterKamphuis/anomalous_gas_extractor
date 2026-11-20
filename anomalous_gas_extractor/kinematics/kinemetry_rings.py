"""Harmonic expansion of a velocity field along ellipses with Kinemetry.

Kinemetry (Krajnovic et al. 2006, MNRAS 366, 787) is not distributed through
PyPI. Download the Python version from the web page of its author and either
install it in the Python environment or give the directory that holds
``kinemetry.py`` with :func:`set_kinemetry_path`.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import logging
import sys
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)

# settings of the ellipse fit that are not exposed to the user
N_HARMONIC_TERMS = 6
AXIS_RATIO_RANGE = (0.3, 1.0)
N_POSITION_ANGLES = 21
N_AXIS_RATIOS = 11
MINIMUM_RING_COVERAGE = 0.5
BOX_PADDING_PIXELS = 3
# innermost rings that the ring measurements may leave out when the centre is blanked
SKIPPED_INNER_RINGS = 3

_kinemetry_path: Optional[str] = None


def set_kinemetry_path(path: Optional[str]) -> None:
    """Directory that holds ``kinemetry.py``; None if the module is installed."""
    global _kinemetry_path
    _kinemetry_path = path
    if path and path not in sys.path:
        sys.path.insert(0, path)


def get_kinemetry_path() -> Optional[str]:
    """Directory given with :func:`set_kinemetry_path` (for worker processes)."""
    return _kinemetry_path


def load_kinemetry():
    """Import the Kinemetry module with a helpful message if it is missing."""
    try:
        return importlib.import_module("kinemetry")
    except ImportError as error:
        raise ImportError(
            "The Kinemetry code of Krajnovic et al. (2006) is required. Install it in "
            "the Python environment or set kinematics.kinemetry_path to the directory "
            "that holds kinemetry.py.") from error


@dataclass
class RingFits:
    """Result of the ellipse fits; one entry per ring.

    Attributes
    ----------
    radius : semi-major axis (pixels)
    position_angle : position angle of the major axis, north to east (degrees)
    axis_ratio : flattening q = b / a of the ellipse
    coefficients : harmonic coefficients in km/s: a0, then the sine and cosine coefficient of
        each fitted order (the odd orders 1, 3, 5 if the centre is fixed)
    x_centre, y_centre : centre of the ellipse (pixels)
    """

    radius: np.ndarray
    position_angle: np.ndarray
    axis_ratio: np.ndarray
    coefficients: np.ndarray
    x_centre: np.ndarray
    y_centre: np.ndarray

    def __len__(self) -> int:
        return len(self.radius)

    @property
    def systemic(self) -> np.ndarray:
        """Zeroth-order term a0: the mean velocity of the ring."""
        return self.coefficients[:, 0]

    @property
    def k1(self) -> np.ndarray:
        """Amplitude of the first-order term: the rotation."""
        return np.hypot(self.coefficients[:, 1], self.coefficients[:, 2])

    @property
    def k5(self) -> np.ndarray:
        """Amplitude of the fifth-order term: deviations from simple rotation."""
        return np.hypot(self.coefficients[:, 5], self.coefficients[:, 6])


def _empty_fits() -> RingFits:
    empty = np.empty(0)
    return RingFits(empty, empty, empty, np.empty((0, 2 * N_HARMONIC_TERMS + 1)), empty, empty)


def _collect(result) -> RingFits:
    return RingFits(radius=np.atleast_1d(result.rad).astype(float),
                    position_angle=np.atleast_1d(result.pa).astype(float),
                    axis_ratio=np.atleast_1d(result.q).astype(float),
                    coefficients=np.atleast_2d(result.cf).astype(float),
                    x_centre=np.atleast_1d(result.xc).astype(float),
                    y_centre=np.atleast_1d(result.yc).astype(float))


def _kinemetry_input(velocity: np.ndarray) -> dict:
    """Pixel lists for Kinemetry on the bounding box of the velocity field.

    Blank pixels are filled with the nearest valid velocity and listed as bad
    pixels, so that nothing is interpolated across holes in the velocity field.
    """
    valid = np.isfinite(velocity)
    rows, columns = np.nonzero(valid)
    y_low = max(rows.min() - BOX_PADDING_PIXELS, 0)
    y_high = min(rows.max() + BOX_PADDING_PIXELS + 1, velocity.shape[0])
    x_low = max(columns.min() - BOX_PADDING_PIXELS, 0)
    x_high = min(columns.max() + BOX_PADDING_PIXELS + 1, velocity.shape[1])
    box = velocity[y_low:y_high, x_low:x_high]
    box_valid = valid[y_low:y_high, x_low:x_high]
    nearest = ndimage.distance_transform_edt(~box_valid, return_distances=False,
                                             return_indices=True)
    y_grid, x_grid = np.indices(box.shape)
    return dict(xbin=(x_grid + x_low).ravel().astype(float),
                ybin=(y_grid + y_low).ravel().astype(float),
                moment=box[tuple(nearest)].ravel().astype(float),
                badpix=np.nonzero(~box_valid.ravel())[0])


def _fit_leading_rings(run, radii: np.ndarray):
    """Kinemetry result for the longest leading part of ``radii`` that can be fitted.

    Returns None if not even the first two rings can be fitted.
    """
    try:
        return run(radii)
    except Exception as error:  # Kinemetry raises several unrelated error types
        logger.debug("Kinemetry failed on %d rings (%s); shortening the list", radii.size, error)

    lowest, highest, best = 2, radii.size - 1, None
    while lowest <= highest:
        middle = (lowest + highest) // 2
        try:
            best = run(radii[:middle])
            lowest = middle + 1
        except Exception:
            highest = middle - 1
    return best


def fit_rings(velocity: np.ndarray, x_centre: float, y_centre: float,
              radii: Sequence[float], fix_centre: bool = True,
              skip_inner_rings: int = 0) -> RingFits:
    """Fit ellipses with free position angle and flattening to a velocity field.

    Parameters
    ----------
    velocity : 2-D array
        Velocity field in km/s; NaN outside the field.
    x_centre, y_centre : float
        Centre of the ellipses (pixels); the starting value if ``fix_centre`` is
        False.
    radii : sequence of float
        Semi-major axes of the ellipses (pixels).
    fix_centre : bool
        Keep the centre fixed or fit it on every ring.
    skip_inner_rings : int
        Largest number of innermost rings that may be left out if Kinemetry
        cannot fit them (too few valid pixels on a blanked centre).

    Returns
    -------
    RingFits
        The fitted rings. Kinemetry stops with an error when a ring cannot be
        fitted. The list of radii is then cut before the first such ring, after
        leaving out up to ``skip_inner_rings`` innermost rings, so fewer rings
        than requested may be returned.
    """
    kinemetry = load_kinemetry()
    radii = np.asarray(radii, dtype=float)
    if radii.size == 0 or not np.isfinite(velocity).any():
        return _empty_fits()
    settings = dict(_kinemetry_input(velocity), x0=x_centre, y0=y_centre,
                    ntrm=N_HARMONIC_TERMS, rangeQ=list(AXIS_RATIO_RANGE),
                    npa=N_POSITION_ANGLES, nq=N_AXIS_RATIOS, cover=MINIMUM_RING_COVERAGE,
                    pad=1.5, fixcen=fix_centre, plot=False, verbose=False)

    def run(ring_radii):
        # Kinemetry prints its progress and warns about rings without pixels
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return kinemetry.kinemetry(radius=ring_radii, **settings)

    for first in range(min(skip_inner_rings + 1, radii.size)):
        fitted = _fit_leading_rings(run, radii[first:])
        if fitted is not None:
            if first:
                logger.debug("Kinemetry could not fit the %d innermost ring(s); left out", first)
            return _collect(fitted)
    return _empty_fits()


def _ellipse_pixels(shape: Tuple[int, int], x_centre: float, y_centre: float, radius: float,
                    position_angle: float, axis_ratio: float, n_points: int):
    """Pixels (x, y) under points spread evenly in angle along an ellipse, and which are on the map."""
    angle = np.radians(position_angle - 90.0)
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    along, across = radius * np.cos(theta), radius * np.sin(theta) * axis_ratio
    x = np.rint(x_centre + along * np.cos(angle) - across * np.sin(angle)).astype(int)
    y = np.rint(y_centre + along * np.sin(angle) + across * np.cos(angle)).astype(int)
    inside = (x >= 0) & (x < shape[1]) & (y >= 0) & (y < shape[0])
    return x, y, inside


def ellipse_coverage(valid: np.ndarray, x_centre: float, y_centre: float, radius: float,
                     position_angle: float, axis_ratio: float, n_points: int = 100) -> float:
    """Fraction of an ellipse that lies on valid pixels of the velocity field."""
    x, y, inside = _ellipse_pixels(valid.shape, x_centre, y_centre, radius, position_angle,
                                   axis_ratio, n_points)
    on_field = np.zeros(n_points, dtype=bool)
    on_field[inside] = valid[y[inside], x[inside]]
    return float(on_field.mean())


def ellipse_velocities(velocity: np.ndarray, x_centre: float, y_centre: float, radius: float,
                       position_angle: float, axis_ratio: float,
                       n_points: int = 100) -> np.ndarray:
    """Velocities of the valid pixels of the velocity field that lie under an ellipse."""
    x, y, inside = _ellipse_pixels(velocity.shape, x_centre, y_centre, radius, position_angle,
                                   axis_ratio, n_points)
    values = velocity[y[inside], x[inside]]
    return values[np.isfinite(values)]


def curve_amplitude(coefficients: np.ndarray, n_angles: int = 360) -> float:
    """Largest deviation from a0 of the velocity curve fitted to one ring.

    The coefficients are those of a fit with a fixed centre: a0, followed by the
    sine and the cosine coefficient of each odd order 1, 3, 5, ...
    """
    angle = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    pairs = np.asarray(coefficients[1:], dtype=float).reshape(-1, 2)
    curve = np.zeros(n_angles)
    for index, (sine, cosine) in enumerate(pairs):
        order = 2 * index + 1
        curve += sine * np.sin(order * angle) + cosine * np.cos(order * angle)
    return float(np.max(np.abs(curve)))
