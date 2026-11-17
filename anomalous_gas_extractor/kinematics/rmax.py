"""Outer radius of the regularly rotating disc (r_max) from Kinemetry.

Along the best-fitting ellipse of a rotating disc the line-of-sight velocity is
``a0 + k1 cos(theta)``. A ring is counted as part of the rotating disc if it
passes the tests selected with ``RingCriteria.tests``:

coverage  : the fraction of the ellipse on the velocity field is at least the
            minimum coverage,
k5k1      : the fifth-order term is at most ``k5k1_max`` of the rotation term,
vsys      : a0 differs from the systemic velocity of the innermost rings by at
            most ``max(vsys_drift_kms, vsys_drift_fraction * k1)``,
amplitude : the fit is constrained by the data: a0 lies within the velocities
            measured on the ring, and the fitted velocity curve deviates from a0
            by at most ``amplitude_ratio_max`` times the largest measured
            deviation. Where part of an ellipse has no data the fit is free
            there, and an unconstrained fit shows up as a curve that runs far
            beyond the measured velocities.

The tests coverage and amplitude both reject rings that the data do not
constrain: the first from the number of sampled points, the second from the
result of the fit.

The rotating disc is the first uninterrupted run of such rings; it has to start
within ``start_within_beams`` of the centre. r_max is the radius at which the
first test is crossed between the last ring of the run and the next ring, each
tested quantity being interpolated linearly between the two rings. The
amplitude ratio is not interpolated: if it is the only test that fails, r_max
is the radius of the last ring of the run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from anomalous_gas_extractor.kinematics.kinemetry_rings import (SKIPPED_INNER_RINGS,
                                                                curve_amplitude,
                                                                ellipse_coverage,
                                                                ellipse_velocities, fit_rings)

HALF_COVERED = 0.5
COVERAGE_TOLERANCE = 1e-9
RING_TESTS = ("coverage", "k5k1", "vsys", "amplitude")
DEFAULT_RING_TESTS = ("coverage", "k5k1", "vsys")
# tests that fail where the data end, as opposed to where the rotation stops being regular
SAMPLING_TESTS = ("coverage", "amplitude")
TEST_NAMES = {"coverage": "coverage", "k5k1": "k5/k1", "vsys": "v_sys drift",
              "amplitude": "unconstrained fit"}


@dataclass(frozen=True)
class RingCriteria:
    """The ring tests, their limits and the sampling of the rings."""

    k5k1_max: float = 0.10
    vsys_drift_kms: float = 15.0
    vsys_drift_fraction: float = 0.15
    amplitude_ratio_max: float = 1.25
    tests: Tuple[str, ...] = DEFAULT_RING_TESTS
    start_within_beams: float = 1.5
    n_reference_rings: int = 3
    ring_step_pixels: float = 2.0
    maximum_radius_arcsec: Optional[float] = None

    def __post_init__(self):
        unknown = [test for test in self.tests if test not in RING_TESTS]
        if unknown or not self.tests:
            raise ValueError(f"ring tests {list(self.tests)}: choose one or more of {list(RING_TESTS)}")

    def vsys_tolerance(self, k1_kms: float) -> float:
        """Allowed deviation of a0 from the systemic velocity (km/s)."""
        return max(self.vsys_drift_kms, self.vsys_drift_fraction * k1_kms)


@dataclass
class Ring:
    """Kinemetry measurements of one ring and the outcome of the ring tests.

    The verdict of every test is kept; ``regular`` is true if the ring passes
    the tests that were selected.
    """

    radius_arcsec: float
    position_angle_deg: float
    axis_ratio: float
    k1_kms: float
    k5_over_k1: float
    vsys_kms: float
    coverage: float
    amplitude_ratio: float = np.nan
    vsys_drift_kms: float = np.nan
    vsys_tolerance_kms: float = np.nan
    pass_coverage: bool = False
    pass_k5k1: bool = False
    pass_vsys: bool = False
    pass_amplitude: bool = False
    regular: bool = False


@dataclass
class DiscExtent:
    """The run of regular rings of a component and its outer radius."""

    rings: List[Ring] = field(default_factory=list)
    first_index: Optional[int] = None
    last_index: Optional[int] = None
    r_max_arcsec: Optional[float] = None
    is_lower_limit: bool = False
    stopped_by: str = ""
    vsys_reference_kms: float = np.nan
    minimum_coverage: float = np.nan

    @property
    def found(self) -> bool:
        return self.last_index is not None

    @property
    def disc_rings(self) -> List[Ring]:
        """Rings of the rotating disc."""
        if not self.found:
            return []
        return self.rings[self.first_index:self.last_index + 1]

    @property
    def position_angle_deg(self) -> Optional[float]:
        """Circular mean of the position angles of the disc rings."""
        if not self.found:
            return None
        angles = np.radians([ring.position_angle_deg for ring in self.disc_rings])
        return float(np.degrees(np.angle(np.mean(np.exp(1j * angles)))) % 360.0)

    @property
    def axis_ratio(self) -> Optional[float]:
        """Median flattening of the disc rings."""
        if not self.found:
            return None
        return float(np.median([ring.axis_ratio for ring in self.disc_rings]))

    @property
    def k1_max_kms(self) -> Optional[float]:
        """Largest rotation amplitude of the disc rings."""
        if not self.found:
            return None
        return float(max(ring.k1_kms for ring in self.disc_rings))


def amplitude_ratio(curve_amplitude_kms: float, mean_kms: float, velocities: np.ndarray) -> float:
    """Amplitude of the fitted velocity curve over the largest measured deviation from its mean.

    Parameters
    ----------
    curve_amplitude_kms : float
        Largest deviation of the fitted curve from a0, anywhere along the ellipse.
    mean_kms : float
        a0 of the fit.
    velocities : array
        Velocities measured on the ellipse.

    Returns
    -------
    float
        At most about one for a fit that the data constrain. Infinite if a0 lies
        outside the measured velocities or if nothing was measured.
    """
    velocities = velocities[np.isfinite(velocities)]
    if velocities.size == 0 or not velocities.min() <= mean_kms <= velocities.max():
        return np.inf
    largest = np.max(np.abs(velocities - mean_kms))
    return float(curve_amplitude_kms / largest) if largest > 0 else np.inf


def measure_rings(velocity: np.ndarray, x_centre: float, y_centre: float,
                  pixel_arcsec: float, criteria: RingCriteria) -> List[Ring]:
    """Run Kinemetry on rings of increasing radius around a fixed centre.

    The innermost rings are skipped until a circle of that radius is at least
    half covered by the velocity field (the centre can be blanked).
    """
    valid = np.isfinite(velocity)
    rows, columns = np.nonzero(valid)
    if rows.size == 0:
        return []
    largest = np.hypot(columns - x_centre, rows - y_centre).max()
    if criteria.maximum_radius_arcsec is not None:
        largest = min(largest, criteria.maximum_radius_arcsec / pixel_arcsec)
    radii = np.arange(criteria.ring_step_pixels, largest, criteria.ring_step_pixels)
    covered = [ellipse_coverage(valid, x_centre, y_centre, radius, 0.0, 1.0) >= HALF_COVERED
               for radius in radii]
    if not any(covered):
        return []
    fits = fit_rings(velocity, x_centre, y_centre, radii[covered.index(True):],
                     skip_inner_rings=SKIPPED_INNER_RINGS)

    rings = []
    for i in range(len(fits)):
        k1 = float(fits.k1[i])
        ellipse = (x_centre, y_centre, fits.radius[i], fits.position_angle[i], fits.axis_ratio[i])
        rings.append(Ring(
            radius_arcsec=float(fits.radius[i] * pixel_arcsec),
            position_angle_deg=float(fits.position_angle[i]),
            axis_ratio=float(fits.axis_ratio[i]),
            k1_kms=k1,
            k5_over_k1=float(fits.k5[i] / k1) if k1 > 0 else np.nan,
            vsys_kms=float(fits.systemic[i]),
            coverage=ellipse_coverage(valid, *ellipse),
            amplitude_ratio=amplitude_ratio(curve_amplitude(fits.coefficients[i]),
                                            float(fits.systemic[i]),
                                            ellipse_velocities(velocity, *ellipse))))
    return rings


def crossing_radius(inner_radius: float, outer_radius: float, inner_value: float,
                    outer_value: float, limit: float) -> float:
    """Radius between a passing and a failing ring at which a limit is reached.

    The tested value rises from ``inner_value`` (below the limit) to
    ``outer_value`` (above it); if it does not, the inner radius is returned.
    """
    if not (np.isfinite(inner_value) and np.isfinite(outer_value)) or outer_value <= inner_value:
        return inner_radius
    fraction = np.clip((limit - inner_value) / (outer_value - inner_value), 0.0, 1.0)
    return inner_radius + (outer_radius - inner_radius) * float(fraction)


def _verdicts(ring: Ring) -> dict:
    return {"coverage": ring.pass_coverage, "k5k1": ring.pass_k5k1, "vsys": ring.pass_vsys,
            "amplitude": ring.pass_amplitude}


def _apply_tests(rings: List[Ring], minimum_coverage: float, criteria: RingCriteria) -> float:
    """Fill in the verdict of every test and of the ring; returns the reference v_sys."""
    reference = float(np.median([ring.vsys_kms for ring in rings[:criteria.n_reference_rings]]))
    for ring in rings:
        ring.vsys_drift_kms = ring.vsys_kms - reference
        ring.vsys_tolerance_kms = criteria.vsys_tolerance(ring.k1_kms)
        ring.pass_coverage = bool(ring.coverage >= minimum_coverage - COVERAGE_TOLERANCE)
        ring.pass_k5k1 = bool(np.isfinite(ring.k5_over_k1)
                              and ring.k5_over_k1 <= criteria.k5k1_max)
        ring.pass_vsys = bool(abs(ring.vsys_drift_kms) <= ring.vsys_tolerance_kms)
        ring.pass_amplitude = bool(ring.amplitude_ratio <= criteria.amplitude_ratio_max)
        verdicts = _verdicts(ring)
        ring.regular = all(verdicts[test] for test in criteria.tests)
    return reference


def _end_of_run(last: Ring, failing: Ring, minimum_coverage: float, criteria: RingCriteria):
    """Radius at which the run ends and the selected tests that the next ring fails."""
    verdicts = _verdicts(failing)
    failed = [test for test in RING_TESTS if test in criteria.tests and not verdicts[test]]
    inner, outer = last.radius_arcsec, failing.radius_arcsec
    crossings = []
    if "coverage" in failed:
        crossings.append(crossing_radius(inner, outer, -last.coverage, -failing.coverage,
                                         -minimum_coverage))
    if "k5k1" in failed:
        crossings.append(crossing_radius(inner, outer, last.k5_over_k1, failing.k5_over_k1,
                                         criteria.k5k1_max))
    if "vsys" in failed:
        crossings.append(crossing_radius(
            inner, outer, abs(last.vsys_drift_kms) / last.vsys_tolerance_kms,
            abs(failing.vsys_drift_kms) / failing.vsys_tolerance_kms, 1.0))
    return (min(crossings) if crossings else inner), failed


def find_rmax(rings: List[Ring], minimum_coverage: float, beam_major_arcsec: float,
              criteria: RingCriteria) -> DiscExtent:
    """Apply the ring tests and locate the outer radius of the rotating disc.

    r_max is a lower limit if the run of regular rings ends only because the
    data end: no ring is left, or the next ring fails nothing but the tests on
    its sampling (coverage, amplitude).
    """
    extent = DiscExtent(rings=rings, minimum_coverage=minimum_coverage)
    if not rings:
        extent.stopped_by = "no ring on the velocity field"
        return extent
    extent.vsys_reference_kms = _apply_tests(rings, minimum_coverage, criteria)

    start_limit = criteria.start_within_beams * beam_major_arcsec + 1e-6
    first = next((i for i, ring in enumerate(rings)
                  if ring.regular and ring.radius_arcsec <= start_limit), None)
    if first is None:
        extent.stopped_by = f"no regular ring within {criteria.start_within_beams} beams"
        return extent
    last = first
    while last + 1 < len(rings) and rings[last + 1].regular:
        last += 1
    extent.first_index, extent.last_index = first, last

    if last + 1 == len(rings):
        extent.r_max_arcsec = rings[last].radius_arcsec
        extent.is_lower_limit = True
        extent.stopped_by = "end of the velocity field"
        return extent
    failing = rings[last + 1]
    extent.r_max_arcsec, failed = _end_of_run(rings[last], failing, minimum_coverage, criteria)
    extent.is_lower_limit = all(test in SAMPLING_TESTS for test in failed)
    extent.stopped_by = (" + ".join(TEST_NAMES[test] for test in failed)
                         + f" at {failing.radius_arcsec:.0f} arcsec")
    return extent
