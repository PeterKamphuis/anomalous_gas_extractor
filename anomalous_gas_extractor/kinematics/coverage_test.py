"""Minimum fraction of a ring that must be sampled for a reliable ring fit.

The limit is measured on the data of each component. Rings that are completely
sampled and pass the k5/k1 and v_sys tests are the reference. Wedges of the
velocity field, seen from the centre, are blanked and Kinemetry is run again.
A trial is corrupted if the gap alone changes a0 or k5/k1 of a reference ring
by more than the limit of the corresponding ring test. The minimum coverage is
the lowest coverage, going down from complete rings, at which at most 5 per cent
of the trials are corrupted.
"""

from __future__ import annotations

import logging
import multiprocessing
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from anomalous_gas_extractor.kinematics import kinemetry_rings
from anomalous_gas_extractor.kinematics.rmax import Ring, RingCriteria

logger = logging.getLogger(__name__)

REFERENCE_COVERAGE = 0.95
REFERENCE_MINIMUM_RADIUS_PIXELS = 4.0
FIELD_MARGIN_PIXELS = 8
BLANKED_FRACTIONS = np.arange(0.05, 0.651, 0.05)
N_WEDGE_DIRECTIONS = 16
BIN_WIDTH_PERCENT = 10
BIN_STEP_PERCENT = 5
LOWEST_BIN_PERCENT = 30
MAXIMUM_CORRUPTED_FRACTION = 0.05
MINIMUM_TRIALS_PER_BIN = 30


@dataclass
class CoverageResult:
    """Outcome of the masking test of one component."""

    minimum_coverage: float
    complete_rings_required: bool
    n_trials: int
    # coverage of the bin centre, number of trials, corrupted fraction (None: too few trials)
    table: List[Tuple[float, int, Optional[float]]]


def wedge_mask(shape: Tuple[int, int], x_centre: float, y_centre: float,
               direction_deg: float, fraction: float) -> np.ndarray:
    """Pixels inside a wedge that covers ``fraction`` of the full circle."""
    rows, columns = np.indices(shape)
    angle = np.degrees(np.arctan2(rows - y_centre, columns - x_centre))
    offset = (angle - direction_deg + 180.0) % 360.0 - 180.0
    return np.abs(offset) < 180.0 * fraction


def _fit_trial(task) -> Dict[float, Tuple[float, float, float, float]]:
    """Ring fits of one masked velocity field: {radius: (a0, k1, k5/k1, coverage)}.

    A trial in which Kinemetry cannot fit the innermost ring gives no rings.
    """
    velocity, x_centre, y_centre, radii, direction, fraction = task
    if fraction > 0:
        velocity = velocity.copy()
        velocity[wedge_mask(velocity.shape, x_centre, y_centre, direction, fraction)] = np.nan
    try:
        fits = kinemetry_rings.fit_rings(velocity, x_centre, y_centre, radii)
    except Exception as error:  # a failed trial must not stop the test
        logger.debug("masking trial failed: %s", error)
        return {}
    valid = np.isfinite(velocity)
    result = {}
    for i in range(len(fits)):
        k1 = float(fits.k1[i])
        coverage = kinemetry_rings.ellipse_coverage(
            valid, x_centre, y_centre, fits.radius[i], fits.position_angle[i], fits.axis_ratio[i])
        result[round(float(fits.radius[i]), 3)] = (
            float(fits.systemic[i]), k1, float(fits.k5[i] / k1) if k1 > 0 else np.nan, coverage)
    return result


def corrupted_fraction_curve(coverage: np.ndarray, corrupted: np.ndarray
                             ) -> Tuple[List[Tuple[float, int, Optional[float]]], float, bool]:
    """Corrupted fraction in bins of coverage and the resulting minimum coverage.

    Returns the table (bin centre, trials, corrupted fraction), the minimum
    coverage and whether complete rings are required (even the smallest gaps
    corrupt too many trials).
    """
    percent = np.rint(100.0 * np.asarray(coverage)).astype(int)
    table = []
    for centre in range(int(100 * REFERENCE_COVERAGE), LOWEST_BIN_PERCENT - 1, -BIN_STEP_PERCENT):
        in_bin = (percent >= centre - BIN_WIDTH_PERCENT // 2) & (percent < centre + BIN_WIDTH_PERCENT // 2)
        n_trials = int(in_bin.sum())
        fraction = float(corrupted[in_bin].mean()) if n_trials >= MINIMUM_TRIALS_PER_BIN else None
        table.append((centre / 100.0, n_trials, fraction))
    minimum = None
    for centre, _, fraction in table:
        if fraction is None or fraction > MAXIMUM_CORRUPTED_FRACTION:
            break
        minimum = centre
    if minimum is None:
        return table, REFERENCE_COVERAGE, True
    return table, minimum, False


def minimum_coverage(velocity: np.ndarray, x_centre: float, y_centre: float, rings: List[Ring],
                     pixel_arcsec: float, criteria: RingCriteria,
                     n_processes: int = 1) -> Optional[CoverageResult]:
    """Run the masking test for one component.

    Parameters
    ----------
    velocity : 2-D array
        Velocity field (km/s) of the component.
    rings : list of Ring
        Rings measured around the same centre, with the verdicts of the tests.

    Returns
    -------
    CoverageResult or None
        None if the component has no completely sampled regular ring.
    """
    radii = np.array([ring.radius_arcsec / pixel_arcsec for ring in rings], dtype=float)
    is_reference = np.array([ring.pass_k5k1 and ring.pass_vsys
                             and ring.coverage >= REFERENCE_COVERAGE for ring in rings], dtype=bool)
    is_reference &= radii >= REFERENCE_MINIMUM_RADIUS_PIXELS - 1e-6
    if not is_reference.any():
        return None
    outermost = radii[is_reference].max()

    # only the disc is needed: blank the velocity field beyond the reference rings
    rows, columns = np.indices(velocity.shape)
    disc = velocity.copy()
    disc[np.hypot(columns - x_centre, rows - y_centre) > outermost + FIELD_MARGIN_PIXELS] = np.nan
    fit_radii = radii[radii <= outermost + 1e-6]

    tasks = [(disc, x_centre, y_centre, fit_radii, 0.0, 0.0)]
    tasks += [(disc, x_centre, y_centre, fit_radii, 360.0 * i / N_WEDGE_DIRECTIONS, float(fraction))
              for fraction in BLANKED_FRACTIONS for i in range(N_WEDGE_DIRECTIONS)]
    if n_processes > 1:
        with multiprocessing.Pool(n_processes, initializer=kinemetry_rings.set_kinemetry_path,
                                  initargs=(kinemetry_rings.get_kinemetry_path(),)) as pool:
            fitted = pool.map(_fit_trial, tasks, chunksize=4)
    else:
        fitted = [_fit_trial(task) for task in tasks]
    complete, trials = fitted[0], fitted[1:]

    coverage, corrupted = [], []
    for trial in trials:
        for radius in np.round(radii[is_reference], 3):
            if radius not in trial or radius not in complete:
                continue
            vsys, _, k5k1, covered = trial[radius]
            vsys_full, k1_full, k5k1_full, _ = complete[radius]
            coverage.append(covered)
            corrupted.append(not np.isfinite(k5k1)
                             or abs(vsys - vsys_full) > criteria.vsys_tolerance(k1_full)
                             or abs(k5k1 - k5k1_full) > criteria.k5k1_max)
    table, minimum, complete_required = corrupted_fraction_curve(
        np.array(coverage), np.array(corrupted, dtype=bool))
    return CoverageResult(minimum_coverage=minimum, complete_rings_required=complete_required,
                          n_trials=len(coverage), table=table)
