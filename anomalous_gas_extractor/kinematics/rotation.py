"""Does a component host a rotating disc?

The ring tests behind r_max are passed by any smooth velocity gradient, also by
a tidal stream. A component is therefore accepted as a rotating disc only if the
run of regular rings

1. is resolved: its radial extent is at least ``minimum_disc_beams`` beams,
2. has a rotation amplitude k1 that does not fall from its first to its last ring,
3. is supported by rotation: the largest k1 exceeds ``minimum_v_over_sigma``
   times the velocity dispersion of the gas inside r_max.

The dispersion is the median of the second moment inside r_max after removal of
the broadening by the velocity gradient across the beam (beam smearing).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from anomalous_gas_extractor.kinematics.datacube import MomentMaps
from anomalous_gas_extractor.kinematics.rmax import DiscExtent

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


@dataclass(frozen=True)
class RotationCriteria:
    """Limits a run of regular rings has to meet to count as a rotating disc."""

    minimum_disc_beams: float = 1.0
    minimum_v_over_sigma: float = 1.0


@dataclass
class RotationVerdict:
    """Measurements behind the classification of one component."""

    is_rotating_disc: bool = False
    disc_extent_beams: float = np.nan
    k1_first_kms: float = np.nan
    k1_last_kms: float = np.nan
    dispersion_kms: float = np.nan
    v_over_sigma: float = np.nan
    failed_tests: List[str] = field(default_factory=list)


def disc_dispersion(maps: MomentMaps, x_centre: float, y_centre: float, r_max_pixels: float,
                    beam_fwhm_pixels: float) -> float:
    """Median velocity dispersion inside r_max, corrected for beam smearing.

    A velocity gradient g (km/s per pixel) across a Gaussian beam of dispersion
    s adds g * s in quadrature to the observed line width.
    """
    rows, columns = np.indices(maps.velocity.shape)
    inside = np.hypot(columns - x_centre, rows - y_centre) <= r_max_pixels
    gradient_y, gradient_x = np.gradient(maps.velocity)
    smearing = np.hypot(gradient_x, gradient_y) * beam_fwhm_pixels * FWHM_TO_SIGMA
    corrected = np.sqrt(np.clip(maps.dispersion ** 2 - smearing ** 2, 0.0, None))
    values = corrected[inside & np.isfinite(corrected)]
    return float(np.median(values)) if values.size else np.nan


def classify_rotation(extent: DiscExtent, maps: MomentMaps, x_centre: float, y_centre: float,
                      pixel_arcsec: float, beam_fwhm_arcsec: float, ring_step_arcsec: float,
                      criteria: RotationCriteria) -> RotationVerdict:
    """Apply the three conditions for a rotating disc to a component."""
    verdict = RotationVerdict()
    if not extent.found:
        verdict.failed_tests.append("no regular rings")
        return verdict
    disc = extent.disc_rings
    # the first ring stands for the radii between the ring inside it and itself
    inner_edge = disc[0].radius_arcsec - ring_step_arcsec
    verdict.disc_extent_beams = (extent.r_max_arcsec - inner_edge) / beam_fwhm_arcsec
    verdict.k1_first_kms, verdict.k1_last_kms = disc[0].k1_kms, disc[-1].k1_kms
    verdict.dispersion_kms = disc_dispersion(maps, x_centre, y_centre,
                                             extent.r_max_arcsec / pixel_arcsec,
                                             beam_fwhm_arcsec / pixel_arcsec)
    if verdict.dispersion_kms > 0:
        verdict.v_over_sigma = extent.k1_max_kms / verdict.dispersion_kms
    elif verdict.dispersion_kms == 0:
        verdict.v_over_sigma = np.inf

    if verdict.disc_extent_beams < criteria.minimum_disc_beams:
        verdict.failed_tests.append("not resolved")
    if verdict.k1_last_kms < verdict.k1_first_kms:
        verdict.failed_tests.append("rotation amplitude falls outwards")
    if not verdict.v_over_sigma > criteria.minimum_v_over_sigma:
        verdict.failed_tests.append("not supported by rotation")
    verdict.is_rotating_disc = not verdict.failed_tests
    return verdict
