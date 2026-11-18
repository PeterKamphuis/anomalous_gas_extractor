"""Where the position-velocity slice of a component lies on the sky.

The slice runs along the kinematic major axis through the centre of the
component:

rotating disc with a model     centre and position angle of the tilted-ring model,
rotating disc without a model  centre and position angle of the Kinemetry ellipses,
no rotating disc               through the kinematic centre, along a given position
                               angle or else along the velocity gradient of the gas
                               (:func:`kinematic_position_angle`).

Offsets along a slice are positive towards its position angle, which points to
the receding side. The slice covers the component along that line, and its
rotating disc, with one beam to spare at each end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from anomalous_gas_extractor.kinematics import hi_mass
from anomalous_gas_extractor.kinematics.datacube import DataCube
from anomalous_gas_extractor.kinematics.figure_data import Component
from anomalous_gas_extractor.kinematics.pv import slice_positions

COMPASS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
# length of a slice that does not cross the component, in beams on each side of the centre
EMPTY_SLICE_BEAMS = 3.0


@dataclass(frozen=True)
class Slice:
    """A position-velocity slice: a strip on the sky through a centre.

    Attributes
    ----------
    ra_deg, dec_deg : float
        Centre, the origin of the offsets.
    position_angle_deg : float
        Direction of positive offsets, north through east.
    first_arcsec, last_arcsec : float
        Offsets of the two ends.
    width_arcsec : float
        Width of the strip.
    source : str
        Where the position angle comes from.
    """

    ra_deg: float
    dec_deg: float
    position_angle_deg: float
    first_arcsec: float
    last_arcsec: float
    width_arcsec: float
    source: str

    @property
    def directions(self) -> Tuple[str, str]:
        """Compass directions of the negative and of the positive end."""
        return compass(self.position_angle_deg + 180.0), compass(self.position_angle_deg)


def compass(position_angle_deg: float) -> str:
    """Compass direction of a position angle (north through east)."""
    return COMPASS[int(round((position_angle_deg % 360.0) / 45.0)) % 8]


def kinematic_position_angle(cube: DataCube, label: int) -> float:
    """Position angle (degrees, north through east) of the velocity gradient of a component.

    A straight line is fitted through the flux-weighted centroids of the channel
    maps of the component, each weighted with the flux of its channel. The line
    is the direction in which the centroids are spread most; the position angle
    points along it to the side of the higher velocities, the receding side.
    Zero is returned for a component in fewer than two channels.
    """
    box = cube.bounding_box(label)
    own = np.where(cube.labels[box] == label, np.nan_to_num(cube.data[box]), 0.0).clip(min=0.0)
    flux = own.sum(axis=(1, 2))
    filled = flux > 0
    if filled.sum() < 2:
        return 0.0
    weights = flux[filled]
    reference = cube.pixel_to_world(box[2].start, box[1].start)
    centroids = np.array([(own[filled] * offsets[box[1:]]).sum(axis=(1, 2)) / weights
                          for offsets in hi_mass.sky_offsets(cube, *reference)])
    centroids -= np.average(centroids, axis=1, weights=weights)[:, None]
    _, directions = np.linalg.eigh(np.cov(centroids, aweights=weights))
    east, north = directions[:, -1]
    velocity = cube.velocity_kms[box[0]][filled]
    velocity = velocity - np.average(velocity, weights=weights)
    if np.sum(weights * velocity * (east * centroids[0] + north * centroids[1])) < 0:
        east, north = -east, -north
    return float(np.degrees(np.arctan2(east, north)) % 360.0)


def _towards(position_angle_deg: float, direction_deg: float) -> float:
    """A position angle turned by 180 degrees if it points away from a direction."""
    if np.cos(np.radians(position_angle_deg - direction_deg)) < 0:
        position_angle_deg += 180.0
    return position_angle_deg % 360.0


def _axis(cube: DataCube, component: Component,
          position_angle_deg: Optional[float]) -> Tuple[float, float, float, str]:
    """Centre, position angle and source of the position angle of the slice of a component."""
    if component.has_disc:
        geometry = component.geometry
        angle = geometry.position_angle_deg
        if not component.has_model:
            # the Kinemetry ellipses give the major axis, not its receding side
            angle = _towards(angle, kinematic_position_angle(cube, component.label))
        return geometry.ra_deg, geometry.dec_deg, angle % 360.0, component.disc_source
    centre = component.row["centre_ra_deg"], component.row["centre_dec_deg"]
    if position_angle_deg is not None:
        return (*centre, position_angle_deg % 360.0, "given")
    return (*centre, kinematic_position_angle(cube, component.label), "velocity gradient")


def component_slice(cube: DataCube, component: Component, width_arcsec: float,
                    position_angle_deg: Optional[float] = None) -> Slice:
    """The position-velocity slice of a component.

    ``position_angle_deg`` is used for a component without rotating disc
    instead of the position angle of its velocity gradient.
    """
    ra, dec, angle, source = _axis(cube, component, position_angle_deg)
    pixel, beam = cube.pixel_arcsec, cube.beam_fwhm_arcsec
    reach = np.hypot(*component.footprint.shape) * pixel
    offsets = np.arange(-reach, reach + pixel, pixel)
    ra_line, dec_line = slice_positions(ra, dec, angle, offsets, [0.0])
    x, y = cube.celestial_wcs.world_to_pixel_values(ra_line[0], dec_line[0])
    x, y = np.rint(x).astype(int), np.rint(y).astype(int)
    rows, columns = component.footprint.shape
    inside = (x >= 0) & (x < columns) & (y >= 0) & (y < rows)
    on_source = offsets[inside][component.footprint[y[inside], x[inside]]]
    first, last = -EMPTY_SLICE_BEAMS * beam, EMPTY_SLICE_BEAMS * beam
    if on_source.size:
        disc = component.geometry.radius_arcsec if component.has_disc else 0.0
        first = min(on_source.min(), -disc) - beam
        last = max(on_source.max(), disc) + beam
    return Slice(ra_deg=ra, dec_deg=dec, position_angle_deg=angle,
                 first_arcsec=float(np.floor(first / pixel) * pixel),
                 last_arcsec=float(np.ceil(last / pixel) * pixel),
                 width_arcsec=width_arcsec, source=source)
