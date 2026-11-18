"""Position-velocity diagrams along a straight line on the sky."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates


def slice_positions(ra_deg: float, dec_deg: float, position_angle_deg: float,
                    along_arcsec: np.ndarray, across_arcsec: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Sky positions of the points of a slice.

    Parameters
    ----------
    ra_deg, dec_deg : float
        Centre of the slice.
    position_angle_deg : float
        Direction of positive offsets along the slice, north to east.
    along_arcsec, across_arcsec : arrays
        Offsets along the slice and perpendicular to it.

    Returns
    -------
    ra, dec : 2-D arrays (across, along) in degrees
    """
    angle = np.radians(position_angle_deg)
    along, across = np.meshgrid(np.asarray(along_arcsec, float), np.asarray(across_arcsec, float))
    east = along * np.sin(angle) + across * np.cos(angle)
    north = along * np.cos(angle) - across * np.sin(angle)
    return (ra_deg + east / 3600.0 / np.cos(np.radians(dec_deg)), dec_deg + north / 3600.0)


def position_velocity(data: np.ndarray, celestial_wcs: WCS, ra_deg: float, dec_deg: float,
                      position_angle_deg: float, offsets_arcsec: np.ndarray,
                      width_arcsec: float = 0.0, pixel_arcsec: float = 1.0) -> np.ndarray:
    """Extract a position-velocity diagram from a cube.

    Parameters
    ----------
    data : 3-D array (channel, y, x)
        The cube; blank voxels count as zero.
    celestial_wcs : astropy.wcs.WCS
        Celestial coordinates of the cube.
    ra_deg, dec_deg, position_angle_deg : float
        Centre and direction of the slice; offsets are positive towards the
        position angle (north to east).
    offsets_arcsec : array
        Offsets along the slice at which the cube is sampled.
    width_arcsec : float
        Width of the slice; the cube is averaged across it in steps of one pixel.
    pixel_arcsec : float
        Pixel size of the cube, the step of the average across the slice.

    Returns
    -------
    2-D array (channel, offset)
        Mean intensity across the slice, linearly interpolated between pixels;
        zero outside the cube.
    """
    n_across = max(int(round(width_arcsec / pixel_arcsec)), 0) + 1
    across = (np.arange(n_across) - 0.5 * (n_across - 1)) * pixel_arcsec
    ra, dec = slice_positions(ra_deg, dec_deg, position_angle_deg, offsets_arcsec, across)
    x, y = celestial_wcs.world_to_pixel_values(ra, dec)
    cube = np.nan_to_num(data)
    diagram = np.empty((cube.shape[0], len(offsets_arcsec)))
    for channel in range(cube.shape[0]):
        sampled = map_coordinates(cube[channel], [y, x], order=1, mode="constant", cval=0.0)
        diagram[channel] = sampled.mean(axis=0)
    return diagram
