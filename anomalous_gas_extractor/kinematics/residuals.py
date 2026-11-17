"""Residuals of the moment maps of a component: data minus tilted-ring model.

The model cube is placed on the grid of the data cube. One mask, the voxels of
the component, then selects the voxels of the data and of the model, and the
moment maps of the model are subtracted from those of the data. Using the same
mask for both makes sure that a residual is a difference between data and model
and not between two selections of voxels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from anomalous_gas_extractor.kinematics.datacube import DataCube, masked_moments, spectral_axis

# largest offset between the grids of model and data, in pixels and in channels
GRID_TOLERANCE = 0.05
CHANNEL_TOLERANCE = 0.5


@dataclass
class ResidualMaps:
    """Moment maps of data and model on the spatial grid of the data cube.

    Attributes
    ----------
    data_intensity, model_intensity : 2-D arrays
        Integrated intensity (Jy/beam km/s) of the voxels of the component;
        zero where the component has no voxels.
    data_velocity, model_velocity, data_dispersion, model_dispersion : 2-D arrays
        Mean velocity and dispersion (km/s); NaN outside ``compared``.
    footprint : 2-D boolean array
        Pixels with voxels of the component.
    compared : 2-D boolean array
        Pixels where the integrated intensity of both data and model exceeds
        the signal-to-noise limit, so that their velocities can be compared.
    """

    data_intensity: np.ndarray
    model_intensity: np.ndarray
    data_velocity: np.ndarray
    model_velocity: np.ndarray
    data_dispersion: np.ndarray
    model_dispersion: np.ndarray
    footprint: np.ndarray
    compared: np.ndarray

    @property
    def intensity(self) -> np.ndarray:
        """Data minus model integrated intensity (Jy/beam km/s)."""
        return self.data_intensity - self.model_intensity

    @property
    def velocity(self) -> np.ndarray:
        """Data minus model mean velocity (km/s)."""
        return self.data_velocity - self.model_velocity

    @property
    def dispersion(self) -> np.ndarray:
        """Data minus model velocity dispersion (km/s)."""
        return self.data_dispersion - self.model_dispersion


def align_model(cube: DataCube, model_file: str) -> Tuple[Tuple[slice, slice, slice], np.ndarray]:
    """Place a model cube on the grid of the data cube.

    The model has to be a cut-out of the data cube: the same pixels and the
    same channels, possibly in reverse order. Returns the position of the
    cut-out in the data cube and the model with its channels in the order of
    the data.
    """
    with fits.open(model_file) as hdul:
        header, model = hdul[0].header, np.nan_to_num(np.squeeze(hdul[0].data))
    ra, dec = WCS(header, naxis=2).pixel_to_world_values(0.0, 0.0)
    x, y = cube.celestial_wcs.world_to_pixel_values(ra, dec)
    x_low, y_low = int(np.rint(x)), int(np.rint(y))
    pixel_ratio = abs(header["CDELT2"]) * 3600.0 / cube.pixel_arcsec
    if max(abs(x - x_low), abs(y - y_low), abs(pixel_ratio - 1.0)) > GRID_TOLERANCE:
        raise ValueError(f"{model_file} is not on the pixel grid of the data cube")

    velocity, _ = spectral_axis(header, model.shape[0])
    channels = np.array([np.argmin(np.abs(cube.velocity_kms - value)) for value in velocity])
    offset = np.abs(cube.velocity_kms[channels] - velocity).max()
    step = np.diff(channels)
    if offset > CHANNEL_TOLERANCE * np.median(cube.channel_width_kms) or not np.all(np.abs(step) == 1):
        raise ValueError(f"{model_file} does not have the channels of the data cube")
    if channels[0] > channels[-1]:
        model, channels = model[::-1], channels[::-1]
    box = (slice(int(channels[0]), int(channels[-1]) + 1),
           slice(y_low, y_low + model.shape[1]), slice(x_low, x_low + model.shape[2]))
    if box[1].start < 0 or box[2].start < 0 or box[1].stop > cube.data.shape[1] \
            or box[2].stop > cube.data.shape[2]:
        raise ValueError(f"{model_file} extends beyond the data cube")
    return box, model


def residual_moments(cube: DataCube, label: int, model_file: str, snr: float = 3.0) -> ResidualMaps:
    """Moment maps of a component and of its model, selected with the same mask.

    Parameters
    ----------
    cube : DataCube
        Data cube with the mask of the components.
    label : int
        Label of the component; its voxels are the mask for data and model.
    model_file : str
        Model cube of the component, a cut-out of the data cube.
    snr : float
        Velocities and dispersions are compared where the integrated intensity
        of both data and model exceeds this many times the noise.
    """
    model_box, model = align_model(cube, model_file)
    own_box = cube.bounding_box(label)
    # smallest box that holds the component and the model
    box = tuple(slice(min(a.start, b.start), max(a.stop, b.stop)) for a, b in zip(own_box, model_box))
    model_on_grid = np.zeros(tuple(part.stop - part.start for part in box), dtype=np.float32)
    inner = tuple(slice(a.start - b.start, a.stop - b.start) for a, b in zip(model_box, box))
    model_on_grid[inner] = model

    mask = cube.labels[box] == label
    velocity, width = cube.velocity_kms[box[0]], cube.channel_width_kms[box[0]]
    data_maps = masked_moments(cube.data[box], mask, velocity, width)
    model_maps = masked_moments(model_on_grid, mask, velocity, width)
    channels = mask.sum(axis=0)
    limit = snr * cube.moment_noise(channels)
    compared = (channels > 0) & (data_maps[0] > limit) & (model_maps[0] > limit)

    shape, plane = cube.data.shape[1:], box[1:]

    def full_frame(values: np.ndarray, valid: np.ndarray, blank: float) -> np.ndarray:
        frame = np.full(shape, blank)
        frame[plane] = np.where(valid, values, blank)
        return frame

    footprint = np.zeros(shape, dtype=bool)
    footprint[plane] = channels > 0
    compared_frame = np.zeros(shape, dtype=bool)
    compared_frame[plane] = compared
    return ResidualMaps(
        data_intensity=full_frame(data_maps[0], channels > 0, 0.0),
        model_intensity=full_frame(model_maps[0], channels > 0, 0.0),
        data_velocity=full_frame(data_maps[1], compared, np.nan),
        model_velocity=full_frame(model_maps[1], compared, np.nan),
        data_dispersion=full_frame(data_maps[2], compared, np.nan),
        model_dispersion=full_frame(model_maps[2], compared, np.nan),
        footprint=footprint, compared=compared_frame)
