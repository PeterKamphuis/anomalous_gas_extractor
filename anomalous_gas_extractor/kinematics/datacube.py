"""Data cube, mask of the deblended components and their moment maps."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property
from typing import Dict, Optional, Tuple

import numpy as np
from astropy import constants
from astropy.io import fits
from astropy.wcs import WCS
from scipy import ndimage

HI_REST_FREQUENCY_HZ = 1420405751.768
SPEED_OF_LIGHT_KMS = constants.c.to_value("km/s")
MAD_TO_SIGMA = 1.4826
# area of a Gaussian beam in units of FWHM_major x FWHM_minor
GAUSSIAN_BEAM_AREA = np.pi / (4.0 * np.log(2.0))


@dataclass
class MomentMaps:
    """Moment maps of one component on the full spatial grid of the cube.

    Attributes
    ----------
    intensity : 2-D array
        Integrated intensity of all voxels of the component (Jy/beam km/s).
    velocity : 2-D array
        Intensity-weighted mean velocity (km/s); NaN where the integrated
        intensity is below the signal-to-noise limit.
    dispersion : 2-D array
        Intensity-weighted velocity dispersion (km/s), blanked like ``velocity``.
    """

    intensity: np.ndarray
    velocity: np.ndarray
    dispersion: np.ndarray

    @property
    def valid(self) -> np.ndarray:
        """Pixels that belong to the velocity field."""
        return np.isfinite(self.velocity)


@dataclass
class DataCube:
    """A spectral line cube with the label mask of its components.

    Attributes
    ----------
    data : 3-D array (channel, y, x)
        Intensities in Jy/beam; blank voxels are NaN.
    labels : 3-D integer array
        0 for voxels without emission, otherwise the label of the component.
    header : astropy.io.fits.Header
        Header of the data cube.
    components : dict
        Name of each component and its label in ``labels``.
    velocity_kms, frequency_hz : 1-D arrays
        Velocity (optical definition unless the cube is in radio velocities)
        and frequency of every channel.
    rms : float
        Noise of the cube outside the mask (Jy/beam).
    """

    data: np.ndarray
    labels: np.ndarray
    header: fits.Header
    components: Dict[str, int]
    velocity_kms: np.ndarray
    frequency_hz: np.ndarray
    rms: float

    @property
    def celestial_wcs(self) -> WCS:
        return WCS(self.header, naxis=2)

    @property
    def pixel_arcsec(self) -> float:
        return abs(self.header["CDELT2"]) * 3600.0

    @property
    def beam_major_arcsec(self) -> float:
        return self.header["BMAJ"] * 3600.0

    @property
    def beam_minor_arcsec(self) -> float:
        return self.header["BMIN"] * 3600.0

    @property
    def beam_fwhm_arcsec(self) -> float:
        """Geometric mean of the major and minor axis of the beam."""
        return float(np.sqrt(self.beam_major_arcsec * self.beam_minor_arcsec))

    @property
    def beam_area_pixels(self) -> float:
        pixel_area = abs(self.header["CDELT1"] * self.header["CDELT2"]) * 3600.0 ** 2
        return GAUSSIAN_BEAM_AREA * self.beam_major_arcsec * self.beam_minor_arcsec / pixel_area

    @property
    def channel_width_kms(self) -> np.ndarray:
        return np.abs(np.gradient(self.velocity_kms))

    @property
    def channel_width_hz(self) -> np.ndarray:
        return np.abs(np.gradient(self.frequency_hz))

    def world_to_pixel(self, ra_deg: float, dec_deg: float) -> Tuple[float, float]:
        """Pixel coordinates (x, y), counted from zero, of a sky position."""
        x, y = self.celestial_wcs.world_to_pixel_values(ra_deg, dec_deg)
        return float(x), float(y)

    def pixel_to_world(self, x: float, y: float) -> Tuple[float, float]:
        """Right ascension and declination (degrees) of a pixel position."""
        ra, dec = self.celestial_wcs.pixel_to_world_values(x, y)
        return float(ra), float(dec)

    @cached_property
    def _bounding_boxes(self) -> list:
        return ndimage.find_objects(self.labels)

    def bounding_box(self, label: int) -> Tuple[slice, slice, slice]:
        """Smallest sub-cube that holds all voxels of a component."""
        boxes = self._bounding_boxes
        if not 1 <= label <= len(boxes) or boxes[label - 1] is None:
            raise ValueError(f"label {label} does not occur in the mask")
        return boxes[label - 1]

    def moment_maps(self, label: int, snr: float = 3.0) -> MomentMaps:
        """Moment maps of the voxels of one component.

        The velocity and dispersion use the positive voxels only. A pixel is
        part of the velocity field if its integrated intensity exceeds ``snr``
        times the noise expected for the number of channels in the mask.
        """
        box = self.bounding_box(label)
        own = self.labels[box] == label
        intensity, mean, dispersion = masked_moments(
            self.data[box], own, self.velocity_kms[box[0]], self.channel_width_kms[box[0]])

        channels = own.sum(axis=0)
        keep = (channels > 0) & (intensity > snr * self.moment_noise(channels)) & np.isfinite(mean)

        shape = self.data.shape[1:]
        maps = MomentMaps(np.zeros(shape), np.full(shape, np.nan), np.full(shape, np.nan))
        plane = box[1:]
        maps.intensity[plane] = np.where(channels > 0, intensity, 0.0)
        maps.velocity[plane] = np.where(keep, mean, np.nan)
        maps.dispersion[plane] = np.where(keep, dispersion, np.nan)
        return maps

    def moment_noise(self, n_channels: np.ndarray) -> np.ndarray:
        """Noise of an integrated intensity summed over ``n_channels`` channels (Jy/beam km/s)."""
        return self.rms * np.median(self.channel_width_kms) * np.sqrt(np.maximum(n_channels, 1))


def masked_moments(data: np.ndarray, mask: np.ndarray, velocity_kms: np.ndarray,
                   width_kms: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Moment maps of the voxels of a cube that a mask selects.

    Parameters
    ----------
    data : 3-D array (channel, y, x)
        Intensities; blank voxels count as zero.
    mask : 3-D boolean array
        Voxels that enter the moments.
    velocity_kms, width_kms : 1-D arrays
        Velocity and width of the channels.

    Returns
    -------
    intensity, velocity, dispersion : 2-D arrays
        Integrated intensity (unit of the data times km/s), intensity-weighted
        mean velocity and velocity dispersion (km/s). The last two use the
        positive voxels only and are NaN where there are none.
    """
    velocity = np.asarray(velocity_kms)[:, None, None]
    weighted = np.where(mask, np.nan_to_num(data), 0.0) * np.asarray(width_kms)[:, None, None]
    intensity = weighted.sum(axis=0)
    positive = np.clip(weighted, 0.0, None)
    norm = positive.sum(axis=0)
    norm = np.where(norm > 0, norm, np.nan)
    mean = (positive * velocity).sum(axis=0) / norm
    variance = (positive * (velocity - mean[None]) ** 2).sum(axis=0) / norm
    return intensity, mean, np.sqrt(variance)


def spectral_axis(header: fits.Header, n_channels: int) -> Tuple[np.ndarray, np.ndarray]:
    """Velocity (km/s) and frequency (Hz) of every channel of a cube.

    Frequency, optical velocity (VOPT, FELO) and radio velocity (VRAD) axes are
    supported; a plain VELO axis is read as optical velocity, the convention of
    the cubes written for and by pyFAT.
    """
    rest = header.get("RESTFRQ", header.get("RESTFREQ", HI_REST_FREQUENCY_HZ))
    values = WCS(header, naxis=3).spectral.pixel_to_world_values(np.arange(n_channels))
    values = np.asarray(values, dtype=float)
    kind = header["CTYPE3"].upper()[:4]
    if kind == "FREQ":
        frequency = values
        velocity = SPEED_OF_LIGHT_KMS * (rest / frequency - 1.0)
    elif kind in ("VOPT", "FELO", "VELO"):
        velocity = values / 1e3
        frequency = rest / (1.0 + velocity / SPEED_OF_LIGHT_KMS)
    elif kind == "VRAD":
        velocity = values / 1e3
        frequency = rest * (1.0 - velocity / SPEED_OF_LIGHT_KMS)
    else:
        raise ValueError(f"unsupported spectral axis {header['CTYPE3']!r}")
    return velocity, frequency


def normalise_name(name: str) -> str:
    """Component name without blanks or punctuation, in upper case."""
    return re.sub(r"[^A-Z0-9]", "", name.upper())


def safe_name(name: str) -> str:
    """Component name in a form that can be used in file names."""
    return re.sub(r"[^A-Za-z0-9_+-]", "", name.replace(" ", "_"))


def component_labels(mask_header: fits.Header, labels: np.ndarray,
                     requested: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    """Name and label of the components in a mask.

    The names are taken, in this order, from ``requested`` (name: label), from
    header keywords ``LABEL<n> = name`` or are set to ``component_<n>``.
    """
    present = [int(value) for value in np.unique(labels) if value > 0]
    if requested:
        missing = [name for name, label in requested.items() if int(label) not in present]
        if missing:
            raise ValueError(f"components {missing} have no voxels in the mask")
        return {name: int(label) for name, label in requested.items()}
    named = {int(key[5:]): str(mask_header[key]).strip()
             for key in mask_header if re.fullmatch(r"LABEL\d+", key)}
    return {named.get(label, f"component_{label}"): label for label in present}


def noise_rms(data: np.ndarray, labels: np.ndarray) -> float:
    """Robust noise (1.4826 x MAD) of the voxels outside the mask."""
    off_source = data[np.isfinite(data) & (labels == 0)]
    if off_source.size == 0:
        raise ValueError("the mask leaves no emission-free voxels to measure the noise")
    return float(MAD_TO_SIGMA * np.median(np.abs(off_source - np.median(off_source))))


def read_data_cube(cube_file: str, mask_file: str,
                   components: Optional[Dict[str, int]] = None) -> DataCube:
    """Read a data cube and the label mask that was made on the same grid."""
    with fits.open(cube_file) as hdul:
        header = hdul[0].header.copy()
        data = np.squeeze(hdul[0].data).astype(np.float32)
    with fits.open(mask_file) as hdul:
        mask_header = hdul[0].header.copy()
        labels = np.squeeze(hdul[0].data)
    if data.ndim != 3:
        raise ValueError(f"{cube_file} is not a three-dimensional cube")
    if labels.shape != data.shape:
        raise ValueError(f"cube {data.shape} and mask {labels.shape} have different shapes")
    for key in ("BMAJ", "BMIN"):
        if key not in header:
            raise KeyError(f"{cube_file} has no {key} keyword")
    labels = np.nan_to_num(labels).astype(np.int32)
    velocity, frequency = spectral_axis(header, data.shape[0])
    return DataCube(data=data, labels=labels, header=header,
                    components=component_labels(mask_header, labels, components),
                    velocity_kms=velocity, frequency_hz=frequency,
                    rms=noise_rms(data, labels))
