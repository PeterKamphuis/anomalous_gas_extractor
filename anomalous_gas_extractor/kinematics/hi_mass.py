"""HI mass of a component inside and outside its rotating disc.

The disc region is the ellipse on the sky with the radius, inclination and
position angle of the tilted-ring model. Gas of the component inside that
ellipse counts as rotating disc, the rest as anomalous. A margin can be added
to the ellipse to include the emission that the beam smears beyond its edge.

HI masses follow Meyer et al. (2017, PASA 34, 52):
``M_HI = 49.7 D_L^2 S`` with the luminosity distance in Mpc and the flux
integrated over frequency in Jy Hz, which holds for any redshift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from astropy.io import fits

from anomalous_gas_extractor.kinematics.datacube import (GAUSSIAN_BEAM_AREA, SPEED_OF_LIGHT_KMS,
                                                         DataCube, spectral_axis)

# solar masses per Jy Hz at a distance of 1 Mpc
HI_MASS_PER_JY_HZ = 49.7
# M_sun pc^-2 of 1 Jy/beam km/s in a Gaussian beam of 1 arcsec^2 (product of the two FWHM):
# N_HI = 1.1055e24 cm^-2 and 1 M_sun pc^-2 = 1.249e20 cm^-2
SURFACE_DENSITY_FACTOR = 8.851e3


@dataclass(frozen=True)
class DiscGeometry:
    """Projected geometry of a rotating disc."""

    ra_deg: float
    dec_deg: float
    radius_arcsec: float
    inclination_deg: float
    position_angle_deg: float


@dataclass
class MassBudget:
    """Fluxes (Jy km/s) and HI masses (solar masses) of one component."""

    distance_mpc: float
    total_flux_jy_kms: float
    disc_flux_jy_kms: float
    total_mass: float
    disc_mass: float

    @property
    def anomalous_flux_jy_kms(self) -> float:
        return self.total_flux_jy_kms - self.disc_flux_jy_kms

    @property
    def anomalous_mass(self) -> float:
        return self.total_mass - self.disc_mass

    @property
    def disc_fraction(self) -> float:
        return self.disc_mass / self.total_mass if self.total_mass > 0 else np.nan


def surface_density(intensity_jy_beam_kms, beam_major_arcsec: float, beam_minor_arcsec: float):
    """HI surface density (solar masses per square parsec) of an integrated intensity.

    The value is along the line of sight: it is not corrected for inclination
    and does not include helium.
    """
    return SURFACE_DENSITY_FACTOR * intensity_jy_beam_kms / (beam_major_arcsec * beam_minor_arcsec)


def hi_mass(flux_jy_hz: float, distance_mpc: float) -> float:
    """HI mass in solar masses of a flux integrated over frequency."""
    return HI_MASS_PER_JY_HZ * distance_mpc ** 2 * flux_jy_hz


def hubble_distance(vsys_kms: float, hubble_constant: float) -> float:
    """Luminosity distance (Mpc) of a recession velocity cz in the Hubble flow.

    Valid at low redshift: ``D_L = (1 + z) cz / H0``.
    """
    redshift = vsys_kms / SPEED_OF_LIGHT_KMS
    return (1.0 + redshift) * vsys_kms / hubble_constant


def sky_offsets(cube: DataCube, ra_deg: float, dec_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """Offsets (arcsec) to the east and to the north of a position, for every pixel."""
    rows, columns = np.indices(cube.data.shape[1:])
    ra, dec = cube.celestial_wcs.pixel_to_world_values(columns, rows)
    east = ((ra - ra_deg + 180.0) % 360.0 - 180.0) * np.cos(np.radians(dec_deg)) * 3600.0
    north = (dec - dec_deg) * 3600.0
    return east, north


def disc_region(cube: DataCube, geometry: DiscGeometry, margin_arcsec: float = 0.0) -> np.ndarray:
    """Pixels inside the projected ellipse of a disc.

    ``margin_arcsec`` is added to both semi-axes of the ellipse.
    """
    east, north = sky_offsets(cube, geometry.ra_deg, geometry.dec_deg)
    angle = np.radians(geometry.position_angle_deg)
    along = east * np.sin(angle) + north * np.cos(angle)
    across = east * np.cos(angle) - north * np.sin(angle)
    major = geometry.radius_arcsec + margin_arcsec
    minor = geometry.radius_arcsec * np.cos(np.radians(geometry.inclination_deg)) + margin_arcsec
    return (along / major) ** 2 + (across / minor) ** 2 <= 1.0


def component_fluxes(cube: DataCube, label: int,
                     region: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Flux of a component, in Jy km/s and in Jy Hz.

    Only the voxels of the component on the pixels of ``region`` are summed;
    all its voxels if no region is given.
    """
    box = cube.bounding_box(label)
    own = cube.labels[box] == label
    if region is not None:
        own = own & region[box[1:]][None]
    spectrum = np.where(own, np.nan_to_num(cube.data[box]), 0.0).sum(axis=(1, 2))
    spectrum = spectrum / cube.beam_area_pixels
    return (float(np.sum(spectrum * cube.channel_width_kms[box[0]])),
            float(np.sum(spectrum * cube.channel_width_hz[box[0]])))


def mass_budget(cube: DataCube, label: int, distance_mpc: float,
                region: Optional[np.ndarray] = None) -> MassBudget:
    """HI mass of a component in total and on the pixels of the disc region.

    Without a region (no rotating disc) all the gas is anomalous.
    """
    total_kms, total_hz = component_fluxes(cube, label)
    disc_kms, disc_hz = component_fluxes(cube, label, region) if region is not None else (0.0, 0.0)
    return MassBudget(distance_mpc=distance_mpc,
                      total_flux_jy_kms=total_kms, disc_flux_jy_kms=disc_kms,
                      total_mass=hi_mass(total_hz, distance_mpc),
                      disc_mass=hi_mass(disc_hz, distance_mpc))


def model_cube_mass(model_file: str, distance_mpc: float) -> float:
    """HI mass of a model cube (Jy/beam) written by the tilted-ring fit."""
    with fits.open(model_file) as hdul:
        header = hdul[0].header
        data = np.nan_to_num(np.squeeze(hdul[0].data))
    _, frequency = spectral_axis(header, data.shape[0])
    width = np.abs(np.gradient(frequency))
    beam_area = (GAUSSIAN_BEAM_AREA * header["BMAJ"] * header["BMIN"]
                 / abs(header["CDELT1"] * header["CDELT2"]))
    flux = float(np.sum(data.sum(axis=(1, 2)) * width) / beam_area)
    return hi_mass(flux, distance_mpc)
