"""Synthetic data shared by the tests.

Tests that need Kinemetry are skipped unless it can be imported; the directory
of kinemetry.py can be given in the environment variable KINEMETRY_PATH.
"""

import os

import numpy as np
import pytest
from astropy.io import fits

from anomalous_gas_extractor.kinematics import kinemetry_rings
from anomalous_gas_extractor.kinematics.datacube import read_data_cube

PIXEL_DEG = 5.0 / 3600.0
BEAM_PIXELS = 3.0
CHANNEL_MS = 5000.0
VSYS_KMS = 1000.0
DISC = dict(x=30.0, y=34.0, radius=14.0, inclination=50.0, position_angle=30.0, vmax=120.0)
NOISE = 1e-4


def disc_velocity_field(shape, x, y, inclination, position_angle, vmax, vsys=VSYS_KMS, scale=4.0):
    """Velocity field and radius in the disc plane of an inclined rotating disc."""
    rows, columns = np.indices(shape)
    angle = np.radians(position_angle)
    east, north = -(columns - x), rows - y
    along = east * np.sin(angle) + north * np.cos(angle)
    across = (east * np.cos(angle) - north * np.sin(angle)) / np.cos(np.radians(inclination))
    radius = np.hypot(along, across)
    rotation = vmax * (2.0 / np.pi) * np.arctan(radius / scale)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_theta = np.where(radius > 0, along / radius, 0.0)
    return vsys + rotation * cos_theta * np.sin(np.radians(inclination)), radius


def cube_header(shape):
    header = fits.Header()
    header["CTYPE1"], header["CTYPE2"], header["CTYPE3"] = "RA---SIN", "DEC--SIN", "VRAD"
    header["CRVAL1"], header["CRVAL2"], header["CRVAL3"] = 30.0, -10.0, VSYS_KMS * 1e3
    header["CRPIX1"], header["CRPIX2"], header["CRPIX3"] = shape[2] / 2, shape[1] / 2, shape[0] // 2 + 1
    header["CDELT1"], header["CDELT2"], header["CDELT3"] = -PIXEL_DEG, PIXEL_DEG, CHANNEL_MS
    header["CUNIT1"], header["CUNIT2"], header["CUNIT3"] = "deg", "deg", "m/s"
    header["BMAJ"] = header["BMIN"] = BEAM_PIXELS * PIXEL_DEG
    header["BPA"], header["BUNIT"], header["RESTFRQ"] = 0.0, "Jy/beam", 1420405751.768
    return header


@pytest.fixture(scope="session")
def cube_files(tmp_path_factory):
    """A cube with a rotating disc (label 1) and a cloud at one velocity (label 2)."""
    shape = (81, 64, 64)
    header = cube_header(shape)
    velocity = (VSYS_KMS * 1e3 + (np.arange(shape[0]) + 1 - header["CRPIX3"]) * CHANNEL_MS) / 1e3
    rng = np.random.default_rng(3)
    data = rng.normal(0.0, NOISE, shape).astype(np.float32)
    labels = np.zeros(shape, dtype=np.int16)

    field, radius = disc_velocity_field(shape[1:], DISC["x"], DISC["y"], DISC["inclination"],
                                        DISC["position_angle"], DISC["vmax"])
    line = 0.01 * np.exp(-0.5 * ((velocity[:, None, None] - field[None]) / 8.0) ** 2)
    line = line * (radius <= DISC["radius"])[None]
    data += line.astype(np.float32)
    labels[line > 5 * NOISE] = 1

    rows, columns = np.indices(shape[1:])
    cloud = np.hypot(columns - 54, rows - 10) <= 5
    line = 0.01 * np.exp(-0.5 * ((velocity[:, None, None] - (VSYS_KMS + 60.0)) / 8.0) ** 2)
    line = line * cloud[None]
    data += line.astype(np.float32)
    labels[line > 5 * NOISE] = 2

    directory = tmp_path_factory.mktemp("cube")
    cube_file, mask_file = str(directory / "cube.fits"), str(directory / "mask.fits")
    fits.writeto(cube_file, data, header)
    mask_header = header.copy()
    mask_header["LABEL1"], mask_header["LABEL2"] = "DISC", "CLOUD"
    fits.writeto(mask_file, labels, mask_header)
    return cube_file, mask_file


@pytest.fixture(scope="session")
def cube(cube_files):
    return read_data_cube(*cube_files)


@pytest.fixture(scope="session")
def kinemetry():
    """Skip the test if the Kinemetry code is not available."""
    kinemetry_rings.set_kinemetry_path(os.environ.get("KINEMETRY_PATH"))
    try:
        return kinemetry_rings.load_kinemetry()
    except ImportError:
        pytest.skip("Kinemetry is not installed")
