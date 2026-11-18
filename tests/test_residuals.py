import numpy as np
import pytest
from astropy.io import fits

from anomalous_gas_extractor.kinematics import hi_mass, residuals
from anomalous_gas_extractor.kinematics.datacube import masked_moments

from conftest import CHANNEL_MS, DISC, VSYS_KMS

BOX = (slice(20, 62), slice(14, 54), slice(10, 50))


def write_model(cube, path, scale=1.0, shift_channels=0, reverse=True, pixel_scale=1.0):
    """A model that is the disc of the fixture itself, as a cut-out with a VELO axis."""
    own = np.where(cube.labels == 1, cube.data, 0.0)
    own = np.roll(own, shift_channels, axis=0)[BOX] * scale
    header = fits.Header()
    for key in ("CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CUNIT1", "CUNIT2", "BMAJ", "BMIN"):
        header[key] = cube.header[key]
    header["CDELT1"] = cube.header["CDELT1"] * pixel_scale
    header["CDELT2"] = cube.header["CDELT2"] * pixel_scale
    header["CRPIX1"] = cube.header["CRPIX1"] - BOX[2].start
    header["CRPIX2"] = cube.header["CRPIX2"] - BOX[1].start
    velocity = cube.velocity_kms[BOX[0]] * 1e3
    if reverse:
        own, velocity = own[::-1], velocity[::-1]
    header["CTYPE3"], header["CRPIX3"], header["CRVAL3"] = "VELO", 1, float(velocity[0])
    header["CDELT3"] = float(velocity[1] - velocity[0])
    fits.writeto(path, own.astype(np.float32), header, overwrite=True)
    return str(path)


def test_masked_moments_of_a_single_line():
    velocity = np.array([10.0, 20.0, 30.0, 40.0])
    data = np.zeros((4, 1, 2))
    data[1:3, 0, 0] = [2.0, 2.0]
    data[:, 0, 1] = [1.0, -5.0, 0.0, 0.0]
    intensity, mean, dispersion = masked_moments(data, np.ones_like(data, bool), velocity, np.full(4, 10.0))
    assert intensity[0].tolist() == [40.0, -40.0]
    assert mean[0].tolist() == [25.0, 10.0]                  # the negative voxel has no weight
    assert dispersion[0].tolist() == [5.0, 0.0]
    intensity, mean, _ = masked_moments(data, np.zeros_like(data, bool), velocity, np.full(4, 10.0))
    assert intensity[0, 0] == 0.0 and np.isnan(mean[0, 0])


@pytest.mark.parametrize("reverse", [True, False])
def test_model_is_placed_on_the_data_grid(cube, tmp_path, reverse):
    box, model = residuals.align_model(cube, write_model(cube, tmp_path / "model.fits", reverse=reverse))
    assert box == BOX
    assert model == pytest.approx(np.where(cube.labels == 1, cube.data, 0.0)[BOX])


def test_models_on_another_grid_are_refused(cube, tmp_path):
    with pytest.raises(ValueError, match="pixel grid"):
        residuals.align_model(cube, write_model(cube, tmp_path / "model.fits", pixel_scale=1.3))


def test_perfect_model_leaves_no_residual(cube, tmp_path):
    maps = residuals.residual_moments(cube, 1, write_model(cube, tmp_path / "model.fits"))
    assert maps.compared.sum() > 200
    assert np.abs(maps.intensity).max() < 1e-6
    assert np.nanmax(np.abs(maps.velocity)) < 1e-6
    assert np.nanmax(np.abs(maps.dispersion)) < 1e-6
    assert maps.footprint[int(DISC["y"]), int(DISC["x"])] and not maps.footprint[0, 0]
    assert np.isnan(maps.velocity[0, 0])


def test_residuals_measure_the_difference(cube, tmp_path):
    # a model with half the flux and shifted by two channels
    model = write_model(cube, tmp_path / "model.fits", scale=0.5, shift_channels=2)
    maps = residuals.residual_moments(cube, 1, model)
    inside = maps.compared
    assert maps.intensity[inside] == pytest.approx(0.5 * maps.data_intensity[inside], rel=0.25)
    # the common mask cuts the part of the shifted model that leaves it: a little less than 10 km/s
    assert -2 * CHANNEL_MS / 1e3 - 0.5 < np.nanmedian(maps.velocity) < -6.0
    assert np.nanmedian(maps.data_velocity) == pytest.approx(VSYS_KMS, abs=10.0)


def test_surface_density():
    # 1 Jy/beam km/s in a 10 x 10 arcsec beam is 1.1055e22 cm^-2 = 88.5 M_sun pc^-2
    assert hi_mass.surface_density(1.0, 10.0, 10.0) == pytest.approx(88.5, rel=1e-3)
