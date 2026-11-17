import numpy as np
import pytest
from astropy.io import fits

from anomalous_gas_extractor.kinematics import datacube

from conftest import BEAM_PIXELS, DISC, NOISE, VSYS_KMS, disc_velocity_field


def test_components_are_named_after_the_mask_header(cube):
    assert cube.components == {"DISC": 1, "CLOUD": 2}


def test_requested_components_must_exist():
    labels = np.array([[[0, 1], [1, 3]]])
    assert datacube.component_labels(fits.Header(), labels) == {"component_1": 1, "component_3": 3}
    assert datacube.component_labels(fits.Header(), labels, {"a": 3}) == {"a": 3}
    with pytest.raises(ValueError):
        datacube.component_labels(fits.Header(), labels, {"a": 2})


def test_noise_and_beam(cube):
    assert cube.rms == pytest.approx(NOISE, rel=0.05)
    assert cube.pixel_arcsec == pytest.approx(5.0)
    assert cube.beam_fwhm_arcsec == pytest.approx(5.0 * BEAM_PIXELS)
    assert cube.beam_area_pixels == pytest.approx(1.1331 * BEAM_PIXELS ** 2, rel=1e-3)


def test_velocity_field_follows_the_disc(cube):
    maps = cube.moment_maps(1)
    expected, radius = disc_velocity_field(maps.velocity.shape, DISC["x"], DISC["y"],
                                           DISC["inclination"], DISC["position_angle"],
                                           DISC["vmax"])
    inside = radius < DISC["radius"] - 2
    assert maps.valid[inside].all()
    assert np.abs(maps.velocity - expected)[inside].max() < 2.0
    assert np.nanmedian(maps.dispersion[inside]) == pytest.approx(8.0, abs=1.5)
    assert not maps.valid[radius > DISC["radius"] + 1].any()


def test_moment_maps_of_unknown_label(cube):
    with pytest.raises(ValueError):
        cube.moment_maps(9)


@pytest.mark.parametrize("ctype", ["VOPT", "VRAD", "VELO"])
def test_velocity_axes(ctype):
    header = fits.Header()
    header["CTYPE3"], header["CRVAL3"], header["CDELT3"], header["CRPIX3"] = ctype, 3.0e6, 5.0e3, 1
    header["CUNIT3"], header["RESTFRQ"] = "m/s", datacube.HI_REST_FREQUENCY_HZ
    velocity, frequency = datacube.spectral_axis(header, 4)
    assert velocity == pytest.approx(3000.0 + 5.0 * np.arange(4))
    assert np.all(np.diff(frequency) < 0)
    assert frequency[0] < datacube.HI_REST_FREQUENCY_HZ


def test_frequency_axis_gives_optical_velocity():
    header = fits.Header()
    rest = datacube.HI_REST_FREQUENCY_HZ
    header["CTYPE3"], header["CRVAL3"], header["CDELT3"], header["CRPIX3"] = "FREQ", rest / 1.01, 1e4, 1
    header["CUNIT3"], header["RESTFRQ"] = "Hz", rest
    velocity, frequency = datacube.spectral_axis(header, 2)
    assert frequency[0] == pytest.approx(rest / 1.01)
    assert velocity[0] == pytest.approx(0.01 * datacube.SPEED_OF_LIGHT_KMS)


def test_names():
    assert datacube.normalise_name("hcg 16-a") == "HCG16A"
    assert datacube.safe_name("NGC 848") == "NGC_848"


def test_cube_and_mask_must_match(cube_files, tmp_path):
    cube_file, _ = cube_files
    wrong = str(tmp_path / "mask.fits")
    fits.writeto(wrong, np.zeros((2, 3, 4), dtype=np.int16))
    with pytest.raises(ValueError):
        datacube.read_data_cube(cube_file, wrong)


def test_systemic_velocity_of_the_fixture(cube):
    assert cube.velocity_kms[cube.velocity_kms.size // 2] == pytest.approx(VSYS_KMS)
