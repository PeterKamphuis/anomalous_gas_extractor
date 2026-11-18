import numpy as np
import pytest

from anomalous_gas_extractor.kinematics import pv

from conftest import DISC, VSYS_KMS


def test_slice_positions_follow_the_position_angle():
    ra, dec = pv.slice_positions(30.0, 0.0, 90.0, np.array([0.0, 3600.0]), np.array([0.0]))
    assert ra[0].tolist() == pytest.approx([30.0, 31.0]) and dec[0].tolist() == pytest.approx([0.0, 0.0])
    ra, dec = pv.slice_positions(30.0, 0.0, 0.0, np.array([3600.0]), np.array([0.0, 3600.0]))
    assert dec[:, 0].tolist() == pytest.approx([1.0, 1.0])       # along the slice: north
    assert ra[:, 0].tolist() == pytest.approx([30.0, 31.0])      # across it: east


def test_rotation_shows_along_the_major_axis(cube):
    ra, dec = cube.pixel_to_world(DISC["x"], DISC["y"])
    offsets = np.arange(-60.0, 61.0, 5.0)
    diagram = pv.position_velocity(np.where(cube.labels == 1, cube.data, 0.0), cube.celestial_wcs,
                                   ra, dec, DISC["position_angle"], offsets,
                                   width_arcsec=15.0, pixel_arcsec=cube.pixel_arcsec)
    assert diagram.shape == (cube.data.shape[0], offsets.size)
    mean = (diagram * cube.velocity_kms[:, None]).sum(axis=0) / diagram.sum(axis=0)
    # positive offsets point to the position angle, the receding side of the fixture
    assert mean[offsets == 40.0][0] - VSYS_KMS > 50.0
    assert mean[offsets == -40.0][0] - VSYS_KMS < -50.0
    assert mean[offsets == 0.0][0] == pytest.approx(VSYS_KMS, abs=5.0)


def test_slice_outside_the_cube_is_empty(cube):
    ra, dec = cube.pixel_to_world(DISC["x"], DISC["y"])
    diagram = pv.position_velocity(cube.data, cube.celestial_wcs, ra, dec, 0.0,
                                   np.array([5000.0]), pixel_arcsec=cube.pixel_arcsec)
    assert np.all(diagram == 0.0)


def test_width_averages_across_the_slice():
    from astropy.wcs import WCS
    wcs = WCS(naxis=2)
    wcs.wcs.ctype, wcs.wcs.crval, wcs.wcs.crpix = ["RA---SIN", "DEC--SIN"], [30.0, 0.0], [11.0, 11.0]
    wcs.wcs.cdelt = [-1.0 / 3600.0, 1.0 / 3600.0]
    data = np.zeros((1, 21, 21))
    data[0, 10, :] = 3.0                                       # a line along RA through the centre
    narrow = pv.position_velocity(data, wcs, 30.0, 0.0, 90.0, np.array([0.0]))
    wide = pv.position_velocity(data, wcs, 30.0, 0.0, 90.0, np.array([0.0]), width_arcsec=2.0)
    assert narrow[0, 0] == pytest.approx(3.0) and wide[0, 0] == pytest.approx(1.0)
