import numpy as np
import pytest

from anomalous_gas_extractor.kinematics import hi_mass
from anomalous_gas_extractor.kinematics.datacube import HI_REST_FREQUENCY_HZ, SPEED_OF_LIGHT_KMS

from conftest import DISC


def test_mass_of_one_jy_kms_at_one_mpc():
    # 1 Jy km/s in radio velocity is nu_0 / c Jy Hz; the classical constant is 2.356e5
    flux_jy_hz = HI_REST_FREQUENCY_HZ / SPEED_OF_LIGHT_KMS
    assert hi_mass.hi_mass(flux_jy_hz, 1.0) == pytest.approx(2.356e5, rel=2e-3)
    assert hi_mass.hi_mass(flux_jy_hz, 10.0) == pytest.approx(2.356e7, rel=2e-3)


def test_hubble_distance():
    assert hi_mass.hubble_distance(7000.0, 70.0) == pytest.approx(100.0 * (1 + 7000.0 / SPEED_OF_LIGHT_KMS))


def test_disc_region_is_the_projected_ellipse(cube):
    ra, dec = cube.pixel_to_world(DISC["x"], DISC["y"])
    geometry = hi_mass.DiscGeometry(ra, dec, 50.0, 60.0, 0.0)
    region = hi_mass.disc_region(cube, geometry)
    assert region.sum() == pytest.approx(np.pi * 10.0 * 5.0, rel=0.05)
    assert region[int(DISC["y"]) + 9, int(DISC["x"])]         # along the major axis (north)
    assert not region[int(DISC["y"]), int(DISC["x"]) + 7]     # beyond the minor axis
    wider = hi_mass.disc_region(cube, geometry, margin_arcsec=5.0)
    assert wider.sum() == pytest.approx(np.pi * 11.0 * 6.0, rel=0.05)


def test_position_angle_runs_from_north_to_east(cube):
    ra, dec = cube.pixel_to_world(32.0, 32.0)
    region = hi_mass.disc_region(cube, hi_mass.DiscGeometry(ra, dec, 60.0, 80.0, 45.0))
    assert region[32 + 6, 32 - 6] and not region[32 + 6, 32 + 6]   # east is at lower x


def test_budget_adds_up(cube):
    ra, dec = cube.pixel_to_world(DISC["x"], DISC["y"])
    region = hi_mass.disc_region(cube, hi_mass.DiscGeometry(ra, dec, 35.0, 50.0, 30.0))
    budget = hi_mass.mass_budget(cube, 1, 20.0, region)
    assert 0.0 < budget.disc_mass < budget.total_mass
    assert budget.disc_mass + budget.anomalous_mass == pytest.approx(budget.total_mass)
    assert budget.disc_fraction == pytest.approx(budget.disc_flux_jy_kms / budget.total_flux_jy_kms,
                                                 rel=1e-3)
    everything = hi_mass.mass_budget(cube, 1, 20.0, np.ones(cube.data.shape[1:], dtype=bool))
    assert everything.anomalous_mass == pytest.approx(0.0, abs=1e-6 * everything.total_mass)


def test_component_without_disc_is_all_anomalous(cube):
    budget = hi_mass.mass_budget(cube, 2, 20.0)
    assert budget.disc_mass == 0.0 and budget.anomalous_mass == budget.total_mass


def test_total_flux_of_the_cloud(cube):
    # 0.01 Jy/beam peak, Gaussian line of 8 km/s dispersion, on the pixels of a circle of radius 5
    flux, _ = hi_mass.component_fluxes(cube, 2)
    pixels = (np.hypot(*np.indices((64, 64)) - np.array([10, 54])[:, None, None]) <= 5).sum()
    expected = 0.01 * np.sqrt(2 * np.pi) * 8.0 * pixels / cube.beam_area_pixels
    assert flux == pytest.approx(expected, rel=0.05)


def test_model_cube_mass(cube, cube_files):
    cube_file, _ = cube_files
    total = hi_mass.model_cube_mass(cube_file, 20.0)
    components = sum(hi_mass.mass_budget(cube, label, 20.0).total_mass for label in (1, 2))
    assert total == pytest.approx(components, rel=0.1)
