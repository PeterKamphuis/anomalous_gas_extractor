from types import SimpleNamespace

import numpy as np
import pytest

from anomalous_gas_extractor.kinematics import kinemetry_rings, rmax
from anomalous_gas_extractor.kinematics.centre import kinematic_centre
from anomalous_gas_extractor.kinematics.kinemetry_rings import (curve_amplitude, ellipse_coverage,
                                                                ellipse_velocities)

from conftest import DISC, VSYS_KMS

CRITERIA = rmax.RingCriteria()
BEAM = 15.0


def ring(radius, k5k1=0.02, vsys=1000.0, coverage=1.0, k1=100.0):
    return rmax.Ring(radius_arcsec=radius, position_angle_deg=30.0, axis_ratio=0.6, k1_kms=k1,
                     k5_over_k1=k5k1, vsys_kms=vsys, coverage=coverage)


def test_crossing_radius_is_interpolated():
    assert rmax.crossing_radius(10.0, 20.0, 0.05, 0.15, 0.10) == pytest.approx(15.0)
    assert rmax.crossing_radius(10.0, 20.0, 0.15, 0.05, 0.10) == 10.0
    assert rmax.crossing_radius(10.0, 20.0, np.nan, 0.5, 0.10) == 10.0


def test_run_ends_where_k5k1_crosses_the_limit():
    rings = [ring(10), ring(20), ring(30, k5k1=0.06), ring(40, k5k1=0.14), ring(50)]
    extent = rmax.find_rmax(rings, 0.6, BEAM, CRITERIA)
    assert (extent.first_index, extent.last_index) == (0, 2)
    assert extent.r_max_arcsec == pytest.approx(35.0)
    assert not extent.is_lower_limit
    assert extent.stopped_by == "k5/k1 at 40 arcsec"
    assert extent.k1_max_kms == 100.0
    assert extent.position_angle_deg == pytest.approx(30.0)


def test_smallest_crossing_is_taken_when_two_tests_fail():
    rings = [ring(10), ring(20), ring(30, k5k1=0.30, vsys=1100.0)]
    extent = rmax.find_rmax(rings, 0.6, BEAM, CRITERIA)
    k5k1_crossing = 20.0 + 10.0 * (0.10 - 0.02) / (0.30 - 0.02)
    vsys_crossing = 20.0 + 10.0 * 15.0 / 100.0
    assert extent.r_max_arcsec == pytest.approx(min(k5k1_crossing, vsys_crossing))
    assert extent.stopped_by == "k5/k1 + v_sys drift at 30 arcsec"


def test_lower_limit_when_only_the_coverage_fails():
    rings = [ring(10), ring(20, coverage=0.9), ring(30, coverage=0.3)]
    extent = rmax.find_rmax(rings, 0.6, BEAM, CRITERIA)
    assert extent.is_lower_limit
    assert extent.r_max_arcsec == pytest.approx(25.0)


def test_lower_limit_at_the_end_of_the_velocity_field():
    extent = rmax.find_rmax([ring(10), ring(20)], 0.6, BEAM, CRITERIA)
    assert extent.is_lower_limit and extent.r_max_arcsec == 20.0


def test_run_has_to_start_near_the_centre():
    rings = [ring(10, k5k1=0.5), ring(20, k5k1=0.5), ring(30), ring(40)]
    assert not rmax.find_rmax(rings, 0.6, BEAM, CRITERIA).found
    rings = [ring(10, k5k1=0.5), ring(20), ring(30), ring(40, k5k1=0.5)]
    extent = rmax.find_rmax(rings, 0.6, BEAM, CRITERIA)
    assert (extent.first_index, extent.last_index) == (1, 2)


def test_vsys_tolerance_scales_with_rotation():
    assert CRITERIA.vsys_tolerance(50.0) == 15.0
    assert CRITERIA.vsys_tolerance(200.0) == pytest.approx(30.0)


def test_no_rings():
    extent = rmax.find_rmax([], 0.6, BEAM, CRITERIA)
    assert not extent.found and extent.r_max_arcsec is None and extent.disc_rings == []


def test_ellipse_coverage():
    valid = np.zeros((40, 40), dtype=bool)
    valid[:, 20:] = True
    assert ellipse_coverage(valid, 20.0, 20.0, 10.0, 0.0, 1.0) == pytest.approx(0.5, abs=0.03)
    assert ellipse_coverage(np.ones((40, 40), dtype=bool), 20.0, 20.0, 10.0, 40.0, 0.5) == 1.0
    assert ellipse_coverage(valid, 20.0, 20.0, 100.0, 0.0, 1.0) == 0.0


def test_kinemetry_recovers_the_disc(cube, kinemetry):
    maps = cube.moment_maps(1)
    rings = rmax.measure_rings(maps.velocity, DISC["x"], DISC["y"], cube.pixel_arcsec, CRITERIA)
    extent = rmax.find_rmax(rings, 0.6, cube.beam_major_arcsec, CRITERIA)
    assert extent.found
    assert extent.r_max_arcsec == pytest.approx(DISC["radius"] * cube.pixel_arcsec, abs=10.0)
    assert extent.vsys_reference_kms == pytest.approx(VSYS_KMS, abs=3.0)
    assert extent.position_angle_deg % 180 == pytest.approx(DISC["position_angle"], abs=5.0)
    assert extent.axis_ratio == pytest.approx(np.cos(np.radians(DISC["inclination"])), abs=0.1)
    expected = DISC["vmax"] * np.sin(np.radians(DISC["inclination"])) * 2 / np.pi * np.arctan(3.0)
    assert extent.k1_max_kms == pytest.approx(expected, rel=0.1)


def test_kinematic_centre_is_recovered(cube, kinemetry):
    maps = cube.moment_maps(1)
    estimate = kinematic_centre(maps.velocity, DISC["x"] + 1.5, DISC["y"] - 1.0, 12.0,
                                cube.beam_major_arcsec / cube.pixel_arcsec, 2.0)
    assert estimate is not None
    assert np.hypot(estimate.x - DISC["x"], estimate.y - DISC["y"]) < 1.0


RELAXED = rmax.RingCriteria(tests=("amplitude", "k5k1"))


def constrained(radius, **keywords):
    """A ring whose fit is constrained by the data."""
    measured = ring(radius, **keywords)
    measured.amplitude_ratio = 0.95
    return measured


def test_unknown_ring_test_is_refused():
    with pytest.raises(ValueError):
        rmax.RingCriteria(tests=("coverage", "symmetry"))
    with pytest.raises(ValueError):
        rmax.RingCriteria(tests=())


def test_curve_amplitude():
    # a0 = 1000; 100 cos(theta) + 20 cos(3 theta) peaks at 120
    assert curve_amplitude(np.array([1000.0, 0.0, 100.0, 0.0, 20.0, 0.0, 0.0])) == pytest.approx(120.0)
    assert curve_amplitude(np.array([1000.0, 60.0, 80.0, 0.0, 0.0, 0.0, 0.0])) == pytest.approx(100.0, rel=1e-3)


def test_amplitude_ratio():
    velocities = np.array([900.0, 950.0, 1000.0, 1080.0, np.nan])
    assert rmax.amplitude_ratio(100.0, 1000.0, velocities) == pytest.approx(1.0)
    assert rmax.amplitude_ratio(250.0, 1000.0, velocities) == pytest.approx(2.5)
    # the mean of the fit outside the measured velocities, or nothing measured
    assert rmax.amplitude_ratio(100.0, 1200.0, velocities) == np.inf
    assert rmax.amplitude_ratio(100.0, 1000.0, np.array([np.nan])) == np.inf


def test_ellipse_velocities():
    velocity = np.full((40, 40), np.nan)
    velocity[:, 20:] = 5.0
    values = ellipse_velocities(velocity, 20.0, 20.0, 10.0, 0.0, 1.0)
    assert 40 <= values.size <= 60 and np.all(values == 5.0)


def test_selected_tests_decide_which_rings_are_regular():
    rings = [constrained(10), constrained(20, vsys=1100.0, coverage=0.2),
             constrained(30, k5k1=0.06), constrained(40, k5k1=0.14)]
    assert rmax.find_rmax(rings, 0.6, BEAM, CRITERIA).last_index == 0
    extent = rmax.find_rmax(rings, 0.6, BEAM, RELAXED)
    assert [measured.regular for measured in rings] == [True, True, True, False]
    assert extent.r_max_arcsec == pytest.approx(35.0)
    assert extent.stopped_by == "k5/k1 at 40 arcsec" and not extent.is_lower_limit
    # the verdicts of the tests that were not selected are still recorded
    assert not rings[1].pass_vsys and not rings[1].pass_coverage


def test_unconstrained_fit_ends_the_run_at_the_last_ring():
    rings = [constrained(10), constrained(20), constrained(30)]
    rings[2].amplitude_ratio = 2.3
    extent = rmax.find_rmax(rings, 0.6, BEAM, RELAXED)
    assert extent.last_index == 1
    assert extent.r_max_arcsec == 20.0 and extent.is_lower_limit
    assert extent.stopped_by == "unconstrained fit at 30 arcsec"


def test_k5k1_sets_the_radius_when_the_fit_is_also_unconstrained():
    rings = [constrained(10), constrained(20), constrained(30, k5k1=0.18)]
    rings[2].amplitude_ratio = np.inf
    extent = rmax.find_rmax(rings, 0.6, BEAM, RELAXED)
    assert extent.r_max_arcsec == pytest.approx(25.0)
    assert extent.stopped_by == "k5/k1 + unconstrained fit at 30 arcsec"
    assert not extent.is_lower_limit


def test_fit_of_a_complete_disc_is_constrained(cube, kinemetry):
    maps = cube.moment_maps(1)
    rings = rmax.measure_rings(maps.velocity, DISC["x"], DISC["y"], cube.pixel_arcsec, RELAXED)
    extent = rmax.find_rmax(rings, 0.6, cube.beam_major_arcsec, RELAXED)
    assert extent.found
    assert all(0.7 < measured.amplitude_ratio < 1.1 for measured in extent.disc_rings)
    reference = rmax.find_rmax(rmax.measure_rings(maps.velocity, DISC["x"], DISC["y"],
                                                  cube.pixel_arcsec, CRITERIA),
                               0.6, cube.beam_major_arcsec, CRITERIA)
    assert extent.r_max_arcsec >= reference.r_max_arcsec - 1e-6



def _kinemetry_failing_inside(smallest_radius):
    """Stand-in for Kinemetry that stops with an error on rings below a radius."""
    def kinemetry(radius, **settings):
        radius = np.asarray(radius, dtype=float)
        if radius.min() < smallest_radius:
            raise TypeError("object of type 'float' has no len()")
        n_rings = radius.size
        return SimpleNamespace(rad=radius, pa=np.full(n_rings, 30.0), q=np.full(n_rings, 0.6),
                               cf=np.ones((n_rings, 2 * kinemetry_rings.N_HARMONIC_TERMS + 1)),
                               xc=np.full(n_rings, 20.0), yc=np.full(n_rings, 20.0))
    return SimpleNamespace(kinemetry=kinemetry)


def test_innermost_rings_that_cannot_be_fitted_are_left_out(monkeypatch):
    monkeypatch.setattr(kinemetry_rings, "load_kinemetry", lambda: _kinemetry_failing_inside(5.0))
    radii = [2.0, 4.0, 6.0, 8.0, 10.0]
    fits = kinemetry_rings.fit_rings(np.zeros((40, 40)), 20.0, 20.0, radii, skip_inner_rings=3)
    assert list(fits.radius) == [6.0, 8.0, 10.0]
    assert len(kinemetry_rings.fit_rings(np.zeros((40, 40)), 20.0, 20.0, radii)) == 0


def test_no_ring_is_returned_when_too_many_inner_rings_fail(monkeypatch):
    monkeypatch.setattr(kinemetry_rings, "load_kinemetry", lambda: _kinemetry_failing_inside(50.0))
    fits = kinemetry_rings.fit_rings(np.zeros((40, 40)), 20.0, 20.0, np.arange(2.0, 40.0, 2.0),
                                     skip_inner_rings=3)
    assert len(fits) == 0
