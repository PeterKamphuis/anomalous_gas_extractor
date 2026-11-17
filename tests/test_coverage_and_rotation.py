import numpy as np
import pytest

from anomalous_gas_extractor.kinematics import coverage_test, rotation
from anomalous_gas_extractor.kinematics.datacube import MomentMaps
from anomalous_gas_extractor.kinematics.rmax import DiscExtent, Ring, RingCriteria, find_rmax

from conftest import DISC


def test_wedge_covers_the_requested_fraction():
    mask = coverage_test.wedge_mask((201, 201), 100.0, 100.0, 45.0, 0.25)
    assert mask.mean() == pytest.approx(0.25, abs=0.01)
    assert mask[150, 150] and not mask[50, 50]


def test_minimum_coverage_from_the_corrupted_fraction():
    coverage = np.repeat([0.95, 0.85, 0.75, 0.65], 100)
    corrupted = np.zeros(400, dtype=bool)
    corrupted[300:340] = True
    table, minimum, complete = coverage_test.corrupted_fraction_curve(coverage, corrupted)
    assert not complete
    assert minimum == pytest.approx(0.75)
    assert table[0][:2] == (0.95, 100)


def test_complete_rings_required_when_small_gaps_corrupt():
    coverage = np.full(100, 0.93)
    _, minimum, complete = coverage_test.corrupted_fraction_curve(coverage, np.ones(100, dtype=bool))
    assert complete and minimum == coverage_test.REFERENCE_COVERAGE


def test_bins_with_few_trials_stop_the_search():
    coverage = np.concatenate([np.full(100, 0.95), np.full(5, 0.85)])
    _, minimum, _ = coverage_test.corrupted_fraction_curve(coverage, np.zeros(105, dtype=bool))
    assert minimum == pytest.approx(0.95)


def test_no_reference_ring_gives_no_result():
    rings = [Ring(10.0, 0.0, 1.0, 50.0, 0.5, 1000.0, 1.0)]
    assert coverage_test.minimum_coverage(np.zeros((20, 20)), 10.0, 10.0, rings, 5.0,
                                          RingCriteria()) is None


def test_masking_test_on_a_regular_disc(cube, kinemetry):
    from anomalous_gas_extractor.kinematics.rmax import measure_rings
    criteria = RingCriteria()
    maps = cube.moment_maps(1)
    rings = measure_rings(maps.velocity, DISC["x"], DISC["y"], cube.pixel_arcsec, criteria)
    find_rmax(rings, 0.6, cube.beam_major_arcsec, criteria)
    result = coverage_test.minimum_coverage(maps.velocity, DISC["x"], DISC["y"], rings,
                                            cube.pixel_arcsec, criteria, n_processes=2)
    assert result is not None and result.n_trials > 100
    assert 0.3 <= result.minimum_coverage <= 0.95


def flat_maps(dispersion, gradient=0.0):
    shape = (41, 41)
    velocity = 1000.0 + gradient * (np.indices(shape)[1] - 20.0)
    return MomentMaps(np.ones(shape), velocity, np.full(shape, dispersion))


def extent_with(k1_values, r_max, step=10.0):
    rings = [Ring(step * (i + 1), 0.0, 1.0, k1, 0.01, 1000.0, 1.0) for i, k1 in enumerate(k1_values)]
    extent = find_rmax(rings, 0.6, 15.0, RingCriteria())
    extent.r_max_arcsec = r_max
    return extent


def classify(extent, maps, **limits):
    return rotation.classify_rotation(extent, maps, 20.0, 20.0, 5.0, 15.0, 10.0,
                                      rotation.RotationCriteria(**limits))


def test_rotating_disc_is_accepted():
    verdict = classify(extent_with([40.0, 80.0, 100.0], 30.0), flat_maps(20.0))
    assert verdict.is_rotating_disc
    assert verdict.disc_extent_beams == pytest.approx(2.0)
    assert verdict.v_over_sigma == pytest.approx(5.0)


def test_unresolved_falling_and_dispersion_dominated_are_rejected():
    verdict = classify(extent_with([40.0], 10.0), flat_maps(60.0))
    assert not verdict.is_rotating_disc
    assert verdict.failed_tests == ["not resolved", "not supported by rotation"]
    verdict = classify(extent_with([60.0, 30.0], 20.0), flat_maps(10.0))
    assert verdict.failed_tests == ["rotation amplitude falls outwards"]


def test_no_regular_rings():
    verdict = classify(DiscExtent(), flat_maps(10.0))
    assert not verdict.is_rotating_disc and verdict.failed_tests == ["no regular rings"]


def test_beam_smearing_is_removed_from_the_dispersion():
    gradient = 4.0                                   # km/s per pixel, beam of 3 pixels
    smearing = gradient * 3.0 * rotation.FWHM_TO_SIGMA
    maps = flat_maps(np.hypot(10.0, smearing), gradient)
    assert rotation.disc_dispersion(maps, 20.0, 20.0, 6.0, 3.0) == pytest.approx(10.0)
