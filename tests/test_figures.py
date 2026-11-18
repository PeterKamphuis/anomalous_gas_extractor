import csv
import logging
import os
import shutil

import numpy as np
import pytest
from omegaconf import OmegaConf

from anomalous_gas_extractor.kinematics.configuration import Configuration
from anomalous_gas_extractor.kinematics import (fat, figure_data, figures, pipeline, pv_figures,
                                                slices)

from conftest import DISC, VSYS_KMS

RMAX_ONLY = ["rmax", "disc_anomalous", "position_velocity"]
NEED_MODELS = ["models", "model_comparison"]


def configuration(cube_files, run_directory):
    cube_file, mask_file = cube_files
    cfg = OmegaConf.structured(Configuration)
    cfg.directories.run_directory = str(run_directory) + "/"
    cfg.kinematics.cube = cube_file
    cfg.kinematics.mask = mask_file
    cfg.kinematics.kinemetry_path = os.environ.get("KINEMETRY_PATH")
    cfg.kinematics.measure_coverage = False
    cfg.hi_mass.distance_mpc = 20.0
    cfg.figures.format = "png"
    return cfg


@pytest.fixture
def cfg(cube_files, tmp_path):
    return configuration(cube_files, tmp_path)


@pytest.fixture(scope="module")
def rmax_run(cube_files, kinemetry, tmp_path_factory):
    """Directory of a run in which the step rmax has been made."""
    directory = tmp_path_factory.mktemp("rmax_run")
    pipeline.run_rmax(configuration(cube_files, directory))
    return directory


def rmax_rows(cube, disc_angle=DISC["position_angle"]):
    """Results of the step rmax as needed by the figures: a rotating disc and a cloud."""
    ra, dec = cube.pixel_to_world(DISC["x"], DISC["y"])
    disc = {"component": "DISC", "label": 1, "rotating_disc": True, "r_max_arcsec": 60.0,
            "r_max_is_lower_limit": False, "centre_ra_deg": ra, "centre_dec_deg": dec,
            "vsys_kms": VSYS_KMS, "position_angle_deg": disc_angle,
            "axis_ratio": float(np.cos(np.radians(DISC["inclination"])))}
    ra, dec = cube.pixel_to_world(54.0, 10.0)
    cloud = {"component": "CLOUD", "label": 2, "rotating_disc": False, "r_max_arcsec": None,
             "r_max_is_lower_limit": False, "centre_ra_deg": ra, "centre_dec_deg": dec,
             "vsys_kms": None, "position_angle_deg": None, "axis_ratio": None}
    return [disc, cloud]


def names(written, run_directory):
    return sorted(os.path.relpath(path, os.path.join(run_directory, "figures")) for path in written)


def test_kinematic_position_angle_points_to_the_receding_side(cube):
    angle = slices.kinematic_position_angle(cube, 1)
    assert angle == pytest.approx(DISC["position_angle"], abs=3.0)
    # a cloud at one velocity has no gradient, but an angle is returned
    assert 0.0 <= slices.kinematic_position_angle(cube, 2) < 360.0


def test_slices_follow_the_disc_or_the_given_angle(cube, cfg):
    # the Kinemetry ellipses may give the approaching side of the major axis
    rows = rmax_rows(cube, disc_angle=DISC["position_angle"] + 180.0)
    disc, cloud = figure_data.load_components(cfg, cube, rows)
    assert disc.has_disc and not disc.has_model and not cloud.has_disc

    strip = slices.component_slice(cube, disc, width_arcsec=15.0)
    assert strip.position_angle_deg == pytest.approx(DISC["position_angle"])
    assert strip.source == "Kinemetry" and strip.directions == ("SW", "NE")
    # the slice covers the disc and the footprint of the component, plus one beam
    assert strip.first_arcsec <= -75.0 and strip.last_arcsec >= 75.0
    assert (strip.ra_deg, strip.dec_deg) == (rows[0]["centre_ra_deg"], rows[0]["centre_dec_deg"])

    given = slices.component_slice(cube, cloud, 15.0, position_angle_deg=90.0)
    assert given.position_angle_deg == 90.0 and given.source == "given"
    assert given.directions == ("W", "E")
    # the cloud has a radius of 5 pixels of 5 arcsec; one beam of 15 arcsec is added
    assert given.first_arcsec == pytest.approx(-40.0, abs=5.0)
    assert given.last_arcsec == pytest.approx(40.0, abs=5.0)
    measured = slices.component_slice(cube, cloud, 15.0)
    assert measured.source == "velocity gradient"


def test_rotation_curve_of_a_model():
    ring = {"radius_arcsec": 0.0, "sbr_approaching": 1e-4, "sbr_receding": 1e-4,
            "vrot_kms_approaching": 0.0, "vrot_kms_receding": 0.0,
            "inclination_deg_approaching": 30.0, "inclination_deg_receding": 90.0}
    rings = [ring, dict(ring, radius_arcsec=10.0, vrot_kms_approaching=100.0,
                        vrot_kms_receding=80.0),
             dict(ring, radius_arcsec=20.0, sbr_receding=0.0, vrot_kms_approaching=120.0)]
    curve = pv_figures.rotation_curve(rings, 1000.0)
    # the receding side has no emission in the outer ring; each side has its own inclination
    assert curve == pytest.approx(np.array([[-20.0, 940.0], [-10.0, 950.0], [0.0, 1000.0],
                                            [10.0, 1080.0]]))
    assert pv_figures.rotation_curve([], 1000.0).shape == (0, 2)


def test_tables_are_read_back(tmp_path):
    table = tmp_path / "rings.csv"
    pipeline.write_table(str(table), [{"radius_arcsec": 6.0, "regular": True, "error": np.nan,
                                       "stopped_by": "k5/k1"}])
    row, = figure_data.read_table(str(table))
    assert row["radius_arcsec"] == 6.0 and row["regular"] is True and np.isnan(row["error"])
    assert row["stopped_by"] == "k5/k1"
    assert figure_data.read_table(str(tmp_path / "missing.csv")) == []


def test_no_figures_without_results(cfg, caplog):
    with caplog.at_level(logging.WARNING):
        assert figures.make_figures(cfg) == []
    assert "run the step 'rmax' first" in caplog.text
    cfg.figures.products = ["rmax", "moment_maps"]
    with pytest.raises(ValueError, match="moment_maps"):
        figures.make_figures(cfg)
    cfg.figures.enable = False
    assert figures.make_figures(cfg) == []


def test_figures_of_the_rmax_results(cube_files, rmax_run, caplog):
    cfg = configuration(cube_files, rmax_run)
    cfg.figures.products = RMAX_ONLY
    cfg.figures.display_names = {"DISC": "The disc"}
    written = figures.make_figures(cfg)
    assert names(written, rmax_run) == sorted([
        "rmax/velocity_fields_rmax.png", "rmax/kinemetry_profiles.png",
        "disc_anomalous/column_density.png", "disc_anomalous/velocity_fields.png",
        "disc_anomalous/velocity_fields_pv_slices.png", "position_velocity/pv_all.png",
        "position_velocity/pv_DISC.png", "position_velocity/pv_CLOUD.png"])
    assert all(os.path.getsize(path) > 5000 for path in written)

    # the figures of the models are skipped as long as there are no models
    cfg.figures.products = NEED_MODELS
    with caplog.at_level(logging.WARNING):
        assert figures.make_figures(cfg) == []
    assert caplog.text.count("run the step 'pyfat' first") == 2


def test_figures_with_a_model(cube_files, rmax_run, tmp_path, monkeypatch):
    run_directory = tmp_path / "run"
    shutil.copytree(rmax_run, run_directory)
    cfg = configuration(cube_files, run_directory)
    cfg.figures.format = "pdf"
    cfg.figures.output_directory = str(tmp_path / "elsewhere")

    def fake_pyfat(main_directory, settings):
        """A final model whose cube is the cube that was given to pyFAT."""
        final = os.path.join(main_directory, "DISC", "Finalmodel")
        os.makedirs(final)
        shutil.copy(os.path.join(main_directory, "DISC", "DISC.fits"),
                    os.path.join(final, "Finalmodel.fits"))
        ra, dec = pipeline.load_cube(cfg).pixel_to_world(DISC["x"], DISC["y"])
        with open(os.path.join(final, "Finalmodel.def"), "w") as definition:
            definition.write("RADI= 0. 20. 40. 60.\nSBR= 1e-4 1e-4 1e-4 0.\n"
                             "VROT= 0. 80. 105. 110.\nINCL= 50. 50. 50. 50.\n"
                             "PA= 30. 30. 30. 30.\nSDIS= 8. 8. 8. 8.\n"
                             f"XPOS= {ra} {ra} {ra} {ra}\nYPOS= {dec} {dec} {dec} {dec}\n"
                             "VSYS= 1000. 1000. 1000. 1000.\n")
        with open(os.path.join(main_directory, fat.RESULTS_CATALOGUE), "w") as results:
            results.write("Directory Name OS Comments\nDISC True The fit has converged\n")
        return 0

    monkeypatch.setattr(fat, "run_pyfat", fake_pyfat)
    pipeline.run_pyfat(cfg)
    written = figures.make_figures(cfg)
    relative = sorted(os.path.relpath(path, tmp_path / "elsewhere") for path in written)
    assert relative == sorted([
        "rmax/velocity_fields_rmax.pdf", "rmax/kinemetry_profiles.pdf",
        "models/model_DISC.pdf", "models/models_all.pdf",
        "disc_anomalous/column_density.pdf", "disc_anomalous/velocity_fields.pdf",
        "disc_anomalous/velocity_fields_pv_slices.pdf", "position_velocity/pv_all.pdf",
        "position_velocity/pv_DISC.pdf", "position_velocity/pv_CLOUD.pdf",
        "model_comparison/model_on_data.pdf", "model_comparison/residual_moments.pdf",
        "model_comparison/residual_statistics.csv"])
    assert all(os.path.getsize(path) > 0 for path in written)

    # the model cube is the data, so the residuals vanish
    with open(written[-1], newline="") as table:
        row, = csv.DictReader(table)
    assert row["component"] == "DISC"
    assert float(row["velocity_median"]) == pytest.approx(0.0, abs=0.5)
    assert float(row["sigma_rms"]) < 0.1
