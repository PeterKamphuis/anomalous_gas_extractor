import csv
import os

import pytest
from omegaconf import OmegaConf

from anomalous_gas_extractor.kinematics.configuration import Configuration
from anomalous_gas_extractor.kinematics import fat, pipeline

from conftest import DISC, VSYS_KMS


@pytest.fixture
def cfg(cube_files, tmp_path):
    cube_file, mask_file = cube_files
    configuration = OmegaConf.structured(Configuration)
    configuration.directories.run_directory = str(tmp_path) + "/"
    configuration.kinematics.cube = cube_file
    configuration.kinematics.mask = mask_file
    configuration.kinematics.kinemetry_path = os.environ.get("KINEMETRY_PATH")
    configuration.hi_mass.distance_mpc = 20.0
    configuration.input.ncpu = 2
    return configuration


def read(path):
    with open(path, newline="") as table:
        return {row["component"]: row for row in csv.DictReader(table)}


def test_centre_table(tmp_path):
    table = tmp_path / "centres.csv"
    table.write_text("#Name,RA,DEC,Velocity\nHCG 16A,32.35,-10.13,4040\n\nNGC848,32.57,-10.32,3962\n")
    assert pipeline.read_centre_table(str(table)) == {"HCG16A": (32.35, -10.13),
                                                      "NGC848": (32.57, -10.32)}


def test_tables_are_rounded(tmp_path):
    pipeline.write_table(str(tmp_path / "table.csv"),
                         [{"component": "A", "value": 45.371086659133105, "flag": True,
                           "missing": float("nan")}])
    assert (tmp_path / "table.csv").read_text().splitlines() == [
        "component,value,flag,missing", "A,45.3710867,True,"]


def test_steps_need_their_input(cfg):
    with pytest.raises(FileNotFoundError, match="run the step 'rmax' first"):
        pipeline.run_pyfat(cfg)
    cfg.kinematics.mask = None
    with pytest.raises(FileNotFoundError, match="run the step 'components'"):
        pipeline.load_cube(cfg)


def test_rmax_and_hi_mass_steps(cfg, kinemetry, tmp_path):
    results = {result.name: result for result in pipeline.run_rmax(cfg)}
    disc, cloud = results["DISC"], results["CLOUD"]
    assert disc.verdict.is_rotating_disc and not cloud.verdict.is_rotating_disc
    assert disc.centre_source == "kinematic"
    assert disc.extent.r_max_arcsec == pytest.approx(DISC["radius"] * 5.0, abs=10.0)
    assert disc.extent.vsys_reference_kms == pytest.approx(VSYS_KMS, abs=3.0)

    table = read(tmp_path / "kinematics" / pipeline.RMAX_TABLE)
    assert table["DISC"]["rotating_disc"] == "True" and table["CLOUD"]["rotating_disc"] == "False"
    assert (tmp_path / "kinematics" / "rings_DISC.csv").is_file()

    # without pyFAT the disc is the ellipse of r_max from Kinemetry
    masses = {row["component"]: row for row in pipeline.run_hi_mass(cfg)}
    assert masses["DISC"]["disc_radius_source"] == "r_max (Kinemetry)"
    assert 0.6 < masses["DISC"]["disc_fraction"] < 1.0
    assert masses["CLOUD"]["disc_hi_mass_msun"] == 0.0
    assert masses["CLOUD"]["anomalous_hi_mass_msun"] == masses["CLOUD"]["total_hi_mass_msun"]
    assert (tmp_path / "hi_mass" / pipeline.HI_MASS_TABLE).is_file()
    assert masses["DISC"]["tilted_ring_model"] == "pyFAT not run"
    assert masses["CLOUD"]["tilted_ring_model"] == "no rotating disc"


def test_small_discs_are_not_modelled(cfg, kinemetry, tmp_path, monkeypatch):
    pipeline.run_rmax(cfg)
    disc = read(tmp_path / "kinematics" / pipeline.RMAX_TABLE)["DISC"]
    across = float(disc["disc_diameter_beams"])
    assert across == pytest.approx(2.0 * float(disc["r_max_arcsec"]) / 15.0)
    assert list(pipeline._fat_targets(cfg)) == ["DISC"]

    def no_pyfat(directory, settings):
        raise AssertionError("pyFAT must not be started")
    monkeypatch.setattr(fat, "run_pyfat", no_pyfat)
    cfg.pyfat.minimum_beams_across = across + 1.0
    assert pipeline.run_pyfat(cfg) == {}
    masses = {row["component"]: row for row in pipeline.run_hi_mass(cfg)}
    assert masses["DISC"]["tilted_ring_model"] == f"disc smaller than {across + 1.0:g} beams"
    assert masses["DISC"]["disc_radius_source"] == "r_max (Kinemetry)"
    assert masses["DISC"]["disc_radius_arcsec"] == pytest.approx(float(disc["r_max_arcsec"]))
    assert 0.6 < masses["DISC"]["disc_fraction"] < 1.0


def test_pyfat_step_prepares_input_and_reads_models(cfg, kinemetry, tmp_path, monkeypatch):
    pipeline.run_rmax(cfg)

    def fake_pyfat(main_directory, settings):
        final = os.path.join(main_directory, "DISC", "Finalmodel")
        os.makedirs(final)
        ra, dec = pipeline.load_cube(cfg).pixel_to_world(DISC["x"], DISC["y"])
        with open(os.path.join(final, "Finalmodel.def"), "w") as definition:
            definition.write("RADI= 0. 15. 30. 45.\nSBR= 1e-4 1e-4 1e-4 0.\nVROT= 0. 50. 90. 90.\n"
                             "INCL= 50. 50. 50. 50.\nPA= 30. 30. 30. 30.\n"
                             f"XPOS= {ra} {ra} {ra} {ra}\nYPOS= {dec} {dec} {dec} {dec}\n"
                             "VSYS= 1000. 1000. 1000. 1000.\n")
        with open(os.path.join(main_directory, fat.RESULTS_CATALOGUE), "w") as results:
            results.write("Directory Name OS Comments\nDISC True The fit has converged\n")
        return 0

    monkeypatch.setattr(fat, "run_pyfat", fake_pyfat)
    models = pipeline.run_pyfat(cfg)
    assert list(models) == ["DISC"] and models["DISC"].disc_radius_arcsec == 45.0
    directory = tmp_path / "pyfat"
    assert (directory / "DISC" / "DISC.fits").is_file()
    assert "r_max" in (directory / fat.CATALOGUE).read_text().splitlines()[0]
    assert not (directory / "CLOUD").exists()

    assert (directory / "model_rings_DISC.csv").read_text().startswith("radius_arcsec,radius_kpc,")
    # the figures of the models are made by the step figures, not here
    assert not (directory / "figures").exists()

    # existing fits are read again without running pyFAT
    def no_pyfat(main_directory, settings):
        raise AssertionError("pyFAT must not be run again")

    monkeypatch.setattr(fat, "run_pyfat", no_pyfat)
    cfg.pyfat.reuse_fits = True
    assert pipeline.run_pyfat(cfg)["DISC"].disc_radius_arcsec == 45.0

    masses = {row["component"]: row for row in pipeline.run_hi_mass(cfg)}
    summary = read(tmp_path / pipeline.SUMMARY_TABLE)
    assert summary["DISC"]["rotating_disc"] == "True" and summary["CLOUD"]["rotating_disc"] == "False"
    assert float(summary["DISC"]["disc_radius_kpc"]) == pytest.approx(45.0 / 206264.806 * 20e3, rel=1e-6)
    assert float(summary["DISC"]["disc_vrot_max_kms"]) == 90.0
    assert float(summary["DISC"]["disc_hi_mass_msun"]) == pytest.approx(
        masses["DISC"]["disc_hi_mass_msun"], rel=1e-8)
    assert summary["CLOUD"]["disc_radius_arcsec"] == ""
    assert float(summary["CLOUD"]["anomalous_fraction"]) == 1.0
    assert masses["DISC"]["disc_radius_source"] == "pyFAT"
    assert masses["DISC"]["disc_radius_arcsec"] == 45.0
    assert masses["DISC"]["disc_inclination_deg"] == 50.0
    assert masses["DISC"]["disc_margin_arcsec"] == 0.0
    cfg.hi_mass.beam_margin = 0.5
    wider = {row["component"]: row for row in pipeline.run_hi_mass(cfg)}
    assert wider["DISC"]["disc_margin_arcsec"] == pytest.approx(0.5 * 15.0)
    assert wider["DISC"]["disc_hi_mass_msun"] > masses["DISC"]["disc_hi_mass_msun"]
    assert masses["CLOUD"]["disc_ra_deg"] is None
    assert 0.0 < masses["DISC"]["disc_fraction"] < 1.0
