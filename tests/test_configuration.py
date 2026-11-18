import os
from pathlib import Path

import pytest
from astropy.io import fits
from omegaconf import OmegaConf

from anomalous_gas_extractor import anomalous_gas_extractor as program
from anomalous_gas_extractor.kinematics import configuration
from anomalous_gas_extractor.kinematics.configuration import Configuration

TEST_DATA = Path(__file__).resolve().parents[1] / "test_data" / "HCG16A"


def write_settings(tmp_path, text=""):
    settings = tmp_path / "run.yml"
    settings.write_text(f"directories:\n  run_directory: {tmp_path}\n"
                        f"input:\n  steps_to_run: [rmax, pyFAT, hi_mass]\n{text}")
    return f"configuration_file={settings}"


def test_default_file_lists_every_setting_with_its_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        configuration.read_configuration(["print_examples=true"])
    text = (tmp_path / configuration.DEFAULT_FILE).read_text()
    assert "# Step pyfat: tilted-ring models of the rotating discs" in text
    assert "  minimum_beams_across: 3.0" in text

    defaults = OmegaConf.structured(Configuration)
    written = OmegaConf.load(tmp_path / configuration.DEFAULT_FILE)
    assert OmegaConf.merge(defaults, written) == defaults
    for name in ("deblend", "components", "kinematics", "pyfat", "hi_mass", "figures"):
        assert set(written[name]) == set(defaults[name])
    assert "sofia_run_directory" not in written.directories and "logging" not in written


def test_settings_are_read_from_file_and_command_line(tmp_path):
    cfg = configuration.read_configuration(
        [write_settings(tmp_path, "hi_mass:\n  distance_mpc: 49.0\n"),
         "pyfat.minimum_beams_across=4", "kinematics.k5k1_max=0.08"])
    assert cfg.input.steps_to_run == ["rmax", "pyfat", "hi_mass"]
    assert cfg.hi_mass.distance_mpc == 49.0 and cfg.pyfat.minimum_beams_across == 4.0
    assert cfg.kinematics.k5k1_max == 0.08
    assert cfg.directories.run_directory == f"{tmp_path}/"
    assert cfg.directories.sofia_run_directory == cfg.directories.run_directory


def test_a_tilted_ring_model_needs_three_beams(tmp_path):
    with pytest.raises(ValueError, match="has to be 3 or larger"):
        configuration.read_configuration([write_settings(tmp_path), "pyfat.minimum_beams_across=2"])
    cfg = configuration.read_configuration([write_settings(tmp_path),
                                            "pyfat.minimum_beams_across=3"])
    assert cfg.pyfat.minimum_beams_across == 3.0


def test_unknown_steps_and_settings_are_refused(tmp_path):
    with pytest.raises(ValueError, match="not steps of the program"):
        configuration.read_configuration([write_settings(tmp_path),
                                          "input.steps_to_run=[rmax,masses]"])
    with pytest.raises(Exception, match="minimum_beams"):
        configuration.read_configuration([write_settings(tmp_path), "pyfat.minimum_beams=4"])
    with pytest.raises(FileNotFoundError):
        configuration.read_configuration(["configuration_file=/no/such/file.yml"])


def test_source_finding_needs_a_cube(tmp_path):
    with pytest.raises(ValueError, match="neither provided a sofia parameter file nor a cube"):
        configuration.read_configuration([write_settings(tmp_path),
                                          "input.steps_to_run=[sofia,deblend]"])


def test_steps_run_in_their_fixed_order(tmp_path, monkeypatch):
    done = []
    for step in configuration.STEPS:
        monkeypatch.setitem(program.STEP_FUNCTIONS, step, lambda cfg, step=step: done.append(step))
    monkeypatch.setattr(program, "setup_log", lambda cfg: None)
    program.main([write_settings(tmp_path), "input.steps_to_run=[figures,hi_mass,rmax]"])
    assert done == ["rmax", "hi_mass", "figures"]


def test_sofia_is_found_by_its_name(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    program.make_sofia_available("sofia")
    assert os.environ["PATH"] == "/usr/bin"
    program.make_sofia_available(f"{tmp_path}/sofia")
    assert os.environ["PATH"].split(os.pathsep)[0] == str(tmp_path)
    with pytest.raises(ValueError, match="has to have that name"):
        program.make_sofia_available("/opt/bin/sofia2")


def test_the_output_directory_of_sofia_is_made(tmp_path):
    (tmp_path / "sofia.par").write_text("input.data = cube.fits\noutput.directory = found\n")
    cfg = OmegaConf.merge(OmegaConf.structured(Configuration), {
        "input": {"sofia_parameter_file": "sofia.par"},
        "directories": {"sofia_run_directory": f"{tmp_path}/"}})
    program.make_sofia_output_directory(cfg)
    assert (tmp_path / "found").is_dir()


def test_the_test_data_are_complete():
    cfg = OmegaConf.merge(OmegaConf.structured(Configuration),
                          OmegaConf.load(TEST_DATA / "HCG16A.yml"))
    configuration.validate(cfg)
    assert "deblend" not in cfg.input.steps_to_run
    assert (TEST_DATA / cfg.input.known_source_table).is_file()
    parameters = program.components.read_sofia_parameters(
        str(TEST_DATA / cfg.input.sofia_parameter_file))
    header = fits.getheader(TEST_DATA / parameters["input.data"])
    assert (header["NAXIS1"], header["NAXIS2"], header["NAXIS3"]) == (83, 83, 105)
    assert header["BUNIT"].strip().upper() == "JY/BEAM" and header["OBJECT"] == "HCG 16A"
