import os
from pathlib import Path

import pytest
from astropy.io import fits
from omegaconf import OmegaConf

from anomalous_gas_extractor import anomalous_gas_extractor as program
from anomalous_gas_extractor.config import config, functions
from anomalous_gas_extractor.config.config import defaults
from anomalous_gas_extractor.kinematics import components
from anomalous_gas_extractor.run_scripts import run_scripts

TEST_DATA = Path(__file__).resolve().parents[1] / "test_data" / "HCG16A"


def write_settings(tmp_path, text=""):
    settings = tmp_path / "run.yml"
    settings.write_text(f"directories:\n  run_directory: {tmp_path}\n"
                        f"input:\n  steps_to_run: [rmax, pyFAT, hi_mass]\n{text}")
    return f"configuration_file={settings}"


def test_default_file_lists_every_setting_with_its_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        functions.process_input(["print_examples=true"])
    text = (tmp_path / functions.DEFAULT_FILE).read_text()
    assert "# Step pyfat: tilted-ring models of the rotating discs" in text
    assert "  minimum_beams_across: 3.0" in text

    standard = OmegaConf.structured(defaults)
    written = OmegaConf.load(tmp_path / functions.DEFAULT_FILE)
    assert OmegaConf.merge(standard, written) == standard
    for name in ("deblend", "components", "kinematics", "pyfat", "hi_mass", "figures"):
        assert set(written[name]) == set(standard[name])
    assert "sofia_run_directory" not in written.directories and "logging" not in written


def test_the_command_age_starts_the_program():
    project = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    assert 'AGE = "anomalous_gas_extractor.anomalous_gas_extractor:main"' in project
    assert callable(program.main)


def test_version_and_help_are_printed(capsys):
    for option, expected in (("-v", "This is version"), ("--version", "This is version"),
                             ("-h", "AGE configuration_file="), ("--help", "print_examples")):
        with pytest.raises(SystemExit):
            program.main([option])
        assert expected in capsys.readouterr().out


def test_a_configuration_file_of_age_itself_is_read(tmp_path):
    """A file with only the settings of AGE, as written before the steps were added."""
    settings = tmp_path / "AGE.yml"
    settings.write_text("cube_name: null\n"
                        "input:\n"
                        "  known_source_table: null\n"
                        "  sofia_parameter_file: Template\n"
                        "  steps_to_run: [rmax, pyFAT]\n"
                        "  verbose: true\n"
                        "  ncpu: 4\n"
                        "  multiprocessing: true\n"
                        "directories:\n"
                        f"  run_directory: {tmp_path}\n"
                        "  sofia_run_directory: ''\n"
                        "  data_directory: ''\n")
    cfg = functions.process_input([f"configuration_file={settings}"])
    assert cfg.input.steps_to_run == ["rmax", "pyfat"] and cfg.input.ncpu == 4
    assert cfg.pyfat.minimum_beams_across == 3.0


def test_settings_are_read_from_file_and_command_line(tmp_path):
    cfg = functions.process_input(
        [write_settings(tmp_path, "hi_mass:\n  distance_mpc: 49.0\n"),
         "pyfat.minimum_beams_across=4", "kinematics.k5k1_max=0.08"])
    assert cfg.input.steps_to_run == ["rmax", "pyfat", "hi_mass"]
    assert cfg.hi_mass.distance_mpc == 49.0 and cfg.pyfat.minimum_beams_across == 4.0
    assert cfg.kinematics.k5k1_max == 0.08
    assert cfg.directories.run_directory == f"{tmp_path}/"
    assert cfg.directories.sofia_run_directory == cfg.directories.run_directory


def test_a_tilted_ring_model_needs_three_beams(tmp_path):
    with pytest.raises(ValueError, match="has to be 3 or larger"):
        functions.process_input([write_settings(tmp_path), "pyfat.minimum_beams_across=2"])
    cfg = functions.process_input([write_settings(tmp_path),
                                            "pyfat.minimum_beams_across=3"])
    assert cfg.pyfat.minimum_beams_across == 3.0


def test_unknown_steps_and_settings_are_refused(tmp_path):
    with pytest.raises(ValueError, match="not steps of the program"):
        functions.process_input([write_settings(tmp_path),
                                          "input.steps_to_run=[rmax,masses]"])
    with pytest.raises(Exception, match="minimum_beams"):
        functions.process_input([write_settings(tmp_path), "pyfat.minimum_beams=4"])
    with pytest.raises(FileNotFoundError):
        functions.process_input(["configuration_file=/no/such/file.yml"])


def test_source_finding_needs_a_cube(tmp_path):
    with pytest.raises(ValueError, match="neither provided a sofia parameter file nor a cube"):
        functions.process_input([write_settings(tmp_path),
                                          "input.steps_to_run=[sofia,deblend]"])


def test_steps_run_in_their_fixed_order(tmp_path, monkeypatch):
    done = []
    for step in config.STEPS:
        monkeypatch.setitem(program.STEP_FUNCTIONS, step, lambda cfg, step=step: done.append(step))
    monkeypatch.setattr(program, "start_step_log", lambda cfg: None)
    program.main([write_settings(tmp_path), "input.steps_to_run=[figures,hi_mass,rmax]"])
    assert done == ["rmax", "hi_mass", "figures"]


def test_the_known_source_table_is_one_file_or_a_list(tmp_path):
    assert functions.source_tables(None) == ["None"]
    assert functions.source_tables(["None"]) == ["None"]
    assert functions.source_tables("known.csv") == ["known.csv"]
    assert functions.source_tables(["known.csv", "more.csv"]) == ["known.csv", "more.csv"]
    for given, expected in (("known.csv", "known.csv"), ("[known.csv,more.csv]", "known.csv")):
        cfg = functions.process_input([write_settings(tmp_path),
                                       f"input.known_source_table={given}"])
        assert functions.first_source_table(cfg) == expected
    assert functions.first_source_table(functions.process_input([write_settings(tmp_path)])) is None


def test_sofia_is_started_with_the_program_of_the_settings(tmp_path, monkeypatch):
    (tmp_path / "sofia.par").write_text("input.data = cube.fits\noutput.directory = found\n")
    started = {}

    def execute(settings, run_directory, sofia_parameter_file):
        started.update(program=settings.internal.sofia, directory=run_directory,
                       parameters=sofia_parameter_file)
        return "Success"

    monkeypatch.setattr(run_scripts, "execute_sofia", execute)
    cfg = functions.process_input([f"directories.run_directory={tmp_path}",
                                   "input.sofia_parameter_file=sofia.par",
                                   "internal.sofia=/opt/bin/sofia2"])
    run_scripts.run_sofia(cfg)
    assert started == {"program": "/opt/bin/sofia2", "directory": f"{tmp_path}/",
                       "parameters": "sofia.par"}
    assert (tmp_path / "found").is_dir() and cfg.cube_name == "cube.fits"
    written = components.read_sofia_parameters(str(tmp_path / "sofia.par"))
    assert written["output.writeCubelets"] == "true" and written["output.writeMask"] == "true"
    # the deblending is run from another directory, so the output directory is written in full
    assert written["output.directory"] == f"{tmp_path}/found"


def test_the_test_data_are_complete():
    cfg = OmegaConf.merge(OmegaConf.structured(defaults),
                          OmegaConf.load(TEST_DATA / "HCG16A.yml"))
    functions.validate(cfg)
    assert "deblend" not in cfg.input.steps_to_run
    assert (TEST_DATA / functions.first_source_table(cfg)).is_file()
    parameters = components.read_sofia_parameters(
        str(TEST_DATA / cfg.input.sofia_parameter_file))
    header = fits.getheader(TEST_DATA / parameters["input.data"])
    assert (header["NAXIS1"], header["NAXIS2"], header["NAXIS3"]) == (83, 83, 105)
    assert header["BUNIT"].strip().upper() == "JY/BEAM" and header["OBJECT"] == "HCG 16A"
