"""The test data are run with the program itself.

These tests need SoFiA-2 and Kinemetry and are skipped without them. SoFiA-2 is
looked for as ``sofia`` on the search path or in the environment variable
AGE_SOFIA; the directory of kinemetry.py can be given in KINEMETRY_PATH. Nothing
is downloaded. The tilted-ring models are left out, so the tests take a few
minutes at most.
"""

import csv
import os
import shutil
from pathlib import Path

import pytest

from anomalous_gas_extractor import anomalous_gas_extractor as program
from anomalous_gas_extractor.kinematics import kinemetry_rings

TEST_DATA = Path(__file__).resolve().parents[1] / "test_data"
SOFIA = shutil.which(os.environ.get("AGE_SOFIA", "sofia"))


def kinemetry_is_available() -> bool:
    kinemetry_rings.set_kinemetry_path(os.environ.get("KINEMETRY_PATH"))
    try:
        kinemetry_rings.load_kinemetry()
    except ImportError:
        return False
    return True


pytestmark = [
    pytest.mark.skipif(SOFIA is None, reason="SoFiA-2 is not installed"),
    pytest.mark.skipif(not kinemetry_is_available(), reason="Kinemetry is not installed"),
]


def run_test_data(name: str, tmp_path, *settings: str) -> dict:
    """Run the program in a copy of a test data directory; the summary by component."""
    run_directory = tmp_path / name
    shutil.copytree(TEST_DATA / name, run_directory)
    arguments = [f"configuration_file={run_directory / (name + '.yml')}",
                 f"directories.run_directory={run_directory}", f"internal.sofia={SOFIA}",
                 "input.ncpu=4", "input.verbose=false", *settings]
    if os.environ.get("KINEMETRY_PATH"):
        arguments.append(f"kinematics.kinemetry_path={os.environ['KINEMETRY_PATH']}")
    program.main(arguments)
    with open(run_directory / "component_summary.csv") as table:
        summary = {row["component"]: row for row in csv.DictReader(table)}
    assert (run_directory / "figures" / "rmax").is_dir()
    return summary


def test_one_galaxy_is_divided_into_disc_and_anomalous_gas(tmp_path):
    summary = run_test_data("HCG16A", tmp_path,
                            "input.steps_to_run=[sofia,components,rmax,hi_mass,figures]")
    assert list(summary) == ["HCG16A"]
    galaxy = summary["HCG16A"]
    assert galaxy["rotating_disc"] == "True"
    assert float(galaxy["r_max_arcsec"]) == pytest.approx(38.7, abs=1.5)
    assert float(galaxy["total_hi_mass_msun"]) == pytest.approx(1.01e9, rel=0.03)
    # without a tilted-ring model the disc is the ellipse of Kinemetry
    assert galaxy["disc_radius_source"] != "pyFAT"
    assert 0.7 < float(galaxy["disc_fraction"]) < 0.95


def test_two_blended_galaxies_are_deblended_and_divided(tmp_path):
    summary = run_test_data("HCG16AB", tmp_path)
    assert sorted(summary) == ["HCG16A", "HCG16B"]
    expected = {"HCG16A": (34.1, 1.02e9, 0.80), "HCG16B": (26.4, 5.97e8, 0.47)}
    for name, (r_max, hi_mass, disc_fraction) in expected.items():
        galaxy = summary[name]
        assert galaxy["rotating_disc"] == "True"
        assert float(galaxy["r_max_arcsec"]) == pytest.approx(r_max, abs=2.0)
        assert float(galaxy["total_hi_mass_msun"]) == pytest.approx(hi_mass, rel=0.05)
        assert float(galaxy["disc_fraction"]) == pytest.approx(disc_fraction, abs=0.08)
