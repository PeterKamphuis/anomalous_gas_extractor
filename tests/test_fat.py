import os

import numpy as np
import pytest
from astropy.io import fits
from omegaconf import OmegaConf

from anomalous_gas_extractor.kinematics import fat

from conftest import DISC, NOISE, VSYS_KMS

DEFINITION = """# final model
NUR= 4
RADI= +0.00000E+00 +8.00000E+00 +1.60000E+01 +2.40000E+01
VROT= +0.00000E+00 +9.00000E+01 +1.50000E+02 +1.40000E+02
SBR= 2.4e-04 2.4e-04 4.7e-04 0.0e+00
SBR_2= 2.4e-04 2.4e-04 4.2e-04 1.0e-16
INCL= +5.5E+01 +5.5E+01 +5.6E+01 +5.7E+01
PA= +4.6E+01 +4.6E+01 +4.7E+01 +4.8E+01
XPOS= +3.2336E+01 +3.2336E+01 +3.2336E+01 +3.2336E+01
YPOS= -1.0132E+01 -1.0132E+01 -1.0132E+01 -1.0132E+01
VSYS= +3.845E+03 +3.845E+03 +3.845E+03 +3.845E+03
INSET= cube_FAT.fits
"""


@pytest.fixture
def target(cube):
    ra, dec = cube.pixel_to_world(DISC["x"], DISC["y"])
    return fat.FatTarget(name="DISC", label=1, ra_deg=ra, dec_deg=dec, vsys_kms=VSYS_KMS,
                         r_max_arcsec=40.0)


def pyfat_output(directory, status="True", definition=DEFINITION):
    os.makedirs(directory / "DISC" / "Finalmodel")
    (directory / fat.RESULTS_CATALOGUE).write_text(
        "Directory Name     OS Comments on Fit Result\n"
        f"DISC    {status} The fit has converged succesfully \n")
    if definition:
        (directory / "DISC" / "Finalmodel" / "Finalmodel.def").write_text(definition)


def test_input_cube_holds_only_the_inner_disc(cube, target, tmp_path):
    cube_file = str(tmp_path / "DISC" / "DISC.fits")
    settings = fat.FatSettings(cut_beyond_rmax_beams=0.5, spatial_margin_beams=3.0,
                               velocity_half_width_kms=150.0)
    fat.write_input_cube(cube, target, settings, cube_file)
    with fits.open(cube_file) as hdul:
        data, header = hdul[0].data, hdul[0].header
    assert header["CTYPE3"] == "VELO" and header["CDELT3"] > 0
    assert header["CRVAL3"] == pytest.approx(VSYS_KMS * 1e3)
    assert np.isfinite(data).all()

    # the centre of the disc is where it was, the cut-out is centred on it
    half = (data.shape[2] - 1) // 2
    assert header["CRPIX1"] == cube.header["CRPIX1"] - (round(DISC["x"]) - half)
    peak = data.max(axis=0)
    rows, columns = np.indices(peak.shape)
    radius = np.hypot(columns - half, rows - half) * cube.pixel_arcsec
    cut = target.r_max_arcsec + 0.5 * cube.beam_major_arcsec
    assert peak[radius < 20.0].min() > 20 * NOISE
    assert peak[radius > cut + 1.0].max() < 12 * NOISE


def test_input_cube_without_cut_keeps_the_component_and_removes_the_others(cube, target, tmp_path):
    cube_file = str(tmp_path / "DISC" / "DISC.fits")
    settings = fat.FatSettings(cut_beyond_rmax_beams=None, spatial_margin_beams=20.0)
    fat.write_input_cube(cube, target, settings, cube_file)
    data = fits.getdata(cube_file)
    assert data.shape[1:] == cube.data.shape[1:]
    peak = data.max(axis=0)
    assert peak[10, 54] < 12 * NOISE                      # the cloud is gone
    assert peak[int(DISC["y"]) + 9, int(DISC["x"])] > 20 * NOISE


def test_catalogue_carries_rmax_and_the_centre_limits(cube, target, tmp_path):
    catalogue = tmp_path / fat.CATALOGUE
    fat.write_catalogue(str(catalogue), [target], cube, fat.FatSettings(distance_mpc=55.0))
    header, line = catalogue.read_text().splitlines()
    row = dict(zip(header.split("|"), line.split("|")))
    assert row["directoryname"] == row["cubename"] == "DISC"
    assert float(row["distance"]) == 55.0 and float(row["r_max"]) == 40.0
    half_beam = 0.5 * cube.beam_major_arcsec / 3600.0
    assert float(row["ypos_max"]) - float(row["ypos_min"]) == pytest.approx(2 * half_beam, abs=2e-6)
    assert float(row["xpos_min"]) < target.ra_deg < float(row["xpos_max"])


def test_configuration(tmp_path):
    overrides = {"fitting": {"ring_size": 1.5}, "advanced": {"force_fit": False}}
    settings = fat.FatSettings(ncpu=3, overrides=overrides)
    fat.write_configuration(str(tmp_path / fat.CONFIGURATION), str(tmp_path), settings)
    configuration = OmegaConf.load(tmp_path / fat.CONFIGURATION)
    assert configuration.ncpu == 3
    assert configuration.fitting.ring_size == 1.5 and not configuration.advanced.force_fit
    assert "INCL" in configuration.fitting.fixed_parameters
    assert configuration.input.catalogue == str(tmp_path / fat.CATALOGUE)

    fat.write_configuration(str(tmp_path / fat.CONFIGURATION), str(tmp_path),
                            fat.FatSettings(flat_disc=False))
    assert "fixed_parameters" not in OmegaConf.load(tmp_path / fat.CONFIGURATION).fitting


def test_model_edge():
    radii = np.array([0.0, 8.0, 16.0, 24.0])
    assert fat.model_edge(radii, np.array([2e-4, 2e-4, 1e-4, 0.0])) == 24.0
    assert fat.model_edge(radii, np.array([2e-4, 2e-4, 1e-4, 5e-5])) == 24.0
    assert fat.model_edge(radii, np.array([2e-4, 1e-4, 1e-16, 0.0])) == 16.0
    assert np.isnan(fat.model_edge(radii, np.zeros(4)))


def test_model_ends_where_its_brightness_reaches_zero(tmp_path):
    pyfat_output(tmp_path)
    model = fat.read_model(str(tmp_path), "DISC")
    assert model.usable
    assert model.disc_radius_arcsec == 24.0
    assert (model.inclination_deg, model.position_angle_deg) == (56.0, 47.0)
    assert model.vrot_max_kms == 150.0 and model.vsys_kms == 3845.0
    assert model.model_cube is None


def test_ring_table_of_the_model(tmp_path):
    pyfat_output(tmp_path, definition=DEFINITION + "# VROT_ERR= 0. 12. 6. 9.\n")
    parameters = fat.read_tirific_def(str(tmp_path / "DISC" / "Finalmodel" / "Finalmodel.def"))
    assert parameters["VROT_ERR"].tolist() == [0.0, 12.0, 6.0, 9.0]
    rings = fat.model_rings(parameters, distance_mpc=10.0)
    assert len(rings) == 4
    third = rings[2]
    assert third["radius_arcsec"] == 16.0
    assert third["radius_kpc"] == pytest.approx(16.0 / 206264.806 * 1e4)
    assert (third["sbr_approaching"], third["sbr_receding"]) == (4.7e-4, 4.2e-4)
    assert third["vrot_kms_approaching_error"] == 6.0
    # the model lists one rotation curve: it holds for both sides, without error for the second
    assert third["vrot_kms_receding"] == third["vrot_kms_approaching"] == 150.0
    assert np.isnan(third["vrot_kms_receding_error"])
    assert "radius_kpc" not in fat.model_rings(parameters)[0]


def test_failed_fits_are_not_usable(tmp_path):
    pyfat_output(tmp_path, status="False")
    assert not fat.read_model(str(tmp_path), "DISC").usable
    assert not fat.read_model(str(tmp_path), "OTHER").usable


def test_missing_final_model(tmp_path):
    pyfat_output(tmp_path, definition=None)
    model = fat.read_model(str(tmp_path), "DISC")
    assert not model.converged and not model.usable


def test_unknown_program():
    with pytest.raises(FileNotFoundError):
        fat.find_executable("no-such-program-anywhere")
