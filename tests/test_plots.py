import numpy as np
import pytest

from anomalous_gas_extractor.kinematics import fat, plots


@pytest.fixture
def rings():
    radii = np.array([0.0, 8.0, 16.0, 24.0, 32.0])
    parameters = {
        "RADI": radii,
        "SBR": np.array([3e-4, 3e-4, 5e-4, 2e-4, 0.0]),
        "SBR_2": np.array([3e-4, 3e-4, 4e-4, 1e-4, 5e-5]),
        "SBR_ERR": np.full(5, 2e-5),
        "VROT": np.array([0.0, 60.0, 110.0, 130.0, 130.0]),
        "VROT_ERR": np.array([0.0, 20.0, 8.0, 6.0, 15.0]),
        "INCL": np.full(5, 55.0), "PA": np.full(5, 46.0), "SDIS": np.full(5, 12.0),
    }
    return fat.model_rings(parameters, distance_mpc=49.0)


@pytest.mark.parametrize("extension", ["png", "pdf"])
def test_model_profiles_are_written(rings, tmp_path, extension):
    figure = tmp_path / f"model.{extension}"
    plots.plot_model_profiles(rings, str(figure), "DISC: tilted-ring model",
                              disc_radius_arcsec=32.0, r_max_arcsec=30.0, distance_mpc=49.0)
    assert figure.stat().st_size > 5000


def test_model_profiles_without_radii_and_distance(rings, tmp_path):
    figure = tmp_path / "model.png"
    plots.plot_model_profiles(rings, str(figure), "DISC")
    assert figure.stat().st_size > 5000


def test_models_side_by_side(rings, tmp_path):
    models = [plots.ModelProfiles("A", rings, 32.0, 30.0), plots.ModelProfiles("B", rings[:4], 24.0)]
    figure = tmp_path / "models.png"
    plots.plot_model_grid(models, str(figure), distance_mpc=49.0)
    assert figure.stat().st_size > 5000


def test_unknown_font_file():
    with pytest.raises(FileNotFoundError):
        plots.figure_style("/no/such/font.ttf")
    assert "font.family" not in plots.figure_style()
