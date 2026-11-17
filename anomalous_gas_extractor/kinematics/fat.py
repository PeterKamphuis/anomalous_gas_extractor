"""Tilted-ring models of the rotating discs with pyFAT.

pyFAT (Kamphuis et al. 2015) fits the brightest source of a cube. Every disc
is therefore given to it in a cube of its own, in which the emission of the
other components, and optionally the gas of the component beyond r_max, is
replaced by noise. The kinematic centre and r_max are passed on through the
catalogue columns ``xpos_min/max``, ``ypos_min/max`` and ``r_max``, which need
pyFAT 1.0.1 or later.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from astropy.io import fits
from omegaconf import OmegaConf

from anomalous_gas_extractor.kinematics.datacube import HI_REST_FREQUENCY_HZ, DataCube

logger = logging.getLogger(__name__)

RESULTS_CATALOGUE = "pyFAT_results.txt"
CATALOGUE = "pyfat_catalogue.txt"
CONFIGURATION = "pyfat_input.yml"
LOG_FILE = "pyfat_run.log"
# surface brightness (Jy km/s arcsec^-2) below which a ring of the model is empty
EMPTY_RING_BRIGHTNESS = 1e-10
ARCSEC_TO_RADIAN = np.pi / (180.0 * 3600.0)
FLAT_DISC = {"fitting": {"fixed_parameters": ["Z0", "XPOS", "YPOS", "VSYS", "INCL", "PA", "SDIS"]},
             "advanced": {"minimum_warp_size": 1000.0}}
COPIED_HEADER_KEYS = ("CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CDELT1", "CDELT2", "CUNIT1",
                      "CUNIT2", "BMAJ", "BMIN", "BPA", "BUNIT")


@dataclass(frozen=True)
class FatSettings:
    """How the input of pyFAT is prepared and how pyFAT is run."""

    executable: str = "pyFAT"
    tirific: str = "tirific"
    sofia: str = "sofia"
    ncpu: int = 1
    distance_mpc: Optional[float] = None
    flat_disc: bool = True
    force_fit: bool = True
    cut_beyond_rmax_beams: Optional[float] = 0.5
    spatial_margin_beams: float = 3.0
    velocity_half_width_kms: float = 300.0
    centre_tolerance_beams: float = 0.5
    random_seed: int = 1
    overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FatTarget:
    """A rotating disc to be modelled."""

    name: str
    label: int
    ra_deg: float
    dec_deg: float
    vsys_kms: float
    r_max_arcsec: float


@dataclass
class FatModel:
    """Final tilted-ring model of a disc.

    The inclination and position angle are those of the outermost ring with
    emission; ``disc_radius_arcsec`` is the radius at which the model ends.
    """

    converged: bool
    message: str = ""
    ra_deg: float = np.nan
    dec_deg: float = np.nan
    vsys_kms: float = np.nan
    inclination_deg: float = np.nan
    position_angle_deg: float = np.nan
    vrot_max_kms: float = np.nan
    disc_radius_arcsec: float = np.nan
    definition_file: Optional[str] = None
    model_cube: Optional[str] = None

    @property
    def usable(self) -> bool:
        return self.converged and np.isfinite(self.disc_radius_arcsec)


def write_input_cube(cube: DataCube, target: FatTarget, settings: FatSettings,
                     file_name: str) -> None:
    """Write the cube of one disc: a cut-out with all other emission replaced by noise.

    The spectral axis is written as a linear velocity axis (VELO, m/s) that
    increases with channel number, which is the form pyFAT handles best.
    """
    pixel, beam = cube.pixel_arcsec, cube.beam_major_arcsec
    x_centre, y_centre = cube.world_to_pixel(target.ra_deg, target.dec_deg)
    rows, columns = np.indices(cube.data.shape[1:])
    radius = np.hypot(columns - x_centre, rows - y_centre) * pixel

    cut = None
    if settings.cut_beyond_rmax_beams is not None:
        cut = target.r_max_arcsec + settings.cut_beyond_rmax_beams * beam
        extent = cut
    else:
        extent = radius[(cube.labels == target.label).any(axis=0)].max()
    half_size = int(np.ceil((extent + settings.spatial_margin_beams * beam) / pixel))
    x_low = max(int(round(x_centre)) - half_size, 0)
    x_high = min(int(round(x_centre)) + half_size + 1, cube.data.shape[2])
    y_low = max(int(round(y_centre)) - half_size, 0)
    y_high = min(int(round(y_centre)) + half_size + 1, cube.data.shape[1])
    channels = np.nonzero(np.abs(cube.velocity_kms - target.vsys_kms)
                          <= settings.velocity_half_width_kms)[0]
    if channels.size < 3:
        raise ValueError(f"{target.name}: v_sys {target.vsys_kms:.0f} km/s is outside the cube")
    box = (slice(channels.min(), channels.max() + 1), slice(y_low, y_high), slice(x_low, x_high))

    data = cube.data[box].copy()
    labels = cube.labels[box]
    replace = ~np.isfinite(data) | ((labels > 0) & (labels != target.label))
    if cut is not None:
        replace |= (labels == target.label) & (radius[box[1:]] > cut)[None]
    rng = np.random.default_rng(settings.random_seed)
    data[replace] = rng.normal(0.0, cube.rms, int(replace.sum()))

    velocity = cube.velocity_kms[box[0]] * 1e3
    if velocity[-1] < velocity[0]:
        data, velocity = data[::-1], velocity[::-1]
    reference = int(np.argmin(np.abs(velocity - target.vsys_kms * 1e3)))

    header = fits.Header()
    for key in COPIED_HEADER_KEYS:
        if key in cube.header:
            header[key] = cube.header[key]
    header["CRPIX1"] = cube.header["CRPIX1"] - x_low
    header["CRPIX2"] = cube.header["CRPIX2"] - y_low
    header["CTYPE3"], header["CUNIT3"] = "VELO", "m/s"
    header["CRPIX3"], header["CRVAL3"] = reference + 1, float(velocity[reference])
    header["CDELT3"] = float(np.gradient(velocity)[reference])
    header["EQUINOX"] = cube.header.get("EQUINOX", 2000.0)
    header["EPOCH"] = header["EQUINOX"]
    header["RESTFREQ"] = cube.header.get("RESTFRQ", cube.header.get("RESTFREQ", HI_REST_FREQUENCY_HZ))
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    fits.writeto(file_name, data.astype(np.float32), header, overwrite=True)


def write_catalogue(file_name: str, targets: List[FatTarget], cube: DataCube,
                    settings: FatSettings) -> None:
    """Write the catalogue of pyFAT with r_max and the limits of the centre."""
    tolerance = settings.centre_tolerance_beams * cube.beam_major_arcsec / 3600.0
    distance = -1 if settings.distance_mpc is None else settings.distance_mpc
    lines = ["id|distance|directoryname|cubename|r_max|xpos_min|xpos_max|ypos_min|ypos_max"]
    for number, target in enumerate(targets, start=1):
        ra_tolerance = tolerance / np.cos(np.radians(target.dec_deg))
        lines.append("|".join([
            str(number), str(distance), target.name, target.name, f"{target.r_max_arcsec:.2f}",
            f"{target.ra_deg - ra_tolerance:.6f}", f"{target.ra_deg + ra_tolerance:.6f}",
            f"{target.dec_deg - tolerance:.6f}", f"{target.dec_deg + tolerance:.6f}"]))
    with open(file_name, "w") as catalogue:
        catalogue.write("\n".join(lines) + "\n")


def write_configuration(file_name: str, main_directory: str, settings: FatSettings) -> None:
    """Write the configuration file of pyFAT.

    ``settings.overrides`` follows the layout of the pyFAT configuration and
    takes precedence over everything that is set here.
    """
    configuration = OmegaConf.create({
        "ncpu": settings.ncpu,
        "input": {"main_directory": os.path.join(main_directory, ""),
                  "catalogue": os.path.join(main_directory, CATALOGUE),
                  "tirific": settings.tirific, "sofia2": settings.sofia},
        "output": {"catalogue": os.path.join(main_directory, RESULTS_CATALOGUE)},
        "fitting": {"fitting_stages": ["Create_FAT_Cube", "Run_Sofia", "Fit_Tirific_OSC"]},
        "advanced": {"force_fit": settings.force_fit},
    })
    if settings.flat_disc:
        configuration = OmegaConf.merge(configuration, FLAT_DISC)
    configuration = OmegaConf.merge(configuration, settings.overrides)
    OmegaConf.save(configuration, file_name)


def find_executable(name: str) -> str:
    """Full path of a program on the search path or next to the Python interpreter."""
    found = shutil.which(name)
    if found is None:
        beside_python = os.path.join(os.path.dirname(sys.executable), name)
        found = beside_python if os.path.isfile(beside_python) else None
    if found is None:
        raise FileNotFoundError(f"the program {name!r} cannot be found")
    return found


def run_pyfat(main_directory: str, settings: FatSettings) -> int:
    """Run pyFAT on the prepared directory; returns its exit status."""
    command = [find_executable(settings.executable),
               f"configuration_file={os.path.join(main_directory, CONFIGURATION)}"]
    log_file = os.path.join(main_directory, LOG_FILE)
    logger.info("running %s (output in %s)", " ".join(command), log_file)
    with open(log_file, "w") as log:
        status = subprocess.run(command, cwd=main_directory, stdout=log,
                                stderr=subprocess.STDOUT, check=False).returncode
    if status != 0:
        logger.warning("pyFAT ended with exit status %d; see %s", status, log_file)
    return status


def read_fit_status(results_file: str) -> Dict[str, FatModel]:
    """Outcome of every fit as listed in the results catalogue of pyFAT."""
    status = {}
    if not os.path.isfile(results_file):
        return status
    with open(results_file) as results:
        for line in results.readlines()[1:]:
            parts = line.split(None, 2)
            if len(parts) >= 2:
                status[parts[0]] = FatModel(converged=parts[1] == "True",
                                            message=parts[2].strip() if len(parts) > 2 else "")
    return status


def read_tirific_def(file_name: str) -> Dict[str, np.ndarray]:
    """Numerical parameters of a TiRiFiC .def file, by keyword.

    pyFAT writes the uncertainties as comment lines ``# <keyword>_ERR= ...``;
    these are returned under ``<keyword>_ERR``.
    """
    parameters = {}
    with open(file_name) as definition:
        for line in definition:
            key, separator, value = line.partition("=")
            if not separator:
                continue
            try:
                parameters[key.strip("# \t").upper()] = np.array(value.split(), dtype=float)
            except ValueError:
                continue
    return parameters


def model_rings(parameters: Dict[str, np.ndarray],
                distance_mpc: Optional[float] = None) -> List[Dict[str, float]]:
    """The rings of a tilted-ring model as a table, one row per ring.

    Disc 1 of the models of pyFAT is the approaching side, disc 2 the receding
    side; a model with one disc has the same values on both sides. Uncertainties
    that the .def file does not list are NaN.
    """
    radii = parameters["RADI"]

    def column(key: str, fallback: Optional[str] = None) -> np.ndarray:
        values = parameters.get(key, parameters.get(fallback) if fallback else None)
        if values is None or len(values) != len(radii):
            return np.full(len(radii), np.nan)
        return values

    table = {"radius_arcsec": radii}
    if distance_mpc is not None:
        table["radius_kpc"] = radii * ARCSEC_TO_RADIAN * distance_mpc * 1e3
    for name, key in (("sbr", "SBR"), ("vrot_kms", "VROT"), ("inclination_deg", "INCL"),
                      ("position_angle_deg", "PA"), ("dispersion_kms", "SDIS")):
        for side, suffix in (("approaching", ""), ("receding", "_2")):
            table[f"{name}_{side}"] = column(key + suffix, fallback=key)
            table[f"{name}_{side}_error"] = column(f"{key}{suffix}_ERR")
    return [{key: float(values[i]) for key, values in table.items()} for i in range(len(radii))]


def model_edge(radii: np.ndarray, brightness: np.ndarray) -> float:
    """Radius at which the surface brightness of a tilted-ring model reaches zero.

    TiRiFiC interpolates linearly between the rings. The disc therefore ends at
    the first ring without emission beyond the last ring with emission, or at
    the last ring of the model if that one still has emission.
    """
    filled = np.nonzero(brightness > EMPTY_RING_BRIGHTNESS)[0]
    if filled.size == 0:
        return np.nan
    return float(radii[min(filled[-1] + 1, len(radii) - 1)])


def read_model(main_directory: str, name: str) -> FatModel:
    """Read the final model of one disc from the output of pyFAT."""
    results = read_fit_status(os.path.join(main_directory, RESULTS_CATALOGUE))
    model = results.get(name, FatModel(converged=False, message="not in the results of pyFAT"))
    final = os.path.join(main_directory, name, "Finalmodel")
    definition_file = os.path.join(final, "Finalmodel.def")
    if not os.path.isfile(definition_file):
        model.converged = False
        model.message = model.message or "pyFAT wrote no final model"
        return model
    rings = read_tirific_def(definition_file)
    brightness = np.maximum(rings["SBR"], rings.get("SBR_2", rings["SBR"]))
    filled = np.nonzero(brightness > EMPTY_RING_BRIGHTNESS)[0]
    if filled.size == 0:
        model.converged = False
        model.message = "the final model has no emission"
        return model
    outer = filled[-1]
    model.definition_file = definition_file
    model.disc_radius_arcsec = model_edge(rings["RADI"], brightness)
    model.ra_deg, model.dec_deg = float(rings["XPOS"][0]), float(rings["YPOS"][0])
    model.vsys_kms = float(rings["VSYS"][0])
    model.inclination_deg = float(rings["INCL"][outer])
    model.position_angle_deg = float(rings["PA"][outer])
    model.vrot_max_kms = float(np.max(rings["VROT"][:outer + 1]))
    model_cube = os.path.join(final, "Finalmodel.fits")
    model.model_cube = model_cube if os.path.isfile(model_cube) else None
    return model
