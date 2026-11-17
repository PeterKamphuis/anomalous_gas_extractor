"""Rotating discs and anomalous gas of the HI components of a data cube.

The HI of every galaxy in a data cube is divided into a regularly rotating disc
and anomalous gas (tails, bridges, clouds). The steps are

sofia       source finding with SoFiA-2 (run by AGE)
deblend     blended detections are divided with deblend_sofia_detections
asymmetry   asymmetry indicators of AGE
components  one label for every component, on the cube used for the kinematics
rmax        kinematic centre and outer radius of the rotating disc (r_max) of
            every component with Kinemetry; a component hosts a rotating disc
            if its regular rings are resolved and dominated by rotation
pyfat       tilted-ring model, limited to r_max, of every rotating disc that
            spans enough beams along its major axis
hi_mass     HI mass inside and outside the rotating disc. The disc is that of
            the tilted-ring model or, for a rotating disc without model, the
            ellipse of Kinemetry with semi-major axis r_max
figures     figures of all steps

Run with

    python -m anomalous_gas_extractor.anomalous_gas_extractor configuration_file=run.yml

``print_examples=true`` writes a file with all settings and their defaults.
"""

import os
import sys

if __name__ == "__main__" and not __package__:
    # started as a file: its directory would hide the package of the same name
    _here = os.path.dirname(os.path.abspath(__file__))
    if sys.path and os.path.abspath(sys.path[0]) == _here:
        sys.path[0] = os.path.dirname(_here)

import json
import logging
from typing import Callable, Dict, List, Optional

from omegaconf import OmegaConf

from deblend_sofia_detections.config.config import defaults as deblend_defaults
from deblend_sofia_detections.main import main_with_input

from anomalous_gas_extractor.kinematics import components, pipeline
from anomalous_gas_extractor.kinematics.configuration import STEPS, read_configuration
from anomalous_gas_extractor.run_scripts.run_scripts import run_asymmetry, run_sofia
from anomalous_gas_extractor.support.logging import logger as age_logger

__author__ = "Roger Ianjamasimanana"

LOGGER_NAME = "anomalous_gas_extractor.kinematics"
LOG_FILE = "run_kinematics.log"
DEBLEND_LOG_FILE = "run_deblend.log"
# the name under which AGE starts SoFiA-2, and the parameter file it writes for the template
AGE_SOFIA_PROGRAM = "sofia"
AGE_SOFIA_PARAMETER_FILE = "sofia_parameter_file.par"

logger = logging.getLogger(f"{LOGGER_NAME}.steps")


def setup_log(cfg) -> None:
    """Send the messages of the steps to the log file and, if verbose, to the screen."""
    log = logging.getLogger(LOGGER_NAME)
    if log.handlers:
        return
    log.setLevel(logging.DEBUG if "ALL" in cfg.logging.debug_functions else logging.INFO)
    layout = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    if cfg.logging.enable_log:
        directory = f"{cfg.directories.run_directory}{cfg.logging.log_directory}"
        os.makedirs(directory, exist_ok=True)
        to_file = logging.FileHandler(os.path.join(directory, LOG_FILE))
        to_file.setFormatter(layout)
        log.addHandler(to_file)
    if cfg.input.verbose or not cfg.logging.enable_log:
        to_screen = logging.StreamHandler()
        to_screen.setFormatter(layout)
        log.addHandler(to_screen)


# ---------------------------------------------------------------------------
# step sofia
# ---------------------------------------------------------------------------

def make_sofia_available(program: str) -> None:
    """Put the directory of SoFiA-2 on the search path: AGE starts it by its name."""
    directory, name = os.path.split(program)
    if name != AGE_SOFIA_PROGRAM:
        raise ValueError(f"internal.sofia is {program}: AGE starts SoFiA-2 as "
                         f"'{AGE_SOFIA_PROGRAM}', so the program has to have that name")
    if directory:
        os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")


def make_sofia_output_directory(cfg) -> None:
    """Create the output directory named in the parameter file; SoFiA-2 stops without it."""
    if cfg.input.sofia_parameter_file == "Template":
        return
    run_directory = cfg.directories.sofia_run_directory
    settings = components.read_sofia_parameters(
        os.path.join(run_directory, cfg.input.sofia_parameter_file))
    if settings.get("output.directory"):
        os.makedirs(os.path.join(run_directory, settings["output.directory"]), exist_ok=True)


def find_sources(cfg) -> None:
    """Step sofia: source finding with SoFiA-2, run by AGE."""
    make_sofia_available(cfg.internal.sofia)
    make_sofia_output_directory(cfg)
    run_sofia(cfg)


# ---------------------------------------------------------------------------
# step deblend
# ---------------------------------------------------------------------------

def flatten_options(options: dict, prefix: str = "") -> Dict[str, object]:
    """Nested settings as a dictionary with dotted keys, e.g. {'input.maximum_no_peaks': 8}."""
    flat = {}
    for key, value in options.items():
        if isinstance(value, dict):
            flat.update(flatten_options(value, prefix=f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def deblend_settings() -> Dict[str, object]:
    """The settings of the installed deblend_sofia_detections with their defaults."""
    return flatten_options(OmegaConf.to_container(OmegaConf.structured(deblend_defaults)))


def deblend_arguments(cfg, log_directory: str,
                      known_settings: Optional[Dict[str, object]] = None) -> List[str]:
    """The arguments with which deblend_sofia_detections is started.

    The settings that follow from the configuration of this program come first,
    then the settings of ``deblend.options``, which are passed on unchanged and
    take precedence. The names of some settings differ between versions of
    deblend_sofia_detections; ``known_settings`` are those of the installed one.
    """
    if known_settings is None:
        known_settings = deblend_settings()
    settings = {
        "input.sofia_parameters": f"{cfg.directories.sofia_run_directory}"
                                  f"{cfg.input.sofia_parameter_file}",
        "logging.log_directory": log_directory,
        "logging.log_file": DEBLEND_LOG_FILE,
        "directories.run_directory": cfg.directories.sofia_run_directory}
    if "internal.sofia" in known_settings:
        settings["internal.sofia"] = cfg.internal.sofia
    if "input.use_peak_deblending" in known_settings:
        settings["input.use_peak_deblending"] = False
    if cfg.input.known_source_table is not None:
        table_setting = ("input.manual_input_tables"
                         if "input.manual_input_tables" in known_settings else "input.tables")
        settings[table_setting] = [cfg.input.known_source_table]
    if "RUN_DEBLEND" in cfg.logging.debug_functions or "ALL" in cfg.logging.debug_functions:
        settings["logging.debug_functions"] = list(cfg.logging.debug_functions)

    options = flatten_options(OmegaConf.to_container(cfg.deblend.options, resolve=True))
    unknown = [key for key in options if key not in known_settings]
    if unknown:
        raise ValueError(f"deblend.options: {unknown} are not settings of the installed "
                         f"deblend_sofia_detections. Its settings are: {sorted(known_settings)}")
    settings.update(options)
    return [f"{key}={json.dumps(value)}" for key, value in settings.items()]


def deblend(cfg) -> None:
    """Step deblend: divide the blended detections of SoFiA-2."""
    log = age_logger(cfg, log_directory=f"{cfg.directories.run_directory}"
                                        f"{cfg.logging.log_directory}",
                     log_file=DEBLEND_LOG_FILE)
    log.print_log(f"Running deblend in the directory {cfg.directories.sofia_run_directory} with "
                  f"the sofia parameter file {cfg.input.sofia_parameter_file}")
    arguments = deblend_arguments(cfg, log.log_directory)
    # deblend_sofia_detections stops if the directory for its downloads does not exist
    options = flatten_options(OmegaConf.to_container(cfg.deblend.options, resolve=True))
    downloads = options.get("directories.ancillary_directory")
    if downloads and os.path.isabs(downloads):
        os.makedirs(downloads, exist_ok=True)
    log.print_log(f"The deblend step is being run with the following arguments: {arguments}",
                  case=["debug"])
    main_with_input(arguments)


# ---------------------------------------------------------------------------
# the other steps
# ---------------------------------------------------------------------------

def make_figures(cfg) -> None:
    """Step figures: the figures of all steps, one directory for every kind."""
    from anomalous_gas_extractor.kinematics import figures
    figures.make_figures(cfg)


STEP_FUNCTIONS: Dict[str, Callable] = {
    "sofia": find_sources,
    "deblend": deblend,
    "asymmetry": run_asymmetry,
    "components": pipeline.run_components,
    "rmax": pipeline.run_rmax,
    "pyfat": pipeline.run_pyfat,
    "hi_mass": pipeline.run_hi_mass,
    "figures": make_figures,
}


def run_steps(cfg) -> None:
    """Run the requested steps in their fixed order."""
    setup_log(cfg)
    if "sofia" not in cfg.input.steps_to_run and cfg.input.sofia_parameter_file == "Template":
        # the parameter file that an earlier run of the step sofia wrote for the template
        cfg.input.sofia_parameter_file = AGE_SOFIA_PARAMETER_FILE
    for step in STEPS:
        if step in cfg.input.steps_to_run:
            logger.info("step %s", step)
            STEP_FUNCTIONS[step](cfg)


def main(argv: Optional[List[str]] = None) -> None:
    run_steps(read_configuration(sys.argv[1:] if argv is None else list(argv)))


if __name__ == "__main__":
    main()
