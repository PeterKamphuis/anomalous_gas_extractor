"""The steps of AGE. Each one takes the settings, writes its results to disk and reads
the results of the earlier steps from disk, so the steps can also be run one at a time."""

import json
import os
import sys
from typing import Dict, List, Optional

from omegaconf import OmegaConf

from deblend_sofia_detections.config.config import defaults as deblend_defaults
from deblend_sofia_detections.deblending.sofia_functions import (
    execute_sofia, load_sofia_input_file, write_sofia)
from deblend_sofia_detections.main import main_with_input

from anomalous_gas_extractor.config.config import NO_TABLE
from anomalous_gas_extractor.config.functions import source_tables
from anomalous_gas_extractor.kinematics import pipeline
from anomalous_gas_extractor.support.logging import logger

SOFIA_LOG_FILE = "run_sofia.log"
DEBLEND_LOG_FILE = "run_deblend.log"
# directory, below the run directory, for the images and tables the deblending downloads
DEBLEND_DOWNLOADS = "ancillary_data"
# the parameter file that is written when the template of SoFiA-2 is used
TEMPLATE_PARAMETER_FILE = "sofia_parameter_file.par"


def _log_directory(cfg) -> str:
    return f"{cfg.directories.run_directory}{cfg.logging.log_directory}"


# ---------------------------------------------------------------------------
# step sofia
# ---------------------------------------------------------------------------

def run_sofia(cfg) -> None:
    """Step sofia: source finding with SoFiA-2."""
    sofia_logger = logger(cfg, log_directory=_log_directory(cfg), log_file=SOFIA_LOG_FILE)
    sofia_logger.print_log(f"Running Sofia in the directory {cfg.directories.sofia_run_directory} "
                           f"with the parameter file {cfg.input.sofia_parameter_file}")

    if cfg.input.sofia_parameter_file == "Template":
        parameters = load_sofia_input_file(filename="Template")
        parameters["input.data"] = cfg.cube_name
        parameters["reliability.enable"] = "true"
        cfg.input.sofia_parameter_file = TEMPLATE_PARAMETER_FILE
    else:
        parameters = load_sofia_input_file(
            filename=f"{cfg.directories.sofia_run_directory}{cfg.input.sofia_parameter_file}")
        cfg.cube_name = parameters["input.data"]
        # the deblending needs the cubelets, the mask and the catalogue
        parameters["output.writeCubelets"] = "true"
        parameters["output.writeMask"] = "true"
        parameters["output.writeCatXML"] = "true"
    parameters["pipeline.threads"] = cfg.input.ncpu
    output_directory = str(parameters.get("output.directory", "")).strip()
    if output_directory:
        # SoFiA-2 starts a relative name at the run directory, the deblending at the directory
        # of the cube; SoFiA-2 stops if the directory does not exist
        output_directory = os.path.join(cfg.directories.sofia_run_directory, output_directory)
        parameters["output.directory"] = output_directory
        os.makedirs(output_directory, exist_ok=True)
    write_sofia(parameters,
                f"{cfg.directories.sofia_run_directory}{cfg.input.sofia_parameter_file}")

    sofia_settings = OmegaConf.create({
        "logging": {"verbose_screen": "false",
                    "debug_functions": cfg.logging.debug_functions,
                    "enable_log": "true",
                    "verbose_log": cfg.logging.verbose,
                    "log_file": sofia_logger.log_file,
                    "log_directory": sofia_logger.log_directory,
                    "enable": "true"},
        "internal": {"sofia": cfg.internal.sofia},
    })
    sofia_output = execute_sofia(sofia_settings,
                                 run_directory=cfg.directories.sofia_run_directory,
                                 sofia_parameter_file=cfg.input.sofia_parameter_file)

    if sofia_output == "Success":
        sofia_logger.print_log("Sofia has finished running successfully",
                               case=["main", "screen"])
        return
    if sofia_output == "No sources found":
        sofia_logger.print_log("Sofia has finished running. It didn't find any sources, you "
                               "need to adapt your sofia settings", case=["main", "screen"])
        sys.exit(8)
    if sofia_output.startswith("No negative sources found"):
        sofia_logger.print_log("Sofia has finished running. It didn't find any negative "
                               "sources, you need to adapt your sofia settings.",
                               case=["main", "screen"])
        sys.exit(1)
    sofia_logger.print_log(f"Sofia has finished running with an unexpected output: "
                           f"{sofia_output}. Check {sofia_logger.log_directory}/sofia_output.txt "
                           f"for details.", case=["main", "screen"])
    sys.exit(1)


# ---------------------------------------------------------------------------
# step deblend
# ---------------------------------------------------------------------------

def flatten_options(options: dict, prefix: str = "") -> Dict[str, object]:
    """Nested settings as a dictionary with dotted keys, e.g. {'input.markers': [...]}."""
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


def _deblend_tables(cfg) -> List[str]:
    """The tables of known sources for the deblending, which is run in another directory."""
    tables = source_tables(cfg.input.known_source_table)
    return [table if table == NO_TABLE or os.path.isabs(table)
            else os.path.join(cfg.directories.run_directory, table) for table in tables]


def deblend_arguments(cfg, log_directory: str,
                      known_settings: Optional[Dict[str, object]] = None) -> List[str]:
    """The arguments with which deblend_sofia_detections is started.

    The settings that follow from the settings of AGE come first, then those of
    ``deblend.options``, which are passed on unchanged and take precedence.
    ``known_settings`` are the settings of the installed deblend_sofia_detections.
    """
    if known_settings is None:
        known_settings = deblend_settings()
    settings = {
        "input.sofia_parameters": f"{cfg.directories.sofia_run_directory}"
                                  f"{cfg.input.sofia_parameter_file}",
        "input.tables": _deblend_tables(cfg),
        "logging.log_directory": log_directory,
        "logging.log_file": DEBLEND_LOG_FILE,
        "directories.run_directory": cfg.directories.sofia_run_directory,
        "directories.ancillary_directory": os.path.join(cfg.directories.run_directory,
                                                        DEBLEND_DOWNLOADS),
        "internal.sofia": cfg.internal.sofia}
    if "RUN_DEBLEND" in cfg.logging.debug_functions or "ALL" in cfg.logging.debug_functions:
        settings["logging.debug_functions"] = list(cfg.logging.debug_functions)

    options = flatten_options(OmegaConf.to_container(cfg.deblend.options, resolve=True))
    unknown = [key for key in options if key not in known_settings]
    if unknown:
        raise ValueError(f"deblend.options: {unknown} are not settings of the installed "
                         f"deblend_sofia_detections. Its settings are: {sorted(known_settings)}")
    settings.update(options)
    return [f"{key}={json.dumps(value)}" for key, value in settings.items()]


def run_deblend(cfg) -> None:
    """Step deblend: divide the blended detections of SoFiA-2."""
    deblend_logger = logger(cfg, log_directory=_log_directory(cfg), log_file=DEBLEND_LOG_FILE)
    deblend_logger.print_log(f"Running deblend in the directory "
                             f"{cfg.directories.sofia_run_directory} with the sofia parameter "
                             f"file {cfg.input.sofia_parameter_file}")
    arguments = deblend_arguments(cfg, deblend_logger.log_directory)
    # deblend_sofia_detections stops if the directory for its downloads does not exist
    downloads = json.loads(dict(argument.split("=", 1) for argument in arguments)[
        "directories.ancillary_directory"])
    if os.path.isabs(downloads):
        os.makedirs(downloads, exist_ok=True)
    deblend_logger.print_log(f"The deblend step is being run with the following arguments: "
                             f"{arguments}", case=["debug"])
    main_with_input(arguments)


# ---------------------------------------------------------------------------
# step asymmetry
# ---------------------------------------------------------------------------

def run_asymmetry(cfg) -> None:
    """Step asymmetry: asymmetry indicators (not yet available)."""
    if cfg.input.verbose:
        print("Running assymetry indicators")


# ---------------------------------------------------------------------------
# rotating discs and anomalous gas
# ---------------------------------------------------------------------------

def run_components(cfg) -> None:
    """Step components: name and select the deblended components."""
    pipeline.run_components(cfg)


def run_rmax(cfg) -> None:
    """Step rmax: kinematic centre, r_max and classification of every component."""
    pipeline.run_rmax(cfg)


def run_pyFAT(cfg) -> None:
    """Step pyfat: tilted-ring models of the rotating discs that are large enough."""
    pipeline.run_pyfat(cfg)


def run_hi_mass(cfg) -> None:
    """Step hi_mass: HI mass of the rotating disc and of the anomalous gas."""
    pipeline.run_hi_mass(cfg)


def run_figures(cfg) -> None:
    """Step figures: the figures of all steps, one directory for every kind."""
    from anomalous_gas_extractor.kinematics import figures
    figures.make_figures(cfg)
