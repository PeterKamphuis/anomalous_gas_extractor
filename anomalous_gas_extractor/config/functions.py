"""Reading and checking the settings of AGE, and the file with their defaults."""

import dataclasses
import os
import sys
from typing import Any, List, Optional

import psutil
import yaml
from omegaconf import OmegaConf

from deblend_sofia_detections.support.system_functions import join_path

import anomalous_gas_extractor
from anomalous_gas_extractor.config.config import (
    MINIMUM_BEAMS_FOR_PYFAT, NO_TABLE, SOURCE_FINDING_STEPS, STEPS, defaults)

PROGRAM = "AGE"
DEFAULT_FILE = "AGE.yml"

# sections of the file with the defaults, and settings that are filled in by the program
PRINTED_SECTIONS = ("cube_name", "input", "directories", "internal", "deblend", "components",
                    "kinematics", "pyfat", "hi_mass", "figures")
HIDDEN_SETTINGS = ("directories.sofia_run_directory", "directories.data_directory")


# ---------------------------------------------------------------------------
# file with the defaults
# ---------------------------------------------------------------------------

def _yaml_value(value: Any) -> str:
    text = yaml.safe_dump(value, default_flow_style=True, width=1000).strip()
    return text[:-3].strip() if text.endswith("...") else text


def _comment(description: Optional[str], indent: str) -> List[str]:
    if not description:
        return []
    return [f"{indent}# {line}" for line in description.split("\n")]


def _setting_lines(settings_class, values: dict, path: str, depth: int) -> List[str]:
    """The settings of one dataclass as lines of yaml with their explanation above them."""
    indent = "  " * depth
    lines = []
    for item in dataclasses.fields(settings_class):
        name = f"{path}{item.name}"
        if name in HIDDEN_SETTINGS or (depth == 0 and item.name not in PRINTED_SECTIONS):
            continue
        description = item.metadata.get("help")
        value = values[item.name]
        nested = item.default_factory if dataclasses.is_dataclass(item.default_factory) else None
        if nested is not None:
            lines += [""] + _comment(description, indent) + [f"{indent}{item.name}:"]
            lines += _setting_lines(nested, value, f"{name}.", depth + 1)
        else:
            lines += _comment(description, indent) + [f"{indent}{item.name}: {_yaml_value(value)}"]
    return lines


def default_file_text() -> str:
    """The file with all settings, their defaults and their explanation."""
    values = OmegaConf.to_container(OmegaConf.structured(defaults))
    header = ["# Settings of AGE (anomalous_gas_extractor) with their defaults.",
              "# Change what is needed and run",
              f"#   {PROGRAM} configuration_file={DEFAULT_FILE}",
              "# Any setting can also be given on the command line, e.g. hi_mass.distance_mpc=49",
              ""]
    return "\n".join(header + _setting_lines(defaults, values, "", 0)) + "\n"


def write_default_file(file_name: str = DEFAULT_FILE) -> str:
    with open(file_name, "w") as output:
        output.write(default_file_text())
    return os.path.abspath(file_name)


def return_help_message(program: str = PROGRAM, file_name: str = DEFAULT_FILE) -> str:
    return f"""
Use {program} in this way:
{program} configuration_file=inputfile.yml   where inputfile.yml holds the settings.
{program} print_examples=true                writes the file {file_name} with all settings, their
                                       defaults and their explanation, in the current directory.
{program} -h                                 prints this message.
{program} -v                                 prints the version.

All settings can be given on the command line as well, e.g. hi_mass.distance_mpc=49
"""


# ---------------------------------------------------------------------------
# checking the settings
# ---------------------------------------------------------------------------

def check_steps_to_run(steps_to_run) -> List[str]:
    """The requested steps in lower case; unknown steps are an error."""
    steps = [str(step).lower() for step in steps_to_run]
    unknown = [step for step in steps if step not in STEPS]
    if unknown:
        raise ValueError(f"input.steps_to_run: {unknown} are not steps of the program. "
                         f"The steps are {list(STEPS)}")
    return steps


def source_tables(known_source_table) -> List[str]:
    """The tables of known sources as a list, [NO_TABLE] if there is none.

    The setting may be one file name, a list of file names or empty.
    """
    if known_source_table is None:
        return [NO_TABLE]
    if isinstance(known_source_table, str):
        tables = [known_source_table]
    else:
        tables = [str(table) for table in known_source_table if table is not None]
    tables = [table for table in tables if table.lower() != NO_TABLE.lower()]
    return tables or [NO_TABLE]


def first_source_table(cfg) -> Optional[str]:
    """The table that names the components, None if no table is given."""
    table = source_tables(cfg.input.known_source_table)[0]
    return None if table == NO_TABLE else table


def validate(cfg) -> None:
    """Check the settings that have limits and bring them to one form."""
    cfg.input.steps_to_run = check_steps_to_run(cfg.input.steps_to_run)
    cfg.input.known_source_table = source_tables(cfg.input.known_source_table)
    if cfg.pyfat.minimum_beams_across < MINIMUM_BEAMS_FOR_PYFAT:
        raise ValueError(f"pyfat.minimum_beams_across is {cfg.pyfat.minimum_beams_across}: a "
                         f"tilted-ring model needs a disc of at least "
                         f"{MINIMUM_BEAMS_FOR_PYFAT:g} beams along its major axis, so the "
                         f"value has to be {MINIMUM_BEAMS_FOR_PYFAT:g} or larger")


def check_run_directory(cfg) -> None:
    if not os.path.isdir(cfg.directories.run_directory):
        raise ValueError(f"The directory {cfg.directories.run_directory} does not exist. "
                         f"Please provide a valid directory")
    # the run directory ends with a slash
    cfg.directories.run_directory = os.path.join(cfg.directories.run_directory, "")


def _directory_of(path: str, run_directory: str) -> str:
    """The directory of a file given relative to the run directory, with a slash at its end."""
    if path == "":
        return run_directory
    path = os.path.join(path, "")
    return path if os.path.isabs(path) else join_path(run_directory, path)


def check_paths(cfg) -> None:
    """Check the input of the source finding and fill in the directories."""
    if cfg.input.sofia_parameter_file == "Template" and cfg.cube_name is None:
        raise ValueError("You have neither provided a sofia parameter file nor a cube name. "
                         "Please provide one")
    check_run_directory(cfg)
    run_directory = cfg.directories.run_directory
    cfg.directories.sofia_run_directory = run_directory
    if cfg.input.sofia_parameter_file != "Template":
        path, file_name = os.path.split(cfg.input.sofia_parameter_file)
        cfg.directories.sofia_run_directory = _directory_of(path, run_directory)
        cfg.input.sofia_parameter_file = file_name
        parameter_file = f"{cfg.directories.sofia_run_directory}{file_name}"
        if not os.path.isfile(parameter_file):
            raise FileNotFoundError(f"The sofia parameter file {parameter_file} can't be found. "
                                    f"Please provide a valid file")
    if cfg.cube_name is not None:
        path, file_name = os.path.split(cfg.cube_name)
        cfg.cube_name = file_name
        cfg.directories.data_directory = _directory_of(path, run_directory)
        cube = f"{cfg.directories.data_directory}{file_name}"
        if not os.path.isfile(cube):
            raise FileNotFoundError(f"The cube {cube} can't be found. Please provide a valid file")


def prepare_paths(cfg) -> None:
    """Check the run directory and, for the steps that need it, the input of the source finding."""
    source_finding = any(step in cfg.input.steps_to_run for step in SOURCE_FINDING_STEPS)
    if source_finding or cfg.input.sofia_parameter_file != "Template" or cfg.cube_name is not None:
        check_paths(cfg)
        return
    check_run_directory(cfg)
    cfg.directories.sofia_run_directory = cfg.directories.run_directory


# ---------------------------------------------------------------------------
# reading the settings
# ---------------------------------------------------------------------------

def process_input(argv: List[str]):
    """The settings: the defaults, then the configuration file, then the command line."""
    if "-v" in argv or "--version" in argv:
        print(f"This is version {anomalous_gas_extractor.__version__} of the program.")
        sys.exit()
    if "-h" in argv or "--help" in argv:
        print(return_help_message())
        sys.exit()
    cfg = OmegaConf.structured(defaults)
    # leave one processor free if all are available
    if cfg.input.ncpu == psutil.cpu_count():
        cfg.input.ncpu -= 1
    command_line = OmegaConf.from_cli(argv)
    requested = OmegaConf.merge(cfg, command_line)
    if requested.print_examples:
        print(f"The settings and their defaults are written to {write_default_file()}")
        sys.exit()
    if requested.configuration_file is not None:
        if not os.path.isfile(requested.configuration_file):
            raise FileNotFoundError(f"The configuration file {requested.configuration_file} "
                                    f"can't be found")
        cfg = OmegaConf.merge(cfg, OmegaConf.load(requested.configuration_file))
    # the command line takes precedence over the file
    cfg = OmegaConf.merge(cfg, command_line)
    validate(cfg)
    prepare_paths(cfg)
    return cfg
