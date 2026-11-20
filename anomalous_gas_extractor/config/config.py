"""Settings of AGE: one section for every step, with the defaults and their explanation.

The explanation of a setting is kept with the setting itself (``setting``); it
is written above the setting in the file with the defaults
(``AGE print_examples=true``, see ``config.functions``).
"""

import copy
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import psutil

# the steps in the order in which they are run
STEPS = ("sofia", "deblend", "asymmetry", "components", "rmax", "pyfat", "hi_mass", "figures")
DEFAULT_STEPS = ("sofia", "deblend", "components", "rmax", "pyfat", "hi_mass", "figures")
# steps that need the parameter file of SoFiA-2 or the name of the cube
SOURCE_FINDING_STEPS = ("sofia", "deblend", "components")

# a tilted-ring model needs a disc of at least this number of beams along its major axis
MINIMUM_BEAMS_FOR_PYFAT = 3.0

# how a table that is not given is written, as in deblend_sofia_detections
NO_TABLE = "None"


def setting(default: Any, description: str):
    """A setting: its default value and the text that explains it."""
    if isinstance(default, (list, dict)):
        return field(default_factory=lambda: copy.deepcopy(default),
                     metadata={"help": description})
    return field(default=default, metadata={"help": description})


def section(section_class, description: str):
    """A group of settings."""
    return field(default_factory=section_class, metadata={"help": description})


def available_cpus() -> int:
    try:
        return len(psutil.Process().cpu_affinity())
    except AttributeError:
        return psutil.cpu_count()


@dataclass
class Input:
    known_source_table: Any = setting(
        [NO_TABLE], "Comma separated table (name, RA, Dec in degrees, ...) of the known "
                    "galaxies, or a list of\nsuch tables. The first one names the components "
                    "and gives the first guess of their centres;\nall are passed on to the "
                    "deblending")
    sofia_parameter_file: str = setting(
        "Template", "Parameter file of SoFiA-2; it names the cube (input.data). With Template "
                    "the\ndefault settings of SoFiA-2 are used on cube_name")
    steps_to_run: List[str] = setting(
        list(DEFAULT_STEPS),
        "Steps that are run. The possible steps, in the order in which they are run:\n"
        + ", ".join(STEPS) + ".\nThe step asymmetry is not run unless it is added to the list")
    verbose: bool = True
    ncpu: int = setting(available_cpus(), "Number of processes")
    multiprocessing: bool = True


@dataclass
class Directories:
    run_directory: str = setting(
        os.getcwd(), "All output is written below this directory, which has to exist")
    sofia_run_directory: str = ""
    data_directory: str = ""


@dataclass
class Logging:
    enable: bool = True
    enable_log: bool = True
    verbose: bool = False
    log_directory: str = f'Logs/{datetime.now().strftime("%d-%m-%Y")}'
    log_file: str = "Log.txt"
    # Functions that are debugged, by their name in the code; ALL debugs every function
    debug_functions: List[str] = field(default_factory=lambda: ["NONE"])


@dataclass
class Internal:
    sofia: str = setting("sofia", "SoFiA-2")
    tirific: str = setting("tirific", "TiRiFiC")
    pyfat: str = setting("pyFAT", "pyFAT")


@dataclass
class Deblend:
    options: Dict[str, Any] = setting(
        {}, "Any setting of deblend_sofia_detections, in the layout of its own configuration "
            "file, e.g.\n{input: {markers: [optical image, 5 HI peaks], optical_images: "
            "[image.fits]}}.\nThe settings are passed on unchanged and have to be settings of "
            "that package")


@dataclass
class Components:
    output_directory: str = setting("components", "Directory of the mask of the components")
    sources: Optional[List[int]] = setting(
        None, "Detections of SoFiA-2 that are used, by their number in the mask of SoFiA-2. "
              "If not set all are used")
    only_named: bool = setting(
        False, "Keep only the components that hold a galaxy of input.known_source_table")
    watershed_directory: str = setting(
        "Watershed_Output", "Directory with the output of the deblending inside the output "
                            "directory of SoFiA-2")
    cutout_margin_beams: Optional[float] = setting(
        3.0, "The next settings are used if kinematics.cube is another cube than the one that "
             "was deblended.\nThe labels are carried to that cube, which is cut to the part of "
             "the sky with components plus\nthis margin in beams. If set to null the whole cube "
             "is kept")
    cutout_region: Optional[str] = setting(
        None, "or to the part of the sky covered by this FITS image or cube, if one is given")
    emission_mask: Optional[str] = setting(
        None, "A label is kept only where that cube has emission: a mask on its grid "
              "(emission where > 0)")
    emission_sofia_parameters: Optional[str] = setting(
        None, "or a parameter file of SoFiA-2 with which such a mask is made")


@dataclass
class Kinematics:
    cube: Optional[str] = setting(
        None, "Cube used for the kinematics; relative paths start at the run directory. "
              "If not set the cube\nthat was deblended is used")
    mask: Optional[str] = setting(
        None, "Mask on the grid of the cube with an integer label for every component and 0 "
              "elsewhere.\nIf not set the mask made by the step components is used")
    components: Optional[Dict[str, int]] = setting(
        None, "Names of the components and their labels in the mask, e.g. {HCG16A: 1}. If not "
              "set all labels\nare used and named after the header keywords LABEL<n> of the "
              "mask, or component_<n>")
    centre_table: Optional[str] = setting(
        None, "Comma separated table (name, RA, Dec in degrees, ...) with the first guess of "
              "the centres.\nIf not set input.known_source_table is used, else the centroid of "
              "the brightest emission")
    kinemetry_path: Optional[str] = setting(
        None, "Directory that holds kinemetry.py if Kinemetry is not installed in the python "
              "environment")
    output_directory: str = setting("kinematics", "Directory of the tables of the step rmax")
    velocity_field_snr: float = setting(
        3.0, "A pixel is part of the velocity field if its integrated intensity exceeds this "
             "signal to noise")
    ring_step_pixels: float = setting(2.0, "Step in semi-major axis between the rings in pixels")
    maximum_radius_arcsec: Optional[float] = setting(
        None, "No rings beyond this radius. If not set the rings run to the edge of the "
              "velocity field")
    ring_tests: List[str] = setting(
        ["coverage", "k5k1", "vsys"],
        "Tests a ring has to pass to count as regular, one or more of\n"
        "  coverage:  the fraction of the ellipse on the velocity field is at least the "
        "minimum coverage\n"
        "  k5k1:      k5/k1 is at most k5k1_max\n"
        "  vsys:      the systemic velocity deviates by at most "
        "max(vsys_drift_kms, vsys_drift_fraction * k1)\n"
        "  amplitude: the fitted velocity curve stays within amplitude_ratio_max times the "
        "velocities\n"
        "             measured on the ring (the fit is constrained by the data)")
    k5k1_max: float = setting(0.10, "Limits of the ring tests")
    vsys_drift_kms: float = 15.0
    vsys_drift_fraction: float = 0.15
    amplitude_ratio_max: float = 1.25
    start_within_beams: float = setting(
        1.5, "The run of regular rings has to start within this number of beams from the centre")
    fit_centre: bool = setting(
        True, "Fit the kinematic centre. If false the first guess of the centre is kept")
    measure_coverage: bool = setting(
        True, "Measure the minimum coverage of a ring with the masking test (only if coverage "
              "is one of the\nring tests). If false, or if a component has no completely "
              "sampled regular ring,\ndefault_coverage is used")
    default_coverage: float = 0.60
    minimum_disc_beams: float = setting(
        1.0, "A run of regular rings is a rotating disc if it is resolved: its radial extent "
             "is at least\nthis number of beams")
    minimum_v_over_sigma: float = setting(
        1.0, "and if it is dominated by rotation: the largest k1 exceeds this number times the "
             "velocity\ndispersion inside r_max (and k1 does not fall from the first to the "
             "last ring)")


@dataclass
class PyFAT:
    output_directory: str = setting("pyfat", "Directory in which pyFAT is run")
    minimum_beams_across: float = setting(
        MINIMUM_BEAMS_FOR_PYFAT,
        "A rotating disc is modelled with pyFAT if it spans at least this number of beams "
        "along its\nmajor axis (2 r_max / major axis of the beam); the value has to be 3 or "
        "larger. The disc of a\nsmaller rotating disc is the ellipse of Kinemetry with "
        "semi-major axis r_max")
    flat_disc: bool = setting(
        True, "Fit a flat disc: inclination, position angle and dispersion do not vary with "
              "radius")
    force_fit: bool = setting(True, "Skip the checks of pyFAT on the quality of the source")
    cut_beyond_rmax_beams: Optional[float] = setting(
        0.5, "Gas of the component beyond r_max plus this number of beams is replaced by noise "
             "in the cube\ngiven to pyFAT. If set to null all gas of the component is kept")
    spatial_margin_beams: float = setting(
        3.0, "Size of the cube given to pyFAT: margin around the emission in beams and half "
             "width in velocity")
    velocity_half_width_kms: float = 300.0
    centre_tolerance_beams: float = setting(
        0.5, "pyFAT has to keep the centre within this number of beams from the kinematic "
             "centre")
    random_seed: int = setting(
        1, "Seed of the noise that replaces the emission of the other components")
    reuse_fits: bool = setting(
        False, "If true pyFAT is not run again when its results in the output directory list "
               "all discs;\nthe existing models are read and the tables are made again")
    overrides: Dict[str, Any] = setting(
        {}, "Any other setting of pyFAT in the layout of its configuration file,\n"
            "e.g. {fitting: {ring_size: 1.1}}. These take precedence over the settings above")


@dataclass
class HIMass:
    output_directory: str = setting("hi_mass", "Directory of the table with the HI masses")
    distance_mpc: Optional[float] = setting(
        None, "Distance in Mpc of all components. If not set the Hubble flow distance of each "
              "component is used")
    hubble_constant: float = 70.0
    beam_margin: float = setting(
        0.0, "The ellipse of the rotating disc can be widened by this fraction of the beam "
             "(FWHM).\nThe rings of the tilted-ring model start at radius 0 and the model ends "
             "at its\noutermost ring, so by default the ellipse is not widened")


@dataclass
class Figures:
    enable: bool = setting(True, "Make figures. If false the step figures does nothing")
    output_directory: str = setting(
        "figures", "All figures are written below this directory, one subdirectory for every "
                   "kind of figure")
    products: List[str] = setting(
        ["rmax", "models", "disc_anomalous", "position_velocity", "model_comparison"],
        "The kinds of figure that are made:\n"
        "  rmax:              velocity field of every component with r_max, and the Kinemetry "
        "profiles\n"
        "  models:            radial profiles of the tilted-ring models\n"
        "  disc_anomalous:    column density and velocity field of the rotating disc and the "
        "anomalous gas\n"
        "  position_velocity: position-velocity diagrams along the major axis\n"
        "  model_comparison:  velocity field of the model on that of the data, and residual "
        "moment maps")
    font_file: Optional[str] = setting(
        None, "Font file (ttf or otf) of the figures. If not set the default font of "
              "matplotlib is used")
    format: str = setting("pdf", "File format of the figures, any format matplotlib can write")
    pv_width_beams: float = setting(1.0, "Width of the position-velocity slices in beams")
    position_angles: Dict[str, float] = setting(
        {}, "Position angle (degrees, north through east) of the slice of a component without "
            "tilted-ring\nmodel, by component name. If not given it is measured from the "
            "velocity gradient of the gas")
    display_names: Dict[str, str] = setting(
        {}, "Names of the components as written in the figures, e.g. {HCG16A: 'HCG 16A'}")


@dataclass
class defaults:
    print_examples: bool = False
    configuration_file: Optional[str] = None
    cube_name: Optional[str] = setting(
        None, "Data cube in which sources are searched if no parameter file of SoFiA-2 is given")
    internal: Internal = section(Internal, "Programs that are started")
    input: Input = section(Input, "Input of the run")
    logging: Logging = field(default_factory=Logging)
    directories: Directories = section(Directories, "Directories")
    deblend: Deblend = section(Deblend, "Step deblend: settings of deblend_sofia_detections")
    components: Components = section(
        Components, "Step components: name and select the deblended components")
    kinematics: Kinematics = section(
        Kinematics, "Step rmax: kinematic centre, r_max and classification of every component")
    pyfat: PyFAT = section(PyFAT, "Step pyfat: tilted-ring models of the rotating discs")
    hi_mass: HIMass = section(
        HIMass, "Step hi_mass: HI mass of the rotating disc and of the anomalous gas")
    figures: Figures = section(Figures, "Step figures")
