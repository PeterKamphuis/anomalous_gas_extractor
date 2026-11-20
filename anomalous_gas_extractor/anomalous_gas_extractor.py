"""Separating HI components into rotating discs and anomalous gas components.

The HI of every galaxy in a data cube is divided into a regularly rotating disc
and anomalous gas (tails, bridges, clouds).

The steps are as follows:

sofia       run source finding with SoFiA-2
deblend     deblend HI components using deblend_sofia_detections
asymmetry   asymmetry indicators (not run by default)
components  name the deblended components after the known galaxies, select
            those to be analysed and put their labels on the cube used for
            the kinematics
rmax        determine kinematic centre and outer radius of the rotating disc (r_max) of
            every component with Kinemetry; a component hosts a rotating disc
            if its regular rings are resolved and dominated by rotation
pyfat       make tilted-ring model, limited to r_max, of every rotating disc that
            spans at least 3 beams along its major axis
hi_mass     estimate HI mass inside and outside the rotating disc. The disc is that of
            the tilted-ring model or, for a rotating disc without model, the
            ellipse of Kinemetry with semi-major axis r_max
figures     figures of all steps

Run with

    AGE configuration_file=run.yml

or, where the command AGE is not installed, with

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

import logging
from typing import Callable, Dict, List, Optional

from anomalous_gas_extractor.config.config import STEPS
from anomalous_gas_extractor.config.functions import process_input
from anomalous_gas_extractor.run_scripts.run_scripts import (
    TEMPLATE_PARAMETER_FILE, run_asymmetry, run_components, run_deblend, run_figures,
    run_hi_mass, run_pyFAT, run_rmax, run_sofia)
from anomalous_gas_extractor.support.logging import STEP_LOGGER, start_step_log

__author__ = "Roger Ianjamasimanana"

logger = logging.getLogger(f"{STEP_LOGGER}.steps")

STEP_FUNCTIONS: Dict[str, Callable] = {
    "sofia": run_sofia,
    "deblend": run_deblend,
    "asymmetry": run_asymmetry,
    "components": run_components,
    "rmax": run_rmax,
    "pyfat": run_pyFAT,
    "hi_mass": run_hi_mass,
    "figures": run_figures,
}


def run_steps(cfg) -> None:
    """Run the requested steps in their fixed order."""
    start_step_log(cfg)
    if "sofia" not in cfg.input.steps_to_run and cfg.input.sofia_parameter_file == "Template":
        # the parameter file that an earlier run of the step sofia wrote for the template
        cfg.input.sofia_parameter_file = TEMPLATE_PARAMETER_FILE
    for step in STEPS:
        if step in cfg.input.steps_to_run:
            logger.info("step %s", step)
            STEP_FUNCTIONS[step](cfg)


def main(argv: Optional[List[str]] = None) -> None:
    run_steps(process_input(sys.argv[1:] if argv is None else list(argv)))


if __name__ == "__main__":
    main()
