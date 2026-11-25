# anomalous_gas_extractor

=====

Introduction
------------

The Anomalous Gas Extractor (AGE) separates the neutral hydrogen (HI) of every galaxy in a data cube into a regularly rotating disc and anomalous gas (tails, bridges, clouds). It is made for interacting systems, such as galaxy groups, in which the HI of several galaxies is connected. AGE combines several packages in one run:

| Step | What is done | With |
|---|---|---|
| `sofia` | source finding | [SoFiA-2](https://gitlab.com/SoFiA-Admin/SoFiA-2) |
| `deblend` | detections that hold several galaxies are divided | [deblend_sofia_detections](https://github.com/PeterKamphuis/deblend_sofia_detections) |
| `components` | the components are named after the known galaxies and selected | |
| `rmax` | kinematic centre and outer radius of the regular rotation (`r_max`) of every component; which components host a rotating disc | [Kinemetry](https://www.aip.de/en/members/davor-krajnovic/kinemetry/) |
| `pyfat` | tilted-ring model, limited to `r_max`, of every rotating disc that spans at least three beams | [pyFAT](https://github.com/PeterKamphuis/pyFAT-astro) and [TiRiFiC](https://gigjozsa.github.io/tirific/) |
| `hi_mass` | HI mass of the rotating disc and of the anomalous gas | |
| `figures` | figures of all steps | |

The full documentation is at https://anomalous-gas-extractor.readthedocs.io.

Requirements
------------

The code requires [python](https://www.python.org/) v3.11 or higher. The python packages it depends on are installed by pip. Three programs have to be installed separately:

* [SoFiA-2](https://gitlab.com/SoFiA-Admin/SoFiA-2)
* [TiRiFiC](https://gigjozsa.github.io/tirific/)
* [Kinemetry](https://www.aip.de/en/members/davor-krajnovic/kinemetry/) (Krajnovic et al. 2006), the python version distributed by its author. It is not on PyPI; download it and give the directory that holds `kinemetry.py` with the setting `kinematics.kinemetry_path`.

Installation
------------

Download the source code from the Github or simply install with pip as:

  	pip install anomalous_gas_extractor

This should also install all required python dependencies.
We recommend the use of python virtual environments. If so desired an anomalous_gas_extractor installation would look like:

  	python3 -m venv anomalous_gas_extractor_venv

  	source anomalous_gas_extractor_venv/bin/activate.csh

    pip install anomalous_gas_extractor

(In case of bash the correct middle line is 	source anomalous_gas_extractor_venv/bin/activate)
You might have to restart the env:

  	deactivate

  	source anomalous_gas_extractor_venv/bin/activate.csh

Once you have installed anomalous_gas_extractor you can check that it has been installed properly by running anomalous_gas_extractor as.

  	AGE -v


Running anomalous_gas_extractor
------------------

You can run anomalous_gas_extractor by providing a configuration file by

    AGE configuration_file=file.yml

an example yaml file with all parameters, their defaults and a line of explanation for each can be printed by running

    AGE print_examples=true

This writes the file `AGE.yml` in the current directory. Every setting can also be given on the command line, where it takes precedence over the file, for example

    AGE configuration_file=file.yml hi_mass.distance_mpc=49

A minimal configuration file for a cube with blended galaxies is

    input:
      sofia_parameter_file: sofia.par       # parameter file of SoFiA-2; its input.data names the cube
      known_source_table: galaxies.csv      # name, RA, Dec (degrees) of the known galaxies
    internal:                               # the programs, with their directory if not on the search path
      sofia: sofia
      tirific: tirific
      pyfat: pyFAT
    deblend:
      options:                              # any setting of deblend_sofia_detections
        input:
          markers: [optical image, manual table, 8 HI peaks]
          optical_images: [optical.fits]    # image on which the galaxies are found
    kinematics:
      kinemetry_path: /directory/with/kinemetry.py
    hi_mass:
      distance_mpc: 49.0

The steps are run in a fixed order and each one writes its results to disk, so a run can be continued or repeated from any step with `input.steps_to_run`, e.g. `input.steps_to_run=[hi_mass,figures]`. The main results of all steps are collected in `component_summary.csv` in the run directory, with one line for every component.

Please see the [documentation](https://anomalous-gas-extractor.readthedocs.io) for an explanation of the steps, their output and all parameters.

Test data
---------

In the GitHub repository the directory `test_data` of the repository holds two small data cubes of the compact group HCG 16 (MeerKAT), with their configuration files and the results that are expected; see the `README.md` in each directory.

* `test_data/HCG16AB`: two galaxies whose HI is connected. A fast test of all steps, including the deblending, that takes one to two minutes (the tilted-ring models are left out).
* `test_data/HCG16A`: one galaxy with its tilted-ring model (some tens of minutes).

These tests can be ran with in there respective directories by invoking 'AGE configuration-file=<HCG16X>.yml. Before running the tests make sure to set the the kinemetry path in the yml file.

The tests directory contains a set of numererical tests that can be ran with:

    KINEMETRY_PATH=/directory/with/kinemetry.py pytest tests
