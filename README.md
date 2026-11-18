# anomalous_gas_extractor

=====

Introduction
------------



Requirements
------------
The code requires full installation of:

    python v3.6 or higher
    


[python](https://www.python.org/)


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

an example yaml file with all parameters can be printed by running

    AGE print_examples=true

please see the advanced input in readthe docs for an explanation of all parameters.