# OmegaConf setups

from dataclasses import dataclass, field
import psutil
import package_name
from omegaconf import MISSING,OmegaConf
from typing import List, Optional
import os
import sys


@dataclass
class General:
    verbose: bool = True
    try:
        ncpu: int = len(psutil.Process().cpu_affinity())
    except AttributeError:
        ncpu: int = psutil.cpu_count()
    directory: str = os.getcwd()
    multiprocessing: bool = True
    #font_file: str = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"

@dataclass
class defaults:
    print_examples: bool = False
    configuration_file: Optional[str] = None
    general: General =  field(default_factory = General)



def process_input(argv):
    if '-v' in argv or '--version' in argv:
        print(f"This is version {package_name.__version__} of the program.")
        sys.exit()
    file_name = 'script_name_defaults.yml'
    program = 'script_name'
    
    if '-h' in argv or '--help' in argv:      
        print(return_help_message(program,file_name))
        sys.exit()
    #First import the defaults 
    cfg = OmegaConf.structured(defaults)
    #if ncpu is the total available remove 1
    if cfg.ncpu == psutil.cpu_count():
        cfg.ncpu -= 1

    # read command line arguments anything list input should be set in brackets '' e.g. pyROTMOD 'rotmass.MD=[1.4,True,True]'
    inputconf = OmegaConf.from_cli(argv)
    cfg_input = OmegaConf.merge(cfg,inputconf)

    # Print examples if requested
    if cfg_input.print_examples:
        with open(file_name,'w') as default_write:
            default_write.write(OmegaConf.to_yaml(cfg_input))
        print(f'''We have printed the file {file_name} in {os.getcwd()}.
''')
        sys.exit()

    #if a configuration file is provided read it
    if not cfg_input.configuration_file is None:
        succes = False
        while not succes:
            try:
                yaml_config = OmegaConf.load(cfg_input.configuration_file)
        #merge yml file with defaults
                cfg = OmegaConf.merge(cfg,yaml_config)
                succes = True
            except FileNotFoundError:
                cfg_input.configuration_file = input(f'''
You have provided a config file ({cfg_input.configuration_file}) but it can't be found.
If you want to provide a config file please give the correct name.
Else press CTRL-C to abort.
configuration_file = ''')
    # make sure the command line overwrite the file
    cfg = OmegaConf.merge(cfg,inputconf)
    
   
    return cfg

def return_help_message(program, file_name):
    help_message = f'''
Use {program} in this way:
{program} configuration_file=inputfile.yml   where inputfile is a yaml config file with the desired input settings.
{program} -h print this message
{program} print_examples=true print a yaml file ({file_name}) with the default setting in the current working directory.
in this file values designated ??? indicated values without defaults.

All config parameters can be set directly from the command line by setting the correct parameters, e.g:
{program} general.directory=/dir/ectory
'''
    return help_message   