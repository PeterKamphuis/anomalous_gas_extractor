# OmegaConf setups

from dataclasses import dataclass, field
import psutil
from typing import List, Optional
import os
from datetime import datetime



@dataclass
class Input:
    known_source_table: Optional[str] = None
    sofia_parameter_file: str = 'Template' # set this to the parameter file you want to run the original sofia with
    steps_to_run: List[str] = field(default_factory=lambda: ['sofia','deblend','asymmetry','pyFAT'])    
    verbose: bool = True
    try:
        ncpu: int = len(psutil.Process().cpu_affinity())
    except AttributeError:
        ncpu: int = psutil.cpu_count()
    multiprocessing: bool = True
  

@dataclass
class Directories:
    run_directory: str = os.getcwd()
    sofia_run_directory: str = ''
    data_directory: str = ''

@dataclass
class Logging:
    enable: bool = True
    enable_log: bool = True
    verbose: bool = False
    log_directory: str = f'Logs/{datetime.now().strftime("%d-%m-%Y")}'
    log_file: str = 'Log.txt'  # Name of the log file
    debug_functions: List[str] = field(default_factory=lambda: ['NONE']) # List of functions to debug. If 'ALL' is in the list, all functions will be debugged. The function names should be the same as the function names in the code. This can be useful to limit the debugging to specific functions that are of interest to the user.



@dataclass
class Internal:
    sofia: str = 'sofia'
    tirific: str = 'tirific'
    
   
@dataclass
class defaults:
    print_examples: bool = False
    configuration_file: Optional[str] = None
    cube_name: Optional[str] = None
    internal: Internal = field(default_factory = Internal)
    input: Input = field(default_factory = Input)
    logging: Logging = field(default_factory = Logging)
    directories: Directories = field(default_factory = Directories)


