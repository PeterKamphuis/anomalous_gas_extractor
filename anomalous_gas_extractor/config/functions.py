import anomalous_gas_extractor
import sys
import psutil
from omegaconf import OmegaConf
from anomalous_gas_extractor.config.config import defaults
from deblend_sofia_detections.support.system_functions import join_path
import os
def process_input(argv):
    if '-v' in argv or '--version' in argv:
        print(f"This is version {anomalous_gas_extractor.__version__} of the program.")
        sys.exit()
    file_name = 'AGE.yml'
    program = 'AGE'
    
    if '-h' in argv or '--help' in argv:      
        print(return_help_message(program,file_name))
        sys.exit()
    #First import the defaults 
    cfg = OmegaConf.structured(defaults)
    #if ncpu is the total available remove 1
    if cfg.input.ncpu == psutil.cpu_count():
        cfg.input.ncpu -= 1

    # read command line arguments anything list input should be set in brackets '' e.g. pyROTMOD 'rotmass.MD=[1.4,True,True]'
    inputconf = OmegaConf.from_cli(argv)
    cfg_input = OmegaConf.merge(cfg,inputconf)

    # Print examples if requested
    if cfg_input.print_examples:
        masked_copy = OmegaConf.masked_copy(cfg_input,\
                    ['input','directories','cube_name'])
        with open(file_name,'w') as default_write:
            default_write.write(OmegaConf.to_yaml(masked_copy))
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
    cfg.input.steps_to_run = check_steps_to_run(cfg.input.steps_to_run) #make sure all steps are lower case
    check_paths(cfg)    
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


def check_steps_to_run(steps_to_run):
    steps_to_run = [step.lower() for step in steps_to_run]
    allowed_steps = ['sofia','deblend','asymmetry','pyfat']
    for step in steps_to_run:
        if not step in allowed_steps:
            raise ValueError(f'''The step {step} is not a valid step to run. Allowed steps are {allowed_steps}''')
    return steps_to_run


def check_paths(cfg):
    if cfg.input.sofia_parameter_file == 'Template' and cfg.cube_name is None:
        raise ValueError(f'''You have neither provided a sofia parameter file nor a cube name. Please provide one''')
    if not os.path.isdir(cfg.directories.run_directory):
        raise ValueError(f'''The directory {cfg.directories.run_directory} does not exist. Please provide a valid directory''')
    if cfg.directories.run_directory[-1] != '/':
        cfg.directories.run_directory += '/' # add a slash to the end of the path if it doesn't have one
    if cfg.input.sofia_parameter_file != 'Template':
        path, file = os.path.split(cfg.input.sofia_parameter_file)
        if path != '':
            path += '/' # add a slash to the end of the path if it doesn't have one
            if path[0] != '/':
                cfg.directories.sofia_run_directory = join_path(cfg.directories.run_directory, path)
            else:
                cfg.directories.sofia_run_directory = path
        else:
            cfg.directories.sofia_run_directory = cfg.directories.run_directory
        cfg.input.sofia_parameter_file = f'{file}'
        if not os.path.isfile(f'{cfg.directories.sofia_run_directory}{cfg.input.sofia_parameter_file}'):
            raise FileNotFoundError(f'''The sofia parameter file {cfg.directories.sofia_run_directory}{cfg.input.sofia_parameter_file} can't be found. Please provide a valid file''')
    else:
        cfg.directories.sofia_run_directory = cfg.directories.run_directory
    if cfg.cube_name is not None:
        path, file = os.path.split(cfg.cube_name)
        
        cfg.cube_name = file
        if path != '':
            path += '/' # add a slash to the end of the path if it doesn't have one
            if path[0] != '/':
                cfg.directories.data_directory = join_path(cfg.directories.run_directory, path)
            else:
                cfg.directories.data_directory = path
        else:
            cfg.directories.data_directory = cfg.directories.run_directory

        
        if not os.path.isfile(f'{cfg.directories.data_directory}{cfg.cube_name}'):
            raise FileNotFoundError(f'''The cube {cfg.directories.data_directory}{cfg.cube_name} can't be found. Please provide a valid file''')