# -*- coding: future_fstrings -*-


from omegaconf import OmegaConf
from deblend_sofia_detections.deblending.sofia_functions import execute_sofia,\
    load_sofia_input_file,write_sofia

from deblend_sofia_detections.main import main_with_input

def run_assymetry(cfg):
    if cfg.input.verbose:
        print('Running assymetry indicators')

def run_deblend(cfg):
    if cfg.input.verbose:
        print(f'Running deblend in the directory {cfg.directories.sofia_run_directory} with the sofia parameter file {cfg.input.sofia_parameter_file}')
    
    args = [f'input.sofia_parameters={cfg.directories.sofia_run_directory}{cfg.input.sofia_parameter_file}']
    args.append(f'directories.run_directory={cfg.directories.sofia_run_directory}')
    args.append(f'input.use_peak_deblending=false')
    if cfg.input.known_source_table is not None:
        args.append(f'input.manual_input_tables=[{', '.join(cfg.input.known_source_table)}]')
    if cfg.input.debug:
        print(f'''The deblend step is being run with the following arguments: {args}''')
        args.append('general.debug=True')
    main_with_input(args)


def run_pyFAT(cfg):
    if cfg.input.verbose:
        print('Running pyFAT')

def run_sofia(cfg):
    if cfg.input.verbose:
        print(f'Running Sofia in the directory {cfg.directories.run_directory} with the parameter file {cfg.input.sofia_parameter_file}')
      
    if cfg.input.debug:
        print(f'''The sofia parameter file is {cfg.input.sofia_parameter_file}''')
    
    if cfg.input.sofia_parameter_file == 'Template':
        sofia_template = load_sofia_input_file(filename='Template')
   
        sofia_template['input.data'] = cfg.cube_name
        sofia_template['reliability.enable'] = True
        cfg.input.sofia_parameter_file = f'sofia_parameter_file.par'
    else:
        sofia_template = load_sofia_input_file(filename=f'{cfg.directories.sofia_run_directory}{cfg.input.sofia_parameter_file}')
        cfg.cube_name = sofia_template['input.data']
        # We need cubelets to deblend
        sofia_template['output.writeCubelets'] = True
        # and the mask
        sofia_template['output.writeMask'] = True
        sofia_template['output.writeCatXML'] = True
    sofia_template['pipeline.threads'] = cfg.input.ncpu
    #write the template back    
    write_sofia(sofia_template, f'{cfg.directories.sofia_run_directory}{cfg.input.sofia_parameter_file}')
    cfg_debl = OmegaConf.create({
        'general': {'verbose': cfg.input.verbose, 'debug': cfg.input.debug},
        'internal': {'sofia': 'sofia'},
    })
    execute_sofia(cfg_debl,run_directory=cfg.directories.sofia_run_directory,
        sofia_parameter_file=cfg.input.sofia_parameter_file)
    


