# -*- coding: future_fstrings -*-

from anomalous_gas_extractor.support.logging import logger
from omegaconf import OmegaConf
from deblend_sofia_detections.deblending.sofia_functions import execute_sofia,\
    load_sofia_input_file,write_sofia

from deblend_sofia_detections.main import main_with_input
import sys
def run_asymmetry(cfg):
    if cfg.input.verbose:
        print('Running assymetry indicators')

def run_deblend(cfg):
    deblend_logger = logger(cfg,log_directory=f'{cfg.directories.run_directory}{cfg.logging.log_directory}',
                log_file=f'run_deblend.log')
    
    deblend_logger.print_log(f'Running deblend in the directory {cfg.directories.sofia_run_directory} with the sofia parameter file {cfg.input.sofia_parameter_file}')
    
    args = [f'input.sofia_parameters={cfg.directories.sofia_run_directory}{cfg.input.sofia_parameter_file}']
    args.append(f'logging.log_directory={deblend_logger.log_directory}')
    args.append(f'logging.log_file=run_deblend.log')
    args.append(f'directories.run_directory={cfg.directories.sofia_run_directory}')
    args.append(f'input.use_peak_deblending=false')
    if cfg.input.known_source_table is not None:
        args.append(f'input.manual_input_tables=[{', '.join(cfg.input.known_source_table)}]')
   
    deblend_logger.print_log(f'''The deblend step is being run with the following arguments: {args}''', 
        case=['debug'])
    if 'RUN_DEBLEND' in cfg.logging.debug_functions or 'ALL' in cfg.logging.debug_functions:
        args.append(f'logging.debug_functions={cfg.logging.debug_functions}')
    deblend_logger.print_log(f'''The deblend step is being run with the following arguments: {args}''', 
        case=['debug'])
    main_with_input(args)


def run_pyFAT(cfg):
    if cfg.input.verbose:
        print('Running pyFAT')

def run_sofia(cfg):
    sofia_logger = logger(cfg,log_directory=f'{cfg.directories.run_directory}{cfg.logging.log_directory}',
            log_file=f'run_sofia.log')

    sofia_logger.print_log(f'Running Sofia in the directory {cfg.directories.run_directory} with the parameter file {cfg.input.sofia_parameter_file}')
      
    sofia_logger.print_log(f'''The sofia parameter file is {cfg.input.sofia_parameter_file}''',
            case = ['debug'])
    
    if cfg.input.sofia_parameter_file == 'Template':
        sofia_template = load_sofia_input_file(filename='Template')
   
        sofia_template['input.data'] = cfg.cube_name
        sofia_template['reliability.enable'] = 'true'
        cfg.input.sofia_parameter_file = f'sofia_parameter_file.par'
    else:
        sofia_template = load_sofia_input_file(filename=f'{cfg.directories.sofia_run_directory}{cfg.input.sofia_parameter_file}')
        cfg.cube_name = sofia_template['input.data']
        # We need cubelets to deblend
        sofia_template['output.writeCubelets'] = 'true'
        # and the mask
        sofia_template['output.writeMask'] = 'true'
        sofia_template['output.writeCatXML'] = 'true'
    sofia_template['pipeline.threads'] = cfg.input.ncpu
    #write the template back    
    write_sofia(sofia_template, f'{cfg.directories.sofia_run_directory}{cfg.input.sofia_parameter_file}')
    cfg_debl = OmegaConf.create({
        'logging': {'verbose_screen': 'false', 
                    'debug_functions': cfg.logging.debug_functions,
                    'enable_log': 'true',
                    'verbose_log': cfg.logging.verbose,
                    'log_file': sofia_logger.log_file,
                    'log_directory': sofia_logger.log_directory,
                    'enable': 'true'},
        'internal': {'sofia': 'sofia'},
    })
    sofia_output = execute_sofia(cfg_debl,run_directory=cfg.directories.sofia_run_directory,
        sofia_parameter_file=cfg.input.sofia_parameter_file)
   
    if sofia_output == 'Success':
        sofia_logger.print_log(f'''Sofia has finished running successfully''',
            case=['main','screen'])
    elif sofia_output == 'No sources found':
        sofia_logger.print_log(f'''Sofia has finished running. It didn't find any sources, you need to adapt your sofia settings''',
            case=['main','screen'])
        sys.exit(8)
    elif sofia_output == 'No negative sources found, Cannot run reliability test.':
        sofia_logger.print_log(f'''Sofia has finished running. It didn't find any negative sources, you need to adapt your sofia settings.''',
            case=['main','screen'])
        sys.exit(1)
    else:
        sofia_logger.print_log(f'''Sofia has finished running with an unexpected output: {sofia_output}. Check {cfg_debl.logging.log_directory}/sofia_output.txt for details.''',
            case=['main','screen'])
        sys.exit(1)

    
    


