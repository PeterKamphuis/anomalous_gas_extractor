# -*- coding: future_fstrings -*-

# This is the stand alone version of the pyFAT moments to create moment maps



import sys
import traceback
import warnings
from anomalous_gas_extractor.run_scripts.run_scripts import run_asymmetry, run_deblend,\
    run_sofia,run_pyFAT

from anomalous_gas_extractor.config.functions import process_input


from multiprocessing import get_context,Manager

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))





def main():
    argv = sys.argv[1:]
    #obtain the config file
    cfg = process_input(argv)
    # Run Sofia.
    if 'sofia' in cfg.input.steps_to_run:
        run_sofia(cfg)
    else:
        if cfg.general.verbose:
            print('Skipping Sofia')
            if cfg.input.sofia_parameter_file == 'Template':
                cfg.input.sofia_parameter_file = f'{cfg.general.directory}/sofia_parameter_file.par'
               
    if 'deblend' in cfg.input.steps_to_run:
        run_deblend(cfg)

    #run asymmetry indicators 
    if 'asymmetry' in cfg.input.steps_to_run:
        run_asymmetry(cfg)

    #run pyFAT
    if 'pyfat' in cfg.input.steps_to_run:
        run_pyFAT(cfg)

    # for some dumb reason pools have to be called from main
    # !!!!!!!!Starts your Main Here


if __name__ =="__main__":
    main()
