
# File for functions related to logging messages to the screen and a log file.
# As we need to have logging every where do not import anything from Sample run Sofia in this file

from datetime import datetime
from inspect import stack
import atexit

from deblend_sofia_detections.support.system_functions import create_directory, join_path
import os



class empty_log():
    def __init__(self,print = True):
        self.print = print
        self.enable=False
    def print_log(self,log_statement, case = ['main']):
        if self.print:
            print(log_statement)
        else:
            pass
    def append_log(self):
        pass
    def write_log(self):
        pass
    def clear_log(self):
        pass
    def close_log(self):
        pass

class logger():
    def __new__(cls, cfg,log_file=None,log_directory=None):
        
        if not cfg.logging.enable:
            return empty_log()
        return super().__new__(cls)

    def __init__(self,cfg,log_file=None,log_directory=None):
        if cfg.logging.enable_log:
            self.set_logging(cfg,log_file,log_directory)
            self.screen= False
            self.enable = True
        else:
            self.set_logging(cfg,log_file,log_directory)
            self.screen = True
            self.enable = False

        self.log_buffer = []
        self.verbose = cfg.logging.verbose
        self.debug_functions = cfg.logging.debug_functions
        atexit.register(self.append_log)
    
    def set_logging(self,cfg,log_file=None,log_directory=None):
        if log_file is None:
            self.log_file = cfg.logging.log_file
        else:            
            self.log_file = log_file
        if log_directory is None:
          
            self.log_directory = cfg.logging.log_directory
        else:
            if log_directory[0] != '/':
                self.log_directory = join_path(log_directory,cfg.logging.log_directory)
            else:
                self.log_directory = log_directory
        if self.log_directory[-1] != '/':
            self.log_directory += '/'
        if self.log_directory[0] != '/':
            self.log_directory = join_path(cfg.directories.run_directory,self.log_directory)
        if self.log_file[0] != '/':
            self.log_file = join_path(self.log_directory, self.log_file)
        tmp = os.path.split(self.log_file)
        self.log_directory = tmp[0]
        if not os.path.exists(self.log_directory):
            create_directory(self.log_directory)
        if os.path.exists(f'{self.log_file}'):
            ext= os.path.splitext(self.log_file)[-1]
            os.rename(self.log_file, self.log_file.replace(ext, f'_previous{ext}'))    
        with open(f'{self.log_file}','w') as log_file:
            log_file.write(f"Log file for Anomalous Gas Extractor run on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
    def print_log(self,log_statement, case = ['main']):
        debugging = False
        # empty tels line number to just add some spacing in front of the log statement,
        debug= 'empty'
        trig=False
        if 'ALL' in self.debug_functions:
            trig = True
        else:
            # get the function  
            current_function= 'NONE' 
            for key in stack():
                if key[3] != 'linenumber' and key[3] != 'print_log' and key[3] != '<module>': 
                    current_function= f"{key[3]}"
                break
            if current_function.lower() in [x.lower() for x in  self.debug_functions]:
                trig=True      
        if trig:
            debugging=True    
            if 'debug_start' in case:
                debug = 'long'
            else:
                debug= 'short'
        log_statement = f"{linenumber(debug=debug)}{log_statement} \n"

        #Now lets check wether we want to print this specifc statement 
        print_screen = False
        print_log = False
        
       
            # If we want a log we have to check this specific message
        if 'main' in case or \
            (debugging and ('debug_start' in case or 'debug_add' in case or 'debug' in case))\
            or ('verbose' in case and (self.verbose or debugging)):
            #if eneable_log  is true self.screen is false 
            if not self.screen:    
                print_log = True
            if self.screen:
                print_screen = True
        #We always print screen messages to the screen unless logging.enable = false
        #This we should use for debugging really not normal functioning that should use main
        if 'screen' in case:
            print_screen = True

        if print_screen:
            print(log_statement)
        if print_log:
            self.log_buffer.append(log_statement)
    print_log.__doc__ =f'''
 NAME:
    print_log
 PURPOSE:
    Print statements to log if existent and screen if Requested
 CATEGORY:
    log_functions 

 INPUTS:
    log_statement = statement to be printed
    Configuration = Standard FAT Configuration

 OPTIONAL INPUTS:


    screen = False
    also print the statement to the screen

 OUTPUTS:
    line in the log or on the screen

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    linenumber, .write

 NOTE:
    If the log is None messages are printed to the screen.
    This is useful for testing functions.
'''
   

    def append_log(self):
        with open(f'{self.log_file}','a') as log_file:
            for message in self.log_buffer:
                log_file.write(message)
        self.log_buffer = []

    def write_log(self):
        with open(self.log_file,'w') as log_write:
            for message in self.log_buffer:
                log_write.write(f'{message}')
        self.log_buffer = []

    def clear_log(self):
        self.log_buffer = []    
    







def linenumber(debug='short'):
    '''get the line number of the print statement in the main.'''
    line = []
    for key in stack():
        if key[1] == 'main.py':
            break
        if key[3] != 'linenumber' and key[3] != 'print_log' and key[3] != '<module>':
            file = key[1].split('/')
            to_add= f"In the function {key[3]} at line {key[2]}"
            if debug == 'long':
                to_add = f"{to_add} in file {file[-1]}."
            else:
                to_add = f"{to_add}."
            line.append(to_add)
    if len(line) > 0:
        if debug == 'long':
            line = ', '.join(line)+f'\n{"":8s}'
        elif debug == 'short':
            line = line[0]+f'\n{"":8s}'
        else:
            line = f'{"":8s}'
    else:
        for key in stack():
            if key[1] == 'main.py':
                line = f"{'('+str(key[2])+')':8s}"
                break
    return line

linenumber.__doc__ =f'''
 NAME:
    linenumber

 PURPOSE:
    get the line number of the print statement in the main. Not sure 
    how well this is currently working.

 CATEGORY:
    log_functions 

 INPUTS:

 OPTIONAL INPUTS:


 OUTPUTS:
    the line number of the print statement

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
    If debug = True the full stack of the line print will be given, 
    in principle the first debug message in every function should set 
    this to true and later messages not.
    !!!!Not sure whether currently the linenumber is produced due to 
    the restructuring.
'''



def check_log(log,warn=True):
    if log is None and warn:
        for key in stack():
            if key[3] != 'linenumber' and key[3] != 'check_log' and key[3] != '<module>': 
                current_function= f"{key[3]}"
                break
        print(f'''{'!'*25:>25s} 
No log provided, in the function {current_function}
{'!'*25:>25s} 
''')
       
       
        return empty_log()
    else:
        return log