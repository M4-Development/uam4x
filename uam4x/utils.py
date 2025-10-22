import os
import json

import numpy as  np

from uam4x.log import FileLogger
from numbers import Real

import sys
DEV = True
if hasattr(sys, '_MEIPASS'):  # this indicates we are in the pyinstaller exe
    DEV = False

log = FileLogger(filename_log=None)

TYPE_MAP = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "ndarray": np.ndarray,
}


def check_and_get_config(conf_path):
    if not os.path.exists(conf_path):
        with open(conf_path, 'w') as configfile:
            json.dump({},configfile)

    with open(conf_path, 'r') as configfile:
        try:
            _config = json.load(configfile)
        except json.decoder.JSONDecodeError:
            log.error(f"Could not load parse json at {conf_path}. Please correct or delete the file.")
            raise IOError(f"Could not load parse json at {conf_path}. Please correct or delete the file.")
    if 'COMPONENTS' not in _config:
        _config['COMPONENTS'] = {}
        with open(conf_path, 'w') as configfile:
            json.dump(_config, configfile, indent=4)
    if 'PLUGINS' not in _config:
        _config['PLUGINS'] = {}
        with open(conf_path, 'w') as configfile:
            json.dump(_config, configfile, indent=4)
    return _config


def check_continuous(val_or_type):
    #Handle values (e.g 1, 2.0, np.array([0.0,0.0,...]),np.array([0,0,...]))
    if is_float(val_or_type):
        return True
    elif is_numpy_array(val_or_type):
        return np.issubdtype(val_or_type.dtype, np.floating)
    #Handle types
    if isinstance(val_or_type, type) and (val_or_type == float or issubclass(val_or_type, (np.floating, np.ndarray))):
        return True

    return False

def check_if_scalar(val_or_type):
    '''
    Checks if a provided value or type is a scalar
    to be used primarily for checking upper and
    lower bounds for constraints and des vars
    '''
    if isinstance(val_or_type, (Real)) and not is_bool(val_or_type):
        return True
    if isinstance(val_or_type, type) and (issubclass(val_or_type, (Real, np.integer, np.floating)) and val_or_type not in (np.bool_,bool)):
        return True
    
    return False

def type_string_to_type(type_name:str):
    return TYPE_MAP[type_name.lower()]


def get_param_type(dotted_path: str, study, plugins= None):
    '''
    Get a the type of a given parameter
    assumes general form vehicles. ... .attribute
    '''
    parts = dotted_path.split('.')
    obj = study
    if parts[0] == 'vehicles':
        for part in parts:
            # Try dict access first (for vehicles, parameters, etc.)
            if isinstance(obj, dict) and part in obj:
                obj = obj[part]
            # Try attribute access
            elif hasattr(obj, part):
                if hasattr(obj,'_modes'):
                    modes = list(obj._modes.keys())
                else:
                    modes = []
                if hasattr(obj, 'get_parameter_type') :
                        if part not in modes:
                            return obj.get_parameter_type(part)
                else:
                    obj = getattr(obj, part)
            else:
                log.error(f"Cannot access '{part}' in path '{dotted_path}'")
                raise AttributeError(f"Cannot access '{part}' in path '{dotted_path}'")
    elif plugins:
        plugin_name = dotted_path.split('.')[0]
        part = '.'.join(dotted_path.split('.')[1:])
        plugin = plugins[plugin_name]
        if part in  plugin.plugin_input_names:
            obj = plugin.plugin_input_names[part]['type']
        elif part in  plugin.plugin_output_names:
            obj = plugin.plugin_output_names[part]['type']
    else:
        raise ValueError(f'No vehicle parameter type found, no plugins declared')

    return obj

def convert_numpy_to_python(value):
    '''
    Converts numpy type instances (np.ndarray,np.floating,np.int)
    to python types for Json serialization
    '''
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return [convert_numpy_to_python(x) for x in value.tolist()]
    if isinstance(value, list):
        return [convert_numpy_to_python(x) for x in value]
    if isinstance(value, dict):
        return {k: convert_numpy_to_python(x) for  k, x in value.items()}

    return value
    

def check_numpy_shape(shape):
    #Integer shapes are allowed
    if is_bool(shape):
        return False    
    elif is_integer(shape):
        return shape >= 0
    elif is_tuple(shape):
        for dim in  shape:
            if is_bool(dim):
                return False
            if not is_integer(dim):
                return False
            if dim < 0:
                return False
        return True
    
    
    return False

def normalize_shape(shape):
    '''
    Given a shape intended for a numpy array
    if the provided shape is a tuple, then
    the returns original value, if it is an int
    then it returns the canonical (N,) form shown
    by numpy when shape is requested
    
    NOTE: ASSUMES check_numpy_shape has been run on the shape
    already. 
    '''
    if is_tuple(shape):
        return shape
    if is_integer(shape):
        return (int(shape), )
# Nicer type checkers

def is_bool(value):
    return isinstance(value, (bool, np.bool_))

def is_integer(value):
    return isinstance(value, (int, np.integer))

def is_float(value):
    return isinstance(value, (float, np.floating))

def is_tuple(value):
    return isinstance(value, tuple)

def is_numpy_array(value):
    return isinstance(value, np.ndarray)

#Get type name from a value
def value_to_type_name(value):
    
    value_type = type(value)
    if hasattr(value_type, '__name__'):
        return value_type.__name__
    elif str(value_type) is  not None:
        return str(value_type)
    else:
        return None
        
    
    
