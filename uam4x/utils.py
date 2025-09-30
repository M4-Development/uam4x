import os
import json

import numpy as  np


TYPE_MAP = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "array": np.ndarray,
}


def check_and_get_config(conf_path):
    if not os.path.exists(conf_path):
        with open(conf_path, 'w') as configfile:
            json.dump({},configfile)

    with open(conf_path, 'r') as configfile:
        try:
            _config = json.load(configfile)
        except json.decoder.JSONDecodeError:
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
    if (isinstance(val_or_type, (float, np.floating))):
        return True
    elif isinstance(val_or_type, np.ndarray):
        return np.issubdtype(val_or_type.dtype, np.floating)
    #Handle types
    if isinstance(val_or_type, type) and (val_or_type == float or issubclass(val_or_type, np.floating, np.ndarray)):
        return True
    
    return False

def type_string_to_type(type_name:str):
    return TYPE_MAP[type_name.lower()]
