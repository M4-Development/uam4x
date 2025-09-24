import os
import json

import numpy as  np

def check_and_get_config(conf_path):
    if not os.path.exists(conf_path):
        with open(conf_path, 'w') as configfile:
            json.dump({},configfile)

    with open(conf_path, 'r') as configfile:
        try:
            _config = json.load(configfile)
        except json.decoder.JSONDecodeError:
            raise json.decoder.JSONDecodeError(f"Could not load parse json at {conf_path}. Please correct or delete the file.")
    if 'COMPONENTS' not in _config:
        _config['COMPONENTS'] = {}
        with open(conf_path, 'w') as configfile:
            json.dump(_config, configfile, indent=4)
    if 'PLUGINS' not in _config:
        _config['PLUGINS'] = {}
        with open(conf_path, 'w') as configfile:
            json.dump(_config, configfile, indent=4)
    return _config

#TODO: combine with existing type parser
def type_parser(type_name):

    type_name_to_type = {'float': float,
                         'str': str,
                         'int': int}

    return type_name_to_type[type_name]

def check_continuous(val_or_type):
    if (isinstance(val_or_type, (float, np.floating))):
        return True
    if isinstance(val_or_type, type) and (val_or_type == float or issubclass(val_or_type, np.floating)):
        return True

    return False
