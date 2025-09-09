
from uam4x.components import register_component, Component, _component_registry
from uam4x.solution import register_plugin, Plugin, _plugin_registry

import os
import json

import importlib.util
import sys
from pathlib import Path

_conf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'conf.json')
if not os.path.exists(_conf_path):
    with open(_conf_path, 'w') as configfile:
        json.dump({},configfile)
with open(_conf_path, 'r') as configfile:
    try:
        _config = json.load(configfile)
    except json.decoder.JSONDecodeError:
        print("WARNING: Could not load conf.json")
        _config = {}

if 'COMPONENTS' not in _config:
    _config['COMPONENTS'] = {}
    with open(_conf_path, 'w') as configfile:
        json.dump(_config, configfile, indent=4)
for _comp_name, _python_file_path in _config['COMPONENTS'].items():
    if not ("uam4x" == os.path.split(_python_file_path)[-2] and os.path.basename() == "components.py"):
        _module_name = Path(_python_file_path).stem
        _spec = importlib.util.spec_from_file_location(_module_name, _python_file_path)
        _module = importlib.util.module_from_spec(_spec)
        sys.modules[_module_name] = _module
        try:
            _spec.loader.exec_module(_module)
            _component_registry[_comp_name] = getattr(_module,_comp_name)
        except FileNotFoundError:
            print(f"WARNING: Component load error ({_comp_name}). Could not find path {_python_file_path}")
        except AttributeError:
            print(f"WARNING: Component load error. Could not find {_comp_name} in {_python_file_path}")

if 'PLUGINS' not in _config:
    _config['PLUGINS'] = {}
    with open(_conf_path, 'w') as configfile:
        json.dump(_config, configfile, indent=4)
for _plugin_name, _python_file_path in _config['PLUGINS'].items():
        _module_name = Path(_python_file_path).stem
        _spec = importlib.util.spec_from_file_location(_module_name, _python_file_path)
        _module = importlib.util.module_from_spec(_spec)
        sys.modules[_module_name] = _module
        try:
            _spec.loader.exec_module(_module)
            _plugin_registry[_plugin_name] = getattr(_module,_plugin_name)
        except FileNotFoundError:
            print(f"WARNING: Plugin load error ({_plugin_name}). Could not find path {_python_file_path}")
        except AttributeError:
            print(f"WARNING: Plugin load error. Could not find {_plugin_name} in {_python_file_path}")
