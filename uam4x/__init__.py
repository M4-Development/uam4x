
from uam4x.components import register_component, Component, _component_registry
from uam4x.solution import register_plugin, Plugin, _plugin_registry
from uam4x.utils import check_and_get_config
import os
import json

import importlib.util
import sys
from pathlib import Path

_conf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'conf.json')
_config = check_and_get_config(_conf_path)

for _comp_name, _python_file_path in _config['COMPONENTS'].items():
    # Check if this is a built-in component from uam4x/uam4x/components.py
    is_builtin = ("uam4x" == os.path.split(_python_file_path)[-2] and
                  os.path.basename(_python_file_path) == "components.py")

    if not is_builtin:
        # External component - create unique module name to avoid conflicts
        _module_name = f"{Path(_python_file_path).parent.name}.{Path(_python_file_path).stem}"
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
    # Built-in components are already loaded via "from uam4x.components import ..." at the top


for _plugin_name, _python_file_path in _config['PLUGINS'].items():
    # Create unique module name to avoid conflicts
    _module_name = f"{Path(_python_file_path).parent.name}.{Path(_python_file_path).stem}"
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
