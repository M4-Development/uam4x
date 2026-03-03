
from uam4x.subsystems import register_subsystem, Subsystem, _subsystem_registry
from uam4x.solution import register_plugin, Plugin, _plugin_registry
from uam4x.utils import check_and_get_config, log
from uam4x.node import register_node, Node, _node_registry
import os

import importlib.util
import sys
from pathlib import Path

_conf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'conf.json')
_config = check_and_get_config(_conf_path)

for _comp_name, _python_file_path in _config['SUBSYSTEMS'].items():
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
            _subsystem_registry[_comp_name] = getattr(_module,_comp_name)
        except FileNotFoundError:
            log.warning(f"Component load error ({_comp_name}). Could not find path {_python_file_path}")
        except AttributeError:
            log.warning(f"Component load error. Could not find {_comp_name} in {_python_file_path}")
    # Built-in components are already loaded via "from uam4x.subsystems import ..." at the top


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
        log.warning(f"Plugin load error ({_plugin_name}). Could not find path {_python_file_path}")
    except AttributeError:
        log.warning(f"Plugin load error. Could not find {_plugin_name} in {_python_file_path}")

# Load registered nodes
for _node_name, _python_file_path in _config.get('NODES', {}).items():
    # Create unique module name to avoid conflicts
    _module_name = f"{Path(_python_file_path).parent.name}.{Path(_python_file_path).stem}"
    _spec = importlib.util.spec_from_file_location(_module_name, _python_file_path)
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_module_name] = _module

    # Add the directory containing the script to sys.path so it can find local imports
    _script_dir = os.path.dirname(_python_file_path)
    if _script_dir not in sys.path:
        sys.path.insert(0, _script_dir)

    try:
        _spec.loader.exec_module(_module)
        # Try to get the node function from the module
        if hasattr(_module, _node_name):
            _node_function = getattr(_module, _node_name)
            # Create a Node object and register it
            _node_obj = Node(_node_name, _node_function.__doc__ or "", _node_function)
            _node_obj.source_file = _python_file_path
            _node_obj.module_path = _module_name
            _node_registry[_node_name] = _node_obj
        else:
            log.warning(f"WARNING: Node load error. Could not find {_node_name} in {_python_file_path}")
    except FileNotFoundError:
        log.warning(f"WARNING: Node load error ({_node_name}). Could not find path {_python_file_path}")
    except Exception as e:
        log.warning(f"WARNING: Node load error ({_node_name}). {str(e)}")
    finally:
        # Remove the added path to avoid cluttering sys.path
        if _script_dir in sys.path:
            sys.path.remove(_script_dir)
