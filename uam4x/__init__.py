
from uam4x.subsystems import register_subsystem, Subsystem, _subsystem_registry
from uam4x.solution import register_plugin, Plugin, _plugin_registry
from uam4x.utils import check_and_get_config, log
import os

import importlib.util
import sys
from pathlib import Path

_conf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'conf.json')
_config = check_and_get_config(_conf_path)

for _subsystem_name, _python_file_path in _config['SUBSYSTEMS'].items():
    # Check if this is a built-in subsystem from uam4x/uam4x/subsystems.py
    is_builtin = ("uam4x" == os.path.split(_python_file_path)[-2] and
                  os.path.basename(_python_file_path) == "subsystems.py")

    if not is_builtin:
        # External subsystem - create unique module name to avoid conflicts
        _module_name = f"{Path(_python_file_path).parent.name}.{Path(_python_file_path).stem}"
        _spec = importlib.util.spec_from_file_location(_module_name, _python_file_path)
        _module = importlib.util.module_from_spec(_spec)
        sys.modules[_module_name] = _module
        try:
            _spec.loader.exec_module(_module)
            _subsystem_registry[_subsystem_name] = getattr(_module,_subsystem_name)
        except FileNotFoundError:
            log.warning(f"Subsystem load error ({_subsystem_name}). Could not find path {_python_file_path}")
        except AttributeError:
            log.warning(f"Subsystem load error. Could not find {_subsystem_name} in {_python_file_path}")
        except Exception as e:
            log.warning(f"Subsystem load error ({_subsystem_name}). {e}")


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
    except Exception as e:
        log.warning(f"Plugin load error ({_plugin_name}). {e}")
