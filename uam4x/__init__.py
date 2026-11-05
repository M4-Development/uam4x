from uam4x.subsystems import register_subsystem, Subsystem, _subsystem_registry
from uam4x.solution import register_plugin, Plugin, _plugin_registry
from uam4x.utils import check_and_get_config, log, _get_class_from_file
import os

_conf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf.json")
_config = (
    {"SUBSYSTEMS": {}, "PLUGINS": {}}
    if os.getenv("UAM4X_DISABLE_REGISTRY_LOAD") == "1"
    else check_and_get_config(_conf_path)
)

for _subsystem_name, _python_file_path in _config["SUBSYSTEMS"].items():
    if _subsystem_name in _subsystem_registry:
        continue
    cls, msg = _get_class_from_file(_subsystem_name, _python_file_path)
    if cls is not None:
        _subsystem_registry[_subsystem_name] = cls
    else:
        log.warning(f"Subsystem '{_subsystem_name}' not found in {_python_file_path}")
        log.warning(msg)

for _plugin_name, _python_file_path in _config["PLUGINS"].items():
    if _plugin_name in _plugin_registry:
        continue
    cls, msg = _get_class_from_file(_plugin_name, _python_file_path)
    if cls is not None:
        _plugin_registry[_plugin_name] = cls
    else:
        log.warning(f"Plugin '{_plugin_name}' not found in {_python_file_path}")
        log.warning(msg)
