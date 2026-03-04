from uam4x.solution import Plugin, register_plugin, get_plugin_class, _plugin_registry
from uam4x.utils import *
import pytest
from uam4x.subsystems import Subsystem, register_subsystem, _subsystem_registry


def test_registery_good():

    file_path = os.path.abspath(__file__)
    test_dir = os.path.dirname(file_path)
    proj_dir = os.path.dirname(test_dir)


    config_path = os.path.join(proj_dir,'uam4x', 'conf.json')

    if os.path.exists(config_path):
        os.remove(config_path)

    config = check_and_get_config(config_path)
    assert "PLUGINS" in config
    assert "SUBSYSTEMS" in config

    @register_plugin
    class P(Plugin):
        pass

    config = check_and_get_config(config_path)
    if os.getenv("UAM4X_DISABLE_REGISTRY_WRITES") != "1":
        assert P.__name__ in config["PLUGINS"]
    assert P.__name__ in _plugin_registry

    @register_subsystem
    class C(Subsystem):
        pass

    config = check_and_get_config(config_path)
    if os.getenv("UAM4X_DISABLE_REGISTRY_WRITES") != "1":
        assert P.__name__ in config["PLUGINS"]
        assert C.__name__ in config["SUBSYSTEMS"]
    assert C.__name__ in _subsystem_registry

