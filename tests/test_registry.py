from uam4x.solution import Plugin, register_plugin, get_plugin_class, _plugin_registry
from uam4x.utils import *
import pytest
from uam4x.components import Component, register_component, _component_registry

    
def test_registery_good():
 
    file_path = os.path.abspath(__file__)
    test_dir = os.path.dirname(file_path)
    proj_dir = os.path.dirname(test_dir)
    
    
    config_path = os.path.join(proj_dir,'uam4x', 'conf.json')
    
    if os.path.exists(config_path):
        os.remove(config_path)
    
    config = check_and_get_config(config_path)
    assert config == {"PLUGINS": {},"COMPONENTS": {}}
    
    @register_plugin
    class P(Plugin):
        pass
    
    config = check_and_get_config(config_path)
    assert config == {"PLUGINS": {P.__name__: file_path},"COMPONENTS": {}}    
    
    @register_component
    class C(Component):
        pass
    
    config = check_and_get_config(config_path)
    assert config == {"PLUGINS": {P.__name__: file_path},"COMPONENTS": {C.__name__: file_path,}}
    

    
