from abc import ABC, abstractmethod
_registry = {}

def register_plugin(cls):
    _registry[cls.__name__] = cls
    return cls

def get_new_plugin_instance(plugin_name):
    if not plugin_name in _registry:
        msg = f"Plugin '{plugin_name}' was not registered. Please check the following:\n"
        msg += f"\t1. Does the plugin exist / did you spell the name correctly?\n"
        msg += f"\t2. Did you import the file containing the plugin?\n"
        msg += f"\t3. Does the plugin definition have the @register_plugin decorator?\n"
        raise ImportError(msg)
    return _registry[plugin_name]()

class Plugin(ABC):
    def __init__(self) -> None:
        self.discrete_names = []


    @abstractmethod
    def setup_plugin(self, plugin_options:dict):
        pass

    @abstractmethod
    def run_plugin(self, links:dict, work_dir:str):
        pass

    def declare_discrete(self, name):
        self.discrete_names.append(name)
