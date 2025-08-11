from abc import ABC, abstractmethod

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