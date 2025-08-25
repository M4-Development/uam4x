from typing import Callable
_plugin_registry = {}

class Solution:
    def __init__(self) -> None:
        self.tracked_outputs = []
        self.links = []
        self.steps = {}

    def add_link(self, independent:str, dependent:str):
        link = [independent, dependent]
        if link not in self.links:
            self.links.append(link)

    def remove_link(self, independent:str, dependent:str):
        link = [independent, dependent]
        if link in self.links:
            self.links.remove(link)

    def get_step_names(self):
        return [step_name for step_name in self.steps]

    def add_step(self, plugin_class, step_name=None, plugin_options:dict={}):
        if not isinstance(plugin_class, Callable):
            raise TypeError("plugin_class should be a callable (class).")
        # plugin_name should be a string, not a class
        if step_name is None:
            step_name = "step" + str(len(self.steps)+1)
        step_info = {"plugin": plugin_class, "plugin_options": plugin_options}
        self.steps[step_name] = step_info

    def remove_step(self, step_name):
        if step_name in self.steps:
            self.steps.pop(step_name)

    def edit_step(self, step_name, plugin_options:dict=None):
        if step_name in self.steps:
            step = self.steps[step_name]
            if plugin_options is not None:
                current_options = step["plugin_options"]
                for option_name, info in plugin_options.items():
                    current_options[option_name] = info
            else:
                plugin_options = {}

    def add_tracked_output(self, output:str):
        if output in self.tracked_outputs:
            raise RuntimeError(f"output already being tracked: {output}")
        self.tracked_outputs.append(output)

    def remove_tracked_output(self, output:str):
        self.tracked_outputs.remove(output)

    def add_plugin_option(self, step_name:str, option_name:str, option_value):
        if step_name in self.steps:
            step = self.steps[step_name]
            step["plugin_options"][option_name] = option_value

    def remove_plugin_option(self, step_name:str, option_name:str):
        if step_name in self.steps:
            step = self.steps[step_name]
            step["plugin_options"].pop(option_name)

    def _to_dict(self) -> dict:
        # Serialize steps, converting plugin class to its name
        steps_serialized = {}
        for step_name, step_info in self.steps.items():
            plugin_class = step_info["plugin"]
            plugin_name = plugin_class.__name__
            steps_serialized[step_name] = {
                "plugin": plugin_name,
                "plugin_options": step_info["plugin_options"]
            }
        return {
            "steps": steps_serialized,
            "tracked_outputs": self.tracked_outputs,
            "links": self.links
        }

    @classmethod
    def _from_dict(cls, data: dict) -> 'Solution':
        sol = cls()
        # Reconstruct steps, mapping plugin name to class if possible
        for step_name, step_data in data["steps"].items():
            plugin_name = step_data["plugin"]
            plugin_class = get_plugin_class(plugin_name)
            sol.add_step(plugin_class, step_name, step_data["plugin_options"])
        sol.tracked_outputs = data["tracked_outputs"]
        sol.links = data["links"]
        return sol

class Plugin():
    def __init__(self) -> None:
        self.discrete_names = []


    def setup_plugin(self, plugin_options:dict):
        pass

    def run_plugin(self, links:dict, work_dir:str):
        pass

    def declare_discrete(self, name):
        self.discrete_names.append(name)


def register_plugin(cls):
    _plugin_registry[cls.__name__] = cls
    return cls

def get_plugin_class(plugin_name):
    if not plugin_name in _plugin_registry:
        msg = f"Plugin '{plugin_name}' was not registered. Please check the following:\n"
        msg += f"\t1. Does the plugin exist / did you spell the name correctly?\n"
        msg += f"\t2. Did you import the file containing the plugin?\n"
        msg += f"\t3. Does the plugin definition have the @register_plugin decorator?\n"
        raise ImportError(msg)
    return _plugin_registry[plugin_name]

