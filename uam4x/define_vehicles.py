import uam4x.components
import os
import sys
import importlib

class Vehicle:

    def __init__(self, study, name, components=None, new_mode=False) -> None:
        """_summary_

        Parameters
        ----------
        name : str
            _description_, name of the vehicle
        components : list, optional
            _description_, by default None, a list of components that define the aircraft [LiftingSurface, Propulsion, etc.]
        modes : list, optional
            _description_, by default None, contains information describing each mode
        """
        self._name = name
        if not new_mode:
            self._modes = []
            self._parent = None
            if components is None:
                components = []
        self._components = components
        self.study = study

    @property
    def name(self):
        return self._name

    @property
    def mode_name(self):
            return self._mode_name

    @mode_name.setter
    def mode_name(self, name):
        self._mode_name = name

    @property
    def parent(self):
        return self._parent

    @property
    def components(self):
        return self._components

    @components.setter
    def components(self, components):
        self._components = components

    @property
    def modes(self):
        return self._modes

    def get_mode(self, name:str):
        for mode in self._modes:
            if mode.mode_name == name:
                return mode

    def get_component(self, name:str):
        for component in self._components:
            if component.name == name:
                return component

    def get_component_names(self):
        component_names = []
        for component in self._components:
            component_names.append(component.name)
        return component_names

    def get_mode_names(self):
        mode_names = []
        for mode in self._modes:
            mode_names.append(mode.mode_name)
        return mode_names

    def new_mode(self, mode_name, components=None):
        mode_names = self.get_mode_names()
        if mode_name not in mode_names:
            if components is None:
                components = {}
                for component in self._components:
                    components[component.name] = component.mode_name
            mode = VehicleMode(study=self.study, name=self._name, mode_name=mode_name, parent=self, components=components)
            self._modes.append(mode)
            return mode
        else:
            return self.get_mode(mode_name)

    def new_component(self, component_type, component_name:str):
        component_names = self.get_component_names()
        if component_name not in component_names:
            if type(component_type) is str:
                if hasattr(uam4x.components, component_type):
                    new_component = getattr(uam4x.components, component_type)(self.study, name=component_name)
                else:
                    work_dir = os.getcwd()
                    script_path = os.path.abspath(sys.argv[0])
                    script_dir = os.path.dirname(script_path)
                    if script_dir not in sys.path:
                        sys.path.append(script_dir)
                    script_dir_relative_path = os.path.relpath(script_dir, work_dir)
                    python_files = []
                    for f in os.listdir(script_dir_relative_path):
                        if f.endswith(".py") and f != "__init__.py":
                            python_files.append(f)
                    for f in python_files:
                        file_name = f[:-3]
                        module = importlib.import_module(file_name)
                        if hasattr(module, component_type):
                            new_component = getattr(module, component_type)(self.study, name=component_name)
            else:
                new_component = component_type(self.study, name=component_name)
            self._components.append(new_component)
            for mode in self._modes:
                mode.components[component_name] = new_component.mode_name
            return new_component
        else:
            return self.get_component(component_name)

    def save_vehicle(self):
        vehicle_dict = {}
        components_dict = {}
        for component in self._components:
            components_dict[component.name] = component.save_component()
        vehicle_dict["components"] = components_dict
        modes_dict = {}
        for mode in self._modes:
            modes_dict[mode.mode_name] = mode.components
        vehicle_dict["modes"] = modes_dict
        return vehicle_dict



class VehicleMode(Vehicle):
    def __init__(self, study, name, mode_name, parent, components) -> None:
        super().__init__(study=study, name=name, components=components, new_mode=True)
        self._parent = parent
        self._mode_name = mode_name

    def get_component(self, name:str):
        if name in self._components:
            mode_name = self._components[name]
            return self._parent.get_component(name).get_mode(mode_name)

    def get_component_names(self):
        component_names = []
        for component in self._parent.components:
            component_names.append(component.name)
        return component_names

    def set_component_mode(self, component_name, mode_name):
        if component_name in self._components:
            mode_names = self._parent.get_component(component_name).get_mode_names()
            if mode_name in mode_names:
                self._components[component_name] = mode_name