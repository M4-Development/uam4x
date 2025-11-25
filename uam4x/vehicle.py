from uam4x.components import Component
from uam4x.log import get_log
from uam4x.utils import log
from collections import  Counter

class VehicleManager():
    log = get_log(log, object_name='vehicle manager')
    def __init__(self):
        self.log = VehicleManager.log
        self.log.debug('initialize')
        self._vehicles = []
    
    def add_vehicle(self, vehicle):
        self.log.debug(f'add vehicle: {vehicle.name}')
        for existing_vehicle in self._vehicles:
            if vehicle.name == existing_vehicle.name:
                self.log.error(f"Vehicle '{vehicle.name}' already exists in the study.")
                raise ValueError(f"Vehicle '{vehicle.name}' already exists in the study.")
        self._vehicles.append(vehicle)


    def get_vehicle(self, name):
        self.log.debug(f'get vehicle: {name}')
        for vehicle in  self._vehicles:
            if vehicle.name == name:
                return vehicle
            
        self.log.error(f"Vehicle with name {name} not found in vehicle manager")
        raise ValueError(f"Vehicle with name {name} not found in vehicle manager")        

    def append(self, vehicle):
        return self.add_vehicle(vehicle)    

    def __getattr__(self, item):
        # Allow attribute access to components by their name
        for veh in self._vehicles:
            if veh.name == item:
                return veh
        self.log.warning(f"'{self.__class__.__name__}' object has no attribute '{item}'")
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")
    
    def __iter__(self):
        return iter(self._vehicles)
    
    def __len__(self):
        return len(self._vehicles)
    
    def __repr__(self):
        names = [v.name for  v in  self._vehicles]
        return f"VehicleManager(vehicles={names})"
    
    

class Vehicle:
    log = get_log(log, object_name='vehicle')
    def __init__(self, name, components=None):
        """
        Parameters
        ----------
        name : str
            Name of the vehicle
        components : list, optional
            List of components that define the vehicle (default: empty list)
        """
        self.log = Vehicle.log        
        self.name = name
        if components is  not None:
            self.check_component_names(components)
        self._components = components if components is not None else []
        self.log.debug(f'initialize')

    @staticmethod
    def check_component_names(components_list):
        c_dict = Counter([c.name for  c in  components_list])
        vio = {name: instances for name, instances in  c_dict.items() if instances > 1}
        msg ='Different components belonging to a vehicle must have different names. The following are in violation:\n'
        if vio:
            for name, instances in  vio.items():
                msg += f'{name}, number of occurences: {instances}\n'
        if vio:
            raise NameError(msg)
    @property
    def name(self):
        self.log.debug('get name')
        return self._name

    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            self.log.error(f"Name must be a string. The name {value} is invalid")
            raise ValueError(f"Name must be a string. The name {value} is invalid")
        if '[' in  value or  ']' in  value:
            self.log.error(f'vehicles cannot have the character "[" or "]" in them. '
                           f'The name: {value} is invalid')            
            raise ValueError(f'vehicles cannot have the character "[" or "]" in them. '
                             f'The name: {value} is invalid')        
        self._name = value

    @property
    def components(self):
        self.log.debug('get component')
        return self._components

    def add_component(self, component):
        parent = getattr(component, '_parent')
        if parent is  not None:
            raise ValueError(f'A particular component may only belong to one vehicle at maximum.\n'
                             f'The component {str(component)} belongs to {str(parent)}')
        self._components.append(component)        
        self.check_component_names(self._components)        
        self.log.debug('add component')
        component._parent = self

    def remove_component(self, component):
        self.log.debug('remove component')
        self._components.remove(component)

    def get_component(self, name):
        self.log.debug(f'get component: {name}')
        for comp in self._components:
            if comp.name == name:
                return comp
        self.log.error(f"Component with name {name} not found in vehicle {self._name}")
        raise ValueError(f"Component with name {name} not found in vehicle {self._name}")

    def get_component_names(self):
        self.log.debug('get component names')
        return [comp.name for comp in self._components]

    def __getattr__(self, item):
        # Allow attribute access to components by their name
        for comp in self._components:
            if comp.name == item:
                return comp
        self.log.warning(f"'{self.__class__.__name__}' object has no attribute '{item}'")
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

    def check_all_parameter_types(self):
        self.log.debug('check all parameter types')
        """Return (all_ok, result_dict): all_ok is True if all parameters in all components and all modes are correct type."""
        result = {comp.name: comp.check_all_parameter_types() for comp in self._components}
        all_ok = all(
            all(param_ok for mode_result in comp_result.values() for param_ok in mode_result.values())
            for comp_result in result.values()
        )
        return all_ok, result

    def __str__(self):
        self.log.debug('write to string')
        comp_info = ', '.join(f"{c.name} ({c.__class__.__name__})" for c in self._components)
        return f"Vehicle(name={self._name}, components=[{comp_info}])"
    def _to_dict(self) -> dict:
        self.log.debug('write to dict')
        return {
            "name": self.name,
            "components": [c._to_dict() for c in self.components]
        }
    @classmethod
    def _from_dict(cls, data: dict) -> 'Vehicle':
        cls.log.debug('vehicle from dict')
        v = Vehicle(data["name"] )
        for cdata in data["components"]:
            v.add_component(Component._from_dict(cdata))
        return v




if __name__ == "__main__":
    from uam4x.components import Wing

    # Create a vehicle
    v = Vehicle("DemoVehicle")
    # Create two wings
    wing1 = Wing("left_wing")
    wing2 = Wing("right_wing")

    # Add wings to the vehicle
    v.add_component(wing1)
    v.add_component(wing2)

    # Set parameters for left wing
    wing1.span = 10.0
    wing1.chord = 2.0
    wing1.sweep = 25.0

    # Set parameters for right wing
    wing2.span = 10.0
    wing2.chord = 3.0
    wing2.three = 25.0

    v.left_wing.add_mode("no_sweep")
    v.left_wing.no_sweep.chord = 4.
    v.left_wing.no_sweep.span = 0.

    print(v.check_all_parameter_types())

    # Print the vehicle and its components
    print(v)
    for comp in v.components:
        print(comp)
        print(f"Modes for {comp.name}:")
        for mode_name, mode in comp._modes.items():
            print(f"    {mode}")
