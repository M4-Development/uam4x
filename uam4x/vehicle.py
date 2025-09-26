from uam4x.components import Component
class VehicleManager(list):

    def __init__(self):
        self._vehicles = []

    def add_vehicle(self, vehicle):
        if vehicle.name in self:
            raise ValueError(f"Vehicle '{vehicle.name}' already exists in the study.")
        self._vehicles.append(vehicle)


    def get_vehicle(self, vehicle):
        return self.vehicle_name

    def __getattr__(self, item):
        # Allow attribute access to components by their name
        for veh in self._vehicles:
            if veh.name == item:
                return veh
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

    def __iter__(self):
        return self._vehicles.__iter__()

class Vehicle:
    def __init__(self, name, components=None):
        """
        Parameters
        ----------
        name : str
            Name of the vehicle
        components : list, optional
            List of components that define the vehicle (default: empty list)
        """
        self._name = name
        self._components = components if components is not None else []

    @property
    def name(self):
        return self._name

    @property
    def components(self):
        return self._components

    def add_component(self, component):
        self._components.append(component)

    def remove_component(self, component):
        self._components.remove(component)

    def get_component(self, name):
        for comp in self._components:
            if comp.name == name:
                return comp
        raise ValueError(f"Component with name '{name}' not found in vehicle '{self._name}'")

    def get_component_names(self):
        return [comp.name for comp in self._components]

    def __getattr__(self, item):
        # Allow attribute access to components by their name
        for comp in self._components:
            if comp.name == item:
                return comp
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

    def check_all_parameter_types(self):
        """Return (all_ok, result_dict): all_ok is True if all parameters in all components and all modes are correct type."""
        result = {comp.name: comp.check_all_parameter_types() for comp in self._components}
        all_ok = all(
            all(param_ok for mode_result in comp_result.values() for param_ok in mode_result.values())
            for comp_result in result.values()
        )
        return all_ok, result

    def __str__(self):
        comp_info = ', '.join(f"{c.name} ({c.__class__.__name__})" for c in self._components)
        return f"Vehicle(name={self._name}, components=[{comp_info}])"
    def _to_dict(self) -> dict:
        return {
            "name": self.name,
            "components": [c._to_dict() for c in self.components]
        }
    @classmethod
    def _from_dict(cls, data: dict) -> 'Vehicle':
        v = Vehicle(data["name"])
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
