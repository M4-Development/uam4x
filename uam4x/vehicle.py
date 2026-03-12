from uam4x.subsystems import Subsystem
from uam4x.log import get_log
from uam4x.utils import log
from collections import  Counter

class VehicleManager():
    """Manage vehiclemanager entities and related operations.

    Notes
    -----
    This class encapsulates related state and behavior used by UAM4X workflows.

    """
    log = get_log(log, object_name='vehicle manager')
    def __init__(self):
        """Initialize object state and default configuration.

        Notes
        -----
        This method performs internal setup or validation logic.

        """
        self.log = VehicleManager.log
        self.log.debug('initialize')
        self._vehicles = []

    def add_vehicle(self, vehicle):
        """Add vehicle.

        Parameters
        ----------
        vehicle : Any
            Input value for `vehicle`.

        """
        self.log.debug(f'add vehicle: {vehicle.name}')
        for existing_vehicle in self._vehicles:
            if vehicle.name == existing_vehicle.name:
                self.log.error(f"Vehicle '{vehicle.name}' already exists in the study.")
                raise ValueError(f"Vehicle '{vehicle.name}' already exists in the study.")
        self._vehicles.append(vehicle)


    def get_vehicle(self, name):
        """Return vehicle.

        Parameters
        ----------
        name : Any
            Identifier used by this operation.

        Returns
        -------
        Any
            Requested value.

        """
        self.log.debug(f'get vehicle: {name}')
        for vehicle in  self._vehicles:
            if vehicle.name == name:
                return vehicle

        self.log.error(f"Vehicle with name {name} not found in vehicle manager")
        raise ValueError(f"Vehicle with name {name} not found in vehicle manager")

    def append(self, vehicle):
        """Execute append.

        Parameters
        ----------
        vehicle : Any
            Input value for `vehicle`.

        Returns
        -------
        Any
            Return value produced by this operation.

        """
        return self.add_vehicle(vehicle)

    def __getattr__(self, item):
        # Allow attribute access to components by their name
        """Execute getattr.

        Parameters
        ----------
        item : Any
            Input value for `item`.

        Returns
        -------
        Any
            Return value produced by this operation.

        """
        for veh in self._vehicles:
            if veh.name == item:
                return veh
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

    def __iter__(self):
        """Execute iter.

        Returns
        -------
        Any
            Return value produced by this operation.

        """
        return iter(self._vehicles)

    def __len__(self):
        """Execute len.

        Returns
        -------
        Any
            Return value produced by this operation.

        """
        return len(self._vehicles)

    def __repr__(self):
        """Return a developer-oriented string representation.

        Returns
        -------
        Any
            Return value produced by this operation.

        """
        names = [v.name for  v in  self._vehicles]
        return f"VehicleManager(vehicles={names})"



class Vehicle:
    """Represent the vehicle component in the UAM4X workflow.

    Notes
    -----
    This class encapsulates related state and behavior used by UAM4X workflows.

    """
    log = get_log(log, object_name='vehicle')
    def __init__(self, name, components=None):
        """Initialize object state and default configuration.

        Parameters
        ----------
        name : Any
            Identifier used by this operation.
        components : Any, optional
            Input value for `components`.

        """
        self.log = Vehicle.log
        self.name = name
        if components is  not None:
            self.check_component_names(components)
        self._components = components if components is not None else []
        self.log.debug(f'initialize')

    @staticmethod
    def check_component_names(components_list):
        """Validate component names.

        Parameters
        ----------
        components_list : Any
            Input value for `components_list`.

        """
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
        """Execute name.

        Returns
        -------
        Any
            Return value produced by this operation.

        """
        return self._name

    @name.setter
    def name(self, value):
        """Execute name.

        Parameters
        ----------
        value : Any
            Input value for `value`.

        """
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
        """Execute components.

        Returns
        -------
        Any
            Return value produced by this operation.

        """
        return self._components

    def add_component(self, component):
        """Add component.

        Parameters
        ----------
        component : Any
            Input value for `component`.

        """
        parent = getattr(component, '_parent')
        if parent is  not None:
            raise ValueError(f'A particular component may only belong to one vehicle at maximum.\n'
                             f'The component {str(component)} belongs to {str(parent)}')
        self._components.append(component)
        self.check_component_names(self._components)
        self.log.debug('add component')
        component._parent = self

    def remove_component(self, component):
        """Remove component.

        Parameters
        ----------
        component : Any
            Input value for `component`.

        """
        self.log.debug('remove component')
        self._components.remove(component)

    def get_component(self, name):
        """Return component.

        Parameters
        ----------
        name : Any
            Identifier used by this operation.

        Returns
        -------
        Any
            Requested value.

        """
        self.log.debug(f'get component: {name}')
        for comp in self._components:
            if comp.name == name:
                return comp
        self.log.error(f"Component with name {name} not found in vehicle {self._name}")
        raise ValueError(f"Component with name {name} not found in vehicle {self._name}")

    def get_component_names(self):
        """Return component names.

        Returns
        -------
        Any
            Requested value.

        """
        self.log.debug('get component names')
        return [comp.name for comp in self._components]

    def __getattr__(self, item):
        # Allow attribute access to components by their name
        """Execute getattr.

        Parameters
        ----------
        item : Any
            Input value for `item`.

        Returns
        -------
        Any
            Return value produced by this operation.

        """
        for comp in self._components:
            if comp.name == item:
                return comp
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

    def check_all_parameter_types(self):
        """Validate all parameter types.

        Returns
        -------
        Any
            Boolean status for the requested check.

        """
        self.log.debug('check all parameter types')
        """Return (all_ok, result_dict): all_ok is True if all parameters in all components and all modes are correct type."""
        result = {comp.name: comp.check_all_parameter_types() for comp in self._components}
        all_ok = all(
            all(param_ok for mode_result in comp_result.values() for param_ok in mode_result.values())
            for comp_result in result.values()
        )
        return all_ok, result

    def __str__(self):
        """Return a human-readable string representation.

        Returns
        -------
        Any
            Return value produced by this operation.

        """
        self.log.debug('write to string')
        comp_info = ', '.join(f"{c.name} ({c.__class__.__name__})" for c in self._components)
        return f"Vehicle(name={self._name}, components=[{comp_info}])"
    def _to_dict(self) -> dict:
        """Execute to dict.

        Returns
        -------
        Any
            JSON-serializable representation.

        """
        self.log.debug('write to dict')
        return {
            "name": self.name,
            "components": [c._to_dict() for c in self.components]
        }
    @classmethod
    def _from_dict(cls, data: dict) -> 'Vehicle':
        """Execute from dict.

        Parameters
        ----------
        data : dict
            Input value for `data`.

        Returns
        -------
        Any
            Reconstructed object instance.

        """
        cls.log.debug('vehicle from dict')
        v = Vehicle(data["name"] )
        for cdata in data["components"]:
            v.add_component(Subsystem._from_dict(cdata))
        return v




if __name__ == "__main__":
    from uam4x.subsystems import Wing

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
