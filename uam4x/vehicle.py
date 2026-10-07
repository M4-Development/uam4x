from __future__ import annotations
import copy
from uam4x.subsystems import Subsystem
from uam4x.log import get_log
from uam4x.utils import log
from collections import Counter


def copy_vehicles(vehicle_manager:VehicleManager):
    execution_vehicles = VehicleManager()
    for vehicle in vehicle_manager:
        vehicle_dict = copy.deepcopy(vehicle._to_dict())
        execution_vehicles.add_vehicle(Vehicle._from_dict(vehicle_dict))
    return execution_vehicles

class VehicleManager:
    """Manage the collection of vehicles in a study.

    A ``VehicleManager`` stores named :class:`Vehicle` objects, enforces unique
    vehicle names, and provides lookup and iteration helpers.

    """

    log = get_log(log, object_name="vehicle manager")

    def __init__(self):
        """Initialize an empty vehicle manager."""
        self.log = VehicleManager.log
        self.log.debug("initialize")
        self._vehicles = []

    def add_vehicle(self, vehicle):
        """Add a vehicle to the manager.

        Parameters
        ----------
        vehicle : Vehicle
            Vehicle object to add.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If a vehicle with the same name is already present.

        """
        self.log.debug(f"add vehicle: {vehicle.name}")
        for existing_vehicle in self._vehicles:
            if vehicle.name == existing_vehicle.name:
                self.log.error(f"Vehicle '{vehicle.name}' already exists in the study.")
                raise ValueError(
                    f"Vehicle '{vehicle.name}' already exists in the study."
                )
        self._vehicles.append(vehicle)

    def get_vehicle(self, name):
        """Return a vehicle by name.

        Parameters
        ----------
        name : str
            Vehicle name.

        Returns
        -------
        Vehicle
            Vehicle object registered under ``name``.

        Raises
        ------
        ValueError
            If no vehicle with name ``name`` is present.

        """
        self.log.debug(f"get vehicle: {name}")
        for vehicle in self._vehicles:
            if vehicle.name == name:
                return vehicle

        self.log.error(f"Vehicle with name {name} not found in vehicle manager")
        raise ValueError(f"Vehicle with name {name} not found in vehicle manager")

    def append(self, vehicle):
        """Append a vehicle to the manager.

        Parameters
        ----------
        vehicle : Vehicle
            Vehicle object to add.

        Returns
        -------
        None

        Notes
        -----
        This is an alias for :meth:`add_vehicle`.

        """
        return self.add_vehicle(vehicle)

    def __getattr__(self, item):
        # Allow attribute access to components by their name
        """Return a vehicle by attribute-style access.

        Parameters
        ----------
        item : str
            Vehicle name requested through attribute-style access.

        Returns
        -------
        Vehicle
            Vehicle object registered under ``item``.

        Raises
        ------
        AttributeError
            If no vehicle or manager attribute with name ``item`` is present.

        Notes
        -----
        This allows attribute-style access such as ``manager.my_vehicle``.

        """

        try:
            return self.get_vehicle(item)
        except ValueError as exc:
            raise AttributeError(
                f"{type(self).__name__!s} has no attribute or vehicle {item!r}"
            ) from exc


    def __iter__(self):
        """Return an iterator over stored vehicles.

        Returns
        -------
        iterator of Vehicle
            Iterator over vehicles in insertion order.

        """
        return iter(self._vehicles)

    def __len__(self):
        """Return the number of stored vehicles.

        Returns
        -------
        int
            Number of vehicles in the manager.

        """
        return len(self._vehicles)

    def __repr__(self):
        """Return a developer-oriented string representation.

        Returns
        -------
        str
            Representation including the stored vehicle names.

        """
        names = [v.name for v in self._vehicles]
        return f"VehicleManager(vehicles={names})"


class Vehicle:
    """Represent a vehicle composed of named subsystems.

    A ``Vehicle`` stores a collection of subsystem components, enforces unique
    component names, and provides helpers for component lookup, parameter type
    validation, and serialization.

    Parameters
    ----------
    name : str
        Vehicle name.
    components : sequence of Subsystem, optional
        Initial subsystem components to attach to the vehicle.

    """

    log = get_log(log, object_name="vehicle")

    def __init__(self, name, components=None):
        """Initialize a vehicle.

        Parameters
        ----------
        name : str
            Vehicle name.
        components : sequence of Subsystem, optional
            Initial subsystem components to attach to the vehicle.

        Raises
        ------
        NameError
            If ``components`` contains duplicate component names.

        """
        self.log = Vehicle.log
        self.name = name
        if components is not None:
            self.check_component_names(components)
        self._components = components if components is not None else []
        self.log.debug(f"initialize")

    @staticmethod
    def check_component_names(components_list):
        """Validate that component names are unique.

        Parameters
        ----------
        components_list : sequence of Subsystem
            Components whose names are checked for duplicates.

        Returns
        -------
        None

        Raises
        ------
        NameError
            If duplicate component names are found.

        """
        c_dict = Counter([c.name for c in components_list])
        vio = {name: instances for name, instances in c_dict.items() if instances > 1}
        msg = "Different components belonging to a vehicle must have different names. The following are in violation:\n"
        if vio:
            for name, instances in vio.items():
                msg += f"{name}, number of occurences: {instances}\n"
        if vio:
            raise NameError(msg)

    @property
    def name(self):
        """Vehicle name.

        Returns
        -------
        str
            Name of the vehicle.

        """
        return self._name

    @name.setter
    def name(self, value):
        """Set the vehicle name.

        Parameters
        ----------
        value : str
            New vehicle name.

        Raises
        ------
        ValueError
            If ``value`` is not a string or contains ``"["`` or ``"]"``.

        """
        if not isinstance(value, str):
            self.log.error(f"Name must be a string. The name {value} is invalid")
            raise ValueError(f"Name must be a string. The name {value} is invalid")
        if "[" in value or "]" in value:
            self.log.error(
                f'vehicles cannot have the character "[" or "]" in them. '
                f"The name: {value} is invalid"
            )
            raise ValueError(
                f'vehicles cannot have the character "[" or "]" in them. '
                f"The name: {value} is invalid"
            )
        self._name = value

    @property
    def components(self):
        """Return the attached subsystem components.

        Returns
        -------
        list of Subsystem
            Components attached to the vehicle.

        """
        return self._components

    def add_component(self, component):
        """Add a subsystem component to the vehicle.

        Parameters
        ----------
        component : Subsystem
            Component to attach to the vehicle.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``component`` already belongs to another vehicle.
        NameError
            If adding ``component`` creates duplicate component names within
            the vehicle.

        Notes
        -----
        Component names must remain unique within the vehicle. The component
        parent reference is updated when the add succeeds.

        """
        parent = getattr(component, "_parent")
        if parent is not None:
            raise ValueError(
                f"A particular component may only belong to one vehicle at maximum.\n"
                f"The component {str(component)} belongs to {str(parent)}"
            )
        self._components.append(component)
        self.check_component_names(self._components)
        self.log.debug("add component")
        component._parent = self

    def remove_component(self, component):
        """Remove a subsystem component from the vehicle.

        Parameters
        ----------
        component : Subsystem or str
            Component object to remove, or the name of the component to
            remove.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``component`` is a string and no component with that name is
            attached to the vehicle, or if the component object is not present
            in the vehicle.

        """
        self.log.debug("remove component")
        if isinstance(component, str):
            component = self.get_component(component)
        self._components.remove(component)
        component._parent = None

    def get_component(self, name):
        """Return a component by name.

        Parameters
        ----------
        name : str
            Component name.

        Returns
        -------
        Subsystem
            Component registered under ``name``.

        Raises
        ------
        ValueError
            If no component with name ``name`` is attached to the vehicle.

        """
        self.log.debug(f"get component: {name}")
        for comp in self._components:
            if comp.name == name:
                return comp
        self.log.error(f"Component with name {name} not found in vehicle {self._name}")
        raise ValueError(
            f"Component with name {name} not found in vehicle {self._name}"
        )

    def get_component_names(self):
        """Return component names.

        Returns
        -------
        list of str
            Names of components attached to the vehicle.

        """
        self.log.debug("get component names")
        return [comp.name for comp in self._components]

    def __getattr__(self, item):
        # Allow attribute access to components by their name
        """Return a component by attribute-style access.

        Parameters
        ----------
        item : str
            Component name requested through attribute-style access.

        Returns
        -------
        Subsystem
            Component registered under ``item``.

        Raises
        ------
        AttributeError
            If no component with name ``item`` is attached to the vehicle.

        Notes
        -----
        This allows attribute-style access such as ``vehicle.wing``.

        """
        for comp in self._components:
            if comp.name == item:
                return comp
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{item}'"
        )

    def check_all_parameter_types(self):
        """Validate parameter types for all attached components.

        Returns
        -------
        all_ok : bool
            True if all component and mode parameters have valid types.
        result : dict
            Nested dictionary containing per-component and per-mode validation
            results.

        Notes
        -----
        This delegates parameter-type validation to each component attached to
        the vehicle.

        """
        self.log.debug("check all parameter types")
        """Return (all_ok, result_dict): all_ok is True if all parameters in all components and all modes are correct type."""
        result = {
            comp.name: comp.check_all_parameter_types() for comp in self._components
        }
        all_ok = all(
            all(
                param_ok
                for mode_result in comp_result.values()
                for param_ok in mode_result.values()
            )
            for comp_result in result.values()
        )
        return all_ok, result

    def __str__(self):
        """Return a human-readable string representation of the vehicle.

        Returns
        -------
        str
            String representation including the vehicle name and component
            names.

        """
        self.log.debug("write to string")
        comp_info = ", ".join(
            f"{c.name} ({c.__class__.__name__})" for c in self._components
        )
        return f"Vehicle(name={self._name}, components=[{comp_info}])"

    def _to_dict(self) -> dict:
        """Serialize the vehicle to a dictionary.

        Returns
        -------
        dict
            JSON-compatible dictionary containing the vehicle name and
            serialized component data.

        """
        self.log.debug("write to dict")
        return {
            "name": self.name,
            "components": [c._to_dict() for c in self.components],
        }

    @classmethod
    def _from_dict(cls, data: dict) -> "Vehicle":
        """Reconstruct a vehicle from a dictionary.

        Parameters
        ----------
        data : dict
            Serialized vehicle dictionary, typically produced by :meth:`_to_dict`.

        Returns
        -------
        Vehicle
            Reconstructed vehicle with restored components.

        """
        cls.log.debug("vehicle from dict")
        v = Vehicle(data["name"])
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
    v.left_wing.no_sweep.chord = 4.0
    v.left_wing.no_sweep.span = 0.0

    print(v.check_all_parameter_types())

    # Print the vehicle and its components
    print(v)
    for comp in v.components:
        print(comp)
        print(f"Modes for {comp.name}:")
        for mode_name, mode in comp._modes.items():
            print(f"    {mode}")
