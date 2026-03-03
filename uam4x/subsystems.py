import os
from uam4x.log import get_log
from uam4x.utils import log

from uam4x.units import (
    UnitType, LengthUnit, AngleUnit, MassUnit, TimeUnit, ForceUnit, TemperatureUnit, PressureUnit, EnergyUnit, PowerUnit, VolumeUnit, AreaUnit, SpeedUnit, AccelerationUnit
)
from uam4x.utils import check_and_get_config, type_string_to_type,is_main_process,registry_writes_enabled

from enum import Enum

import json
import importlib.util
import numpy as np


_subsystem_registry = {}
_subsystem_names = []
_subsystem_to_param_name_type_list = {}

class Parameter:
    """
    Represents a parameter with type, value, units, and description.

    Parameters
    ----------
    name : str
        Name of the parameter.
    default_value : Any
        Default value for the parameter.
    dtype : type
        Python type for the parameter (e.g., float, int).
    unit_type : UnitType, optional
        The type of unit (e.g., LENGTH, MASS).
    unit : Enum or str, optional
        The unit (e.g., METER, KG).
    description : str, optional
        Description of the parameter.
    """
    log = get_log(log, object_name=f'parameter')
    def __init__(self, name: str, default_value, dtype, unit_type: UnitType = None, unit: Enum = None, description: str = ""):
        from uam4x.units import UnitType
        self.name = name
        self.dtype = dtype
        self.log = Parameter.log
        if not isinstance(default_value, dtype):
            self.log.error(f"Parameter '{name}' must be of type {dtype.__name__}, got {type(default_value).__name__}")
            raise TypeError(f"Parameter '{name}' must be of type {dtype.__name__}, got {type(default_value).__name__}")
        self._value = default_value

        self.unit_type = unit_type if unit_type is not None else UnitType.NONE
        self.unit = unit if unit is not None else ""
        self.description = description

    def __str__(self):
        """
        Returns a string representation of the parameter.

        Returns
        -------
        str
        """
        unit_display = self.unit if self.unit else "—"
        self.log.debug(f'return string representation')
        return f"{self.name}: {self.value} [{self.unit_type}, {unit_display}] ({self.description})"

    def __repr__(self):
        """
        Returns a string representation of the parameter (same as __str__).

        Returns
        -------
        str
        """
        self.log.debug(f'return string representation')
        return self.__str__()
    @property
    def value(self):
        """
        Get or set the value of the parameter.

        Returns
        -------
        Any
        """
        return self._value

    @value.setter
    def value(self, v):
        """
        Set the value of the parameter, enforcing type.

        Parameters
        ----------
        v : Any
            Value to set. Must match dtype.
        """

        if not isinstance(v, self.dtype):
            self.log.error(f"Value for parameter '{self.name}' must be of type {self.dtype.__name__}, got {type(v).__name__}")
            raise TypeError(f"Value for parameter '{self.name}' must be of type {self.dtype.__name__}, got {type(v).__name__}")
        self._value = v

    #def check_type(self):
        #"""
        #Check if the value matches the dtype.

        #Returns
        #-------
        #bool
        #"""
        #self.log.debug(f'check type')
        #return isinstance(self.value, self.dtype)

    def _to_dict(self) -> dict:
        """
        Serialize the parameter to a JSON-serializable dictionary.

        Returns
        -------
        dict
        """
        # Only include JSON-serializable fields, never include .dtype or any type/class
        # Ensure unit_type and unit are always serializable (string or None)
        self.log.debug(f'write to dict')
        unit_type = self.unit_type
        if hasattr(unit_type, '__name__'):
            unit_type = unit_type.__name__
        #elif not isinstance(unit_type, str):
            #unit_type = str(unit_type) if unit_type is not None else None

        unit = self.unit
        #if hasattr(unit, 'value'):
            #unit = unit.value
        #if hasattr(unit, '__name__'):
            #unit = unit.__name__
        #elif not isinstance(unit, str):
            #unit = str(unit) if unit is not None else None

        return {
            "name": self.name,
            "value": self.value,
            "dtype": self.dtype.__name__,
            "unit_type": unit_type,
            "unit": unit,
            "description": self.description
        }
    @classmethod
    def _from_dict(cls, data: dict) -> 'Parameter':
        """
        Deserialize a parameter from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary representation of the parameter.

        Returns
        -------
        Parameter
        """
        cls.log.debug('Deserialize a parameter from a dictionary')
        dtype = type_string_to_type(data["dtype"])
        if dtype == np.ndarray:
            value = np.array(data["value"])
        else:
            value = dtype(data["value"])
        return cls(
            name=data["name"],
            default_value=value,
            dtype=dtype,
            unit_type=data["unit_type"],
            unit=data["unit"],
            description=data["description"],
        )

class _Mode:
    """
    Represents a mode for a component, allowing parameter overrides.

    Parameters
    ----------
    name : str
        Name of the mode.
    component : Subsystem, optional
        The parent component.
    """
    log = get_log(log, object_name=f'mode')
    def __init__(self, name: str, component: "Subsystem"):
        self.name = name
        self._overrides = {}
        self.component = component  # Reference to parent component for inheritance
        self.log = _Mode.log

    def set_parameter(self, name: str, value):
        """
        Set a parameter override for this mode.

        Parameters
        ----------
        name : str
            Parameter name.
        value : Any
            Value to set.
        """
        self.log.debug(f'set parameter')
        if name not in self.component._parameters:
            self.log.error(f"Parameter '{name}' does not exist in component '{self.component.name}' for mode '{self.name}'")
            raise AttributeError(f"Parameter '{name}' does not exist in component '{self.component.name}' for mode '{self.name}'")
        dtype = self.component._parameters[name].dtype
        if not isinstance(value, dtype):
            self.log.error(f"Override for parameter '{name}' in mode '{self.name}' must be of type {dtype.__name__}, got {type(value).__name__}")
            raise TypeError(f"Override for parameter '{name}' in mode '{self.name}' must be of type {dtype.__name__}, got {type(value).__name__}")
        self._overrides[name] = value

    def get_parameter(self, name: str):
        """
        Get the value of a parameter override or inherited from the component.

        Parameters
        ----------
        name : str
            Parameter name.

        Returns
        -------
        Any
            The value of the parameter.
        """
        self.log.debug(f'get parameter')
        if name in self._overrides:
            return self._overrides[name]
        if name in self.component._parameters:
            return self.component._parameters[name].value
        self.log.error(f"Parameter '{name}' does not exist in component '{self.component.name}' for mode '{self.name}'")
        raise AttributeError(f"Parameter '{name}' does not exist in component '{self.component.name}' for mode '{self.name}'")

    def __getattr__(self, item):
        """
        Get an attribute or parameter override from the mode or parent component.

        Parameters
        ----------
        item : str
            Attribute or parameter name.

        Returns
        -------
        Any
        """
        # Only called if attribute not found in normal places
        if item in self._overrides:
            return self._overrides[item]
        if self.component and hasattr(self.component, item) and not isinstance(getattr(self.component, item), _Mode) and not callable(getattr(self.component, item)):
            return getattr(self.component, item)
        self.log.error(f"'{self.__class__.__name__}' object has no attribute '{item}'")
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        """
        Set a parameter override or attribute.

        Parameters
        ----------
        key : str
            Attribute or parameter name.
        value : Any
            Value to set.
        """
        if key in ("name", "_overrides", "component"):
            object.__setattr__(self, key, value)
        elif key in self.component._parameters:
            if not isinstance(value, self.component._parameters[key].dtype):
                self.log.error(f"Parameter '{key}' in mode '{self.name}' must be of type {self.component._parameters[key].dtype.__name__}, got {type(value).__name__}")
                raise TypeError(f"Parameter '{key}' in mode '{self.name}' must be of type {self.component._parameters[key].dtype.__name__}, got {type(value).__name__}")
            self._overrides[key] = value
        elif key == 'log':
            pass
        else:
            self.log.error(f"'{self.__class__.__name__}' object has no parameter '{key}' to override")
            raise AttributeError(f"'{self.__class__.__name__}' object has no parameter '{key}' to override")

    @property
    def parameters(self):
        """
        Get all effective parameters for this mode (overrides + inherited).

        Returns
        -------
        dict
            Mapping of parameter names to Parameter objects.
        """
        # Return a dict of all effective Parameter objects: overridden in mode, else from component
        self.log.debug(f'Get parameters for mode {self.name}')
        params = {}
        for pname, param in self.component._parameters.items():
            if pname in self._overrides:
                override_value = self._overrides[pname]
                dtype = param.dtype
                if dtype is not None and not isinstance(override_value, dtype):
                    self.log.error(f"Override for parameter '{pname}' in mode '{self.name}' must be of type {dtype.__name__}, got {type(override_value).__name__}")
                    raise TypeError(f"Override for parameter '{pname}' in mode '{self.name}' must be of type {dtype.__name__}, got {type(override_value).__name__}")
                overridden_param = Parameter(
                    param.name,
                    override_value,
                    dtype,
                    unit_type=param.unit_type,
                    unit=param.unit,
                    description=param.description
                )
                params[pname] = overridden_param
            else:
                params[pname] = param
        # Add any extra overrides not in component
        for pname, value in self._overrides.items():
            if pname not in params:
                self.log.error(f"Parameter '{pname}' does not exist in component '{self.component.name}' for mode '{self.name}'")
                raise AttributeError(f"Parameter '{pname}' does not exist in component '{self.component.name}' for mode '{self.name}'")
        return params

    def __str__(self):
        """
        Returns a string representation of the mode and its overrides.

        Returns
        -------
        str
        """
        self.log.debug(f'return a string representation')
        if not self._overrides:
            return f"Mode('{self.name}') (no parameter overrides)"
        param_strs = [f"{k}: {v}" for k, v in self._overrides.items()]
        params = ", ".join(param_strs)
        return f"Mode('{self.name}') overrides: {params}"

class Subsystem:
    """
    Base class for all subsystems. Supports parameters, modes, and dynamic properties.

    Parameters
    ----------
    name : str
        Name of the component.
    """
    log = get_log(log, object_name=f'component')
    def __init__(self, name: str):
        self.name = name
        if '[' in  name:
            self.log.error(f'components cannot have the character "[" in them')
            raise ValueError(f'components cannot have the character "[" in them')
        self._parameters = {}
        self._modes = {}
        self._is_setup = True
        self.log = Subsystem.log
        self.setup()
        self._is_setup = False
        setattr(self, 'baseline', self)
        self._parent = None

    def setup(self):
        """
        Override in subclasses to add parameters using add_parameter().
        """
        self.log.debug(f'Override in subclasses to add parameters using add_parameter()')

    def add_parameter(self, name: str, default_value, dtype, unit_type=None, unit=None, description=""):
        """
        Add a parameter to the component (only allowed in setup()).

        Parameters
        ----------
        name : str
            Name of the parameter.
        default_value : Any
            Default value.
        dtype : type
            Python type (e.g., float, int).
        unit_type : UnitType, optional
            Unit type (e.g., LENGTH, MASS).
        unit : Enum or str, optional
            Unit (e.g., METER, KG).
        description : str, optional
            Description.
        """

        self.log.debug(f'add parameter {name} to {type(self).__name__}')
        if not self._is_setup:
            self.log.error("Cannot add parameters outside of setup().\nAdd them in the setup() method or use SetParameter() after setup.")
            raise RuntimeError("Cannot add parameters outside of setup().\nAdd them in the setup() method or use SetParameter() after setup.")
        if '[' in  name or ']' in  name:
            self.log.error(f'parameters cannot have the character "[" or "]" in them')
            raise ValueError(f'parameters cannot have the character "[" or "]" in them')
        self._parameters[name] = Parameter(name, default_value, dtype, unit_type=unit_type, unit=unit, description=description)
        # Dynamically add property to the instance for this parameter
        def getter(self):
            return self._parameters[name].value
        def setter(self, v):
            self.set_parameter(name, v)
        prop = property(getter, setter)
        setattr(self.__class__, name, prop)

    def set_parameter(self, name: str, value: float, unit=None):
        """
        Set the value (and optionally unit) of a parameter.

        Parameters
        ----------
        name : str
            Parameter name.
        value : Any
            Value to set.
        unit : Enum or str, optional
            Unit to set.
        """
        self.log.debug(f'set {type(self).__name__} parameter: {name}')
        from uam4x.units import UNIT_TYPE_TO_UNIT_CLASS
        if name in self._parameters:
            param = self._parameters[name]
            param.value = value

            unit_type = param.unit_type
            if unit_type and unit_type in UNIT_TYPE_TO_UNIT_CLASS:
                unit_class = UNIT_TYPE_TO_UNIT_CLASS[unit_type]
                # Accept both class attribute and value
                if unit not in unit_class.__dict__.values():
                    self.log.error(f"Unit '{unit}' does not match unit type '{unit_type}' for parameter '{name}'.")
                    raise ValueError(f"Unit '{unit}' does not match unit type '{unit_type}' for parameter '{name}'.")
            param.unit = unit
        else:
            self.log.error(f"Parameter '{name}' not defined for {self.__class__.__name__}.")
            raise AttributeError(f"Parameter '{name}' not defined for {self.__class__.__name__}.")

    def add_mode(self, mode_name: str):
        """
        Add a mode to the component.

        Parameters
        ----------
        mode_name : str
            Name of the mode.

        Returns
        -------
        Mode
        """
        self.log.debug(f"add mode: {mode_name}")
        # Only accept a string (mode name), auto-create a Mode
        if not isinstance(mode_name, str):
            self.log.error("add_mode only accepts a string as the mode name.")
            raise TypeError("add_mode only accepts a string as the mode name.")
        if '[' in mode_name or  ']' in  mode_name:
            self.log.error(f'modes cannot have the character "[" or "]" in them')
            raise ValueError(f'modes cannot have the character "[" or "]" in them')
        mode = _Mode(mode_name, self)
        self._modes[mode.name] = mode
        setattr(self, mode.name, mode)
        return mode

    def get_parameter(self, param_name: str):
        """
        Get the value of a parameter by name.

        Parameters
        ----------
        param_name : str
            Name of the parameter.

        Returns
        -------
        Any or None
            The value of the parameter, or None if not found.
        """
        self.log.debug(f'get parameter: {param_name}')
        if param_name not in self._parameters:
            self.log.error(f"Parameter '{param_name}' not defined for {self.__class__.__name__}.")
            raise AttributeError(f"Parameter '{param_name}' not defined for {self.__class__.__name__}.")
        return self._parameters[param_name].value

    def get_parameter_type(self, param_name: str):
        """
        Get the type of a parameter by name.

        Parameters
        ----------
        param_name : str
            Name of the parameter.

        Returns
        -------
        Any or None
            The type of the parameter, or None if not found.
        """
        self.log.debug(f'get parameter type: {param_name}')
        if param_name not in self._parameters:
            self.log.error(f"Parameter '{param_name}' not defined for {self.__class__.__name__}.")
            raise AttributeError(f"Parameter '{param_name}' not defined for {self.__class__.__name__}.")
        return self._parameters[param_name].dtype

    def check_all_parameter_types(self):
        """
        Check types of all parameters and mode overrides.

        Returns
        -------
        dict
            {'main': ..., mode_name: ...} for all parameters and all modes.
        """
        self.log.debug('check all parameter types')
        #result = {'main': {name: param.check_type() for name, param in self._parameters.items()}}
        result = {}
        for mode_name, mode in self._modes.items():
            mode_result = {}
            for pname, override_value in mode._overrides.items():
                if pname not in self._parameters:
                    self.log.error(f"Parameter '{pname}' does not exist in subsystem '{self.name}' for mode '{mode_name}'")
                    raise AttributeError(f"Parameter '{pname}' does not exist in subsystem '{self.name}' for mode '{mode_name}'")
                dtype = self._parameters[pname].dtype
                mode_result[pname] = isinstance(override_value, dtype)
            result[mode_name] = mode_result
        return result

    def __str__(self):
        """
        Returns a string representation of the component and its parameters.

        Returns
        -------
        str
        """
        self.log.debug('returns a string representation of the component and its parameters.')
        param_strs = [str(param) for param in self._parameters.values()]
        params = "\n    ".join(param_strs)
        return f"{self.__class__.__name__}('{self.name}')\n    {params}"

    def _to_dict(self) -> dict:
        """
        Serialize the component to a JSON-serializable dictionary.

        Returns
        -------
        dict
        """
        self.log.debug('write to dict')
        return {
            "name": self.name,
            "subsystem_type": type(self).__name__,
            "parameters": {k: v._to_dict() for k, v in self._parameters.items()},
            "modes": {k: {"overrides": dict(mode._overrides)} for k, mode in self._modes.items()}
        }

    @classmethod
    def _from_dict(cls, data: dict) -> 'Subsystem':
        """
        Deserialize a component from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary representation of the component.

        Returns
        -------
        Subsystem
        """
        cls.log.debug('Deserialize a component from a dictionary')
        # Parameters
        subsystem_type = data["subsystem_type"]
        subsystem_class = get_subsystem_class(subsystem_type)
        comp = subsystem_class(data["name"])
        for k, v in data["parameters"].items():
            comp._parameters[k] = Parameter._from_dict(v)

        # Modes
        for mode_name, mode_data in data["modes"].items():
            mode = comp.add_mode(mode_name)
            for pname, value in mode_data["overrides"].items():
                mode._overrides[pname] = value
        return comp

    def get_param_dict(self) -> dict:
        """
        Get a dictionary of all parameters and their values.

        Returns
        -------
        dict
            Mapping of parameter names to their values.
        """
        self.log.debug('get dictionary of parameters and their values')
        return {name: param.value for name, param in self._parameters.items()}

def register_subsystem(cls):
    subsystem = cls.__name__
    log.debug(f'register subsystem: {subsystem}')
    # Only register in the registry, don't modify conf.json for built-in subsystems
    # when running components.py directly
    if cls.__module__ == "__main__":
        _subsystem_registry[subsystem] = cls
        _update_consts(subsystem , cls)
        return cls

    # For external subsystems, update conf.json
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'conf.json')
    _config = check_and_get_config(config_path)


    if subsystem  in _config['SUBSYSTEMS'] and _config['SUBSYSTEMS'][subsystem] != importlib.util.find_spec(cls.__module__).origin:
        log.warning(f"WARNING: Subsystem '{subsystem }' already registered\nOverwritting existing data")
    if not "uam4x.subsystems" in cls.__module__.replace('.', '/'):
        _config['SUBSYSTEMS'][subsystem] = importlib.util.find_spec(cls.__module__).origin
        if is_main_process() and registry_writes_enabled():
            with open(config_path, 'w') as configfile:
                json.dump(_config, configfile, indent=4)

    _subsystem_registry[subsystem] = cls
    _update_consts(subsystem, cls)
    return cls

def _update_consts(subsystem , subsystem_class):
    if not subsystem  in _subsystem_names:
        _subsystem_names.append(subsystem)
    dummy = subsystem_class('dummy')
    params = list(dummy.get_param_dict().keys())
    _subsystem_to_param_name_type_list[subsystem] = []
    for param  in params:
        _subsystem_to_param_name_type_list[subsystem ].append((param, dummy._parameters[param].dtype.__name__))

def get_subsystem_class(subsystem_class_name):
    if not subsystem_class_name in _subsystem_registry:
        msg = f"Subsystem '{subsystem_class_name}' was not registered. Please check the following:\n"
        msg += f"\t1. Did you register the subsystem using the @register_subsystem?\n"
        msg += f"\t2. Were there Plugin Load Errors when you imported uam4x?\n"
        log.error(msg)
        raise ImportError(msg)
    return _subsystem_registry[subsystem_class_name]

def get_subsystem_names():
    return  _subsystem_names

def get_param_name_and_type_list(subsystem):
    return _subsystem_to_param_name_type_list[subsystem]

# Wing subsystem with only span, chord, and sweep
@register_subsystem
class Wing(Subsystem):
    def setup(self):
        self.add_parameter("area", 0., float, unit_type=UnitType.AREA, description="Wing area" )
        self.add_parameter("span", 0., float, unit_type=UnitType.LENGTH, description="Wing span")
        self.add_parameter("chord", 0., float, unit_type=UnitType.LENGTH, description="Wing chord")
        self.add_parameter("sweep", 0., float, unit_type=UnitType.ANGLE, description="Wing sweep angle")
        self.add_parameter("aspect_ratio", 1., float, unit_type=UnitType.ANGLE, description="Wing sweep angle")

@register_subsystem
class Rotor(Subsystem):
    def setup(self):
        self.add_parameter("vtip", 550., float, unit_type=UnitType.SPEED, description="Rotor tip speed")
        self.add_parameter("tilt", 1.0, float, unit_type=UnitType.LENGTH, description="Rotor tilt")
        self.add_parameter("radius", 1.0, float, unit_type=UnitType.LENGTH, description="Rotor radius")
        self.add_parameter("nblade", 2, int, description="Number of blades")
        self.add_parameter("rpm", 1000.0, float, unit_type=UnitType.FREQUENCY if hasattr(UnitType, 'FREQUENCY') else None, description="Rotor RPM")
        self.add_parameter("blade_chord", 0.1, float, unit_type=UnitType.LENGTH, description="Blade chord length")
        self.add_parameter("collective_pitch", 0.0, float, unit_type=UnitType.ANGLE, description="Collective pitch angle (deg)")

@register_subsystem
class RotorBladeRegion(Subsystem):
    def setup(self):
        self.add_parameter("spanwise_start", 0.0, float, unit_type=UnitType.LENGTH, description="start of section as a fraction of the total span")
        self.add_parameter("spanwise_stop", 0.0, float, unit_type=UnitType.LENGTH, description="end of section as a fration of the total span")
        self.add_parameter("chord_start", 0.0, float, unit_type=UnitType.LENGTH, description="chord at the start of the section")
        self.add_parameter("chord_stop", 0.0, float, unit_type=UnitType.LENGTH, description="chord at the end of the section")
        self.add_parameter("sweep", 0.0, float, unit_type=UnitType.ANGLE, description="sweep angle")
        self.add_parameter("dihedral", 0.0, float, unit_type=UnitType.ANGLE, description="dihedral angle")
        self.add_parameter("twist_start", 0.0, float, unit_type=UnitType.ANGLE, description="twist angle at the start of the section")
        self.add_parameter("twist_stop", 0.0, float, unit_type=UnitType.ANGLE, description="twist angle at the end of the section")
        self.add_parameter("airfoil_start", "", str, unit_type=None, description="airfoil at the begining of the section")
        self.add_parameter("airfoil_stop", "", str, unit_type=None, description="airfoil at the end of the section")
        self.add_parameter("horizontal_shift_start", 0.0, float, unit_type=UnitType.LENGTH, description="")
        self.add_parameter("horizontal_shift_stop", 0.0, float, unit_type=UnitType.LENGTH, description="")
        self.add_parameter("vertical_shift_start", 0.0, float, unit_type=UnitType.LENGTH, description="")
        self.add_parameter("vertical_shift_stop", 0.0, float, unit_type=UnitType.LENGTH, description="")
        self.add_parameter("mainsparweb_thickness", 0.0, float, unit_type=UnitType.LENGTH, description="")
        self.add_parameter("mainsparweb_angle", 0.0, float, unit_type=UnitType.ANGLE, description="")

@register_subsystem
class Aircraft(Subsystem):
    def setup(self):
        self.add_parameter("npass", 4, int, description="Number of passengers")
        self.add_parameter("wpass", 200.0, float, unit_type=UnitType.MASS, description="Mass per passenger")

# --- Test script ---
if __name__ == "__main__":
    # Example: create, write, read, and print a Wing
    wing = Wing("wing1")
    wing.span = 10.0
    print(wing.baseline.span)  # Should print 10.0
    wing.set_parameter("chord", 2.0)
    wing.sweep = 25.0

    wing.set_parameter("span", 10.0, unit=LengthUnit.METER)

    # Add a mode where sweep is set to 0 (e.g., "unswept")
    wing.add_mode("unswept")
    wing.unswept.sweep = 0.

    print(f"Wing parameters (default):")
    for pname, param in wing._parameters.items():
        ut = param.unit_type
        u = param.unit
        print(param)

    # Show all effective parameters for the unswept mode (overrides + inherited)
    print(f"\nWing parameters (unswept mode, all effective):")
    for pname, param in wing.unswept._parameters.items():
        ut = param.unit_type
        u = param.unit
        print(param)

    import json
    with open('temp.json', 'w') as f:
        json.dump(wing._to_dict(), f, indent=4)
