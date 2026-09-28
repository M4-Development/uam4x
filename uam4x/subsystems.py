import os
from uam4x.log import get_log
from uam4x.utils import log, canonicalize_path

from uam4x.units import (
    UnitType,
    LengthUnit,
    AngleUnit,
    MassUnit,
    TimeUnit,
    ForceUnit,
    TemperatureUnit,
    PressureUnit,
    EnergyUnit,
    PowerUnit,
    VolumeUnit,
    AreaUnit,
    SpeedUnit,
    AccelerationUnit,
)
from uam4x.utils import (
    check_and_get_config,
    type_string_to_type,
    is_main_process,
    registry_writes_enabled,
)

from enum import Enum

import json
import importlib.util
import numpy as np

_subsystem_registry = {}
_subsystem_names = []
_subsystem_to_param_name_type_list = {}


class Parameter:
    """Represent a typed subsystem parameter.

    A ``Parameter`` stores a name, value, declared Python type, optional unit
    metadata, and a free-text description. Parameter values are type-checked at
    construction and when reassigned.

    Parameters
    ----------
    name : str
        Parameter name.
    default_value : object
        Initial parameter value. The value must be an instance of ``dtype``.
    dtype : type
        Declared Python type for the parameter value.
    unit_type : UnitType, optional
        Unit category associated with the parameter. If omitted, the parameter
        uses :attr:`uam4x.units.UnitType.NONE`.
    unit : Enum or str, optional
        Unit label associated with the parameter value.
    description : str, optional
        Human-readable parameter description.

    """

    log = get_log(log, object_name=f"parameter")

    def __init__(
        self,
        name: str,
        default_value,
        dtype,
        unit_type: UnitType = None,
        unit: Enum = None,
        description: str = "",
        comment: str = "",
    ):
        """Initialize a parameter.

        Parameters
        ----------
        name : str
            Parameter name.
        default_value : object
            Initial parameter value. The value must be an instance of ``dtype``.
        dtype : type
            Declared Python type for the parameter value.
        unit_type : UnitType, optional
            Unit category associated with the parameter.
        unit : Enum or str, optional
            Unit label associated with the parameter value.
        description : str, optional
            Human-readable parameter description.
        comment : str, optional
            Source, reference, or rationale associated with the current value.

        Raises
        ------
        TypeError
            If ``default_value`` is not an instance of ``dtype``.

        """
        from uam4x.units import UnitType

        self.name = name
        self.dtype = dtype
        self.log = Parameter.log
        if not isinstance(default_value, dtype):
            self.log.error(
                f"Parameter '{name}' must be of type {dtype.__name__}, got {type(default_value).__name__}"
            )
            raise TypeError(
                f"Parameter '{name}' must be of type {dtype.__name__}, got {type(default_value).__name__}"
            )
        self._value = default_value

        self.unit_type = unit_type if unit_type is not None else UnitType.NONE
        self.unit = unit if unit is not None else ""
        self.description = description
        self.comment = comment

    def __str__(self):
        """Return a human-readable string representation of the parameter.

        Returns
        -------
        str
            String representation including the parameter name, value, unit
            metadata, and description.

        """
        unit_display = self.unit if self.unit else "—"
        self.log.debug(f"return string representation")
        return f"{self.name}: {self.value} [{self.unit_type}, {unit_display}] ({self.description})"

    def __repr__(self):
        """Return a developer-oriented string representation of the parameter.

        Returns
        -------
        str
            Representation of the parameter. This implementation matches
            :meth:`__str__`.

        """
        self.log.debug(f"return string representation")
        return self.__str__()

    @property
    def value(self):
        """Parameter value.

        Returns
        -------
        object
            Current parameter value.

        """
        return self._value

    @value.setter
    def value(self, v):
        """Set the parameter value.

        Parameters
        ----------
        v : object
            New parameter value. The value must be an instance of the declared
            parameter type.

        Raises
        ------
        TypeError
            If ``v`` is not an instance of :attr:`dtype`.

        """

        if not isinstance(v, self.dtype):
            self.log.error(
                f"Value for parameter '{self.name}' must be of type {self.dtype.__name__}, got {type(v).__name__}"
            )
            raise TypeError(
                f"Value for parameter '{self.name}' must be of type {self.dtype.__name__}, got {type(v).__name__}"
            )
        self._value = v

    def _to_dict(self) -> dict:
        """Serialize the parameter to a dictionary.

        Returns
        -------
        dict
            JSON-compatible dictionary containing the parameter name, value,
            type, unit metadata, and description.

        """
        # Only include JSON-serializable fields, never include .dtype or any type/class
        # Ensure unit_type and unit are always serializable (string or None)
        self.log.debug(f"write to dict")
        unit_type = self.unit_type
        if hasattr(unit_type, "__name__"):
            unit_type = unit_type.__name__

        unit = self.unit

        return {
            "name": self.name,
            "value": self.value,
            "dtype": self.dtype.__name__,
            "unit_type": unit_type,
            "unit": unit,
            "description": self.description,
            "comment": self.comment,
        }

    @classmethod
    def _from_dict(cls, data: dict) -> "Parameter":
        """Reconstruct a parameter from a dictionary.

        Parameters
        ----------
        data : dict
            Serialized parameter dictionary, typically produced by
            :meth:`_to_dict`.

        Returns
        -------
        Parameter
            Reconstructed parameter.

        """
        cls.log.debug("Deserialize a parameter from a dictionary")
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
            comment=data.get("comment", ""),
        )


class _Mode:
    """Represent a subsystem operating mode with parameter overrides.

    A ``_Mode`` stores parameter values that override the baseline parameters
    of a parent :class:`Subsystem`. Parameters not explicitly overridden are
    read from the parent subsystem.

    Parameters
    ----------
    name : str
        Mode name.
    component : Subsystem
        Parent subsystem whose baseline parameters are inherited.

    """

    log = get_log(log, object_name=f"mode")

    def __init__(self, name: str, component: "Subsystem"):
        """Initialize a mode.

        Parameters
        ----------
        name : str
            Mode name.
        component : 'Subsystem'
            Parent subsystem whose baseline parameters are inherited.

        """
        self.name = name
        self._overrides = {}
        self._override_comments = {}
        self.component = component  # Reference to parent component for inheritance
        self.log = _Mode.log

    def set_parameter(self, name: str, value, comment: str = ""):
        """Set a parameter override for this mode.

        Parameters
        ----------
        name : str
            Parameter name.
        value : object
            Override value to assign.
        comment : str, optional
            Source, reference, or rationale associated with the override value.

        Returns
        -------
        None

        Raises
        ------
        AttributeError
            If ``name`` is not a parameter on the parent subsystem.
        TypeError
            If ``value`` is not an instance of the parent parameter's declared
            type.

        """
        self.log.debug(f"set parameter")
        if name not in self.component._parameters:
            self.log.error(
                f"Parameter '{name}' does not exist in component '{self.component.name}' for mode '{self.name}'"
            )
            raise AttributeError(
                f"Parameter '{name}' does not exist in component '{self.component.name}' for mode '{self.name}'"
            )
        dtype = self.component._parameters[name].dtype
        if not isinstance(value, dtype):
            self.log.error(
                f"Override for parameter '{name}' in mode '{self.name}' must be of type {dtype.__name__}, got {type(value).__name__}"
            )
            raise TypeError(
                f"Override for parameter '{name}' in mode '{self.name}' must be of type {dtype.__name__}, got {type(value).__name__}"
            )
        self._overrides[name] = value
        self._override_comments[name] = comment

    def get_parameter(self, name: str):
        """Return a mode override or inherited subsystem parameter value.

        Parameters
        ----------
        name : str
            Parameter name.

        Returns
        -------
        object
            Override value if present, otherwise the parent subsystem parameter
            value.

        Raises
        ------
        AttributeError
            If ``name`` is not a parameter on the parent subsystem.

        """
        self.log.debug(f"get parameter")
        if name in self._overrides:
            return self._overrides[name]
        if name in self.component._parameters:
            return self.component._parameters[name].value
        self.log.error(
            f"Parameter '{name}' does not exist in component '{self.component.name}' for mode '{self.name}'"
        )
        raise AttributeError(
            f"Parameter '{name}' does not exist in component '{self.component.name}' for mode '{self.name}'"
        )

    def get_parameter_comment(self, name: str):
        """Return the comment for a mode override or inherited parameter.

        Parameters
        ----------
        name : str
            Parameter name.

        Returns
        -------
        str
            Override comment if present, otherwise the baseline comment.

        Raises
        ------
        AttributeError
            If the parent subsystem has no parameter named name.

        """
        self.log.debug("get parameter comment")
        if name in self._overrides:
            return self._override_comments.get(name, "")
        if name in self.component._parameters:
            return self.component.get_parameter_comment(name)
        message = (
            f"Cannot get comment for parameter '{name}': parameter does not "
            f"exist in component '{self.component.name}' for mode '{self.name}'."
        )
        self.log.error(message)
        raise AttributeError(message)

    def __getattr__(self, item):
        """Return an override, inherited parameter, or parent attribute.

        Parameters
        ----------
        item : str
            Attribute or parameter name.

        Returns
        -------
        object
            Override value, inherited parameter value, or parent subsystem
            attribute.

        Raises
        ------
        AttributeError
            If ``item`` cannot be resolved on the mode or parent subsystem.

        """
        # Only called if attribute not found in normal places
        if item in self._overrides:
            return self._overrides[item]
        if (
            self.component
            and hasattr(self.component, item)
            and not isinstance(getattr(self.component, item), _Mode)
            and not callable(getattr(self.component, item))
        ):
            return getattr(self.component, item)
        self.log.error(f"'{self.__class__.__name__}' object has no attribute '{item}'")
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{item}'"
        )

    def __setattr__(self, key, value):
        """Assign a mode override or regular attribute.

        Parameters
        ----------
        key : str
            Attribute or parameter name to assign.
        value : object
            Value to assign.

        Returns
        -------
        None

        Notes
        -----
        If ``key`` matches a parent subsystem parameter name, the assignment is
        stored as a mode override. Otherwise, special attributes are assigned
        directly and unknown names raise :class:`AttributeError`.

        """
        if key in ("name", "_overrides", "_override_comments", "component"):
            object.__setattr__(self, key, value)
        elif key in self.component._parameters:
            self.set_parameter(key, value)
        elif key == "log":
            pass
        else:
            self.log.error(
                f"'{self.__class__.__name__}' object has no parameter '{key}' to override"
            )
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no parameter '{key}' to override"
            )

    @property
    def parameters(self):
        """Return effective parameter values for the mode.

        Returns
        -------
        dict
            Mapping from parameter names to :class:`Parameter` objects,
            including inherited subsystem defaults and mode overrides.

        """
        # Return a dict of all effective Parameter objects: overridden in mode, else from component
        self.log.debug(f"Get parameters for mode {self.name}")
        params = {}
        for pname, param in self.component._parameters.items():
            if pname in self._overrides:
                override_value = self._overrides[pname]
                dtype = param.dtype
                override_comment = self._override_comments.get(pname, "")
                if dtype is not None and not isinstance(override_value, dtype):
                    self.log.error(
                        f"Override for parameter '{pname}' in mode '{self.name}' must be of type {dtype.__name__}, got {type(override_value).__name__}"
                    )
                    raise TypeError(
                        f"Override for parameter '{pname}' in mode '{self.name}' must be of type {dtype.__name__}, got {type(override_value).__name__}"
                    )
                overridden_param = Parameter(
                    param.name,
                    override_value,
                    dtype,
                    unit_type=param.unit_type,
                    unit=param.unit,
                    description=param.description,
                    comment=override_comment,
                )
                params[pname] = overridden_param
            else:
                params[pname] = param
        # Add any extra overrides not in component
        for pname, value in self._overrides.items():
            if pname not in params:
                self.log.error(
                    f"Parameter '{pname}' does not exist in component '{self.component.name}' for mode '{self.name}'"
                )
                raise AttributeError(
                    f"Parameter '{pname}' does not exist in component '{self.component.name}' for mode '{self.name}'"
                )
        return params

    def __str__(self):
        """Return a human-readable string representation of the mode.

        Returns
        -------
        str
            String representation including the mode name and active
            overrides.

        """
        self.log.debug(f"return a string representation")
        if not self._overrides:
            return f"Mode('{self.name}') (no parameter overrides)"
        param_strs = [f"{k}: {v}" for k, v in self._overrides.items()]
        params = ", ".join(param_strs)
        return f"Mode('{self.name}') overrides: {params}"


class Subsystem:
    """Represent a physical vehicle component with parameters and modes.

    ``Subsystem`` is the base class for concrete physical component models such
    as wings, rotors, and aircraft. A subsystem manages baseline parameters,
    optional mode-specific overrides, parameter lookup and validation, and
    serialization.

    Parameters
    ----------
    name : str
        Component name.

    Notes
    -----
    Concrete subclasses typically implement :meth:`setup` to register their
    component parameters.

    """

    log = get_log(log, object_name=f"component")

    def __init__(self, name: str):
        """Initialize a physical component instance.

        Parameters
        ----------
        name : str
            Component name.

        Raises
        ------
        ValueError
            If ``name`` contains ``"["``.

        Notes
        -----
        Construction initializes parameter and mode storage, calls
        :meth:`setup` to register baseline parameters, and creates a
        ``baseline`` alias to the subsystem itself.

        """
        self.name = name
        if "[" in name:
            self.log.error(f'components cannot have the character "[" in them')
            raise ValueError(f'components cannot have the character "[" in them')
        self._parameters = {}
        self._modes = {}
        self._is_setup = True
        self.log = Subsystem.log
        self.setup()
        self._is_setup = False
        setattr(self, "baseline", self)
        self._parent = None

    def setup(self):
        """Register baseline parameters for the component.

        Returns
        -------
        None

        Notes
        -----
        Subclasses override this method and call :meth:`add_parameter` to
        declare their baseline parameters during construction.

        """
        self.log.debug(
            f"Override in subclasses to add parameters using add_parameter()"
        )

    def add_parameter(
        self, name: str, default_value, dtype, unit_type=None, unit=None, description=""
    ):
        """Add a baseline parameter to the component.

        Parameters
        ----------
        name : str
            Parameter name.
        default_value : object
            Initial parameter value. The value must be an instance of ``dtype``.
        dtype : type
            Declared Python type for the parameter value.
        unit_type : UnitType, optional
            Unit category associated with the parameter.
        unit : Enum or str, optional
            Unit label associated with the parameter value.
        description : str, optional
            Human-readable parameter description.
        comment : str, optional
            Source, reference, or rationale associated with the current value.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If called outside :meth:`setup`.
        ValueError
            If ``name`` contains ``"["`` or ``"]"``.
        TypeError
            If ``default_value`` is not an instance of ``dtype``.

        Notes
        -----
        This method creates a :class:`Parameter` object and dynamically
        installs a property on the subsystem class for attribute-style access
        to the parameter.

        """

        self.log.debug(f"add parameter {name} to {type(self).__name__}")
        if not self._is_setup:
            self.log.error(
                "Cannot add parameters outside of setup().\nAdd them in the setup() method or use SetParameter() after setup."
            )
            raise RuntimeError(
                "Cannot add parameters outside of setup().\nAdd them in the setup() method or use SetParameter() after setup."
            )
        if "[" in name or "]" in name:
            self.log.error(f'parameters cannot have the character "[" or "]" in them')
            raise ValueError(f'parameters cannot have the character "[" or "]" in them')
        self._parameters[name] = Parameter(
            name,
            default_value,
            dtype,
            unit_type=unit_type,
            unit=unit,
            description=description,
        )

        # Dynamically add property to the instance for this parameter
        def getter(self):
            """Return the current value of this dynamically declared parameter.

            Returns
            -------
            object
                Current parameter value.

            """
            return self._parameters[name].value

        def setter(self, v):
            """Set the value of this dynamically declared parameter.

            Parameters
            ----------
            v : object
                Value to assign to the parameter.

            """
            self.set_parameter(name, v)

        prop = property(getter, setter)
        setattr(self.__class__, name, prop)

    def set_parameter(self, name: str, value, unit=None, comment: str = ""):
        """Set the value of a baseline parameter.

        Parameters
        ----------
        name : str
            Parameter name.
        value : object
            New parameter value.
        unit : Enum or str, optional
            Unit label to assign to the parameter.
        comment : str, optional
            Source, reference, or rationale associated with the new value.

        Returns
        -------
        None

        Raises
        ------
        AttributeError
            If ``name`` is not a defined parameter on the component.
        TypeError
            If ``value`` is not an instance of the parameter's declared type.
        ValueError
            If ``unit`` is incompatible with the parameter's declared unit
            type.

        """
        self.log.debug(f"set {type(self).__name__} parameter: {name}")
        from uam4x.units import UNIT_TYPE_TO_UNIT_CLASS

        if name in self._parameters:
            param = self._parameters[name]
            param.value = value
            param.comment = comment
            unit_type = param.unit_type
            if unit is not None and unit_type in UNIT_TYPE_TO_UNIT_CLASS:
                unit_class = UNIT_TYPE_TO_UNIT_CLASS[unit_type]
                # Accept both class attribute and value
                if unit not in unit_class.__dict__.values():
                    self.log.error(
                        f"Unit '{unit}' does not match unit type '{unit_type}' for parameter '{name}'."
                    )
                    raise ValueError(
                        f"Unit '{unit}' does not match unit type '{unit_type}' for parameter '{name}'."
                    )
                else:
                    param.unit = unit if unit is not None else ""
        else:
            self.log.error(
                f"Parameter '{name}' not defined for {self.__class__.__name__}."
            )
            raise AttributeError(
                f"Parameter '{name}' not defined for {self.__class__.__name__}."
            )

    def add_mode(self, mode_name: str):
        """Add an operating mode to the component.

        Parameters
        ----------
        mode_name : str
            Mode name.

        Returns
        -------
        _Mode
            Newly created mode object.

        Raises
        ------
        TypeError
            If ``mode_name`` is not a string.
        ValueError
            If ``mode_name`` contains ``"["`` or ``"]"``.

        Notes
        -----
        The created mode is attached as an attribute on the subsystem using
        ``mode_name``.

        """
        self.log.debug(f"add mode: {mode_name}")
        # Only accept a string (mode name), auto-create a Mode
        if not isinstance(mode_name, str):
            self.log.error("add_mode only accepts a string as the mode name.")
            raise TypeError("add_mode only accepts a string as the mode name.")
        if "[" in mode_name or "]" in mode_name:
            self.log.error(f'modes cannot have the character "[" or "]" in them')
            raise ValueError(f'modes cannot have the character "[" or "]" in them')
        mode = _Mode(mode_name, self)
        self._modes[mode.name] = mode
        setattr(self, mode.name, mode)
        return mode

    def get_parameter(self, param_name: str):
        """Return the value of a baseline parameter.

        Parameters
        ----------
        param_name : str
            Parameter name.

        Returns
        -------
        object
            Current parameter value.

        Raises
        ------
        AttributeError
            If ``param_name`` is not a defined parameter on the component.

        """
        self.log.debug(f"get parameter: {param_name}")
        if param_name not in self._parameters:
            self.log.error(
                f"Parameter '{param_name}' not defined for {self.__class__.__name__}."
            )
            raise AttributeError(
                f"Parameter '{param_name}' not defined for {self.__class__.__name__}."
            )
        return self._parameters[param_name].value

    def get_parameter_comment(self, param_name: str):
        """Return the comment associated with a baseline parameter value.

        Parameters
        ----------
        param_name : str
            Parameter name.

        Returns
        -------
        str
            Source, reference, or rationale for the current value.

        Raises
        ------
        AttributeError
            If param_name is not a defined parameter on the component.

        """
        self.log.debug(f"get parameter comment: {param_name}")
        if param_name not in self._parameters:
            message = (
                f"Cannot get comment for parameter '{param_name}': parameter "
                f"is not defined for {self.__class__.__name__}."
            )
            self.log.error(message)
            raise AttributeError(message)
        return self._parameters[param_name].comment

    def get_parameter_type(self, param_name: str):
        """Return the declared type of a baseline parameter.

        Parameters
        ----------
        param_name : str
            Parameter name.

        Returns
        -------
        type
            Declared Python type of the parameter.

        Raises
        ------
        AttributeError
            If ``param_name`` is not a defined parameter on the component.

        """
        self.log.debug(f"get parameter type: {param_name}")
        if param_name not in self._parameters:
            self.log.error(
                f"Parameter '{param_name}' not defined for {self.__class__.__name__}."
            )
            raise AttributeError(
                f"Parameter '{param_name}' not defined for {self.__class__.__name__}."
            )
        return self._parameters[param_name].dtype

    def check_all_parameter_types(self):
        """Validate parameter types for all mode overrides.

        Returns
        -------
        dict
            Mapping from mode names to dictionaries of parameter-name
            validation results.

        Raises
        ------
        AttributeError
            If a mode override references a parameter that does not exist on
            the baseline component.

        Notes
        -----
        Baseline parameters are validated when they are created or reassigned.
        This method checks only mode override values.

        """
        self.log.debug("check all parameter types")
        result = {}
        for mode_name, mode in self._modes.items():
            mode_result = {}
            for pname, override_value in mode._overrides.items():
                if pname not in self._parameters:
                    self.log.error(
                        f"Parameter '{pname}' does not exist in subsystem '{self.name}' for mode '{mode_name}'"
                    )
                    raise AttributeError(
                        f"Parameter '{pname}' does not exist in subsystem '{self.name}' for mode '{mode_name}'"
                    )
                dtype = self._parameters[pname].dtype
                mode_result[pname] = isinstance(override_value, dtype)
            result[mode_name] = mode_result
        return result

    def __str__(self):
        """Return a human-readable string representation of the component.

        Returns
        -------
        str
            String representation including the component type, name, and
            baseline parameters.

        """
        self.log.debug(
            "returns a string representation of the component and its parameters."
        )
        param_strs = [str(param) for param in self._parameters.values()]
        params = "\n    ".join(param_strs)
        return f"{self.__class__.__name__}('{self.name}')\n    {params}"

    def _to_dict(self) -> dict:
        """Serialize the component to a dictionary.

        Returns
        -------
        dict
            JSON-compatible dictionary containing the component name, concrete
            subsystem type, serialized parameters, and serialized mode
            overrides.

        """
        self.log.debug("write to dict")
        return {
            "name": self.name,
            "subsystem_type": type(self).__name__,
            "parameters": {k: v._to_dict() for k, v in self._parameters.items()},
            "modes": {
                k: {
                    "overrides": dict(mode._overrides),
                    "comments": dict(mode._override_comments),
                }
                for k, mode in self._modes.items()
            },
        }

    @classmethod
    def _from_dict(cls, data: dict) -> "Subsystem":
        """Reconstruct a component from a dictionary.

        Parameters
        ----------
        data : dict
            Serialized component dictionary, typically produced by
            :meth:`_to_dict`.

        Returns
        -------
        Subsystem
            Reconstructed concrete subsystem with restored parameters and
            modes.

        """
        cls.log.debug("Deserialize a component from a dictionary")
        # Parameters
        subsystem_type = data["subsystem_type"]
        subsystem_class = get_subsystem_class(subsystem_type)
        comp = subsystem_class(data["name"])
        for k, v in data["parameters"].items():
            comp._parameters[k] = Parameter._from_dict(v)

        # Modes
        for mode_name, mode_data in data["modes"].items():
            mode = comp.add_mode(mode_name)
            comments = mode_data.get("comments", {})
            for pname, value in mode_data["overrides"].items():
                mode._overrides[pname] = value
                mode._override_comments[pname] = comments.get(pname, "")
        return comp

    def get_param_dict(self) -> dict:
        """Return a dictionary of parameter values.

        Returns
        -------
        dict
            Mapping from parameter names to their current baseline values.

        """
        self.log.debug("get dictionary of parameters and their values")
        return {name: param.value for name, param in self._parameters.items()}


def register_subsystem(cls):
    """Register a subsystem class for lookup and deserialization.

    Parameters
    ----------
    cls : type
        Subsystem class to register.

    Returns
    -------
    type
        Registered subsystem class. This allows the function to be used as a
        class decorator.

    Notes
    -----
    Built-in subsystem classes are added to the in-memory registry only.
    External subsystem classes may also update ``conf.json`` so they can be
    discovered in future sessions.

    """
    subsystem = cls.__name__
    log.debug(f"register subsystem: {subsystem}")
    # Only register in the registry, don't modify conf.json for built-in subsystems
    # when running subsystems.py directly
    if cls.__module__ == "__main__":
        _subsystem_registry[subsystem] = cls
        _update_consts(subsystem, cls)
        return cls

    # For external subsystems, update conf.json
    config_path = canonicalize_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf.json"))
    _config = check_and_get_config(config_path)
    import inspect
    if (
        subsystem in _config["SUBSYSTEMS"]
        and canonicalize_path(_config["SUBSYSTEMS"][subsystem])
        != canonicalize_path(os.path.abspath(inspect.getfile(cls)))
    ):
        log.warning(
            f"WARNING: Subsystem '{subsystem }' already registered\nOverwritting existing data"
        )
    if not "uam4x.subsystems" in cls.__module__.replace(".", "/"):
        _config["SUBSYSTEMS"][subsystem] = canonicalize_path(os.path.abspath(inspect.getfile(cls)))
        if is_main_process() and registry_writes_enabled():
            with open(config_path, "w") as configfile:
                json.dump(_config, configfile, indent=4)

    _subsystem_registry[subsystem] = cls
    _update_consts(subsystem, cls)
    return cls


def _update_consts(subsystem, subsystem_class):
    """Update cached subsystem metadata tables.

    Parameters
    ----------
    subsystem : str
        Registered subsystem class name.
    subsystem_class : type
        Subsystem class associated with ``subsystem``.

    Returns
    -------
    None

    Notes
    -----
    This helper instantiates a dummy subsystem to collect parameter names and
    declared types for metadata queries.

    """
    if not subsystem in _subsystem_names:
        _subsystem_names.append(subsystem)
    dummy = subsystem_class("dummy")
    params = list(dummy.get_param_dict().keys())
    _subsystem_to_param_name_type_list[subsystem] = []
    for param in params:
        _subsystem_to_param_name_type_list[subsystem].append(
            (param, dummy._parameters[param].dtype.__name__)
        )


def get_subsystem_class(subsystem_class_name):
    """Return a registered subsystem class by name.

    Parameters
    ----------
    subsystem_class_name : str
        Registered subsystem class name.

    Returns
    -------
    type
        Registered subsystem class.

    Raises
    ------
    ImportError
        If ``subsystem_class_name`` is not registered.

    """
    if not subsystem_class_name in _subsystem_registry:
        msg = f"Subsystem '{subsystem_class_name}' was not registered. Please check the following:\n"
        msg += f"\t1. Did you register the subsystem using the @register_subsystem?\n"
        msg += f"\t2. Were there Plugin Load Errors when you imported uam4x?\n"
        log.error(msg)
        raise ImportError(msg)
    return _subsystem_registry[subsystem_class_name]


def get_subsystem_names():
    """Return registered subsystem names.

    Returns
    -------
    list of str
        Registered subsystem class names.

    """
    return _subsystem_names


def get_param_name_and_type_list(subsystem):
    """Return parameter name/type metadata for a subsystem class.

    Parameters
    ----------
    subsystem : str
        Registered subsystem class name.

    Returns
    -------
    list of tuple[str, str]
        Sequence of ``(parameter_name, dtype_name)`` pairs for the subsystem.

    """
    return _subsystem_to_param_name_type_list[subsystem]

def get_param_type(subsystem, param):
    """Return the declared dtype name for one subsystem parameter.

    Parameters
    ----------
    subsystem : str
        Registered subsystem class name.
    param : str
        Parameter name to look up.

    Returns
    -------
    str or None
        Declared dtype name for ``param``, or None if the parameter is not
        defined for ``subsystem``.

    """
    subsystem_params_to_type = get_param_name_and_type_list(subsystem)
    for parameter, dtype in subsystem_params_to_type:
        if parameter == param:
            return dtype
    return None

# Wing subsystem with only span, chord, and sweep
@register_subsystem
class Wing(Subsystem):
    """Represent a wing component."""

    def setup(self):
        """Register baseline wing parameters.

        Returns
        -------
        None

        """
        self.add_parameter(
            "area", 0.0, float, unit_type=UnitType.AREA, description="Wing area"
        )
        self.add_parameter(
            "span", 0.0, float, unit_type=UnitType.LENGTH, description="Wing span"
        )
        self.add_parameter(
            "chord", 0.0, float, unit_type=UnitType.LENGTH, description="Wing chord"
        )
        self.add_parameter(
            "sweep",
            0.0,
            float,
            unit_type=UnitType.ANGLE,
            description="Wing sweep angle",
        )
        self.add_parameter(
            "aspect_ratio",
            1.0,
            float,
            unit_type=UnitType.ANGLE,
            description="Wing sweep angle",
        )


@register_subsystem
class Rotor(Subsystem):
    """Represent a rotor component."""

    def setup(self):
        """Register baseline rotor parameters.

        Returns
        -------
        None

        """
        self.add_parameter(
            "vtip",
            550.0,
            float,
            unit_type=UnitType.SPEED,
            description="Rotor tip speed",
        )
        self.add_parameter(
            "tilt", 1.0, float, unit_type=UnitType.LENGTH, description="Rotor tilt"
        )
        self.add_parameter(
            "radius", 1.0, float, unit_type=UnitType.LENGTH, description="Rotor radius"
        )
        self.add_parameter("nblade", 2, int, description="Number of blades")
        self.add_parameter(
            "rpm",
            1000.0,
            float,
            unit_type=UnitType.FREQUENCY if hasattr(UnitType, "FREQUENCY") else None,
            description="Rotor RPM",
        )
        self.add_parameter(
            "blade_chord",
            0.1,
            float,
            unit_type=UnitType.LENGTH,
            description="Blade chord length",
        )
        self.add_parameter(
            "collective_pitch",
            0.0,
            float,
            unit_type=UnitType.ANGLE,
            description="Collective pitch angle (deg)",
        )
        self.add_parameter(
                "disk_loading",
                0.0,
                float,
                unit_type=None,
                description="Diskloading (thrust/rotor area)")

@register_subsystem
class RotorBladeRegion(Subsystem):
    """Represent a rotor blade region component."""

    def setup(self):
        """Register baseline rotor blade region parameters.

        Returns
        -------
        None

        """
        self.add_parameter(
            "spanwise_start",
            0.0,
            float,
            unit_type=UnitType.LENGTH,
            description="start of section as a fraction of the total span",
        )
        self.add_parameter(
            "spanwise_stop",
            0.0,
            float,
            unit_type=UnitType.LENGTH,
            description="end of section as a fration of the total span",
        )
        self.add_parameter(
            "chord_start",
            0.0,
            float,
            unit_type=UnitType.LENGTH,
            description="chord at the start of the section",
        )
        self.add_parameter(
            "chord_stop",
            0.0,
            float,
            unit_type=UnitType.LENGTH,
            description="chord at the end of the section",
        )
        self.add_parameter(
            "sweep", 0.0, float, unit_type=UnitType.ANGLE, description="sweep angle"
        )
        self.add_parameter(
            "dihedral",
            0.0,
            float,
            unit_type=UnitType.ANGLE,
            description="dihedral angle",
        )
        self.add_parameter(
            "twist_start",
            0.0,
            float,
            unit_type=UnitType.ANGLE,
            description="twist angle at the start of the section",
        )
        self.add_parameter(
            "twist_stop",
            0.0,
            float,
            unit_type=UnitType.ANGLE,
            description="twist angle at the end of the section",
        )
        self.add_parameter(
            "airfoil_start",
            "",
            str,
            unit_type=None,
            description="airfoil at the begining of the section",
        )
        self.add_parameter(
            "airfoil_stop",
            "",
            str,
            unit_type=None,
            description="airfoil at the end of the section",
        )
        self.add_parameter(
            "horizontal_shift_start",
            0.0,
            float,
            unit_type=UnitType.LENGTH,
            description="",
        )
        self.add_parameter(
            "horizontal_shift_stop",
            0.0,
            float,
            unit_type=UnitType.LENGTH,
            description="",
        )
        self.add_parameter(
            "vertical_shift_start",
            0.0,
            float,
            unit_type=UnitType.LENGTH,
            description="",
        )
        self.add_parameter(
            "vertical_shift_stop", 0.0, float, unit_type=UnitType.LENGTH, description=""
        )
        self.add_parameter(
            "mainsparweb_thickness",
            0.0,
            float,
            unit_type=UnitType.LENGTH,
            description="",
        )
        self.add_parameter(
            "mainsparweb_angle", 0.0, float, unit_type=UnitType.ANGLE, description=""
        )

        self.add_parameter(
            'RICWG',0.0,float
        )
        self.add_parameter(
            'chord_array',np.array([0.0,0.0]),np.ndarray
        )



@register_subsystem
class Aircraft(Subsystem):
    """Represent an aircraft-level component."""
    def setup(self):
        """Register baseline aircraft-level parameters.

        Returns
        -------
        None

        """
        self.add_parameter('Wpass',0.0,float,description='Weight per passenger')

        self.add_parameter('Wenviron',75.,float,description='Environmental Control Lumped Weight')

        self.add_parameter('Wfurnish',0.0,float,'Weight of all furnishings (seats, carpet, etc.)')

        self.add_parameter('Welectrical',0.0,float,'Weight of misc electrical systems (lights, entertainment)')

        self.add_parameter('Npass',1,int,'Number of passengers')




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
    wing.unswept.sweep = 0.0

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

    with open("temp.json", "w") as f:
        json.dump(wing._to_dict(), f, indent=4)
