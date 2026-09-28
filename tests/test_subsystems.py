import pytest
from uam4x.subsystems import Wing, Rotor, RotorBladeRegion, Aircraft
from uam4x.subsystems import Parameter, get_subsystem_class, Subsystem, _Mode, _subsystem_registry, register_subsystem
from uam4x import  check_and_get_config
import os
import numpy as np
from uam4x.units import (

    UnitType, LengthUnit, AngleUnit, MassUnit, TimeUnit, ForceUnit, TemperatureUnit, PressureUnit, EnergyUnit, PowerUnit, VolumeUnit, AreaUnit, SpeedUnit, AccelerationUnit

)

from operator import attrgetter
#----------------------------------Parameter Class Checks
#Positive checks
def test_Parameter_class():

    test_parameter = Parameter(name='test_param', default_value=1.0, dtype=float, unit_type=LengthUnit,unit='in', description='this is a test parameter')
    assert test_parameter.name == 'test_param'
    assert test_parameter.value == 1.0
    assert test_parameter.unit_type == LengthUnit
    assert test_parameter.unit == 'in'
    assert test_parameter.description == 'this is a test parameter'
    assert str(test_parameter) == f"{test_parameter.name}: {test_parameter.value} [{test_parameter.unit_type}, in] ({test_parameter.description})"
    assert test_parameter.__repr__() == str(test_parameter)
    test_parameter.value=5.0
    assert test_parameter.value == 5.0
    assert test_parameter._value == test_parameter.value == 5.0

#def test_Parameter_check_type():
    #test_parameter = Parameter(name='test_param', default_value=1.0, dtype=float, unit_type=LengthUnit,unit='in', description='this is a test parameter')
    #assert test_parameter.check_type() == True


@pytest.mark.parametrize(
    "value, dtype, unit_type, unit, desc, expected_unit_type, expected_unit, expected_desc",
    [
        # 1) unit_type is a UnitType string, unit is a LengthUnit constant (string)
        (
            10.0,
            float,
            UnitType.LENGTH,      # "length"
            LengthUnit.INCH,      # "in"
            "sample param",
            "length",             # expected unit_type string
            "in",                 # expected unit string
            "sample param",
        ),

        # 2) unit_type is the LengthUnit *class*, unit is a bare string
        #    -> exercises the `hasattr(unit_type, '__name__')` branch
        (
            11.0,
            float,
            LengthUnit,           # class LengthUnit -> "LengthUnit"
            "in",
            "",
            "LengthUnit",         # from LengthUnit.__name__
            "in",                 # left as-is (already a string)
            "",                   # description passed through unchanged
        ),

        # 3) No units at all -> __init__ normalizes to UnitType.NONE ("none") and ""
        (
            5.0,
            float,
            None,
            None,
            "no units",
            UnitType.NONE,        # "none"
            "",                   # empty string from __init__
            "no units",
        ),
    ],
)
def test_Parameter_to_dict(
    value, dtype, unit_type, unit, desc,
    expected_unit_type, expected_unit, expected_desc
):
    name = "sample_param"

    param = Parameter(
        name=name,
        default_value=value,
        dtype=dtype,
        unit_type=unit_type,
        unit=unit,
        description=desc,
    )

    param_dict = param._to_dict()

    # Basic fields
    assert param_dict["name"] == name
    assert param_dict["value"] == value
    assert param_dict["dtype"] == dtype.__name__

    # Unit-related fields
    assert param_dict["unit_type"] == expected_unit_type
    assert param_dict["unit"] == expected_unit

    # Description
    assert param_dict["description"] == expected_desc

@pytest.mark.parametrize(
    "value, dtype, unit_type, unit, desc",
    [
        # Simple float with length units
        (10.0, float, UnitType.LENGTH, LengthUnit.INCH, "span in inches"),
        # Int with no units (will normalize to UnitType.NONE and "")
        (3, int, None, None, "integer param with no units"),
        # Bool, angular units
        (True, bool, UnitType.ANGLE, None, "boolean angle flag"),
    ],
)
def test_Parameter_from_dict_roundtrip(value, dtype, unit_type, unit, desc):
    name = "sample_param"

    # Create original parameter
    orig = Parameter(
        name=name,
        default_value=value,
        dtype=dtype,
        unit_type=unit_type,
        unit=unit,
        description=desc,
    )

    # Serialize to dict, then deserialize
    data = orig._to_dict()
    restored = Parameter._from_dict(data)

    # Basic identity checks
    assert isinstance(restored, Parameter)
    assert restored.name == orig.name
    assert restored.dtype is orig.dtype
    assert restored.description == orig.description

    # Unit fields after round-trip
    assert restored.unit_type == orig.unit_type
    assert restored.unit == orig.unit

    # Value check (scalar types)
    assert restored.value == orig.value

def test_Parameter_from_dict_ndarray():
    # Manually build a dict as it would appear in JSON
    data = {
        "name": "array_param",
        "value": [1.0, 2.0, 3.0],    # list, should become np.ndarray
        "dtype": "ndarray",
        "unit_type": UnitType.LENGTH,  # "length"
        "unit": LengthUnit.INCH,       # "in"
        "description": "array-valued parameter",
    }

    param = Parameter._from_dict(data)

    # Check basic fields
    assert isinstance(param, Parameter)
    assert param.name == "array_param"
    assert param.dtype is np.ndarray
    assert param.unit_type == UnitType.LENGTH
    assert param.unit == LengthUnit.INCH
    assert param.description == "array-valued parameter"

    # Value must be a numpy array with the same contents
    assert isinstance(param.value, np.ndarray)
    assert np.array_equal(param.value, np.array([1.0, 2.0, 3.0]))
#negative checks
def test_value_type_incoherance_in_constructor():

    with pytest.raises(check=TypeError):
        Parameter(name='lol', default_value='idk', dtype=float)

def test_value_type_incoherance_setter():
    test_parameter = Parameter(name='test_param', default_value=1.0, dtype=float, unit_type=LengthUnit,unit='in', description='this is a test parameter')

    with  pytest.raises(TypeError):
        test_parameter.value = '67'


# TODO parameter to/from dict
# Unclear if check_type method required for parameter based on allowed access...

#-----------------------------------Mode checks
def test_Mode_instantiation():
    #Mode should really be instantiated through a component
    W = Wing('test_wing')
    W.add_mode('vertical_mode')

    assert W.vertical_mode.name == 'vertical_mode'
    assert W.vertical_mode._overrides == {}
    assert W.vertical_mode.component is  W



def test_Mode_set_parameter_good():
    W = Wing('test_wing')
    W.set_parameter('chord', 11.0)
    W.set_parameter('span', 15.0)

    W.add_mode('vertical_mode')
    assert W.vertical_mode.chord == 11.0
    assert W.vertical_mode.span == 15.0

    W.vertical_mode.span = 14.0
    W.vertical_mode.set_parameter('chord', 15.0)

    assert W.vertical_mode.chord == 15.0
    assert W.vertical_mode.span == 14.0

def test_Mode_get_parameter():
    W = Wing('test_wing')
    W.set_parameter('chord', 11.0)
    W.set_parameter('span', 15.0)


    W.add_mode('vertical_mode')
    assert W.vertical_mode.chord == 11.0
    assert W.vertical_mode.span == 15.0

    W.vertical_mode.span = 14.0

    assert W.vertical_mode.chord is W.vertical_mode.get_parameter('chord')
    assert W.vertical_mode.span is W.vertical_mode.get_parameter('span')


def test_Mode_parameters_property():
    R = Rotor('test_rotor')
    R.set_parameter('vtip', 670.)
    R.set_parameter('nblade', 8)

    assert R.vtip == 670.
    assert R.nblade == 8
    assert R.tilt == 1.0
    assert R.radius == 1.0
    assert R.rpm == 1000.0
    assert R.blade_chord == 0.1
    assert R.collective_pitch == 0.0

    R.add_mode('vertical_rotor')
    R.vertical_rotor.set_parameter('tilt', 90.0)
    R.vertical_rotor.set_parameter('vtip', 80.0)

    vt_mode_params = R.vertical_rotor.parameters

    rotor_param_names = ['vtip', 'tilt', 'radius', 'nblade', 'rpm', 'blade_chord', 'collective_pitch']
    for param_name in  rotor_param_names:
        assert param_name in  vt_mode_params.keys()

    assert vt_mode_params['vtip'].value == 80.0
    assert vt_mode_params['tilt'].value == 90.0
    assert vt_mode_params['radius'].value == 1.0
    assert vt_mode_params['rpm'].value == 1000.0

def test_Mode_str():
    A = Wing('Main')
    A.add_mode('vertical')

    # No overrides yet
    assert str(A.vertical) == "Mode('vertical') (no parameter overrides)"

    # Add some overrides
    A.vertical.set_parameter('chord', 32.0)
    A.vertical.set_parameter('aspect_ratio', 13.0)

    # Exact string match with both overrides in order
    assert str(A.vertical) == "Mode('vertical') overrides: chord: 32.0, aspect_ratio: 13.0"

#----------Negative mode checks
def test_Mode_set_parameter_bad_name():
    W = Wing('test_wing')
    W.set_parameter('chord', 11.0)
    W.set_parameter('span', 15.0)

    W.add_mode('vertical_mode')
    with pytest.raises(AttributeError):
        W.vertical_mode.set_parameter('random_param', 1.0)

    with  pytest.raises(AttributeError):
        W.vertical_mode.random_param = 1.0


def test_Mode_set_parameter_bad_value():
    W = Wing('test_wing')
    W.set_parameter('chord', 11.0)
    W.set_parameter('span', 15.0)

    W.add_mode('vertical_mode')
    with pytest.raises(TypeError):
        W.vertical_mode.set_parameter('span', 1)

    with  pytest.raises(TypeError):
        W.vertical_mode.chord = 'five'

def test_Mode_get_parameter_bad():

    W = Wing('test_wing')
    W.set_parameter('chord', 11.0)
    W.set_parameter('span', 15.0)

    W.add_mode('vertical_mode')

    with pytest.raises(AttributeError):
        W.vertical_mode.get_parameter('random_var')

    with  pytest.raises(AttributeError):
        W.vertical_mode.random_var

@pytest.mark.parametrize('name,value,error', [
    ('vtip', 90, TypeError),
    ('random_var', 67., AttributeError)
])
def test_Mode_parameters_property_bad(name, value, error):
    R = Rotor('test_rotor')
    R.set_parameter('vtip', 670.)
    R.set_parameter('nblade', 8)

    assert R.vtip == 670.
    assert R.nblade == 8
    assert R.tilt == 1.0
    assert R.radius == 1.0
    assert R.rpm == 1000.0
    assert R.blade_chord == 0.1
    assert R.collective_pitch == 0.0

    R.add_mode('vertical_rotor')
    #User really shouldn't mess with encapsulation, but we cover this
    R.vertical_rotor._overrides[name] = value
    with  pytest.raises(error):
        vt_mode_params = R.vertical_rotor.parameters

#-----------------------------------Subsystem checks
def test_Subsystem_instantiation():

    C = Subsystem('sample_component')
    assert  C._parameters == {}
    assert C._modes == {}
    assert C._is_setup == False
    assert C._parent is  None

def test_Subsystem_add_param():

    class CustomSubsystem(Subsystem):
        def setup(self):
            self.add_parameter('king',3.0, float)
            self.add_parameter('ace', 4.0, float)

    C = CustomSubsystem('bridge')
    assert C.king == 3.0

    C.ace = 5.0

    assert C.ace == 5.0

def test_Subsystem_set_parameter():

    W = Wing('sample_wing')

    assert W._parameters['chord'].unit == ""
    assert W._parameters['chord'].unit_type is UnitType.LENGTH

    W.set_parameter('chord', 4.0, LengthUnit.INCH)

    assert W._parameters['chord'].unit is LengthUnit.INCH
    assert W._parameters['chord'].unit_type is UnitType.LENGTH


def test_Subsystem_parameter_comments():
    W = Wing("sample_wing")

    W.set_parameter("span", 15.0, comment="NASA reference")
    assert W.get_parameter_comment("span") == "NASA reference"

    W.span = 16.0
    assert W.get_parameter_comment("span") == ""

    with pytest.raises(AttributeError, match="Cannot get comment"):
        W.get_parameter_comment("missing")


def test_Mode_parameter_comments_and_serialization():
    W = Wing("sample_wing")
    W.set_parameter("span", 15.0, comment="Baseline reference")
    mode = W.add_mode("cruise")

    assert mode.get_parameter_comment("span") == "Baseline reference"

    mode.set_parameter("span", 20.0, comment="Cruise reference")
    assert mode.get_parameter_comment("span") == "Cruise reference"
    assert mode.parameters["span"].comment == "Cruise reference"

    data = W._to_dict()
    assert data["modes"]["cruise"]["comments"]["span"] == "Cruise reference"

    rebuilt = Subsystem._from_dict(data)
    assert rebuilt.get_parameter_comment("span") == "Baseline reference"
    assert rebuilt.cruise.get_parameter_comment("span") == "Cruise reference"

    data["parameters"]["span"].pop("comment")
    data["modes"]["cruise"].pop("comments")
    rebuilt_legacy = Subsystem._from_dict(data)
    assert rebuilt_legacy.get_parameter_comment("span") == ""
    assert rebuilt_legacy.cruise.get_parameter_comment("span") == ""

    with pytest.raises(AttributeError, match="Cannot get comment"):
        mode.get_parameter_comment("missing")


def test_Subsystem_add_mode():
    R = Rotor('sample_rotor')

    R.add_mode('vertical')

    R.vertical.vtip = 5.0
    R.vertical.set_parameter('rpm', 2000.0)

    assert R.vertical.vtip == 5.0
    assert R.vtip == 550.
    assert R.rpm == 1000.
    assert R.vertical.rpm == 2000.0


def test_RotorBladeRegion_get_parameter_defaults():
    RBR = RotorBladeRegion('sample_rotor_blade_region')

    # expected default values from setup()
    expected_defaults = {
        "spanwise_start": 0.0,
        "spanwise_stop": 0.0,
        "chord_start": 0.0,
        "chord_stop": 0.0,
        "sweep": 0.0,
        "dihedral": 0.0,
        "twist_start": 0.0,
        "twist_stop": 0.0,
        "airfoil_start": "",
        "airfoil_stop": "",
        "horizontal_shift_start": 0.0,
        "horizontal_shift_stop": 0.0,
        "vertical_shift_start": 0.0,
        "vertical_shift_stop": 0.0,
        "mainsparweb_thickness": 0.0,
        "mainsparweb_angle": 0.0,
    }

    # Check value + consistency with attribute + underlying Parameter
    for name, expected in expected_defaults.items():
        # via get_parameter
        val_from_getter = RBR.get_parameter(name)
        # via dynamic attribute
        val_from_attr = getattr(RBR, name)
        # via underlying Parameter object
        param_obj = RBR._parameters[name]

        assert val_from_getter == expected
        assert val_from_attr == expected
        assert param_obj.value == expected

        # And all three ways are consistent with each other
        assert val_from_getter == val_from_attr
        assert val_from_getter == param_obj.value


def test_RotorBladeRegion_get_parameter_after_set():
    RBR = RotorBladeRegion('sample_rotor_blade_region')

    # Set a few parameters to non-default values
    RBR.set_parameter("chord_start", 1.5, unit=LengthUnit.INCH)
    RBR.set_parameter("dihedral", 10.0)  # ANGLE; unit optional in your current logic
    RBR.set_parameter("airfoil_start", "NACA2412")  # string, no unit_type

    # chord_start
    assert RBR.get_parameter("chord_start") == 1.5
    assert RBR.chord_start == 1.5
    assert RBR._parameters["chord_start"].value == 1.5
    assert RBR._parameters["chord_start"].unit == LengthUnit.INCH

    # dihedral
    assert RBR.get_parameter("dihedral") == 10.0
    assert RBR.dihedral == 10.0
    assert RBR._parameters["dihedral"].value == 10.0

    # airfoil_start
    assert RBR.get_parameter("airfoil_start") == "NACA2412"
    assert RBR.airfoil_start == "NACA2412"
    assert RBR._parameters["airfoil_start"].value == "NACA2412"


def test_Subsystem_get_parameter_type():
    class CustomSubsystem(Subsystem):
        def setup(self):
            self.add_parameter('sample_array', np.arange(1, 10), np.ndarray)

    RBR = RotorBladeRegion('sample_rotor_blade_region')
    assert RBR.get_parameter_type('spanwise_start') is float
    assert RBR.get_parameter_type('airfoil_start') is  str

    R = Rotor('sample_rotor')
    assert R.get_parameter_type('nblade') is int

    CC = CustomSubsystem('sample_custom')
    assert CC.get_parameter_type('sample_array') is np.ndarray

def test_check_all_parameter_types_good():
    W = Wing('sample_wing')

    # Add a mode and set some valid overrides
    vertical_mode = W.add_mode('vertical_mode')
    vertical_mode.set_parameter('span', 20.0)   # dtype float
    vertical_mode.set_parameter('sweep', 30.0)  # dtype float

    result = W.check_all_parameter_types()

    # There should be an entry for the mode
    assert 'vertical_mode' in result

    mode_result = result['vertical_mode']

    # Only overridden parameters should appear
    assert 'span' in mode_result
    assert 'sweep' in mode_result
    assert 'chord' not in mode_result  # never overridden

    # Both overrides are correct types, so they should be True
    assert mode_result['span'] is True
    assert mode_result['sweep'] is True


def test_Subsystem_str_with_mode():
    rbr = RotorBladeRegion("sample_rotor_blade_region")

    # Add a mode and set some overrides to make sure values show up
    mode = rbr.add_mode("cruise")
    mode.set_parameter("chord_start", 1.5)
    mode.set_parameter("dihedral", 10.0)

    s = str(rbr)
    lines = s.splitlines()

    # First line: subsystem header
    assert lines[0] == "RotorBladeRegion('sample_rotor_blade_region')"

    # All parameters (base values) should appear in the string as "name: value"
    for pname, param in rbr._parameters.items():
        expected_fragment = f"{pname}: {param.value}"
        assert expected_fragment in s, f"Missing '{expected_fragment}' in Subsystem.__str__ output"

def test_Subsystem_to_dict_with_mode():
    rbr = RotorBladeRegion("sample_rotor_blade_region")

    # Set some non-default values
    rbr.set_parameter("chord_start", 2.5, unit=LengthUnit.INCH)
    rbr.set_parameter("sweep", 15.0)

    # Add a mode with overrides
    mode = rbr.add_mode("cruise")
    mode.set_parameter("chord_start", 3.0)  # override base value
    mode.set_parameter("dihedral", 7.5)

    data = rbr._to_dict()

    # Top-level keys
    assert data["name"] == "sample_rotor_blade_region"
    assert data["subsystem_type"] == "RotorBladeRegion"
    assert "parameters" in data
    assert "modes" in data

    # Parameters dict keys should match the component's _parameters keys
    assert set(data["parameters"].keys()) == set(rbr._parameters.keys())

    # Check a couple of specific parameters
    chord_start_dict = data["parameters"]["chord_start"]
    assert chord_start_dict["name"] == "chord_start"
    assert chord_start_dict["value"] == rbr.chord_start  # 2.5
    assert chord_start_dict["dtype"] == "float"
    # unit_type/unit representation depends on your UnitType/LengthUnit setup,
    # but we can at least check presence:
    assert "unit_type" in chord_start_dict
    assert "unit" in chord_start_dict

    sweep_dict = data["parameters"]["sweep"]
    assert sweep_dict["name"] == "sweep"
    assert sweep_dict["value"] == rbr.sweep  # 15.0
    assert sweep_dict["dtype"] == "float"

    # Modes section: should contain our "cruise" mode with overrides
    assert "cruise" in data["modes"]
    cruise_mode_data = data["modes"]["cruise"]
    assert "overrides" in cruise_mode_data

    # Overrides should match what we set on the mode
    assert cruise_mode_data["overrides"]["chord_start"] == 3.0
    assert cruise_mode_data["overrides"]["dihedral"] == 7.5
    # And no unexpected overrides
    assert set(cruise_mode_data["overrides"].keys()) == {"chord_start", "dihedral"}

def test_Subsystem_from_dict():
    # Hand-crafted dict mimicking JSON for a Wing with a mode + overrides
    data = {
        "name": "test_wing",
        "subsystem_type": "Wing",
        "parameters": {
            "span": {
                "name": "span",
                "value": 10.0,
                "dtype": "float",
                "unit_type": "length",   # matches what _to_dict would emit
                "unit": "m",
                "description": "Wing span",
            },
            "sweep": {
                "name": "sweep",
                "value": 25.0,
                "dtype": "float",
                "unit_type": "angle",
                "unit": "deg",
                "description": "Wing sweep angle",
            },
        },
        "modes": {
            "cruise": {
                "overrides": {
                    "sweep": 30.0,  # mode-specific override
                }
            }
        },
    }

    # Call the generic Subsystem loader; it should resolve Wing via get_subsystem_class
    comp = Subsystem._from_dict(data)

    # We got a Wing with the right name
    assert isinstance(comp, Wing)
    assert comp.name == "test_wing"

    # Parameters from dict should override defaults
    assert "span" in comp._parameters
    assert "sweep" in comp._parameters

    span_param = comp._parameters["span"]
    sweep_param = comp._parameters["sweep"]

    assert span_param.value == 10.0
    assert span_param.dtype is float
    assert span_param.name == "span"

    assert sweep_param.value == 25.0
    assert sweep_param.dtype is float
    assert sweep_param.name == "sweep"

    # Dynamic properties should reflect the values
    assert comp.span == 10.0
    assert comp.sweep == 25.0

    # Modes should be reconstructed with overrides
    assert "cruise" in comp._modes
    cruise = comp.cruise  # _Mode instance attached as attribute

    # Underlying overrides dict matches what we put in the data
    assert cruise._overrides["sweep"] == 30.0

    # And get_parameter should see the override, not the base component value
    assert cruise.get_parameter("sweep") == 30.0
    # Base component still has 25.0
    assert comp.sweep == 25.0

def test_Wing_get_param_dict():
    W = Wing("test_wing")

    # Set some non-default values so we can see changes propagate
    W.set_parameter("area", 42.0)
    W.set_parameter("span", 15.5)
    W.set_parameter("chord", 3.2)

    param_dict = W.get_param_dict()

    # 1) Keys should match the component's _parameters
    assert set(param_dict.keys()) == set(W._parameters.keys())

    # 2) Values should match the underlying Parameter.value
    for name, param in W._parameters.items():
        assert param_dict[name] == param.value

    # 3) And for a few known parameters, also match the dynamic attributes
    assert param_dict["area"] == W.area == W._parameters["area"].value
    assert param_dict["span"] == W.span == W._parameters["span"].value
    assert param_dict["chord"] == W.chord == W._parameters["chord"].value
#--------------Negative checks
def test_Subsystem_bad_name():
    with pytest.raises(ValueError):
        C = Subsystem('rand[om')

def test_Subsystem_bad_add_parameter():
    class CustomSubsystem(Subsystem):
        def setup(self):
            self.add_parameter('ki[ng',3.0, float)
            self.add_parameter('ace', 4.0, float)

    with pytest.raises(ValueError):
        C = CustomSubsystem('bridge')


def test_Subsystem_bad_add_parameter_outside_setup():
    class CustomSubsystem(Subsystem):
        def setup(self):
            self.add_parameter('king',3.0, float)
            self.add_parameter('ace', 4.0, float)

    C = CustomSubsystem('sample_component')
    with pytest.raises(RuntimeError):
        C.add_parameter('jack', 1.0, float)

def test_Subsystem_set_parameter_bad_unit():

    W = Wing('sample_wing')

    assert W._parameters['chord'].unit == ''
    assert W._parameters['chord'].unit_type is UnitType.LENGTH

    with pytest.raises(ValueError):

        W.set_parameter('chord', 4.0, 'hello')

def test_Subsystem_set_parameter_bad_name():
    W = Wing('sample_wing')

    with  pytest.raises(AttributeError):
        W.set_parameter('random_var', 8.)

@pytest.mark.parametrize(
    "name,error",
    [
        # Bad characters in a string -> ValueError
        ("bad[mode", ValueError),
        ("bad]mode", ValueError),

        # Non-string types -> TypeError
        (3.0, TypeError),                  # float
        (42, TypeError),                   # int
        (True, TypeError),                 # bool
        (None, TypeError),                 # NoneType
        (["list"], TypeError),             # list
        ({"mode": "vertical"}, TypeError), # dict
        ((1, 2), TypeError),               # tuple
        (object(), TypeError),             # arbitrary object
    ],
)
def test_Subsystem_add_mode_bad_name(name, error):
    R = Rotor("sample_rotor")

    with pytest.raises(error):
        R.add_mode(name)


def test_Subsystem_get_parameter_missing_name():

    RBR = RotorBladeRegion('sample_rotor_blade_region')

    with pytest.raises(AttributeError):
        RBR.get_parameter('random_param')


def test_Subsystem_get_parameter_type_missing_param():
    class CustomSubsystem(Subsystem):
        def setup(self):
            self.add_parameter('sample_array', np.arange(1, 10), np.ndarray)

    RBR = RotorBladeRegion('sample_rotor_blade_region')
    with pytest.raises(AttributeError):
        RBR.get_parameter_type('random_var')

    CC = CustomSubsystem('sample_custom')

    with pytest.raises(AttributeError):
        CC.get_parameter('non_existent_param')




def test_check_all_parameter_types_bad_param_name():
    W = Wing('sample_wing')
    vertical_mode = W.add_mode('vertical_mode')

    # Bypass type/name checks by directly poking _overrides
    vertical_mode._overrides['not_a_real_param'] = 123.0

    # Now check_all_parameter_types should raise, since the param doesn't exist
    with pytest.raises(AttributeError):
        W.check_all_parameter_types()
#-----------------------------------Wing checks

def test_wing_parameters():
    wing = Wing("wing1")
    assert wing.name == "wing1"
    # Set and get
    wing.span = 12.5
    assert wing.span == 12.5
    wing.set_parameter("chord", 2.0)
    assert wing.chord == 2.0
    # Test mode override
    wing.add_mode("cruise")
    wing.cruise.span = 15.0
    assert wing.cruise.span == 15.0
    assert wing.span == 12.5
    assert wing.baseline.span == 12.5
#-----------------------------------Rotor checks
def test_rotor_parameters():
    rotor = Rotor("rotor1")
    assert rotor.name == "rotor1"
    assert rotor.radius == 1.0
    rotor.set_parameter("rpm", 1200.0)
    assert rotor.rpm == 1200.0
    rotor.add_mode("hover")
    rotor.hover.rpm = 1500.0
    assert rotor.hover.rpm == 1500.0
    assert rotor.rpm == 1200.0
#-----------------------------------RotorBladeRegion checks

#-----------------------------------Aircraft checks
def test_aircraft_parameters():
    aircraft = Aircraft("ac1")
    assert aircraft.name == "ac1"
    assert aircraft.Npass == 1
    aircraft.set_parameter("Wpass", 180.0)
    assert aircraft.Wpass == 180.0

#----------------------------------Subsystem registry
def test_register_subsystem_good():

    file_path = os.path.abspath(__file__)
    test_dir = os.path.dirname(file_path)
    proj_dir = os.path.dirname(test_dir)


    config_path = os.path.join(proj_dir,'uam4x', 'conf.json')

    if os.path.exists(config_path):
        os.remove(config_path)

    config = check_and_get_config(config_path)
    assert "PLUGINS" in config
    assert "SUBSYSTEMS" in config


    @register_subsystem
    class C(Subsystem):
        pass

    config = check_and_get_config(config_path)
    if os.getenv("UAM4X_DISABLE_REGISTRY_WRITES") != "1":
        assert C.__name__ in config["SUBSYSTEMS"]
    assert _subsystem_registry[C.__name__] == C


def test_subsystem_registry():
    for name in ("Wing", "Rotor", "Aircraft"):
        subsystem_class = get_subsystem_class(name)
        assert subsystem_class.__name__ == name
        assert isinstance(subsystem_class("registry_test"), Subsystem)
    with pytest.raises(ImportError):
        get_subsystem_class("NotASubsystem")
if __name__ == "__main__":
    test_Subsystem_set_parameter()
    pass
