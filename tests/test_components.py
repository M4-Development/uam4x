import pytest
from uam4x.components import Wing, Rotor, Aircraft, Parameter, get_commponent_class

class ChildComponent(Wing):
    def setup(self):
        super().setup()
        self.add_parameter("float_param", 1.23, float, description="A float parameter")
        self.add_parameter("int_param", 42, int, description="An int parameter")
        self.add_parameter("str_param", "hello", str, description="A string parameter")
        self.add_parameter("bool_param", True, bool, description="A bool parameter")


def test_child_component_all_types():
    child = ChildComponent("child1")
    # Attribute access
    assert isinstance(child.float_param, float)
    assert child.float_param == 1.23
    assert isinstance(child.int_param, int)
    assert child.int_param == 42
    assert isinstance(child.str_param, str)
    assert child.str_param == "hello"
    assert isinstance(child.bool_param, bool)
    assert child.bool_param is True
    # Test setting via attribute
    child.float_param = 2.34
    child.int_param = 99
    child.str_param = "world"
    child.bool_param = False
    assert child.float_param == 2.34
    assert child.int_param == 99
    assert child.str_param == "world"
    assert child.bool_param is False
    # Access via get_parameter
    assert child.get_parameter("float_param") == 2.34
    assert child.get_parameter("int_param") == 99
    assert child.get_parameter("str_param") == "world"
    assert child.get_parameter("bool_param") is False
    # Set via set_parameter
    child.set_parameter("float_param", 3.45)
    child.set_parameter("int_param", 123)
    child.set_parameter("str_param", "alt")
    child.set_parameter("bool_param", True)
    assert child.float_param == 3.45
    assert child.int_param == 123
    assert child.str_param == "alt"
    assert child.bool_param is True
    # Baseline access
    assert child.baseline.float_param == 3.45
    assert child.baseline.int_param == 123
    assert child.baseline.str_param == "alt"
    assert child.baseline.bool_param is True
    # Add and test modes
    child.add_mode("mode1")
    child.mode1.float_param = 10.0
    child.mode1.int_param = 7
    child.mode1.str_param = "mode1str"
    child.mode1.bool_param = False
    assert child.mode1.float_param == 10.0
    assert child.mode1.int_param == 7
    assert child.mode1.str_param == "mode1str"
    assert child.mode1.bool_param is False
    # Baseline remains unchanged
    assert child.baseline.float_param == 3.45
    assert child.baseline.int_param == 123
    assert child.baseline.str_param == "alt"
    assert child.baseline.bool_param is True
    # Add another mode
    child.add_mode("mode2")
    child.mode2.float_param = -1.0
    child.mode2.int_param = 0
    child.mode2.str_param = "zero"
    child.mode2.bool_param = True
    assert child.mode2.float_param == -1.0
    assert child.mode2.int_param == 0
    assert child.mode2.str_param == "zero"
    assert child.mode2.bool_param is True
    assert child.baseline.float_param == 3.45
    assert child.baseline.int_param == 123
    assert child.baseline.str_param == "alt"
    assert child.baseline.bool_param is True

    # Add another mode
    mode3 = child.add_mode("mode3")
    mode3.set_parameter("float_param", 5.0)
    mode3.set_parameter("int_param", 10)
    mode3.set_parameter("str_param", "ten")
    mode3.set_parameter("bool_param", False)
    assert child.mode3.float_param == 5.0
    assert child.mode3.int_param == 10
    assert child.mode3.str_param == "ten"
    assert child.mode3.bool_param is False
    assert child.baseline.float_param == 3.45
    assert child.baseline.int_param == 123
    assert child.baseline.str_param == "alt"
    assert child.baseline.bool_param is True
    # Incorrect usage: wrong type

    with pytest.raises(TypeError):
        child.float_param = "not a float"
    with pytest.raises(TypeError):
        child.set_parameter("int_param", "not an int")
    with pytest.raises(TypeError):
        child.mode1.bool_param = "not a bool"

    # Incorrect usage: non-existent parameter
    with pytest.raises(AttributeError):
        _ = child.nonexistent_param
    with pytest.raises(AttributeError):
        _ = child.bad_mode.float_param
    with pytest.raises(AttributeError):
        child.set_parameter("not_a_param", 123)

    # Incorrect usage: non-existent parameter
    with pytest.raises(AttributeError):
        child.get_parameter("nonexistent_param")
    with pytest.raises(AttributeError):
        _ = child.bad_mode.get_parameter("float_param")


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


def test_aircraft_parameters():
    aircraft = Aircraft("ac1")
    assert aircraft.name == "ac1"
    assert aircraft.npass == 4
    aircraft.set_parameter("wpass", 180.0)
    assert aircraft.wpass == 180.0


def test_component_registry():
    assert get_commponent_class("Wing") is Wing
    assert get_commponent_class("Rotor") is Rotor
    assert get_commponent_class("Aircraft") is Aircraft
    with pytest.raises(ImportError):
        get_commponent_class("NotAComponent")


def test_parameter_type_check():
    wing = Wing("wing2")
    wing.set_parameter("span", 10.0)
    assert isinstance(wing.span, float)
    with pytest.raises(TypeError):
        wing.set_parameter("span", "not_a_float")


def test_to_dict_and_from_dict():
    wing = Wing("wing3")
    wing.set_parameter("span", 8.0)
    d = wing._to_dict()
    wing2 = Wing._from_dict(d)
    assert wing2.name == wing.name
    assert wing2.span == 8.0
    # Modes
    wing.add_mode("takeoff")
    wing.takeoff.span = 20.0
    d2 = wing._to_dict()
    wing3 = Wing._from_dict(d2)
    assert "takeoff" in wing3._modes
    assert wing3._modes["takeoff"]._overrides["span"] == 20.0



