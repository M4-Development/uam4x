import pytest
from uam4x.subsystems import Subsystem, Wing, Rotor, RotorBladeRegion, Rotorcraft, _Mode
import numpy as np
from uam4x.vehicle import Vehicle, VehicleManager
import  re





#-----------------------------------------------------------All fixtures

@pytest.fixture
def demo_vehicle():
    return Vehicle('vehicle_demo')

@pytest.fixture
def demo_wing() -> Wing:
    return Wing("wing_demo")

@pytest.fixture
def demo_rotor() -> Rotor:
    return Rotor("rotor_demo")

@pytest.fixture
def demo_rotor_blade_region() -> RotorBladeRegion:
    return RotorBladeRegion('rotor_blade_region_demo')

@pytest.fixture
def demo_rotorcraft() -> Rotorcraft:
    return Rotorcraft('rotorcraft_demo')

@pytest.fixture
def demo_mode()-> _Mode:
    return _Mode('mode_demo')

@pytest.fixture
def demo_vehicle_manager():
    return VehicleManager()
#-----------------------------------------------------------Test VehicleManager Class
def test_adding_vehicle_to_vehicle_manager_attr_access(demo_vehicle_manager, demo_vehicle, demo_wing):

    demo_vehicle.add_component(demo_wing)
    demo_vehicle_manager.add_vehicle(demo_vehicle)

    assert demo_vehicle_manager.vehicle_demo is  demo_vehicle
    assert demo_vehicle_manager.get_vehicle('vehicle_demo') is  demo_vehicle

def test_vehicle_manager_repr_with_vehicles(demo_vehicle_manager, demo_vehicle):
    # demo_vehicle.name should be "vehicle_demo" from your fixture
    vm = demo_vehicle_manager

    # Add first vehicle
    vm.add_vehicle(demo_vehicle)

    # Add a second vehicle with a different name
    another_vehicle = Vehicle("another_vehicle")
    vm.add_vehicle(another_vehicle)

    expected_names = [demo_vehicle.name, another_vehicle.name]
    expected_repr = f"VehicleManager(vehicles={expected_names})"

    assert repr(vm) == expected_repr

def test_vehicle_iteration(demo_vehicle_manager):

    arr = np.arange(1, 20)
    veh_names = []
    for num in arr:
        demo_vehicle_manager.add_vehicle(Vehicle(f'test_veh_{num}'))
        veh_names.append(f'test_veh_{num}')

    assert len(veh_names) == len(demo_vehicle_manager)
    for vehicle in  demo_vehicle_manager:
        assert vehicle.name in  veh_names

def test_nonexistent_vehicle_access(demo_vehicle_manager, demo_vehicle, demo_wing):
    demo_vehicle.add_component(demo_wing)
    demo_vehicle_manager.add_vehicle(demo_vehicle)

    with pytest.raises(ValueError):
        demo_vehicle_manager.get_vehicle('non_existent_vehicle')

    with pytest.raises(AttributeError):
        demo_vehicle_manager.non_existent_vehicle

def test_adding_same_vehicle(demo_vehicle_manager):

    test_veh1 = Vehicle('TestVehicle')
    test_veh2 = Vehicle('TestVehicle')
    assert test_veh1 is  not test_veh2

    demo_vehicle_manager.add_vehicle(test_veh1)
    with pytest.raises(ValueError):
        demo_vehicle_manager.add_vehicle(test_veh2)



#-------------------------------------------------Test Vehicle Class

# Expected behavior checks
#Test adding components
def test_adding_vehicle_components_good(demo_rotor, demo_wing, demo_rotorcraft):
    veh = Vehicle('SampleVehicle', [demo_wing, demo_rotorcraft])
    veh.add_component(demo_rotor)

    assert veh.wing_demo is demo_wing
    assert veh.rotorcraft_demo is demo_rotorcraft
    assert veh.rotor_demo is demo_rotor



#Check querying of attributes via name
def test_vehicle_component_attribute_values(
    demo_vehicle, demo_wing, demo_rotor, demo_rotor_blade_region, demo_rotorcraft
):
    # -------------------------
    # Set some non-default values
    # -------------------------
    demo_wing.area = 2320.0
    demo_wing.set_parameter("span", 30.0)
    demo_wing.chord = 15.0
    # sweep, aspect_ratio left at defaults

    demo_rotor.vtip = 60.0
    demo_rotor.set_parameter("incidence", 10.0)
    demo_rotor.radius = 3.14
    demo_rotor.nblade = 5
    # omega, root_chord, pitch_angle left at defaults

    demo_rotor_blade_region.spanwise_start = 40.0
    demo_rotor_blade_region.set_parameter("spanwise_stop", 50.0)
    demo_rotor_blade_region.chord_start = 0.0
    demo_rotor_blade_region.chord_stop = 69.0
    # everything else left at defaults

    demo_rotorcraft.npass = 15.0
    # Wpass left at default

    # -------------------------
    # Attach components to vehicle
    # -------------------------
    demo_vehicle.add_component(demo_wing)
    demo_vehicle.add_component(demo_rotor)
    demo_vehicle.add_component(demo_rotor_blade_region)
    demo_vehicle.add_component(demo_rotorcraft)

    # -------------------------
    # Identity: access via name returns the same objects
    # -------------------------
    assert demo_vehicle.wing_demo is demo_wing
    assert demo_vehicle.rotor_demo is demo_rotor
    assert demo_vehicle.rotor_blade_region_demo is demo_rotor_blade_region
    assert demo_vehicle.rotorcraft_demo is demo_rotorcraft

    # -------------------------
    # Wing parameters (explicit + defaults)
    # -------------------------
    # explicit
    assert demo_vehicle.wing_demo.area == demo_wing.area == 2320.0
    assert demo_vehicle.wing_demo.span == demo_wing.span == 30.0
    assert demo_vehicle.wing_demo.chord == demo_wing.chord == 15.0

    # defaults
    assert demo_vehicle.wing_demo.sweep == demo_wing.sweep == 0.0
    assert demo_vehicle.wing_demo.aspect_ratio == demo_wing.aspect_ratio == 1.0

    # -------------------------
    # Rotor parameters (explicit + defaults)
    # -------------------------
    # explicit
    assert demo_vehicle.rotor_demo.vtip == demo_rotor.vtip == 60.0
    assert demo_vehicle.rotor_demo.incidence == demo_rotor.incidence == 10.0
    assert demo_vehicle.rotor_demo.radius == demo_rotor.radius == 3.14
    assert demo_vehicle.rotor_demo.nblade == demo_rotor.nblade == 5

    # defaults
    assert demo_vehicle.rotor_demo.omega == demo_rotor.omega == 1000.0
    assert demo_vehicle.rotor_demo.root_chord == demo_rotor.root_chord == 1.0
    assert demo_vehicle.rotor_demo.pitch_angle == demo_rotor.pitch_angle == 0.0

    # -------------------------
    # RotorBladeRegion parameters (explicit + defaults)
    # -------------------------
    # explicit
    assert (
        demo_vehicle.rotor_blade_region_demo.spanwise_start
        == demo_rotor_blade_region.spanwise_start
        == 40.0
    )
    assert (
        demo_vehicle.rotor_blade_region_demo.spanwise_stop
        == demo_rotor_blade_region.spanwise_stop
        == 50.0
    )
    assert (
        demo_vehicle.rotor_blade_region_demo.chord_start
        == demo_rotor_blade_region.chord_start
        == 0.0
    )
    assert (
        demo_vehicle.rotor_blade_region_demo.chord_stop
        == demo_rotor_blade_region.chord_stop
        == 69.0
    )

    # defaults
    assert demo_vehicle.rotor_blade_region_demo.sweep == demo_rotor_blade_region.sweep == 0.0
    assert (
        demo_vehicle.rotor_blade_region_demo.dihedral
        == demo_rotor_blade_region.dihedral
        == 0.0
    )
    assert (
        demo_vehicle.rotor_blade_region_demo.twist_start
        == demo_rotor_blade_region.twist_start
        == 0.0
    )
    assert (
        demo_vehicle.rotor_blade_region_demo.twist_stop
        == demo_rotor_blade_region.twist_stop
        == 0.0
    )
    assert (
        demo_vehicle.rotor_blade_region_demo.airfoil_start
        == demo_rotor_blade_region.airfoil_start
        == ""
    )
    assert (
        demo_vehicle.rotor_blade_region_demo.airfoil_stop
        == demo_rotor_blade_region.airfoil_stop
        == ""
    )
    assert (
        demo_vehicle.rotor_blade_region_demo.horizontal_shift_start
        == demo_rotor_blade_region.horizontal_shift_start
        == 0.0
    )
    assert (
        demo_vehicle.rotor_blade_region_demo.horizontal_shift_stop
        == demo_rotor_blade_region.horizontal_shift_stop
        == 0.0
    )
    assert (
        demo_vehicle.rotor_blade_region_demo.vertical_shift_start
        == demo_rotor_blade_region.vertical_shift_start
        == 0.0
    )
    assert (
        demo_vehicle.rotor_blade_region_demo.vertical_shift_stop
        == demo_rotor_blade_region.vertical_shift_stop
        == 0.0
    )
    assert (
        demo_vehicle.rotor_blade_region_demo.mainsparweb_thickness
        == demo_rotor_blade_region.mainsparweb_thickness
        == 0.0
    )
    assert (
        demo_vehicle.rotor_blade_region_demo.mainsparweb_angle
        == demo_rotor_blade_region.mainsparweb_angle
        == 0.0
    )

    # -------------------------
    # Rotorcraft parameters (explicit + defaults)
    # -------------------------
    # explicit
    assert demo_vehicle.rotorcraft_demo.npass == demo_rotorcraft.npass == 15.0

    # default
    assert demo_vehicle.rotorcraft_demo.Wpass == demo_rotorcraft.Wpass == 0.0


def test_vehicle_component_attribute_identity(
    demo_vehicle, demo_wing, demo_rotor, demo_rotor_blade_region, demo_rotorcraft
):
    # -------------------------
    # Set some non-default values
    # -------------------------
    demo_wing.area = 2320.0
    demo_wing.set_parameter("span", 30.0)
    demo_wing.chord = 15.0
    # sweep, aspect_ratio left at defaults

    demo_rotor.vtip = 60.0
    demo_rotor.set_parameter("incidence", 10.0)
    demo_rotor.radius = 3.14
    demo_rotor.nblade = 5
    # omega, root_chord, pitch_angle left at defaults

    demo_rotor_blade_region.spanwise_start = 40.0
    demo_rotor_blade_region.set_parameter("spanwise_stop", 50.0)
    demo_rotor_blade_region.chord_start = 0.0
    demo_rotor_blade_region.chord_stop = 69.0
    # everything else left at defaults

    demo_rotorcraft.npass = 15.0
    # Wpass left at default

    # -------------------------
    # Attach components to vehicle
    # -------------------------
    demo_vehicle.add_component(demo_wing)
    demo_vehicle.add_component(demo_rotor)
    demo_vehicle.add_component(demo_rotor_blade_region)
    demo_vehicle.add_component(demo_rotorcraft)

    # -------------------------
    # Identity: access via name returns the same subsystem objects
    # -------------------------
    assert demo_vehicle.wing_demo is demo_wing
    assert demo_vehicle.rotor_demo is demo_rotor
    assert demo_vehicle.rotor_blade_region_demo is demo_rotor_blade_region
    assert demo_vehicle.rotorcraft_demo is demo_rotorcraft

    # -------------------------
    # Wing parameters (explicit + defaults) – identity
    # -------------------------
    # explicit
    assert demo_vehicle.wing_demo.area is demo_wing.area
    assert demo_vehicle.wing_demo.span is demo_wing.span
    assert demo_vehicle.wing_demo.chord is demo_wing.chord

    # defaults
    assert demo_vehicle.wing_demo.sweep is demo_wing.sweep
    assert demo_vehicle.wing_demo.aspect_ratio is demo_wing.aspect_ratio

    # -------------------------
    # Rotor parameters (explicit + defaults) – identity
    # -------------------------
    # explicit
    assert demo_vehicle.rotor_demo.vtip is demo_rotor.vtip
    assert demo_vehicle.rotor_demo.incidence is demo_rotor.incidence
    assert demo_vehicle.rotor_demo.radius is demo_rotor.radius
    assert demo_vehicle.rotor_demo.nblade is demo_rotor.nblade

    # defaults
    assert demo_vehicle.rotor_demo.omega is demo_rotor.omega
    assert demo_vehicle.rotor_demo.root_chord is demo_rotor.root_chord
    assert demo_vehicle.rotor_demo.pitch_angle is demo_rotor.pitch_angle

    # -------------------------
    # RotorBladeRegion parameters (explicit + defaults) – identity
    # -------------------------
    # explicit
    assert (
        demo_vehicle.rotor_blade_region_demo.spanwise_start
        is demo_rotor_blade_region.spanwise_start
    )
    assert (
        demo_vehicle.rotor_blade_region_demo.spanwise_stop
        is demo_rotor_blade_region.spanwise_stop
    )
    assert (
        demo_vehicle.rotor_blade_region_demo.chord_start
        is demo_rotor_blade_region.chord_start
    )
    assert (
        demo_vehicle.rotor_blade_region_demo.chord_stop
        is demo_rotor_blade_region.chord_stop
    )

    # defaults
    assert (
        demo_vehicle.rotor_blade_region_demo.sweep
        is demo_rotor_blade_region.sweep
    )
    assert (
        demo_vehicle.rotor_blade_region_demo.dihedral
        is demo_rotor_blade_region.dihedral
    )
    assert (
        demo_vehicle.rotor_blade_region_demo.twist_start
        is demo_rotor_blade_region.twist_start
    )
    assert (
        demo_vehicle.rotor_blade_region_demo.twist_stop
        is demo_rotor_blade_region.twist_stop
    )
    assert (
        demo_vehicle.rotor_blade_region_demo.airfoil_start
        is demo_rotor_blade_region.airfoil_start
    )
    assert (
        demo_vehicle.rotor_blade_region_demo.airfoil_stop
        is demo_rotor_blade_region.airfoil_stop
    )
    assert (
        demo_vehicle.rotor_blade_region_demo.horizontal_shift_start
        is demo_rotor_blade_region.horizontal_shift_start
    )
    assert (
        demo_vehicle.rotor_blade_region_demo.horizontal_shift_stop
        is demo_rotor_blade_region.horizontal_shift_stop
    )
    assert (
        demo_vehicle.rotor_blade_region_demo.vertical_shift_start
        is demo_rotor_blade_region.vertical_shift_start
    )
    assert (
        demo_vehicle.rotor_blade_region_demo.vertical_shift_stop
        is demo_rotor_blade_region.vertical_shift_stop
    )
    assert (
        demo_vehicle.rotor_blade_region_demo.mainsparweb_thickness
        is demo_rotor_blade_region.mainsparweb_thickness
    )
    assert (
        demo_vehicle.rotor_blade_region_demo.mainsparweb_angle
        is demo_rotor_blade_region.mainsparweb_angle
    )

    # -------------------------
    # Rotorcraft parameters (explicit + defaults) – identity
    # -------------------------
    # explicit
    assert demo_vehicle.rotorcraft_demo.npass is demo_rotorcraft.npass

    # default
    assert demo_vehicle.rotorcraft_demo.Wpass is demo_rotorcraft.Wpass


#Check adding component then querying
def test_vehicle_component_access(demo_vehicle,demo_wing):

    demo_vehicle.add_component(demo_wing)
    assert demo_vehicle.get_component('wing_demo') is demo_wing
    assert demo_vehicle.get_component('wing_demo').span==demo_wing.span
    assert demo_vehicle.get_component('wing_demo').span is demo_wing.span
    assert demo_vehicle.get_component('wing_demo') is demo_vehicle.wing_demo

# Ensure test_get_component_names is correct
def test_get_component_names(demo_vehicle,demo_wing,demo_rotorcraft):


    demo_vehicle.add_component(demo_wing)
    demo_vehicle.add_component(demo_rotorcraft)
    assert 'wing_demo' in demo_vehicle.get_component_names()
    assert 'rotorcraft_demo' in demo_vehicle.get_component_names()
    assert ['wing_demo','rotorcraft_demo']==demo_vehicle.get_component_names()

# Ensure that removing a component does what is expected

def test_remove_component(demo_vehicle,demo_wing,demo_rotorcraft):
    demo_vehicle.add_component(demo_wing)
    demo_vehicle.add_component(demo_rotorcraft)
    assert 'wing_demo' in demo_vehicle.get_component_names()
    demo_vehicle.remove_component(demo_wing)
    assert 'wing_demo' not in demo_vehicle.get_component_names()
    assert demo_vehicle.get_component_names()==['rotorcraft_demo']

#Check the components property
def test_components_property(demo_vehicle,demo_wing,demo_rotorcraft):
    demo_vehicle.add_component(demo_wing)
    demo_vehicle.add_component(demo_rotorcraft)
    assert demo_vehicle.components == [demo_wing,demo_rotorcraft]
    assert demo_vehicle.components[0] is demo_wing
    assert demo_vehicle.components[1] is demo_rotorcraft

#check expected positive behavior of check_all_parameter_types
def test_check_all_parameter_types_positive(demo_vehicle,demo_wing):
    demo_vehicle.add_component(demo_wing)

    demo_vehicle.check_all_parameter_types()


#Add a few components then query names

#Move to dict, then query names?
# Move to dict, then query names?
def test_vehicle_to_dict(demo_vehicle, demo_wing):
    # Attach one component
    demo_vehicle.add_component(demo_wing)

    demo_dict = demo_vehicle._to_dict()

    # ---- Vehicle-level checks ----
    # Name should match the Vehicle instance
    assert demo_dict["name"] == demo_vehicle.name
    assert isinstance(demo_dict["components"], list)
    assert len(demo_dict["components"]) == 1

    # ---- Component-level checks ----
    component_dict = demo_dict["components"][0]
    assert component_dict["name"] == demo_wing.name
    assert component_dict["subsystem_type"] == "Wing"
    assert isinstance(component_dict["parameters"], dict)

    wing_param_dict = component_dict["parameters"]

    # Make sure all expected parameters exist
    expected_param_names = {"area", "span", "chord", "sweep", "aspect_ratio"}
    assert expected_param_names.issubset(wing_param_dict.keys())

    # Helper to check one parameter entry
    def check_param(
        param_name,
        expected_value,
        expected_unit_type,
        expected_description,
        expected_dtype="float",
        expected_unit="",
    ):
        p = wing_param_dict[param_name]
        assert p["name"] == param_name
        assert p["value"] == expected_value
        assert p["dtype"] == expected_dtype
        assert p["unit_type"] == expected_unit_type
        assert p["unit"] == expected_unit
        assert p["description"] == expected_description

    # ---- Parameter-level checks (using Wing.setup defaults) ----
    check_param(
        "area",
        expected_value=0.0,
        expected_unit_type="area",
        expected_description="Wing area",
    )
    check_param(
        "span",
        expected_value=0.0,
        expected_unit_type="length",
        expected_description="Wing span",
    )
    check_param(
        "chord",
        expected_value=0.0,
        expected_unit_type="length",
        expected_description="Wing chord",
    )
    check_param(
        "sweep",
        expected_value=0.0,
        expected_unit_type="angle",
        expected_description="Wing sweep angle",
    )
    check_param(
        "aspect_ratio",
        expected_value=1.0,
        expected_unit_type="angle",
        expected_description="Wing sweep angle",
    )

# Test string access
def test_vehicle__str__(demo_vehicle, demo_wing, demo_rotorcraft):
    demo_vehicle.add_component(demo_wing)
    demo_vehicle.add_component(demo_rotorcraft)
    assert demo_vehicle.__str__() == 'Vehicle(name=vehicle_demo, components=[wing_demo (Wing), rotorcraft_demo (Rotorcraft)])'

def test_vehicle_from_dict_roundtrip(demo_wing, demo_rotorcraft):
    # -------------------------
    # Create original vehicle with components
    # -------------------------
    demo_wing.area = 123.0
    demo_wing.span = 42.0
    demo_rotorcraft.npass = 7.0

    original = Vehicle("demo_vehicle_from_dict", [demo_wing, demo_rotorcraft])

    # Sanity: original vehicle
    assert original.name == "demo_vehicle_from_dict"
    assert original.get_component_names() == ["wing_demo", "rotorcraft_demo"]

    # -------------------------
    # Serialize to dict
    # -------------------------
    data = original._to_dict()
    assert isinstance(data, dict)
    assert data["name"] == original.name

    # -------------------------
    # Reconstruct from dict
    # -------------------------
    rebuilt = Vehicle._from_dict(data)

    # Vehicle-level checks
    assert isinstance(rebuilt, Vehicle)
    assert rebuilt is not original  # new instance
    assert rebuilt.name == original.name

    # Component-level checks: same names and types
    rebuilt_names = rebuilt.get_component_names()
    assert set(rebuilt_names) == {"wing_demo", "rotorcraft_demo"}

    rebuilt_wing = rebuilt.get_component("wing_demo")
    rebuilt_rotorcraft = rebuilt.get_component("rotorcraft_demo")

    assert isinstance(rebuilt_wing, Wing)
    assert isinstance(rebuilt_rotorcraft, Rotorcraft)

    # Parameter values preserved (not identity, just equality)
    assert rebuilt_wing.area == demo_wing.area == 123.0
    assert rebuilt_wing.span == demo_wing.span == 42.0
    assert rebuilt_rotorcraft.npass == demo_rotorcraft.npass == 7.0

    # Make sure new components are distinct objects from originals
    assert rebuilt_wing is not demo_wing
    assert rebuilt_rotorcraft is not demo_rotorcraft

#----------------------------------------------------------Negative checks
def test_component_access_wrong_name(demo_vehicle, demo_wing):

    demo_vehicle.add_component(demo_wing)
    name = 'random_name'
    msg = f'Component with name {name} not found in vehicle {demo_vehicle._name}'

    with  pytest.raises(ValueError, match=re.escape(msg)):
        demo_vehicle.get_component(name)




def test_component_duplicate_names():

    test_wing1 = Wing('duplicate_name')
    test_wing2 = Wing('duplicate_name')

    assert test_wing1 is  not test_wing2
    with  pytest.raises(NameError):
        sample_veh = Vehicle('SampleVehicle',[test_wing1, test_wing2])

    sample_veh = Vehicle('SampleVehicle',[test_wing1])
    with  pytest.raises(NameError):
        sample_veh.add_component(test_wing2)

    sample_veh = Vehicle('SampleVehicle')
    sample_veh.add_component(test_wing1)
    with  pytest.raises(NameError):
        sample_veh.add_component(test_wing2)





@pytest.mark.parametrize("var_name", [
    (67),
    ('has]init'),
    ('has[init')
])
def test_vehicle_invalid_name(var_name):

    with  pytest.raises(ValueError):
        bad_name_veh = Vehicle(var_name)


#Test non existent attribute access
def test_bad_component_accesss(demo_vehicle, demo_rotor):

    with pytest.raises(AttributeError):
        demo_vehicle.non_existent_attribute




if __name__ == '__main__':

    test_component_duplicate_names()
