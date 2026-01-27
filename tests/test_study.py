import pytest
import  sys
import platform
from importlib.metadata import distributions
from pathlib import  Path
import subprocess
import copy
import numpy as np

from uam4x.study import Study
from uam4x.solution import Solution
from uam4x.control import Control
from uam4x.vehicle import Vehicle, VehicleManager
from uam4x.control import ParameterSweep
from uam4x.components import Component, _Mode, Parameter, Wing, Rotor, RotorBladeRegion, Aircraft



SAMPLE_PARAM_STATE = {
        "vehicles.multi_objective_vehicle.weighting_factor.weighting_factor": 0,
        "optimization": True,
        "type": "Optimization",
        "des_vars": {
            "vehicles.multi_objective_vehicle.wing1.aspect_ratio": {
                "lower": 1.0,
                "upper": 10.0,
                "indices": None
            },
            "vehicles.multi_objective_vehicle.wing1.span": {
                "lower": 100.0,
                "upper": 1000.0,
                "indices": None
            }
        },
        "constraints": {
            "calc_area.area": {
                "lower": 49999.0,
                "upper": 50001.0,
                "indices": None
            }
        },
        "objective_name": "calc_objective.objective",
        "calc_area.area": 49998.999999998494,
        "calc_objective.CD": 0.042767071432663215,
        "calc_objective.wing_weight": 88.92178070843795,
        "calc_objective.objective": 88.92178070843795,
        "calc_area.area_array_test[1,:]": [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0
        ],
        "vehicles.multi_objective_vehicle.wing1.aspect_ratio": 1.0000000000000364,
        "vehicles.multi_objective_vehicle.wing1.span": 223.6045616708217
    }

@pytest.fixture
def sample_param_state():
    return copy.deepcopy(SAMPLE_PARAM_STATE)

    
#--------------------------------------------------------Study
@pytest.fixture
def sample_study():
    class ExtraComponent(Component):
        def setup(self):
            self.add_parameter(
                name='sample_array',
                default_value=np.arange(1, 10),
                dtype=np.ndarray,
            )

    EC = ExtraComponent('array_based_component')
    W = Wing('swept_wing')
    R = Rotor('rotor_one')
    RBR = RotorBladeRegion('rotor_region_one')
    A = Aircraft('sample_uam')

    # --- Wing params ---
    W.set_parameter("area", 12.5)
    W.set_parameter("span", 10.0)
    W.set_parameter("sweep", 25.0)

    # --- Rotor params ---
    R.set_parameter("vtip", 600.0)
    R.set_parameter("radius", 2.5)
    R.set_parameter("nblade", 4)
    R.set_parameter("tilt", 0.0)

    # --- RotorBladeRegion params ---
    RBR.set_parameter("spanwise_start", 0.1)
    RBR.set_parameter("spanwise_stop", 0.9)
    RBR.set_parameter("twist_start", -5.0)

    # --- Aircraft params ---
    A.set_parameter("npass", 6)
    A.set_parameter("wpass", 180.0)

    # --- Rotor mode ---
    R.add_mode('vertical_rotor')
    R.vertical_rotor.set_parameter('rpm', 500.)
    R.vertical_rotor.set_parameter('tilt', 90.0)

    # --- Vehicles ---
    Vehicle1 = Vehicle('commercial_uam')
    Vehicle1.add_component(W)
    Vehicle1.add_component(R)
    Vehicle1.add_component(EC)

    Vehicle2 = Vehicle('military_uam')
    Vehicle2.add_component(RBR)
    Vehicle2.add_component(A)

    # --- Study ---
    S = Study()
    S.vehicles.append(Vehicle1)
    S.add_vehicle(Vehicle2)
    
    return S

def test_Study_instance():
    
    s = Study()
    
    assert isinstance(s.vehicles, VehicleManager)
    assert isinstance(s.solution, Solution)
    assert isinstance(s.control, Control)
    
    
    
def test_Study_get_environment():
    s = Study()
    s._capture_environment('sample_trade')

    # New location of the environment info
    assert 'sample_trade' in s._runtime_meta_data
    assert 'environment' in s._runtime_meta_data['sample_trade']
    env = s._runtime_meta_data['sample_trade']['environment']

    assert isinstance(env, dict)
    assert set(env.keys()) == {'Machine', 'Python', 'Packages','Timestamp'}

    machine = env['Machine']

    # Machine info has expected keys
    for key in ["os_system", "os_release", "os_version", "platform", "machine", "processor"]:
        assert key in machine
        assert isinstance(machine[key], str)

    # Python info looks sane
    py = env["Python"]
    assert "python_spec" in py and isinstance(py["python_spec"], str)
    assert "python_executable" in py and isinstance(py["python_executable"], str)

    # Packages is a dict of name -> version strings (may be empty but usually not)
    pkgs = env["Packages"]
    assert isinstance(pkgs, dict)
    for name, version in list(pkgs.items())[:10]:  # sample a few
        assert isinstance(name, str)
        assert isinstance(version, str)

        
def test_Study_add_trade():
    sample_param_dict = {'sample_param1': [1.0, 10.0, 100.0, 1000.0, 10000.0],
                    'sample_param2': [6.0, 7.0, 67.0, 6767., 676767.]}
    
    PS = ParameterSweep(sample_param_dict)
    
    S = Study()
    
    S.add_trade('sample_trade', PS)
    
    assert S.control.trade_studies['sample_trade'] is PS
    

def test_Study_add_vehicle():
    v1 = Vehicle('sample_vehicle1')
    v2 = Vehicle('sample_vehicle2')
    
    S = Study()
    S.add_vehicle(v1)
    S.add_vehicle(v2)
    
    assert S.vehicles.sample_vehicle1 is v1
    assert S.vehicles.get_vehicle('sample_vehicle2') is v2
    
def test_Study_get_vehicle():
    v1 = Vehicle('sample_vehicle1')
    v2 = Vehicle('sample_vehicle2')
    
    S = Study()
    S.add_vehicle(v1)
    S.add_vehicle(v2)
    
    assert S.get_vehicle('sample_vehicle1') is v1
    assert S.get_vehicle('sample_vehicle2') is v2
    
#Test to dict

#Test from dict

#Test save to json

#Test load from json
    
def test_Study_get_plugin_input_and_output_names_positive():
    S = Study()	

    # --- Add seven links in UAM4x style: add_link(source, target) ---

    # 1. stepA → stepB (stepB is dependent)
    S.solution.add_link("stepA.out1", "stepB.in1")

    # 2. stepB → stepA (stepB is independent)
    S.solution.add_link("stepB.out2", "stepA.in2")

    # 3. stepC → stepB (dependent)
    S.solution.add_link("stepC.signal", "stepB.command")

    # 4. stepB → vehicles (independent)
    S.solution.add_link("stepB.thrust", "vehicles.input")

    # 5. vehicles → stepB (dependent)
    S.solution.add_link("vehicles.mass", "stepB.load")

    # 6. Indexed dependent target (brackets removed)
    S.solution.add_link("stepA.force", "stepB.vec")

    # 7. Indexed independent source (brackets removed)
    S.solution.add_link("stepB.state[2]", "stepC.status")

    # --- Add a tracked output belonging to stepB ---
    S.solution.add_tracked_output("stepB.pressure")
    S.solution.add_tracked_output("stepA.out1")

    # ----------------------------
    # Compute names for stepB
    # ----------------------------
    inputs, outputs = S.get_plugin_input_and_output_names("stepB")

    # --- Expected results ---

    expected_inputs = {
        "stepB.in1",      # from stepA.out1 → stepB.in1
        "stepB.command",  # from stepC.signal → stepB.command
        "stepB.load",     # from vehicles.mass → stepB.load
        "stepB.vec",      # from stepA.force → stepB.vec[1,3] (bracket stripped)
    }

    expected_outputs = {
        "stepB.out2",     # from stepB.out2 → stepA.in2
        "stepB.thrust",   # from stepB.thrust → vehicles.input
        "stepB.state",    # from stepB.state[2] → stepC.status (bracket stripped)
        "stepB.pressure", # tracked output
    }

    assert inputs == expected_inputs
    assert outputs == expected_outputs
    
def test_Study_get_value(sample_study):
    
    
    S = sample_study
    
    # ---------- Wing parameters ----------
    assert S.get_value('vehicles.commercial_uam.swept_wing.area') == 12.5
    assert S.get_value('vehicles.commercial_uam.swept_wing.span') == 10.0
    assert S.get_value('vehicles.commercial_uam.swept_wing.sweep') == 25.0

    # ---------- Rotor parameters ----------
    assert S.get_value('vehicles.commercial_uam.rotor_one.vtip') == 600.0
    assert S.get_value('vehicles.commercial_uam.rotor_one.radius') == 2.5
    assert S.get_value('vehicles.commercial_uam.rotor_one.nblade') == 4
    assert S.get_value('vehicles.commercial_uam.rotor_one.tilt') == 0.0

    # ---------- Rotor mode parameters ----------
    assert S.get_value('vehicles.commercial_uam.rotor_one.vertical_rotor.rpm') == 500.0
    assert S.get_value('vehicles.commercial_uam.rotor_one.vertical_rotor.tilt') == 90.0

    # ---------- RotorBladeRegion parameters ----------
    assert S.get_value('vehicles.military_uam.rotor_region_one.spanwise_start') == 0.1
    assert S.get_value('vehicles.military_uam.rotor_region_one.spanwise_stop') == 0.9
    assert S.get_value('vehicles.military_uam.rotor_region_one.twist_start') == -5.0

    # ---------- Aircraft parameters ----------
    assert S.get_value('vehicles.military_uam.sample_uam.npass') == 6
    assert S.get_value('vehicles.military_uam.sample_uam.wpass') == 180.0

    # ---------- ExtraComponent: whole array ----------
    full_arr = S.get_value(
        'vehicles.commercial_uam.array_based_component.sample_array'
    )
    expected_arr = np.arange(1, 10)
    assert np.array_equal(full_arr, expected_arr)

    # ---------- ExtraComponent: subindex of array ----------
    elem = S.get_value(
        'vehicles.commercial_uam.array_based_component.sample_array[3]'
    )
    assert elem == expected_arr[3]    

def test_Study_get_all_names_by_category(sample_study):
    # sample_study fixture returns ONLY the Study instance
    S = sample_study

    # -------------------------
    # Mock links + tracked_outputs on this Study
    # -------------------------
    S.solution.links = [
        # (independent/output, dependent/input)
        ('step1.weights.wing_weight',      'step2.performance.weight_in'),
        ('step1.mass.mtow',                'step2.performance.mtow_in'),
        ('vehicles.commercial_uam.swept_wing.span',
         'step3.performance.span_in'),
    ]

    S.solution.tracked_outputs = [
        'step2.performance.weight_in',          # valid step output
        'step4.sensitivity.grad_out',           # valid step output
        'vehicles.commercial_uam.rotor_one.vtip',  # NOT a step -> should be ignored
    ]

    result = S.get_all_names_by_category()

    vehicle_names = result['vehicle_names']
    step_inputs   = result['step_inputs']
    step_outputs  = result['step_outputs']

    # -------------------------
    # Vehicle names: string-based checks only (no direct objects)
    # These must match how your fixture builds the Study.
    # -------------------------

    # Wing (commercial_uam)
    assert 'vehicles.commercial_uam.swept_wing.area' in vehicle_names
    assert 'vehicles.commercial_uam.swept_wing.span' in vehicle_names
    assert 'vehicles.commercial_uam.swept_wing.sweep' in vehicle_names

    # Rotor (commercial_uam)
    assert 'vehicles.commercial_uam.rotor_one.vtip'   in vehicle_names
    assert 'vehicles.commercial_uam.rotor_one.radius' in vehicle_names
    assert 'vehicles.commercial_uam.rotor_one.nblade' in vehicle_names
    assert 'vehicles.commercial_uam.rotor_one.tilt'   in vehicle_names

    # Rotor mode (vertical_rotor) – this also validates using mode._parameters
    assert (
        'vehicles.commercial_uam.rotor_one.vertical_rotor.rpm'
        in vehicle_names
    )
    assert (
        'vehicles.commercial_uam.rotor_one.vertical_rotor.tilt'
        in vehicle_names
    )

    # RotorBladeRegion (military_uam)
    assert (
        'vehicles.military_uam.rotor_region_one.spanwise_start'
        in vehicle_names
    )
    assert (
        'vehicles.military_uam.rotor_region_one.spanwise_stop'
        in vehicle_names
    )
    assert (
        'vehicles.military_uam.rotor_region_one.twist_start'
        in vehicle_names
    )

    # Aircraft (military_uam)
    assert 'vehicles.military_uam.sample_uam.npass' in vehicle_names
    assert 'vehicles.military_uam.sample_uam.wpass' in vehicle_names

    # Extra array-based component (assuming fixture added this)
    assert (
        'vehicles.commercial_uam.array_based_component.sample_array'
        in vehicle_names
    )

    # -------------------------
    # step_inputs checks (from links: dependents that start with 'step')
    # -------------------------
    assert set(step_inputs) == {
        'step2.performance.weight_in',
        'step2.performance.mtow_in',
        'step3.performance.span_in',
    }

    # -------------------------
    # step_outputs checks
    #   - from links: independents starting with 'step'
    #   - from tracked_outputs: entries starting with 'step'
    # -------------------------
    assert set(step_outputs) == {
        'step1.weights.wing_weight',
        'step1.mass.mtow',
        'step2.performance.weight_in',   # from both links + tracked_outputs
        'step4.sensitivity.grad_out',
    }

    # -------------------------
    # Sanity: names are actually usable via S.get_value
    # (this only needs the Study instance)
    # -------------------------

    # simple scalar
    assert S.get_value('vehicles.commercial_uam.swept_wing.span') == 10.0

    # full array
    arr = S.get_value(
        'vehicles.commercial_uam.array_based_component.sample_array'
    )
    assert np.array_equal(arr, np.arange(1, 10))

    # indexed element via bracket handling
    elem = S.get_value(
        'vehicles.commercial_uam.array_based_component.sample_array[3]'
    )
    assert elem == np.arange(1, 10)[3]

def test_Study_get_all_names(sample_study):
    # sample_study fixture returns ONLY the Study instance
    S = sample_study

    # Wire up the same links / tracked_outputs used in the category test
    S.solution.links = [
        ('step1.weights.wing_weight',      'step2.performance.weight_in'),
        ('step1.mass.mtow',                'step2.performance.mtow_in'),
        ('vehicles.commercial_uam.swept_wing.span',
         'step3.performance.span_in'),
    ]
    S.solution.tracked_outputs = [
        'step2.performance.weight_in',
        'step4.sensitivity.grad_out',
        'vehicles.commercial_uam.rotor_one.vtip',  # non-step, should be ignored
    ]

    # Call the category method and the wrapper
    all_by_cat = S.get_all_names_by_category()
    vehicle_names = all_by_cat['vehicle_names']
    step_inputs   = all_by_cat['step_inputs']
    step_outputs  = all_by_cat['step_outputs']

    all_names = S.get_all_names()

    # 1) It should be exactly the concatenation order
    assert all_names == vehicle_names + step_inputs + step_outputs

    # 2) No name is dropped
    for name in vehicle_names:
        assert name in all_names
    for name in step_inputs:
        assert name in all_names
    for name in step_outputs:
        assert name in all_names

        from uam4x.study import Study
        from uam4x.vehicle import Vehicle
        from uam4x.components import Wing
        
        
def test_Study_from_dict_roundtrip():
    # --- Build original study ---
    study = Study()

    # Vehicle + component
    veh = Vehicle("TestVehicle")
    wing = Wing("left_wing")
    wing.set_parameter("span", 5.0)
    wing.set_parameter("area", 20.0)
    veh.add_component(wing)

    study.add_vehicle(veh)

    # Optional: add a mode + override to ensure it survives component._from_dict
    mode = wing.add_mode("cruise")
    mode.set_parameter("span", 7.5)

    # --- Convert to dict and reconstruct ---
    data = study._to_dict()
    rebuilt = Study._from_dict(data)

    # --- Basic type checks ---
    assert isinstance(rebuilt, Study)

    # Vehicle exists and has same name
    rebuilt_veh = rebuilt.get_vehicle("TestVehicle")
    assert isinstance(rebuilt_veh, Vehicle)
    assert rebuilt_veh.name == "TestVehicle"

    # We expect a new Vehicle object, not the same instance
    assert rebuilt_veh is not veh

    # Find the wing on the rebuilt vehicle
    rebuilt_wings = [c for c in rebuilt_veh.components if isinstance(c, Wing)]
    assert len(rebuilt_wings) == 1
    rebuilt_wing = rebuilt_wings[0]

    # Component name preserved
    assert rebuilt_wing.name == "left_wing"

    # Base parameter values preserved
    assert rebuilt_wing.span == 5.0
    assert rebuilt_wing.area == 20.0

    # Mode and overrides preserved
    assert "cruise" in rebuilt_wing._modes
    rebuilt_mode = rebuilt_wing.cruise
    assert rebuilt_mode.get_parameter("span") == 7.5

    # And ensure mode override is not the same dict object
    assert rebuilt_mode is not mode

if __name__ == "__main__":
    test_Study_get_value()