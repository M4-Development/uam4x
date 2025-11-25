import numpy as np
from uam4x.execution_engine import  Engine
from uam4x.study import Study
from uam4x.vehicle import Vehicle
from uam4x.components import  Component, Wing, register_component
from uam4x.solution import Plugin, register_plugin, Solution
import pytest
from uam4x_openvsp.plugin import OpenvspPlugin
from uam4x.control import Optimization, ParameterSweep
import numpy as np
import os
#@pytest.fixture
#def sample_study():
    
    #class Aero(Component):
        #def setup(self):
            ##Flow params:
            #self.add_parameter('Mach', default_value=0.0, dtype=float)
            #self.add_parameter('Q', default_value=2.56, dtype=float)
            ##Mass params:
            #self.add_parameter('Mass', 1000., dtype=float)
            #self.add_parameter('Iyy', 3000., dtype=float)
            
            ##Derivatives
            #self.add_parameter('ClA', default_value=2*np.pi, dtype=float)
            #self.add_parameter('CMa', default_value=-3.5., dtype=float)
            #self.add_parameter('vert_control_derivs', np.array([5., 6.]), float)
            #self.add_parameter('pitch_control_derivs', np.array([1., 2.]), float)
            
            ##Fixed aero states (note rates are assumed zero already and 2D problem)
            #self.add_parameter('AOA', 0.0, dtype=float)
            #self.add_parameter('URDD3', default_value=386.09, dtype=float)

            
            ##Free aero states: angle of flaps and elevators 
            #self.add_parameter('control_states', default_value=np.zeros(), float)            

    
    #W = Wing('sample_wing')
    #W.set_parameter('chord', 70.)
    #W.set_parameter('span', 400.)
    #W.set_parameter('area', 2800.)
    
    
    #class VerticalTrim(Plugin):
        #def setup_plugin(self):
            ##flight point: 
            #self.declare_input('Mach', float)
            #self.declare_input('Q', float)
            #self.declare_input('AOA', float)
            #self.declare_input('URDD3', float)
            #self.declare_input('Mass', float)
            #self.declare_input('CLA', float)
            #self.declare_input('DPitch', float)
            
            #self.declare_input('chord', float)
            #self.declare_input('span', float)
            #self.declare_input('area', float)
            #self.declare_input('vert_control_derivs', np.ndarray, shape=(2, ))
            
            #self.declare_output('objective', float)
            
        #def run_plugin(self, links, workdir):
            
            ##solve simple vertical trim equation: qS*(CSPITCH*dpitch+ANGLEA*AOA)-mg=0 so steady level flight at a presribed angle of attack
            ##what elevator needed to balance. 
            #obj = links['Q'] * links['area'] * (links['CSPITCH'] * links['Dpitch'])
            #obj += links['Q'] * links['area'] *(links['CLA'] * links['AOA']) - links['Mass'] * links['URDD3']
            
            #return {'objective': obj}
            
    #class PitchTrim(Plugin):
        #def setup_plugin(self):
            
@pytest.fixture
def sample_engine_setup():
    """
    Toy Study/Engine for testing Engine.check_* methods.

    - Vehicle:
        * CustomComponent with 5 scalar params + 1 array param (test_array).
        * Wing with chord/span/area.
    - Plugins:
        * Plugin1:
            - Inputs: 3 vehicle scalars, wing area, slice of test_array[1, :]
            - Outputs: 2 scalars, 1 dead-end scalar, 1 array_out (2x2)
        * Plugin2:
            - Inputs: Plugin1.out1_to_p2, 2 vehicle scalars, all wing geom,
                      slice of Plugin1.array_out[0, :]
            - Outputs: 1 scalar to Plugin3, 1 dead-end scalar
        * Plugin3:
            - Inputs: 2 vehicle scalars, Plugin1.out2_to_p3, Plugin2.out_to_p3
            - Outputs: objective, dead-end scalar

    Array links (only the *source* is indexed):
        vehicles.sample_vehicle.custom_component.test_array[1,:] -> plugin1.array_slice_in
        plugin1.array_out[0,:] -> plugin2.array_from_p1_slice
    """

    # -------------------------
    # 1) Vehicle + components
    # -------------------------
    class CustomComponent(Component):
        def setup(self):
            # 5 scalar parameters
            self.add_parameter("p1", default_value=1.0, dtype=float)
            self.add_parameter("p2", default_value=2.0, dtype=float)
            self.add_parameter("p3", default_value=3.0, dtype=float)
            self.add_parameter("p4", default_value=4.0, dtype=float)
            self.add_parameter("p5", default_value=5.0, dtype=float)

            # Array parameter to be sliced in links (shape (3,3))
            self.add_parameter(
                "test_array",
                default_value=np.array(
                    [
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0],
                    ]
                ),
                dtype=np.ndarray,
            )

    custom_comp = CustomComponent("custom_component")

    wing = Wing("sample_wing")
    wing.set_parameter("chord", 10.0)
    wing.set_parameter("span", 50.0)
    wing.set_parameter("area", 500.0)

    vehicle = Vehicle("sample_vehicle", [custom_comp, wing])

    # -------------------------
    # 2) Plugins
    # -------------------------

    class Plugin1(Plugin):
        """
        Plugin1:
        - Inputs:
            veh_p1, veh_p2, veh_p3 (vehicle scalars)
            wing_area
            array_slice_in (slice of test_array[1,:], shape (3,))
        - Outputs:
            out1_to_p2 (float) -> Plugin2
            out2_to_p3 (float) -> Plugin3
            dead1 (float, dead-end)
            array_out (np.ndarray, shape (2,2)) -> sliced to Plugin2
        """
        def setup_plugin(self, plugin_options=None):
            # vehicle parameters
            self.declare_input("veh_p1", float)
            self.declare_input("veh_p2", float)
            self.declare_input("veh_p3", float)

            # wing geometry
            self.declare_input("wing_area", float)

            # slice of test_array[1,:] (shape (3,))
            self.declare_input("array_slice_in", np.ndarray, shape=(3,))

            # outputs used by other plugins
            self.declare_output("out1_to_p2", float)
            self.declare_output("out2_to_p3", float)

            # dead-end output
            self.declare_output("dead1", float)

            # array output to be sliced in a link
            self.declare_output("array_out", np.ndarray, shape=(2, 2))

        def run_plugin(self, links, workdir):
            # No actual computation needed for these tests
            pass

    class Plugin2(Plugin):
        """
        Plugin2:
        - Inputs:
            from_p1 (Plugin1.out1_to_p2)
            veh_p4, veh_p5
            wing_chord, wing_span, wing_area
            array_from_p1_slice (Plugin1.array_out[0,:], shape (2,))
        - Outputs:
            out_to_p3 (float) -> Plugin3
            dead2 (float, dead-end)
        """
        def setup_plugin(self, plugin_options=None):
            # from Plugin1
            self.declare_input("from_p1", float)

            # vehicle parameters
            self.declare_input("veh_p4", float)
            self.declare_input("veh_p5", float)

            # wing geometry
            self.declare_input("wing_chord", float)
            self.declare_input("wing_span", float)
            self.declare_input("wing_area", float)

            # slice of Plugin1.array_out[0,:], shape (2,)
            self.declare_input("array_from_p1_slice", np.ndarray, shape=(2,))

            # output to Plugin3
            self.declare_output("out_to_p3", float)

            # dead-end output
            self.declare_output("dead2", float)

        def run_plugin(self, links, workdir):
            pass

    class Plugin3(Plugin):
        """
        Plugin3:
        - Inputs:
            veh_p1, veh_p2
            from_p1_out2 (Plugin1.out2_to_p3)
            from_p2_out (Plugin2.out_to_p3)
        - Outputs:
            objective
            dead3 (dead-end)
        """
        def setup_plugin(self, plugin_options=None):
            # vehicle parameters
            self.declare_input("veh_p1", float)
            self.declare_input("veh_p2", float)

            # from Plugin1
            self.declare_input("from_p1_out2", float)

            # from Plugin2
            self.declare_input("from_p2_out", float)

            # "real" output
            self.declare_output("objective", float)

            # dead-end output
            self.declare_output("dead3", float)

        def run_plugin(self, links, workdir):
            pass

    # -------------------------
    # 3) Study + Solution + links
    # -------------------------
    work_dir = os.path.join(os.getcwd(), "work_dir_engine_tests")
    os.makedirs(work_dir, exist_ok=True)

    study = Study()
    study.add_vehicle(vehicle)

    solution = study.solution

    # NOTE: use same signature style as your real code:
    # study.solution.add_step(PluginClass, "step_name", plugin_options=None)
    solution.add_step(Plugin1, "plugin1")
    solution.add_step(Plugin2, "plugin2")
    solution.add_step(Plugin3, "plugin3")

    # Vehicle parameter paths
    base = "vehicles.sample_vehicle.custom_component"
    wing_base = "vehicles.sample_vehicle.sample_wing"

    # -------- Plugin1 scalar inputs --------
    solution.add_link(f"{base}.p1", "plugin1.veh_p1")
    solution.add_link(f"{base}.p2", "plugin1.veh_p2")
    solution.add_link(f"{base}.p3", "plugin1.veh_p3")
    solution.add_link(f"{wing_base}.area", "plugin1.wing_area")

    # -------- array slice param -> Plugin1 --------
    # test_array has shape (3, 3); row index 1 gives shape (3,)
    solution.add_link(f"{base}.test_array[1,:]", "plugin1.array_slice_in")

    # -------- Plugin2 inputs --------
    # from Plugin1
    solution.add_link("plugin1.out1_to_p2", "plugin2.from_p1")

    # vehicle scalars
    solution.add_link(f"{base}.p4", "plugin2.veh_p4")
    solution.add_link(f"{base}.p5", "plugin2.veh_p5")

    # wing geom
    solution.add_link(f"{wing_base}.chord", "plugin2.wing_chord")
    solution.add_link(f"{wing_base}.span", "plugin2.wing_span")
    solution.add_link(f"{wing_base}.area", "plugin2.wing_area")

    # array slice plugin output -> Plugin2
    # array_out has shape (2, 2); row index 0 gives shape (2,)
    solution.add_link("plugin1.array_out[0,:]", "plugin2.array_from_p1_slice")

    # -------- Plugin3 inputs --------
    solution.add_link(f"{base}.p1", "plugin3.veh_p1")
    solution.add_link(f"{base}.p2", "plugin3.veh_p2")
    solution.add_link("plugin1.out2_to_p3", "plugin3.from_p1_out2")
    solution.add_link("plugin2.out_to_p3", "plugin3.from_p2_out")

    # -------------------------
    # 4) Instantiate plugins dict for Engine.check_model
    # -------------------------
    plugins = {}
    for step_name, step_info in study.solution.steps.items():
        plugin_cls = step_info["plugin"]
        plugin_opts = step_info.get("plugin_options", {})
        plugin = plugin_cls()
        plugin.setup_plugin(plugin_opts)
        plugins[step_name] = plugin

    engine = Engine(study, work_dir=str(work_dir))

    # Minimal state you can tweak in tests
    state = {
        "optimization": False,
        "des_vars": {},
        "constraints": {},
        "objective_name": "plugin3.objective",
    }

    return study, engine, plugins, state


#------Positive checks
def test_check_link_types_good(sample_engine_setup):
    study, engine, plugins, state = sample_engine_setup
    
    engine.check_link_types(plugins)
    
def test_check_links_to_plugins_good(sample_engine_setup):
    study, engine, plugins, state = sample_engine_setup
    
    engine.check_links_to_plugins(plugins)
    


def test_check_link_types_indexed_array_to_scalar_ok(sample_engine_setup):
    """
    Positive test for check_link_types:

    Indexed ndarray output (e.g. plugin1.array_out[0]) feeding a continuous
    scalar input. This should hit:

        indexes_specified == True
        output_type == np.ndarray
        check_continuous(input_type) == True
        input_type != np.ndarray

    and therefore be allowed with no error.
    """
    study, engine, plugins, state = sample_engine_setup

    # Modify Plugin3 to declare a scalar input that will receive the indexed array element
    class Plugin3WithScalar(plugins["plugin3"].__class__):
        def setup_plugin(self, plugin_options=None):
            # original inputs:
            self.declare_input("veh_p1", float)
            self.declare_input("veh_p2", float)
            self.declare_input("from_p1_out2", float)
            self.declare_input("from_p2_out", float)

            # new scalar input fed by an indexed array element
            self.declare_input("scalar_from_array", float)

            # outputs
            self.declare_output("objective", float)
            self.declare_output("dead3", float)

        def run_plugin(self, links, workdir):
            pass

    # Swap plugin3 in the study with our modified class
    study.solution.steps["plugin3"]["plugin"] = Plugin3WithScalar
    plugins["plugin3"] = Plugin3WithScalar()
    plugins["plugin3"].setup_plugin(study.solution.steps["plugin3"]["plugin_options"])

    # Add the link using scalar-style indexing:
    # array_out is declared as np.ndarray, indexing [0] is conceptually a scalar
    study.solution.add_link("plugin1.array_out[0]", "plugin3.scalar_from_array")

    # This should not raise
    engine.check_link_types(plugins)
    
    
    
def test_check_link_indices_ok(sample_engine_setup):
    """
    Existing sample_engine_setup has:
      - parameter slice: vehicles...test_array[1,:] -> plugin1.array_slice_in (3,) -> (3,)
      - plugin slice: plugin1.array_out[0,:] -> plugin2.array_from_p1_slice (2,) -> (2,)
    These should all pass check_link_indices with no error.
    """
    study, engine, plugins, state = sample_engine_setup

    # Should not raise
    engine.check_link_indices(plugins)
 
 
def test_check_link_indices_plugin_src_value_array_ok(sample_engine_setup):
    """
    Force plugin1.array_out to have a concrete value so that the
    'a is not None' branch is executed and indexing succeeds.
    """
    study, engine, plugins, state = sample_engine_setup

    # Force a non-None value for the output being indexed
    plugins["plugin1"].plugin_output_names["array_out"]["value"] = np.ones((2, 2))
    # Shape can stay as-is or be set explicitly
    plugins["plugin1"].plugin_output_names["array_out"]["shape"] = (2, 2)

    # This uses the existing link:
    #   plugin1.array_out[0,:] -> plugin2.array_from_p1_slice
    # and should not raise
    engine.check_link_indices(plugins)
    
    
def test_check_link_indices_plugin_src_declared_shape_ok(sample_engine_setup):
    """
    Force plugin1.array_out to have value=None but shape=(2,2) so that the
    'a_shape is not None' branch is exercised and indexing succeeds via np.ones.
    """
    study, engine, plugins, state = sample_engine_setup

    plugins["plugin1"].plugin_output_names["array_out"]["value"] = None
    plugins["plugin1"].plugin_output_names["array_out"]["shape"] = (2, 2)

    # Again uses plugin1.array_out[0,:] -> plugin2.array_from_p1_slice
    engine.check_link_indices(plugins)
    
    
    
def test_check_link_indices_scalar_index_to_float_ok(sample_engine_setup):
    """
    Index a single scalar from a plugin array output and send it to a float input.
    This should take the src_size == 1 and b_type == float branch and pass.
    """
    study, engine, plugins, state = sample_engine_setup

    # Declare a new float input on plugin2
    plugins["plugin2"].plugin_input_names["scalar_from_p1"] = {
        "type": float,
        "value": None,
        "shape": None,
    }

    # Add the corresponding link at the solution level
    study.solution.add_link("plugin1.array_out[0,0]", "plugin2.scalar_from_p1")

    # Make sure array_out has a usable shape/value
    plugins["plugin1"].plugin_output_names["array_out"]["value"] = np.ones((2, 2))
    plugins["plugin1"].plugin_output_names["array_out"]["shape"] = (2, 2)

    # Should not raise
    engine.check_link_indices(plugins)

    
    
    

def test_check_desvars_scalar_ok(sample_engine_setup):
    """
    Design var on a scalar vehicle parameter (float) with valid scalar bounds.
    Expect: no exception.
    """
    study, engine, plugins, _ = sample_engine_setup

    # Design var name corresponds to a float parameter on the CustomComponent
    dv_name = "vehicles.sample_vehicle.custom_component.p1"

    state = {
        "des_vars": {
            dv_name: {
                "lower": 0.0,
                "upper": 10.0,
                "indices": None,
            }
        }
    }

    # Should not raise
    engine.check_desvars(plugins, state)    
    
def test_check_desvars_array_ok(sample_engine_setup):
    """
    Design var on an ndarray vehicle parameter with matching ndarray bounds.
    Expect: no exception.
    """
    study, engine, plugins, _ = sample_engine_setup

    dv_name = "vehicles.sample_vehicle.custom_component.test_array"

    # test_array is shape (3,3) in the fixture
    lower = np.zeros((3, 3))
    upper = np.ones((3, 3)) * 10.0

    state = {
        "des_vars": {
            dv_name: {
                "lower": lower,
                "upper": upper,
                "indices": None,
            }
        }
    }

    engine.check_desvars(plugins, state)
    
def test_check_constraints_scalar_ok(sample_engine_setup):
    """
    Simple scalar float constraint on a vehicle parameter should pass.
    """
    study, engine, plugins, _ = sample_engine_setup

    state = {
        "constraints": {
            "vehicles.sample_vehicle.custom_component.p1": {
                "lower": 0.0,
                "upper": 10.0,
                "indices": None,
            }
        }
    }

    # Should not raise
    engine.check_constraints(plugins, state)
    
def test_check_constraints_non_continuous_type_raises(sample_engine_setup):
    """
    Constraint backed by a non-continuous type (e.g. int) should raise TypeError.
    """
    study, engine, plugins, _ = sample_engine_setup

    # Inject a "bad" plugin output whose type is int
    plugins["plugin1"].plugin_output_names["bad_int_output"] = {
        "type": int,
        "value": 1,
        "shape": None,
    }

    constraint_name = "plugin1.bad_int_output"

    state = {
        "constraints": {
            constraint_name: {
                "lower": 0,
                "upper": 1,
                "indices": None,
            }
        }
    }

    with pytest.raises(TypeError, match=r"Only floats allowed as constraints"):
        engine.check_constraints(plugins, state)

def test_check_upper_lower_array_scalar_bounds_return(sample_engine_setup):
    """
    For an ndarray variable, scalar lower/upper bounds are allowed (broadcasting),
    and should hit the early 'return' branch with no error.
    """
    study, engine, plugins, _ = sample_engine_setup

    var_type = np.ndarray
    dv_or_const = "Constraint"
    var_name = "plugin1.array_out"  # declared as ndarray in Plugin1

    data_lower = 0.0   # scalar
    data_upper = 1.0   # scalar

    # Should NOT raise – this exercises the scalar-scalar early-return path.
    engine.check_upper_lower(
        var_type,
        data_lower,
        data_upper,
        dv_or_const,
        var_name,
        plugins,
        study,
    )
    
def test_check_objective_continuous_ok(sample_engine_setup):
    """
    Objective is a standard float output: plugin3.objective.

    This should pass silently (no exception) because the type is continuous.
    """
    study, engine, plugins, _ = sample_engine_setup

    state = {
        "objective_name": "plugin3.objective",
    }

    # Should NOT raise
    engine.check_objective(plugins, state)
    
    
def test_check_optimization_discrete_all_continuous_passes(sample_engine_setup):
    """
    All plugin-to-plugin links use continuous outputs (float or float arrays),
    so check_optimization_discrete should NOT raise.
    """
    study, engine, plugins, state = sample_engine_setup

    # Even though state['optimization'] is not used inside this method,
    # we set it for consistency with how check_model would call it.
    state["optimization"] = True

    # Should not raise
    engine.check_optimization_discrete(plugins, state)
    
    
    
    
def test_check_model_full_optimization_path_passes(sample_engine_setup):
    """
    End-to-end positive test for Engine.check_model.

    Uses the sample_engine_setup (which already has consistent links,
    types, and indices), and supplies a clean optimization state:
    - one continuous design variable (vehicle scalar)
    - one continuous constraint (plugin scalar output)
    - a continuous objective (plugin3.objective)

    Expectation: check_model completes without raising.
    """
    study, engine, plugins, _ = sample_engine_setup

    state = {
        "optimization": True,
        "des_vars": {
            # Vehicle scalar parameter (float)
            "vehicles.sample_vehicle.custom_component.p1": {
                "lower": 0.0,
                "upper": 10.0,
                "indices": None,
            }
        },
        "constraints": {
            # Plugin scalar output (float) used as a constraint
            "plugin1.out1_to_p2": {
                "lower": -1.0,
                "upper": 1.0,
                "indices": None,
            }
        },
        # Objective is a float plugin output in plugin3
        "objective_name": "plugin3.objective",
    }

    # Should run all sub-checks without error:
    # - check_links_to_plugins
    # - check_link_types
    # - check_link_indices
    # - check_desvars
    # - check_constraints
    # - check_objective
    # - check_optimization_discrete
    engine.check_model(plugins, state)
    
    
def test_sort_steps_orders_by_links(sample_engine_setup):
    """
    sort_steps should reorder steps into a topological order determined
    by the links, regardless of how steps were originally added.

    We:
      - Take the sample fixture (plugins 1→2→3 dependency chain).
      - Scramble the public steps order to [plugin3, plugin2, plugin1].
      - Call engine.sort_steps().
      - Expect the *internal* _steps dict to be ordered as
        [plugin1, plugin2, plugin3] based on link dependencies.
    """
    study, engine, plugins, _ = sample_engine_setup

    # Sanity: original order from fixture
    original_order = study.solution.get_step_names()
    assert original_order == ["plugin1", "plugin2", "plugin3"]

    # Scramble the public steps dict
    steps = study.solution.steps
    scrambled = {
        "plugin3": steps["plugin3"],
        "plugin2": steps["plugin2"],
        "plugin1": steps["plugin1"],
    }
    study.solution.steps = scrambled

    # Confirm we actually scrambled it
    assert list(study.solution.steps.keys()) == ["plugin3", "plugin2", "plugin1"]

    # Now run the sort – it populates solution._steps
    engine.sort_steps()

    # Assert on the internal sorted representation
    sorted_order_internal = list(study.solution._steps.keys())
    assert sorted_order_internal == ["plugin1", "plugin2", "plugin3"]
#----------------------Negative checks    
def test_check_links_to_plugins_bad_input(sample_engine_setup):
    """
    Add a link whose destination (plugin input) is NOT declared in the plugin,
    and verify Engine.check_links_to_plugins() raises a ValueError.
    """
    study, engine, plugins, state = sample_engine_setup

    # Add a bogus link: plugin2 has no input named "not_a_real_input"
    study.solution.add_link(
        "vehicles.sample_vehicle.custom_component.p1",
        "plugin2.not_a_real_input",
    )

    with pytest.raises(ValueError):
        engine.check_links_to_plugins(plugins)
        
def test_check_links_to_plugins_bad_output(sample_engine_setup):
    """
    Ensure Engine.check_links_to_plugins raises ValueError when a link
    references a plugin output that was never declared in that plugin.
    """
    study, engine, plugins, state = sample_engine_setup

    # Add a bogus output link: source is an undeclared output on plugin1
    # Destination is a valid input on plugin2 so we only trigger the *output* side.
    study.solution.add_link("plugin1.missing_output", "plugin2.from_p1")

    with pytest.raises(ValueError):
        engine.check_links_to_plugins(plugins)
    
def test_check_link_types_invalid_indexing_on_scalar_output(sample_engine_setup):
    """
    Negative test 1:
    - Add a link that *indexes* a scalar output:
        plugin1.out1_to_p2[0] -> plugin2.from_p1

    Since plugin1.out1_to_p2 is a float (not np.ndarray),
    Engine.check_link_types should raise a ValueError:
        'Indexes may only be specified for arrays...'
    """
    study, engine, plugins, state = sample_engine_setup

    # This is intentionally invalid: out1_to_p2 is a scalar, but we index it
    study.solution.add_link("plugin1.out1_to_p2[0]", "plugin2.from_p1")

    with pytest.raises(ValueError, match="Indexes may only be specified for arrays"):
        engine.check_link_types(plugins)
        
def test_check_link_types_mismatched_types_vehicle_array_to_scalar_input(sample_engine_setup):
    """
    Negative test 2:
    - Add a link from an array parameter (np.ndarray) to a scalar plugin input (float):
        vehicles.sample_vehicle.custom_component.test_array -> plugin1.veh_p1

    No indexing here, so:
        output_type = np.ndarray
        input_type  = float
    -> mismatch, so Engine.check_link_types should collect a bad_link and
       raise TypeError with 'Link types must match' in the message.
    """
    study, engine, plugins, state = sample_engine_setup

    base = "vehicles.sample_vehicle.custom_component"
    # test_array is declared as np.ndarray in the fixture
    study.solution.add_link(f"{base}.test_array", "plugin1.veh_p1")

    with pytest.raises(TypeError) as excinfo:
        engine.check_link_types(plugins)

    assert "Link types must match" in str(excinfo.value)
    
    
def test_check_link_indices_invalid_parameter_indices(sample_engine_setup):
    """
    Invalid parameter indexing:
    vehicles.sample_vehicle.custom_component.test_array[10,:]
    is out of bounds for a 3x3 test_array.

    Should raise the 'The indices in ... provided are not a valid means to index' ValueError.
    """
    study, engine, plugins, state = sample_engine_setup

    base = "vehicles.sample_vehicle.custom_component"
    bad_src = f"{base}.test_array[10,:]"

    # Destination can be any plugin input; reuse plugin1.array_slice_in
    study.solution.add_link(bad_src, "plugin1.array_slice_in")

    with pytest.raises(
        ValueError,
        match=r"The indices in vehicles\.sample_vehicle\.custom_component\.test_array\[10,:] provided are not a valid means to index",
    ):
        engine.check_link_indices(plugins)


def test_check_link_indices_dest_value_mismatch(sample_engine_setup):
    """
    Force plugin2.array_from_p1_slice to have a concrete value of the wrong shape
    so that the 'b is not None' branch is taken and a shape-mismatch ValueError
    is raised.
    """
    study, engine, plugins, state = sample_engine_setup

    # plugin1.array_out[0,:] has src_shape (2,)
    plugins["plugin1"].plugin_output_names["array_out"]["value"] = np.ones((2, 2))
    plugins["plugin1"].plugin_output_names["array_out"]["shape"] = (2, 2)

    # Destination value has mismatched shape (3,)
    plugins["plugin2"].plugin_input_names["array_from_p1_slice"]["value"] = np.ones((3,))
    plugins["plugin2"].plugin_input_names["array_from_p1_slice"]["shape"] = None

    with pytest.raises(ValueError, match=r"The source indices must match the target shape\. The input plugin1\.array_out\[0,:]"):
        engine.check_link_indices(plugins)
        
        
def test_check_link_indices_dest_declared_shape_mismatch(sample_engine_setup):
    """
    Force plugin2.array_from_p1_slice to have value=None but a declared shape
    that does not match the source slice, so that the 'b_shape is not None'
    branch is taken and raises.
    """
    study, engine, plugins, state = sample_engine_setup

    # src: plugin1.array_out[0,:] -> shape (2,)
    plugins["plugin1"].plugin_output_names["array_out"]["value"] = None
    plugins["plugin1"].plugin_output_names["array_out"]["shape"] = (2, 2)

    # dest: mismatched declared shape
    plugins["plugin2"].plugin_input_names["array_from_p1_slice"]["value"] = None
    plugins["plugin2"].plugin_input_names["array_from_p1_slice"]["shape"] = (3,)

    with pytest.raises(ValueError, match=r"The source indices must match the target shape\. The input plugin1\.array_out\[0,:]"):
        engine.check_link_indices(plugins)
        
def test_check_link_indices_plugin_src_value_invalid_index(sample_engine_setup):
    """
    Plugin-out case: a is not None, but indices are invalid for its shape,
    so a[inds] raises, and we hit the ValueError('') branch.
    """
    study, engine, plugins, state = sample_engine_setup

    # Make plugin1.array_out have a concrete 2x2 value
    plugins["plugin1"].plugin_output_names["array_out"]["value"] = np.ones((2, 2))
    plugins["plugin1"].plugin_output_names["array_out"]["shape"] = (2, 2)

    # Add a destination input so the link is well-formed
    plugins["plugin2"].plugin_input_names["array_bad"] = {
        "type": np.ndarray,
        "value": None,
        "shape": (2,),   # what we "intend" to receive
    }

    # Invalid row index 5 for a (2,2) array => a[inds] will raise IndexError
    study.solution.add_link("plugin1.array_out[5,:]", "plugin2.array_bad")

    with pytest.raises(ValueError):
        engine.check_link_indices(plugins)
        
        
def test_check_link_indices_plugin_src_shape_invalid_index(sample_engine_setup):
    """
    Plugin-out case: a is None but a_shape is not None, and the indices are
    invalid for that shape, so np.ones(shape=a_shape)[inds] raises and we
    hit the ValueError('') branch.
    """
    study, engine, plugins, state = sample_engine_setup

    # Force value to None but keep a declared shape
    plugins["plugin1"].plugin_output_names["array_out"]["value"] = None
    plugins["plugin1"].plugin_output_names["array_out"]["shape"] = (2, 2)

    # Destination input
    plugins["plugin2"].plugin_input_names["array_bad_shape"] = {
        "type": np.ndarray,
        "value": None,
        "shape": (2,),   # intended, but irrelevant for this branch
    }

    # Same invalid index; now this will fail at np.ones((2,2))[inds]
    study.solution.add_link("plugin1.array_out[5,:]", "plugin2.array_bad_shape")

    with pytest.raises(ValueError):
        engine.check_link_indices(plugins)
        
def test_check_desvars_non_continuous_type_raises(sample_engine_setup):
    """
    Design var whose underlying type is non-continuous (e.g. int).
    Expect: TypeError from check_desvars.
    """
    study, engine, plugins, _ = sample_engine_setup

    # Inject a fake discrete plugin output into plugin1
    plugins["plugin1"].plugin_output_names["bad_disc"] = {
        "type": int,
        "value": 0,
        "shape": None,
    }

    dv_name = "plugin1.bad_disc"

    # Bounds are continuous and well-formed; the failure should be on the DV type
    state = {
        "des_vars": {
            dv_name: {
                "lower": 0.0,
                "upper": 1.0,
                "indices": None,
            }
        }
    }

    with pytest.raises(TypeError, match=r"Only floats allowed as design variables"):
        engine.check_desvars(plugins, state)
        
def test_check_constraints_array_bound_shape_mismatch_raises(sample_engine_setup):
    """
    When a constraint is on an ndarray variable, lower/upper array bounds must
    have matching shapes. Mismatched shapes should raise ValueError via
    Engine.check_upper_lower.
    """
    study, engine, plugins, _ = sample_engine_setup

    # plugin1.array_out is declared as np.ndarray, shape (2, 2)
    constraint_name = "plugin1.array_out"

    lower = np.zeros((2, 2))
    upper = np.ones((3, 2))  # different shape -> should trigger shape mismatch

    state = {
        "constraints": {
            constraint_name: {
                "lower": lower,
                "upper": upper,
                "indices": None,
            }
        }
    }

    # This should hit the "Constraint shapes must match" branch in check_upper_lower
    with pytest.raises(ValueError, match=r"Constraint shapes must match"):
        engine.check_constraints(plugins, state)
        
        
def test_check_upper_lower_array_lower_vs_value_shape_mismatch_raises(sample_engine_setup):
    """
    If the variable has an ndarray value and the lower bound is an ndarray with
    a mismatched shape, check_upper_lower should raise a ValueError.
    """
    study, engine, plugins, _ = sample_engine_setup

    var_type = np.ndarray
    dv_or_const = "Constraint"
    var_name = "plugin1.array_out"

    # Make plugin1.array_out have an actual ndarray value with known shape.
    plugins["plugin1"].plugin_output_names["array_out"]["value"] = np.ones((2, 2))
    plugins["plugin1"].plugin_output_names["array_out"]["shape"] = (2, 2)

    # Lower bound with mismatched shape
    data_lower = np.zeros((3, 2))
    data_upper = None  # upper doesn't matter for this branch

    with pytest.raises(ValueError, match=r"If arrays are declared as bounds for Constraint"):
        engine.check_upper_lower(
            var_type,
            data_lower,
            data_upper,
            dv_or_const,
            var_name,
            plugins,
            study,
        )
        
def test_check_upper_lower_array_upper_vs_declared_shape_mismatch_raises(sample_engine_setup):
    """
    If the variable has no value but a declared shape, and the upper bound's shape
    doesn't match that declared shape, we should get a ValueError.
    """
    study, engine, plugins, _ = sample_engine_setup

    var_type = np.ndarray
    dv_or_const = "Constraint"
    var_name = "plugin1.array_out"

    # Simulate 'shape-only' declaration: no value, only shape.
    plugins["plugin1"].plugin_output_names["array_out"]["value"] = None
    plugins["plugin1"].plugin_output_names["array_out"]["shape"] = (2, 2)

    data_lower = None
    data_upper = np.ones((3, 2))  # mismatched with (2, 2)

    with pytest.raises(ValueError, match=r"If arrays are declared as bounds for Constraint"):
        engine.check_upper_lower(
            var_type,
            data_lower,
            data_upper,
            dv_or_const,
            var_name,
            plugins,
            study,
        )

def test_check_upper_lower_array_upper_vs_value_shape_mismatch_raises(sample_engine_setup):
    """
    data_upper is a non-scalar array, and the backing var_value is a non-scalar
    array with a *different* shape -> mismatch vs var_value branch for upper.
    Expect ValueError from the 'upper vs var_value.shape' check.
    """
    study, engine, plugins, _ = sample_engine_setup

    var_type = np.ndarray
    dv_or_const = "Constraint"
    var_name = "plugin1.array_out"

    # Again, actual value with shape (2, 2)
    plugins["plugin1"].plugin_output_names["array_out"]["value"] = np.ones((2, 2))
    plugins["plugin1"].plugin_output_names["array_out"]["shape"] = (2, 2)

    data_lower = None
    data_upper = np.ones((3, 2))  # mismatched with (2, 2)

    with pytest.raises(ValueError, match=r"If arrays are declared as bounds for Constraint"):
        engine.check_upper_lower(
            var_type,
            data_lower,
            data_upper,
            dv_or_const,
            var_name,
            plugins,
            study,
        )
        

def test_check_upper_lower_scalar_var_with_array_bounds_raises(sample_engine_setup):
    """
    var_type is scalar (float), but both data_lower and data_upper are arrays.
    This should hit the final 'scalar but bounds are non-scalars' branch.
    """
    study, engine, plugins, _ = sample_engine_setup

    var_type = float
    dv_or_const = "Constraint"
    var_name = "plugin3.objective"  # declared as float in Plugin3

    # Both bounds are arrays, which is illegal when var_type is scalar
    data_lower = np.array([0.0, 1.0])
    data_upper = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match=r"was declared as <class 'float'> which is a scalar"):
        engine.check_upper_lower(
            var_type,
            data_lower,
            data_upper,
            dv_or_const,
            var_name,
            plugins,
            study,
        )
        
def test_check_upper_lower_array_lower_vs_declared_shape_mismatch_raises(sample_engine_setup):
    """
    If the variable has no value but a declared shape, and the LOWER bound's shape
    doesn't match that declared shape, we should get a ValueError.

    This specifically hits the 'elif var_shape:' branch for data_lower.
    """
    study, engine, plugins, _ = sample_engine_setup

    var_type = np.ndarray
    dv_or_const = "Constraint"
    var_name = "plugin1.array_out"

    # Shape-only declaration: no value, only shape
    plugins["plugin1"].plugin_output_names["array_out"]["value"] = None
    plugins["plugin1"].plugin_output_names["array_out"]["shape"] = (2, 2)

    # Lower bound shape (3, 2) mismatches declared shape (2, 2)
    data_lower = np.ones((3, 2))
    data_upper = None  # irrelevant for this branch

    with pytest.raises(ValueError, match=r"If arrays are declared as bounds for Constraint"):
        engine.check_upper_lower(
            var_type,
            data_lower,
            data_upper,
            dv_or_const,
            var_name,
            plugins,
            study,
        )
        
def test_check_objective_non_continuous_raises(sample_engine_setup):
    study, engine, plugins, _ = sample_engine_setup

    plugins["plugin3"].plugin_output_names["objective"]["type"] = int

    state = {
        "objective_name": "plugin3.objective",
    }

    with pytest.raises(TypeError):
        engine.check_objective(plugins, state)

def test_check_optimization_discrete_discrete_link_raises(sample_engine_setup):
    """
    If a plugin-to-plugin link originates from a discrete type (e.g. int),
    check_optimization_discrete must raise a TypeError.
    """
    study, engine, plugins, state = sample_engine_setup

    # Corrupt Plugin1's output metadata: mark out1_to_p2 as discrete (int)
    plugins["plugin1"].plugin_output_names["out1_to_p2"]["type"] = int

    state["optimization"] = True

    # This link exists in the fixture:
    #   plugin1.out1_to_p2 -> plugin2.from_p1
    # and both plugin1 and plugin2 are in solution.steps,
    # so it will be checked.
    with pytest.raises(TypeError, match=r"Links between optimization components must be continuous"):
        engine.check_optimization_discrete(plugins, state)

def test_check_optimization_discrete_discrete_indexed_link_raises(sample_engine_setup):
    """
    Use an indexed plugin output link (plugin1.array_out[0,:]) and
    make the underlying declared type discrete to exercise the
    indexing path in check_optimization_discrete.
    """
    study, engine, plugins, state = sample_engine_setup

    # Corrupt the underlying type for array_out
    plugins["plugin1"].plugin_output_names["array_out"]["type"] = int

    state["optimization"] = True

    # Fixture has:
    #   plugin1.array_out[0,:] -> plugin2.array_from_p1_slice
    with pytest.raises(TypeError, match=r"Links between optimization components must be continuous"):
        engine.check_optimization_discrete(plugins, state)
        
def test_sort_steps_circular_dependency_raises(sample_engine_setup):
    """
    sort_steps should raise RuntimeError if there is a circular dependency
    between solution steps.

    We create a cycle:
        plugin1 -> plugin2 -> plugin3 -> plugin1

    by adding an extra link from plugin3 back to plugin1.
    """
    study, engine, plugins, _ = sample_engine_setup

    # Sanity: original graph is acyclic and ordered
    assert study.solution.get_step_names() == ["plugin1", "plugin2", "plugin3"]

    # Existing links already give:
    #   plugin1 -> plugin2
    #   plugin1 -> plugin3
    #   plugin2 -> plugin3
    #
    # Add a back-edge to create a cycle: plugin3 -> plugin1
    # Names after the step token can be anything; sort_steps only checks
    # the first part (step name) on each side of the link.
    study.solution.add_link("plugin3.fake_output", "plugin1.fake_input")

    # Now the dependency graph has a cycle: 1 -> 2 -> 3 -> 1
    # sort_steps should detect this (no step can be added) and raise.
    with pytest.raises(RuntimeError, match=r"Circular Solution Sequence Detected"):
        engine.sort_steps()
if __name__ == '__main__':
    pass