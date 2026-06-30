import numpy as np
from uam4x.utils import *
from uam4x.utils import _is_valid_square_brackets, _is_valid_index, _is_simple, _is_slice, _is_fancy, _parse_simple, _parse_fancy, _parse_slice, _parse_complicated, _parse_index
import pytest
import os
import json
from uam4x.study import Study
from uam4x.solution import Plugin, register_plugin
from uam4x.subsystems import Subsystem, Wing, register_subsystem, Rotor, RotorBladeRegion, Aircraft
from uam4x.vehicle import Vehicle

def test_check_and_get_config_good():
    file_path = os.path.abspath(__file__)
    test_dir = os.path.dirname(file_path)
    proj_dir = os.path.dirname(test_dir)


    config_path = os.path.join(proj_dir,'uam4x', 'conf.json')

    if os.path.exists(config_path):
        os.remove(config_path)

    config = check_and_get_config(config_path)
    assert config == {"PLUGINS": {},"SUBSYSTEMS": {}}


    sample_dict = {
    "SUBSYSTEMS": {
        "Wing": "e:\\nasa_uam\\uam4x\\uam4x\\subsystems.py",
        "Rotor": "e:\\nasa_uam\\uam4x\\uam4x\\subsystems.py",
        "Aircraft": "e:\\nasa_uam\\uam4x\\uam4x\\subsystems.py",
        "WeightingFactor": "E:\\NASA_UAM\\uam4x_examples\\multi_objective\\custom.py",
        "RotorBladeRegion": "e:\\nasa_uam\\uam4x\\uam4x\\subsystems.py",
        "NdarcTechFactors": "E:\\NASA_UAM\\uam4x_examples\\tilt_rotor\\custom.py",
        "LineSegment": "E:\\NASA_UAM\\uam4x_examples\\time_tracking\\n_segments\\increasing_input_des_vars\\n_segments_exact\\custom.py",
        "MultiConstraintInp": "E:\\NASA_UAM\\uam4x_examples\\time_tracking\\multi_constraint\\constant_output_variable_constraint\\custom.py",
        "Sphere": "E:\\NASA_UAM\\uam4x_examples\\time_tracking\\sphere_optimization\\variable_plugins_flops\\custom.py"
    },
    "PLUGINS": {
        "CalcArea": "E:\\NASA_UAM\\uam4x_examples\\multi_objective\\custom.py",
        "CalcObjective": "E:\\NASA_UAM\\uam4x_examples\\time_tracking\\sphere_optimization\\variable_plugins_flops\\custom.py",
        "OpenvspPlugin": "e:\\nasa_uam\\uam4x_plugins\\openvsp\\uam4x_openvsp\\plugin.py",
        "M4SSPlugin": "e:\\nasa_uam\\uam4x_plugins\\m4ss\\uam4x_m4ss\\plugin.py",
        "FakeFastOpenvspM4ssPlugin": "E:\\NASA_UAM\\uam4x_examples\\tilt_rotor\\custom.py",
        "FakeFastM4SS": "E:\\NASA_UAM\\uam4x_examples\\tilt_rotor\\custom.py",
        "Residual": "E:\\NASA_UAM\\uam4x_examples\\tilt_rotor\\custom.py",
        "SumComponents": "E:\\NASA_UAM\\uam4x_examples\\tilt_rotor\\custom.py",
        "CalcTotalPassWeight": "E:\\NASA_UAM\\uam4x_examples\\tilt_rotor\\custom.py",
        "NdarcPlugin": "e:\\nasa_uam\\uam4x_plugins\\ndarc\\uam4x_ndarc\\plugin.py",
        "CalcConstraint": "E:\\NASA_UAM\\uam4x_examples\\time_tracking\\multi_constraint\\constant_output_variable_constraint\\custom.py",
    }
    }
    with  open(config_path, 'w') as  f:
        json.dump(sample_dict, f, indent=4)

    config = check_and_get_config(config_path)

    assert config == sample_dict

@pytest.mark.parametrize("inp,expected_result", [
    (np.ndarray, True),
    (None, False),
    (bool, False),
    (list, False),
    (int, False),
    (float, True),
    (tuple, False),
    (1., True),
    (1, False),
    ('bad', False),
    ([1., 1., 1.], False),
    (np.full(5, 1), False),
    (np.full(5, 1.), True),
    (True, False),
    (False, False)
])
def test_check_continuous(inp, expected_result):

    assert is_continuous_type_or_value(inp) ==  expected_result



@pytest.mark.parametrize("inp,expected_result", [

    (np.ndarray, False),
    (float, True),
    (int, True),
    (bool, False),
    (tuple, False),
    (list, False),
    (None, False),
    (np.full(6, 7.), False),
    (np.full(6, 7), False),
    (1., True),
    (1, True),
    (-1, True),
    (0, True),
    (False, False),
    (True, False)
])
def test_check_scalar(inp, expected_result):
    assert is_scalar_type_or_value(inp) == expected_result

@pytest.mark.parametrize("name,expected_type", [
    ('int', int),
    ('float', float),
    ('str', str),
    ('bool', bool),
    ('ndarray', np.ndarray)
])
def test_type_string_to_type_good(name, expected_type):
    assert type_string_to_type(name) == expected_type


# Test convert numpy to python


#Test check_numpy_shape
@pytest.mark.parametrize("value,expected_output", [
    (6, True),
    (7., False),
    (-1, False),
    (np.nan, False),
    (True, False),
    (False, False),
    ((1, 'five'), False),
    ((1, 4, 5, 7), True),
    ((1., 4, 5, 6), False),
    ((1, False, 3), False),
    ([1, 2, 3], False),
    ((1, -1, 2), False)
])
def test_check_numpy_shape(value, expected_output):
    assert check_numpy_shape(value) == expected_output


#test normalize_shape
@pytest.mark.parametrize("value,expected_result", [
    ((1, 5, 6), (1, 5, 6)),
     (6, (6, ))
])
def test_normalize_shape(value, expected_result):
    assert normalize_shape(value) == expected_result

# is_bool
@pytest.mark.parametrize("value,expected_output", [
    (True, True),
    (False, True),
    (np.bool_(True), True),
    (np.bool_(False), True),
    (1, False),
    (0, False),
    (1.0, False),
    ("True", False),
    (None, False),
    (np.array([True, False]), False),
])
def test_is_bool(value, expected_output):
    assert is_bool(value) is expected_output


# is_integer
@pytest.mark.parametrize("value,expected_output", [
    (0, True),
    (1, True),
    (-5, True),
    (np.int32(3), True),
    (np.int64(-10), True),
    (True, False),
    (False, False),
    (1.0, False),
    (np.float64(2.0), False),
    ("3", False),
    (None, False),
    (np.array([1, 2, 3]), False),
])
def test_is_integer(value, expected_output):
    assert is_integer(value) is expected_output


# is_float
@pytest.mark.parametrize("value,expected_output", [
    (0.0, True),
    (1.0, True),
    (-3.14, True),
    (np.float32(2.5), True),
    (np.float64(-0.001), True),
    (1, False),
    (np.int32(4), False),
    (True, False),
    (False, False),
    ("1.0", False),
    (None, False),
    (np.array([1.0, 2.0]), False),
])
def test_is_float(value, expected_output):
    assert is_float(value) is expected_output


# is_tuple
@pytest.mark.parametrize("value,expected_output", [
    ((), True),
    ((1,), True),
    ((1, 2, 3), True),
    (("a", 1, 3.0), True),
    ([1, 2, 3], False),
    ("abc", False),
    ({1, 2, 3}, False),
    ({"a": 1}, False),
    (np.array([1, 2, 3]), False),
    (None, False),
])
def test_is_tuple(value, expected_output):
    assert is_tuple(value) is expected_output


# is_numpy_array
@pytest.mark.parametrize("value,expected_output", [
    (np.array([1, 2, 3]), True),
    (np.zeros((2, 2)), True),
    (np.ones(5), True),
    (np.array(5), True),      # scalar array is still an ndarray
    ([], False),
    ([1, 2, 3], False),
    ((1, 2, 3), False),
    (1, False),
    (1.0, False),
    (True, False),
    (None, False),
])
def test_is_numpy_array(value, expected_output):
    assert is_numpy_array(value) is expected_output


@pytest.mark.parametrize("value,expected", [
    (1, int.__name__),
    (67., float.__name__),
    (['hello', 1, 99.], list.__name__),
    ((1, 'a', [1, 2.]), tuple.__name__),
    (np.array([1, 2, 3]), np.ndarray.__name__),
    (None,type(None).__name__)
])
def test_value_to_type_name(value, expected):
    assert value_to_type_name(value) == expected


@pytest.mark.parametrize('value', [
    ('[1,2,3,4]'),
    ('[[1,2,3],[4,5,6]]'),
    ('[1:5:3]'),
    ('[1::-5]'),
    ('[[6,7],3:-4:2,4]')
])
def test_is_valid_square_brackets_positive(value):
    assert _is_valid_square_brackets(value) == True


@pytest.mark.parametrize('value', [
    ('[1,2,3,4]'),
    ('[[1,2,3],[4,5,6]]'),
    ('[1:5:3]'),
    ('[1::-5]'),
    ('[[6,7],3:-4:2,4]')
])
def test_is_valid_index_positive(value):
    assert _is_valid_index(value) == True


@pytest.mark.parametrize("value, expected_output", [
    # Simple lists (no slices, no nested brackets) -> True
    ("1", True),
    ("1,2,3,4", True),
    ("0, 5, 10", True),
    ("10,-3,7", True),

    # Slices (contain ':') -> False
    ("1:5:3", False),
    ("1:5", False),
    ("3:-4:2", False),
    (":", False),

    # Nested / fancy indexing (brackets inside) -> False
    ("[1,2,3]", False),
    ("[1,2], [3,4]", False),
    ("3, [4,5]", False),

    # Mixed but still simple list of numbers -> True
    ("1, 0, -5, 12", True),
])
def test_is_simple(value, expected_output):
    assert _is_simple(value) is expected_output


@pytest.mark.parametrize("value, expected_output", [
    # No colon → not a slice
    ("1", False),
    ("0", False),
    ("-3", False),
    ("[1,2,3]", False),
    ("10,20,30", False),

    # Simple slices
    ("1:5", True),
    ("2:3:4", True),
    (":", True),
    ("-1:10", True),

    # Bracketed slices / mixed but still contain colon → slice
    ("[1:5]", True),
    ("3,-4:2", True),
])
def test_is_slice(value, expected_output):
    assert _is_slice(value) is expected_output

@pytest.mark.parametrize("value, expected_output", [
    # No brackets -> not fancy
    ("1", False),
    ("2:3:4", False),
    ("1:5", False),
    ("-3", False),

    # Proper fancy indexing: has [ and ] but no :
    ("[1,2,3]", True),
    ("[0]", True),
    ("[1, 2, 3, 4]", True),
    ("[[1,2],[3,4]]", True),

    # Bracket-mismatched or weird but still no ':' -> False
    ("[1,2,3", False),
    ("1,2,3]", False),
])
def test_is_fancy(value, expected_output):
    assert _is_fancy(value) is expected_output


@pytest.mark.parametrize("value, expected_output", [
    # Single indices -> int
    ("1", 1),
    ("10", 10),
    ("-3", -3),
    ("   5   ", 5),
    ("0", 0),

    # Multiple indices -> tuple of ints
    ("1,2,3", (1, 2, 3)),
    ("1, 2, 3", (1, 2, 3)),
    ("0,-1,10", (0, -1, 10)),
    ("  4 ,  5 ,  6  ", (4, 5, 6)),
    ("-1, -2, -3", (-1, -2, -3)),
])
def test_parse_simple(value, expected_output):
    assert _parse_simple(value) == expected_output


@pytest.mark.parametrize("value, expected_output", [
    # Simple fancy indexing
    ("[1]", [1]),
    ("[0]", [0]),
    ("[1,2,3]", [1, 2, 3]),
    ("[ 1,  2,  3 ]", [1, 2, 3]),
    ("[10, -5, 7]", [10, -5, 7]),

    # Nested lists still just flatten to all ints
    ("[[1,2],[3,4]]", [1, 2, 3, 4]),
    ("[[0,-1], [10, 20]]", [0, -1, 10, 20]),

    # Mixed spacing
    ("[  -1 ,2 ,  3 ]", [-1, 2, 3]),
])
def test_parse_fancy(value, expected_output):
    assert _parse_fancy(value) == expected_output

@pytest.mark.parametrize("value, expected_output", [
    ("[1]", [1]),
    ("[0]", [0]),
    ("[-1]", [-1]),
    ("[1,2,3]", [1, 2, 3]),
    ("[10,-5,7]", [10, -5, 7]),
    ("[-1,-2,-3]", [-1, -2, -3]),
])
def test_parse_fancy(value, expected_output):
    assert _parse_fancy(value) == expected_output





@pytest.mark.parametrize("value, expected", [
    # Full start:stop:step
    ("1:5:2",  (1, 5, 2)),
    ("-3:10:1", (-3, 10, 1)),
    ("0:10:3", (0, 10, 3)),
    ("-5:-1:2", (-5, -1, 2)),

    # start:stop (step -> None)
    ("1:5",    (1, 5, None)),
    ("-3:10",  (-3, 10, None)),
    ("0:10",   (0, 10, None)),

    # start: (stop, step -> None)
    ("1:",     (1, None, None)),
    ("-3:",    (-3, None, None)),
    ("0:",     (0, None, None)),

    # :stop (start, step -> None)
    (":5",     (None, 5, None)),
    (":-10",   (None, -10, None)),

    # Bare ':' and '::' style
    (":",      (None, None, None)),   # [:]
    ("::",     (None, None, None)),   # [::] → all None
    ("::2",    (None, None, 2)),      # [::2]
    ("::-1",   (None, None, -1)),     # [::-1]

    # start::step (stop -> None)
    ("5::2",   (5, None, 2)),
    ("-3::2",  (-3, None, 2)),
    ("-3::",   (-3, None, None)),
])
def test_parse_slice_valid(value, expected):
    s = _parse_slice(value)
    assert isinstance(s, slice)
    assert (s.start, s.stop, s.step) == expected



@pytest.mark.parametrize("value, expected", [
    # Mixed: slice + simple scalars
    ("[1:,2,3]",
     (slice(1, None, None), 2, 3)),

    # Pure fancy indexing (nested lists)
    ("[[1,2],[3,4]]",
     ([1, 2], [3, 4])),

    # Fancy + simple
    ("[[1,2],[3,4],5]",
     ([1, 2], [3, 4], 5)),

    # Pure slices
    ("[1:5,2:10:2]",
     (slice(1, 5, None), slice(2, 10, 2))),

    # Simple + slice + simple
    ("[1,2:5,-3]",
     (1, slice(2, 5, None), -3)),

    # Fancy + slice + simple
    ("[[6,7],3:-4:2,4]",
     ([6, 7], slice(3, -4, 2), 4)),

    # All slices with omitted bounds
    ("[:,:]",
     (slice(None, None, None), slice(None, None, None))),

    # Fancy + slice with omitted start
    ("[[1,2],:5]",
     ([1, 2], slice(None, 5, None))),

    # Slice with negatives + fancy with negatives
    ("[-1:-5:-2,[1,-2]]",
     (slice(-1, -5, -2), [1, -2])),
])
def test_parse_complicated_valid(value, expected):
    result = _parse_complicated(value)
    assert isinstance(result, tuple)
    assert result == expected



@pytest.mark.parametrize("value, expected", [
    # 1) Pure simple indexing (is_simple branch)
    ("[3]", 3),                          # single index -> int
    ("[1,2,3]", (1, 2, 3)),
    ("[-1,5,-2]", (-1, 5, -2)),

    # 2) Pure fancy (advanced) indexing (parse_complicated + parse_fancy)
    ("[[1,2],[3,4]]", ([1, 2], [3, 4])),
    ("[[1,5,3],[1,2],[8]]", ([1, 5, 3], [1, 2], [8])),

    # 3) Simple + fancy mix
    ("[[1,2],[3,4],5]", ([1, 2], [3, 4], 5)),

    # 4) Pure slicing (parse_complicated + parse_slice)
    ("[:]", (slice(None, None, None),)),                    # [:]
    ("[::2]", (slice(None, None, 2),)),                     # [::2]
    ("[1:,2:-1,3]",
     (slice(1, None, None), slice(2, -1, None), 3)),
    ("[1::2]", (slice(1, None, 2),)),                       # start + step

    # 5) Fancy + slice mix
    ("[[6,7],3:-4:2,4]",
     ([6, 7], slice(3, -4, 2), 4)),
    ("[[1,2],:5]",
     ([1, 2], slice(None, 5, None))),

    # 6) Multiple slices
    ("[1:5,2:10:2]",
     (slice(1, 5, None), slice(2, 10, 2))),
    ("[:,:]",
     (slice(None, None, None), slice(None, None, None))),
])
def test_parse_index_positive(value, expected):
    assert _parse_index(value) == expected


@pytest.mark.parametrize("dotted_path, expected_path, expected_inds", [
    # Simple scalar index
    ("vehicle[3]", "vehicle", 3),

    # Simple tuple index
    ("vehicle.array[1,2,3]", "vehicle.array", (1, 2, 3)),
    ("a.b.c[-1,5,-2]", "a.b.c", (-1, 5, -2)),

    # Slicing on last term
    ("vehicles.test_vehicle.chord.span[1:3]",
     "vehicles.test_vehicle.chord.span",
     (slice(1, 3, None),)),

    ("obj.data[1:,2:-1,3]",
     "obj.data",
     (slice(1, None, None), slice(2, -1, None), 3)),

    ("x.y[::2]",
     "x.y",
     (slice(None, None, 2),)),

    # Fancy (advanced) indexing on last term
    ("system.array[[1,2],[3,4]]",
     "system.array",
     ([1, 2], [3, 4])),

    ("obj.indices[[1,5,3],[1,2],[8]]",
     "obj.indices",
     ([1, 5, 3], [1, 2], [8])),

    # Mixed fancy + slice
    ("a.b.c[[6,7],3:-4:2,4]",
     "a.b.c",
     ([6, 7], slice(3, -4, 2), 4)),

    ("vehicle.grid[[1,2],:5]",
     "vehicle.grid",
     ([1, 2], slice(None, 5, None))),

    # Multiple slices
    ("solver.field[:,:]",
     "solver.field",
     (slice(None, None, None), slice(None, None, None))),
])
def test_handle_brackets_positive(dotted_path, expected_path, expected_inds):
    path, inds = split_dotted_path_and_indices(dotted_path)
    assert path == expected_path
    assert inds == expected_inds

@pytest.mark.parametrize(
    "value, expected, check_idempotence",
    [
        # ---- NumPy scalar types ----
        (np.int32(5), 5, False),
        (np.int64(7), 7, False),
        (np.float32(3.14), pytest.approx(3.14), False),
        (np.float64(2.718), pytest.approx(2.718), False),
        (np.bool_(True), True, False),
        (np.bool_(False), False, False),

        # ---- 1D array ----
        (np.array([1, 2, 3], dtype=np.int64), [1, 2, 3], False),

        # ---- 2D array ----
        (
            np.array([[1, 2], [3, 4]], dtype=np.float64),
            [[1.0, 2.0], [3.0, 4.0]],
            False,
        ),

        # ---- Bool array ----
        (
            np.array([True, False, True], dtype=bool),
            [True, False, True],
            False,
        ),

        # ---- List with mixed NumPy ----
        (
            [np.int32(1), np.float64(2.5), np.array([3, 4]), np.bool_(True)],
            [1, 2.5, [3, 4], True],
            False,
        ),

        # ---- Dict with mixed NumPy ----
        (
            {
                "i": np.int64(10),
                "f": np.float32(1.5),
                "b": np.bool_(False),
                "arr": np.array([1, 2, 3]),
            },
            {
                "i": 10,
                "f": pytest.approx(1.5),
                "b": False,
                "arr": [1, 2, 3],
            },
            False,
        ),

        # ---- Nested dict + list (also used for idempotence) ----
        (
            {
                "a": [
                    np.int32(1),
                    np.array([2, 3]),
                    {"x": np.float64(4.5)},
                ],
                "b": {
                    "sub": np.array([[1, 2], [3, 4]], dtype=np.int32),
                },
            },
            {
                "a": [1, [2, 3], {"x": 4.5}],
                "b": {"sub": [[1, 2], [3, 4]]},
            },
            True,  # also check idempotence on this one
        ),

        # ---- Non-NumPy: primitives + custom object should pass through structurally ----
        (
            {
                "int": 1,
                "float": 2.0,
                "str": "hello",
                "bool": True,
                "none": None,
            },
            {
                "int": 1,
                "float": 2.0,
                "str": "hello",
                "bool": True,
                "none": None,
            },
            False,
        ),
    ],
)
def test_convert_numpy_to_python_parametrized(value, expected, check_idempotence):
    result = convert_numpy_to_python(value)

    # Equality check handles lists, dicts, and pytest.approx for floats
    assert result == expected

    if check_idempotence:
        # Running it again should not change the already-converted structure
        result2 = convert_numpy_to_python(result)
        assert result2 == result

def test_get_param_type():

    @register_plugin
    class TestPlug(Plugin):
        """
        test plugin to check get_param_type functionality for plugin links/components
        """
        def setup_plugin(self, plugin_options):
            # declare inputs
            Plugin.declare_input(self, "testin_cont", float, 0.0)
            Plugin.declare_input(self, "testin_discrete", int, 1)

            # declare outputs
            Plugin.declare_output(self, "testout_cont", float, 0.0)
            Plugin.declare_output(self, "testout_discrete", int, 1)

        def run_plugin(self, links: dict, work_dir: str):
            results = {}
            results["testout_cont"] = 0.06
            results["testout_discrete"] = 6
            return results

    @register_subsystem
    class BridgeSubsystem(Subsystem):
        def setup(self):
            self.add_parameter("Ace",   default_value=4.0, dtype=float)
            self.add_parameter("King",  default_value=3.0, dtype=float)
            self.add_parameter("Queen", default_value=2.0, dtype=float)
            self.add_parameter("Jack",  default_value=1.0, dtype=float)
            # Array-valued parameter
            self.add_parameter(
                "DeckValues",
                default_value=np.arange(1, 5),
                dtype=np.ndarray,
            )

    study = Study()

    # -------- Vehicle 1: mtr --------
    mtr = Vehicle("mtr")

    wing = Wing("wing")
    wing.set_parameter("span", 35.56)
    wing.set_parameter("area", 142.1)
    wing.set_parameter("sweep", 0.5)

    aircraft = Aircraft("aircraft")
    aircraft.set_parameter("Npass", 6)
    aircraft.set_parameter("Upass", 200.0)

    rotor = Rotor("rotor")
    rotor.set_parameter("nblade", 8)
    vflight_rotor = rotor.add_mode("vflight")
    vflight_rotor.set_parameter("tilt", 90.0)
    cruise_rotor = rotor.add_mode("cruise")
    cruise_rotor.set_parameter("tilt", 0.0)

    mtr.add_component(wing)
    mtr.add_component(rotor)
    mtr.add_component(aircraft)

    # -------- Vehicle 2: Bridge --------
    BC = BridgeSubsystem("bridge_face_cards")
    V2 = Vehicle("Bridge")
    V2.add_component(BC)

    # Add vehicles to study
    study.add_vehicle(mtr)
    study.add_vehicle(V2)

    # Define and add test plugin
    study.solution.add_step(TestPlug, "test_plug", {})

    # Add links for test plugin
    study.solution.add_link("vehicles.mtr.wing.span", "test_plug.testin_cont")
    study.solution.add_link("vehicles.mtr.aircraft.Npass", "test_plug.testin_discrete")

    plugins = study.solution.generate_plugins()

    # -------- Vehicle 1 checks --------
    actual = study.get_param_type("vehicles.mtr.wing.span")
    assert actual == float, f"expected float, got {actual}"

    actual = study.get_param_type("vehicles.mtr.wing.area")
    assert actual == float, f"expected float, got {actual}"

    actual = study.get_param_type("vehicles.mtr.wing.sweep")
    assert actual == float, f"expected float, got {actual}"

    actual = study.get_param_type("vehicles.mtr.aircraft.Upass")
    assert actual == float, f"expected float, got {actual}"

    actual = study.get_param_type("vehicles.mtr.aircraft.Npass")
    assert actual == int, f"expected int, got {actual}"

    actual = study.get_param_type("vehicles.mtr.rotor.nblade")
    assert actual == int, f"expected int, got {actual}"

    actual = study.get_param_type("vehicles.mtr.rotor.vflight.tilt")
    assert actual == float, f"expected float, got {actual}"

    actual = study.get_param_type("vehicles.mtr.rotor.cruise.tilt")
    assert actual == float, f"expected float, got {actual}"

    # -------- Plugin I/O checks --------
    actual = study.solution.get_var_type("test_plug.testin_cont", plugins)
    assert actual == float, f"expected float, got {actual}"

    actual = study.solution.get_var_type("test_plug.testin_discrete", plugins)
    assert actual == int, f"expected int, got {actual}"

    actual = study.solution.get_var_type("test_plug.testout_cont", plugins)
    assert actual == float, f"expected float, got {actual}"

    actual = study.solution.get_var_type("test_plug.testout_discrete", plugins)
    assert actual == int, f"expected int, got {actual}"

    # -------- Vehicle 2 checks (including array param) --------
    actual = study.get_param_type("vehicles.Bridge.bridge_face_cards.Ace")
    assert actual == float, f"expected float, got {actual}"

    actual = study.get_param_type("vehicles.Bridge.bridge_face_cards.King")
    assert actual == float, f"expected float, got {actual}"

    actual = study.get_param_type("vehicles.Bridge.bridge_face_cards.Queen")
    assert actual == float, f"expected float, got {actual}"

    actual = study.get_param_type("vehicles.Bridge.bridge_face_cards.Jack")
    assert actual == float, f"expected float, got {actual}"

    actual = study.get_param_type("vehicles.Bridge.bridge_face_cards.DeckValues")
    assert actual == np.ndarray, f"expected np.ndarray, got {actual}"
#---------------------------------------------------------------------Negative checks
def test_type_string_to_type_bad():
    with pytest.raises(KeyError):
        type_string_to_type('random')


@pytest.mark.parametrize('value',[
    ('[[]['),
    ('[]]')
])
def test_is_valid_square_brackets_vio(value):
    with  pytest.raises(ValueError):
        _is_valid_square_brackets(value)


@pytest.mark.parametrize('value', [
    ('[1:::3]'),
    ('[1,2,3,5-,6]'),
    ('[1,b,3]'),
    ('[4.,5]'),
    ('6,7,9'),
    ('[6,07,9]'),
    ('[[[1,2,3,4]]]'),
    ('[1,,2,4,5]'),
    ('1,2,3,4]'),
    ('[1,2,3,4'),
    ('[[1,2,3]4,5]')

])
def test_is_valid_index_vio(value):
    with pytest.raises(ValueError):
        _is_valid_index(value)

@pytest.mark.parametrize("value", [
    "[1:5]",
    "[1:5:2]",
    "[1,2:5]",
    "[0:10, 2]",
    "[[1:2],3]",
])
def test_is_fancy_vio(value):
    with pytest.raises(ValueError):
        _is_fancy(value)



@pytest.mark.parametrize("value", [
    "1:2:3:4",   # more than 2 colons
    "1:5:0",     # zero step
    "0:10:-0",   # '-0' → 0 → zero step
])
def test_parse_slice_vio(value):
    with pytest.raises(ValueError):
        _parse_slice(value)



@pytest.mark.parametrize("value", [
    # Slicing inside fancy indexing is explicitly disallowed by is_fancy
    "[[1:2],[3,4]]",
    "[[1,2],[3:4]]",
])
def test_parse_complicated_fancy_slice_violation(value):
    with pytest.raises(ValueError, match="Slicing not allowed in fancy indexing"):
        _parse_complicated(value)


@pytest.mark.parametrize("value", [
    # 1) Bad brackets (is_valid_square_brackets)
    "[[][",          # unbalanced
    "[]]",           # extra closing
    "1,2,3]",        # missing opening
    "[1,2,3",        # missing closing

    # 2) Invalid characters / patterns (is_valid_index)
    "[1,b,3]",       # non-digit char
    "[4.,5]",        # decimal point not allowed
    "[1,,2,4,5]",    # empty entry
    "[1:5a]",        # junk after digit
    "[6,07,9]",      # leading zero after comma
    "[[1,2,3]4,5]",  # internal ']' not followed by comma

    # 3) Too many colons in a group
    "[1:::3]",       # ':::' not allowed

    # 4) Slicing not allowed *inside* fancy indexing (is_fancy)
    "[[1:2],[3,4]]",
    "[[1,2],[3:4]]",

    # 5) Invalid slice: zero step (parse_slice)
    "[1:5:0]",
    "[::0]",

    # 6) Minus sign not followed by digit
    "[-,1,2]",
])
def test_parse_index_negative(value):
    with pytest.raises(ValueError):
        _parse_index(value)



def test_check_and_get_config_bad_json():
    file_path = os.path.abspath(__file__)
    test_dir = os.path.dirname(file_path)
    proj_dir = os.path.dirname(test_dir)


    config_path = os.path.join(proj_dir,'uam4x', 'conf.json')
    os.remove(config_path)
    with  open(config_path, 'w') as  f:
        f.writelines('bogus stuff')

    with pytest.raises(IOError):
        check_and_get_config(config_path)

    if os.path.exists(config_path):
        os.remove(config_path)

def test_get_param_type_raises():

    @register_plugin
    class TestPlug(Plugin):
        def setup_plugin(self, plugin_options):
            Plugin.declare_input(self, "testin_cont", float, 0.0)
            Plugin.declare_input(self, "testin_discrete", int, 1)
            Plugin.declare_output(self, "testout_cont", float, 0.0)
            Plugin.declare_output(self, "testout_discrete", int, 1)

        def run_plugin(self, links: dict, work_dir: str):
            return {"testout_cont": 0.06, "testout_discrete": 6}

    # ----- build the SAME minimal study as in your example -----
    study = Study()

    mtr = Vehicle("mtr")

    wing = Wing("wing")
    wing.set_parameter("span", 35.56)
    mtr.add_component(wing)

    rotor = Rotor("rotor")
    rotor.set_parameter("nblade", 8)
    mtr.add_component(rotor)

    aircraft = Aircraft("aircraft")
    aircraft.set_parameter("Npass", 6)
    mtr.add_component(aircraft)

    study.add_vehicle(mtr)

    # Add plugin step
    study.solution.add_step(TestPlug, "test_plug", {})

    plugins = study.solution.generate_plugins()

    with pytest.raises(AttributeError):
        study.get_param_type("vehicles.mtr.wing.NotARealParam")

    with pytest.raises(AttributeError):
        study.get_param_type("vehicles.mtr.NotAComponent.height")

    with pytest.raises(ValueError):
        study.get_param_type("vehicles.NotAVehicle.wing.span")

    with pytest.raises(ValueError):
        study.get_param_type("vehicles.mtr.wing")

    with pytest.raises(KeyError):
        study.solution.get_var_type("test_plug.NotARealIO", plugins)

    with pytest.raises(KeyError):
        study.solution.get_var_type("NotAPlugin.testin_cont", plugins)

if __name__ == '__main__':
    test_check_and_get_config_good()
