import numpy as np
import pytest

from uam4x.control import Optimization, ParameterSweep, Control

DESIGN_VAR_NAME = "vehicles.multi_objective_vehicle.wing1.aspect_ratio"
CONSTRAINT_NAME = "analysis.wing.drag_constraint"





# -------------------- Optimization class --------------------

#-------Positive checks
def test_add_dv_no_bounds():

    opt = Optimization()

    opt.add_design_var(DESIGN_VAR_NAME)
    assert DESIGN_VAR_NAME in opt.des_vars


def test_add_dv_upper():
    opt = Optimization()
    opt.add_design_var(DESIGN_VAR_NAME, upper=10.0)
    assert opt.des_vars[DESIGN_VAR_NAME]["upper"] == 10.0


def test_add_dv_lower():
    opt = Optimization()
    opt.add_design_var(DESIGN_VAR_NAME, lower=1.0)
    assert opt.des_vars[DESIGN_VAR_NAME]["lower"] == 1.0


def test_add_dv_both_bounds():
    opt = Optimization()
    opt.add_design_var(DESIGN_VAR_NAME, lower=1.0, upper=10.0)
    assert opt.des_vars[DESIGN_VAR_NAME]["lower"] == 1.0
    assert opt.des_vars[DESIGN_VAR_NAME]["upper"] == 10.0


def test_add_dv_array_bounds():
    opt = Optimization()
    lower = np.array([0.0, 1.0], dtype=float)
    upper = np.array([1.0, 3.0], dtype=float)
    opt.add_design_var(DESIGN_VAR_NAME, lower=lower, upper=upper)
    np.testing.assert_array_equal(opt.des_vars[DESIGN_VAR_NAME]["lower"], lower)
    np.testing.assert_array_equal(opt.des_vars[DESIGN_VAR_NAME]["upper"], upper)


def test_add_constraint_lower_only():
    opt = Optimization()
    opt.add_constraint(CONSTRAINT_NAME, lower=0.0)
    assert opt.constraints[CONSTRAINT_NAME]["lower"] == 0.0
    assert opt.constraints[CONSTRAINT_NAME]["upper"] is None


def test_add_constraint_upper_only():
    opt = Optimization()
    opt.add_constraint(CONSTRAINT_NAME, upper=5.0)
    assert opt.constraints[CONSTRAINT_NAME]["upper"] == 5.0
    assert opt.constraints[CONSTRAINT_NAME]["lower"] is None


def test_add_constraint_array_bounds():
    opt = Optimization()
    lower = np.array([0.0, 1.0], dtype=float)
    upper = np.array([1.0, 3.0], dtype=float)
    opt.add_constraint(CONSTRAINT_NAME, lower=lower, upper=upper)
    np.testing.assert_array_equal(opt.constraints[CONSTRAINT_NAME]["lower"], lower)
    np.testing.assert_array_equal(opt.constraints[CONSTRAINT_NAME]["upper"], upper)


def test_add_objective_name():
    opt = Optimization()
    opt.add_objective('random_objective')

    assert opt.objective_name == 'random_objective'

def test_to_dict_good():
    opt = Optimization()
    opt.add_design_var(DESIGN_VAR_NAME, lower=1.0, upper=10.0)
    lower = np.array([0.0, 1.0], dtype=float)
    upper = np.array([1.0, 3.0], dtype=float)
    opt.add_constraint(CONSTRAINT_NAME, lower=lower, upper=upper)
    opt.add_objective('random_objective')

    opt_dict = opt._to_dict()

    assert opt_dict == {
        'type': 'Optimization',
        'des_vars': {DESIGN_VAR_NAME: {'lower': 1.0,'upper': 10.0,'indices': None}},
        'constraints':{CONSTRAINT_NAME: {'lower': lower,'upper': upper,'indices': None}},
        'objective_name': 'random_objective',
        'driver_options': {
            'optimizer': 'SLSQP',
        },
        'optimizer_options': {},
        'optimizer': 'ScipyOptimizeDriver',
    }

def test_Optimization_from_dict_array_and_scalar_bounds():
    # Hand-crafted dict that mimics what would come from JSON
    data = {
        "type": "optimization",
        "des_vars": {
            # scalar bounds
            "x": {"lower": 0.0, "upper": 10.0, 'indices': None},
            # vector bounds (as lists)
            "y": {"lower": [0.0, 1.0], "upper": [2.0, 3.0], 'indices': None},
        },
        "constraints": {
            # vector bounds (as lists)
            "c1": {"lower": [0.0, -1.0], "upper": [5.0, 1.0], 'indices': None},
            # scalar bounds
            "c2": {"lower": -1.0, "upper": 2.0, 'indices': None},
        },
        "objective_name": "obj",
        "driver_options": {"optimizer": "SLSQP", "maxiter": 200, "tol": 1e-6},
        "optimizer_options": {},
        "optimizer": "ScipyOptimizeDriver",
    }

    opt = Optimization._from_dict(data)

    # Basic class + attributes
    assert isinstance(opt, Optimization)
    assert opt.objective_name == "obj"

    # ---- Design variables ----
    dv_x = opt.des_vars["x"]
    dv_y = opt.des_vars["y"]

    # x: scalar bounds should stay scalars (not arrays)
    assert isinstance(dv_x["lower"], (int, float, np.floating))
    assert isinstance(dv_x["upper"], (int, float, np.floating))
    assert not isinstance(dv_x["lower"], np.ndarray)
    assert not isinstance(dv_x["upper"], np.ndarray)
    assert dv_x["lower"] == 0.0
    assert dv_x["upper"] == 10.0

    # y: list bounds should become numpy arrays
    assert isinstance(dv_y["lower"], np.ndarray)
    assert isinstance(dv_y["upper"], np.ndarray)
    assert np.array_equal(dv_y["lower"], np.array([0.0, 1.0]))
    assert np.array_equal(dv_y["upper"], np.array([2.0, 3.0]))

    # ---- Constraints ----
    c1 = opt.constraints["c1"]
    c2 = opt.constraints["c2"]

    # c1: lists → arrays
    assert isinstance(c1["lower"], np.ndarray)
    assert isinstance(c1["upper"], np.ndarray)
    assert np.array_equal(c1["lower"], np.array([0.0, -1.0]))
    assert np.array_equal(c1["upper"], np.array([5.0, 1.0]))

    # c2: scalars remain scalars
    assert isinstance(c2["lower"], (int, float, np.floating))
    assert isinstance(c2["upper"], (int, float, np.floating))
    assert not isinstance(c2["lower"], np.ndarray)
    assert not isinstance(c2["upper"], np.ndarray)
    assert c2["lower"] == -1.0
    assert c2["upper"] == 2.0


def test_optimization_from_dict_applies_optimizer_options():
    data = {
        "type": "Optimization",
        "des_vars": {},
        "constraints": {},
        "objective_name": "obj",
        "driver_options": {"optimizer": "SLSQP"},
        "optimizer_options": {"print_level": 0},
        "optimizer": "pyOptSparseDriver",
    }
    opt = Optimization._from_dict(data)
    assert opt.driver == "pyOptSparseDriver"
    assert opt._optimizer_options["print_level"] == 0

# -------------------- Failure paths: shared validator branches --------------------


@pytest.mark.parametrize(
    ("adder", "kwargs", "exc_type", "match"),
    [
        ("add_design_var", {"lower": "one"}, TypeError, r"The lower bound.*not continuos"),
        ("add_design_var", {"upper": "two"}, TypeError, r"The upper bound.*not continuos"),
        ("add_constraint", {"lower": "three"}, TypeError, r"The lower bound.*not continuos"),
        ("add_constraint", {"upper": "four"}, TypeError, r"The upper bound.*not continuos"),
        ("add_design_var", {"lower": [0.0, 1.0]}, TypeError, r"The lower bound.*not continuos"),
        ("add_design_var", {"upper": [0.0, 1.0]}, TypeError, r"The upper bound.*not continuos"),
        ("add_constraint", {"lower": [0.0, 1.0]}, TypeError, r"The lower bound.*not continuos"),
        ("add_constraint", {"upper": [0.0, 1.0]}, TypeError, r"The upper bound.*not continuos"),
        ("add_design_var", {"lower": (0.0, 1.0)}, TypeError, r"The lower bound.*not continuos"),
        ("add_design_var", {"upper": (0.0, 1.0)}, TypeError, r"The upper bound.*not continuos"),
        ("add_constraint", {"lower": (0.0, 1.0)}, TypeError, r"The lower bound.*not continuos"),
        ("add_constraint", {"upper": (0.0, 1.0)}, TypeError, r"The upper bound.*not continuos"),
        ("add_design_var", {"lower": {0.0, 1.0}}, TypeError, r"The lower bound.*not continuos"),
        ("add_design_var", {"upper": {0.0, 1.0}}, TypeError, r"The upper bound.*not continuos"),
        ("add_constraint", {"lower": {0.0, 1.0}}, TypeError, r"The lower bound.*not continuos"),
        ("add_constraint", {"upper": {0.0, 1.0}}, TypeError, r"The upper bound.*not continuos"),
        ("add_design_var", {"lower": 1}, TypeError, r"The lower bound.*not continuos"),
        ("add_design_var", {"upper": 1}, TypeError, r"The upper bound.*not continuos"),
        ("add_constraint", {"lower": 1}, TypeError, r"The lower bound.*not continuos"),
        ("add_constraint", {"upper": 1}, TypeError, r"The upper bound.*not continuos"),
        (
            "add_design_var",
            {"lower": 0.0, "upper": np.array([1.0, 2.0], dtype=float)},
            TypeError,
            r"Upper and lower bounds must be of the same type",
        ),
        (
            "add_constraint",
            {"lower": 0.0, "upper": np.array([1.0, 2.0], dtype=float)},
            TypeError,
            r"Upper and lower bounds must be of the same type",
        ),
        (
            "add_design_var",
            {"lower": np.array([0.0, 1.0], dtype=float), "upper": np.array([1.0], dtype=float)},
            ValueError,
            r"shapes must match",
        ),
        (
            "add_constraint",
            {"lower": np.array([0.0, 1.0], dtype=float), "upper": np.array([1.0], dtype=float)},
            ValueError,
            r"shapes must match",
        ),
        ("add_design_var", {"lower": 2.0, "upper": 1.0}, ValueError, r"upper bounds must be greater than or equal to the lower bounds"),
        ("add_constraint", {"lower": 2.0, "upper": 1.0}, ValueError, r"upper bounds must be greater than or equal to the lower bounds"),
        (
            "add_design_var",
            {"lower": np.array([0.0, 1.0], dtype=float), "upper": np.array([0.5, 0.0], dtype=float)},
            ValueError,
            r"upper bounds must be greater than or equal to the lower bounds",
        ),
        (
            "add_constraint",
            {"lower": np.array([0.0, 1.0], dtype=float), "upper": np.array([0.5, 0.0], dtype=float)},
            ValueError,
            r"upper bounds must be greater than or equal to the lower bounds",
        ),
    ],
)
def test_shared_bound_validation_failures(adder, kwargs, exc_type, match):
    opt = Optimization()
    name = DESIGN_VAR_NAME if "design" in adder else CONSTRAINT_NAME
    method = getattr(opt, adder)
    with pytest.raises(exc_type, match=match):
        method(name, **kwargs)


# -------------------- Constraint-only special rule --------------------


def test_add_constraint_requires_bound():
    opt = Optimization()
    with pytest.raises(ValueError, match="At least one constraint must be declared"):
        opt.add_constraint(CONSTRAINT_NAME)


# -------------------- Indices passthrough --------------------


def test_add_design_var_indices_passthrough():

    opt = Optimization()
    indices = [0, 2]
    opt.add_design_var(DESIGN_VAR_NAME, lower=0.0, upper=1.0, indices=indices)
    assert opt.des_vars[DESIGN_VAR_NAME]["indices"] is indices


def test_optimization_setters_and_get_desvar_names():
    opt = Optimization()
    opt.set_driver("pyOptSparseDriver")
    opt.add_driver_option("maxiter", 25)
    opt.add_optimizer_option("tol", 1e-8)
    opt.add_design_var("x", lower=0.0, upper=1.0)

    assert opt.driver == "pyOptSparseDriver"
    assert opt._mdao_driver_options["maxiter"] == 25
    assert opt._optimizer_options["tol"] == 1e-8
    assert opt.get_desvar_names() == ["x"]


def test_add_constraint_indices_passthrough():
    opt = Optimization()
    indices = np.array([1, 3])
    opt.add_constraint(CONSTRAINT_NAME, lower=0.0, upper=1.0, indices=indices)
    assert opt.constraints[CONSTRAINT_NAME]["indices"] is indices


#-----------------------------------Parameter Class

#--------Positive checks
def test_childless_parametersweep_class():
    sample_param_dict = {'sample_param1': [1.0, 10.0, 100.0, 1000.0, 10000.0],
                    'sample_param2': [6.0, 7.0, 67.0, 6767., 676767.]}

    PS = ParameterSweep(sample_param_dict)

    assert PS.child is None
    assert PS.param_name_to_value_list == sample_param_dict

def test_optimization_child_parameter_sweep():
    opt = Optimization()

    sample_param_dict = {'sample_param1': [1.0, 10.0, 100.0, 1000.0, 10000.0],
                    'sample_param2': [6.0, 7.0, 67.0, 6767., 676767.]}
    PS = ParameterSweep(param_name_to_value_list=sample_param_dict, child=opt)

    assert PS.child is  opt

def test_set_child_good():
    opt = Optimization()

    sample_param_dict = {'sample_param1': [1.0, 10.0, 100.0, 1000.0, 10000.0],
                    'sample_param2': [6.0, 7.0, 67.0, 6767., 676767.]}
    PS = ParameterSweep(param_name_to_value_list=sample_param_dict, child=opt)

    assert PS.child is  opt

    new_opt = Optimization()
    PS.set_child(new_opt)

    assert PS.child is  not opt
    assert PS.child is new_opt

def test_get_states():
    pass


def test_parameter_sweep_get_optim_child_and_names():
    opt = Optimization()
    opt.add_design_var("x", lower=0.0, upper=2.0)

    inner = ParameterSweep({"b": [3.0, 4.0]}, child=opt)
    outer = ParameterSweep({"a": [1.0, 2.0]}, child=inner)

    assert outer.get_optim_child() is opt
    assert inner.get_desvar_names() == ["x"]
    assert outer.get_desvar_names() == ["x"]
    assert outer.get_all_param_names() == {"a", "b"}


def test_parameter_sweep_get_optim_child_none_and_desvars_none():
    ps = ParameterSweep({"a": [1.0, 2.0]})
    assert ps.get_optim_child() is None
    assert ps.get_desvar_names() == []
    assert ps.get_all_param_names() == {"a"}


def test_parameter_sweep_get_states_with_optimization_child():
    opt = Optimization()
    opt.add_design_var("x", lower=0.0, upper=1.0)
    opt.add_constraint("c", lower=0.0)
    opt.add_objective("obj")

    ps = ParameterSweep({"a": [1.0, 2.0]}, child=opt)
    states = ps.get_states()

    assert len(states) == 2
    assert states[0]["optimization"] is True
    assert "des_vars" in states[0]
    assert states[1]["a"] == 2.0


def test_parameter_sweep_get_states_with_nested_child_sweep():
    inner = ParameterSweep({"b": [10.0, 20.0]})
    outer = ParameterSweep({"a": [1.0, 2.0]}, child=inner)

    states = outer.get_states()
    assert len(states) == 4
    assert {s["a"] for s in states} == {1.0, 2.0}
    assert {s["b"] for s in states} == {10.0, 20.0}
    assert all(s["optimization"] is False for s in states)


def test_parameter_sweep_from_dict_branches():
    no_child_data = {
        "type": "ParameterSweep",
        "param_name_to_value_list": {"a": [1.0, 2.0]},
        "child": None,
    }
    ps = ParameterSweep._from_dict(no_child_data)
    assert ps.child is None

    nested_data = {
        "type": "ParameterSweep",
        "param_name_to_value_list": {"a": [1.0, 2.0]},
        "child": {
            "type": "ParameterSweep",
            "param_name_to_value_list": {"b": [3.0, 4.0]},
            "child": None,
        },
    }
    ps_nested = ParameterSweep._from_dict(nested_data)
    assert isinstance(ps_nested.child, ParameterSweep)

#-------Negative checks
@pytest.mark.parametrize('input_dict,err_type', [
    ({'sample_param1': [1.0, 10.0, 100.0, 1000.0, 10000.0],
                    'sample_param2': (6.0, 7.0, 67.0, 6767., 676767.)},TypeError),
    ({'sample_param1': [1.0, 10.0, 100.0, 1000.0, 10000.0],
                    'sample_param2': [6.0, 7.0, 67.0, 676767.]}, ValueError)
])
def test_parameter_sweep_bad_inputs(input_dict, err_type):

    with pytest.raises(err_type):
        ParameterSweep(param_name_to_value_list=input_dict)

@pytest.mark.parametrize('opt',[
    (1.),
    (2),
    ('string'),
    ([1, 2., 'five']),
    ({'hello': 'goodbye'})
])
def test_set_child_bad_type(opt):
    sample_param_dict = {'sample_param1': [1.0, 10.0, 100.0, 1000.0, 10000.0],
                    'sample_param2': [6.0, 7.0, 67.0, 6767., 676767.]}
    PS = ParameterSweep(sample_param_dict)
    with pytest.raises(TypeError):
        PS.set_child(opt)


#------------------------------Control class

def test_Control_instance():
    C = Control()
    assert C.title == 'Default Title'
    assert C.author == 'Default Author'
    assert C.description == 'Default Description'

    assert C.trade_studies == {}


def test_control_add_metadata_and_reserved_trade_name():
    c = Control()
    c.add_metadata(title="T", author="A", description="D")
    assert c.title == "T"
    assert c.author == "A"
    assert c.description == "D"

    with pytest.raises(ValueError, match="reserved"):
        c.add_tradestudy("Default State", Optimization())


def test_Control_add_trade():
    opt = Optimization()



    sample_param_dict1 = {'sample_param1': [1.0, 10.0, 100.0, 1000.0, 10000.0],
                    'sample_param2': [6.0, 7.0, 67.0, 6767., 676767.]}

    param_sweep1 = ParameterSweep(sample_param_dict1, opt)

    C = Control()
    C.add_tradestudy('sample_trade1', param_sweep1)



    sample_param_dict2 = {'sample_param3': [8.0, 10.0, 100.0, 1000.0, 10000.0],
                    'sample_param4': [6.0, 7.0, 67.0, 6767., 676767.]}

    param_sweep2 = ParameterSweep(sample_param_dict2, opt)
    C.add_tradestudy('sample_trade2', param_sweep2)

    assert C.trade_studies['sample_trade2'] is param_sweep2
    assert C.trade_studies['sample_trade1'] is param_sweep1

def test_Control_to_dict():
    opt = Optimization()

    # First sweep
    sample_param_dict1 = {
        'sample_param1': [1.0, 10.0, 100.0, 1000.0, 10000.0],
        'sample_param2': [6.0, 7.0, 67.0, 6767.0, 676767.0],
    }
    param_sweep1 = ParameterSweep(sample_param_dict1, opt)

    # Second sweep
    sample_param_dict2 = {
        'sample_param3': [8.0, 10.0, 100.0, 1000.0, 10000.0],
        'sample_param4': [6.0, 7.0, 67.0, 6767.0, 676767.0],
    }
    param_sweep2 = ParameterSweep(sample_param_dict2, opt)

    C = Control()
    C.title = "Test Control"
    C.author = "Unit Tester"
    C.description = "Control used for _to_dict test"

    C.add_tradestudy('sample_trade1', param_sweep1)
    C.add_tradestudy('sample_trade2', param_sweep2)

    data = C._to_dict()

    # ---- Top-level Control fields ----
    expected_top = {
        "title": "Test Control",
        "author": "Unit Tester",
        "description": "Control used for _to_dict test",
    }

    assert data["title"] == expected_top["title"]
    assert data["author"] == expected_top["author"]
    assert data["description"] == expected_top["description"]

    # ---- trade_studies structure ----
    assert "trade_studies" in data
    trade_studies = data["trade_studies"]

    # Keys preserved
    assert set(trade_studies.keys()) == {"sample_trade1", "sample_trade2"}

    # Manually build the expected inner dict structure *for the parts we care about*
    ts1 = trade_studies["sample_trade1"]
    ts2 = trade_studies["sample_trade2"]

    # At minimum, we expect the parameter sweep dicts to carry the param grid
    expected_ts1 = {
        "param_name_to_value_list": sample_param_dict1,
    }
    expected_ts2 = {
        "param_name_to_value_list": sample_param_dict2,
    }

    # Make sure keys we expect exist
    for key in expected_ts1.keys():
        assert key in ts1, f"Missing key '{key}' in sample_trade1 dict"
    for key in expected_ts2.keys():
        assert key in ts2, f"Missing key '{key}' in sample_trade2 dict"

    # And values match exactly
    assert ts1["param_name_to_value_list"] == expected_ts1["param_name_to_value_list"]
    assert ts2["param_name_to_value_list"] == expected_ts2["param_name_to_value_list"]


def test_Control_from_dict_mixed_trades():
    # ----- Manually construct the dict that Control._from_dict expects -----
    sweep_params = {
        "altitude": [0.0, 5000.0, 10000.0],
        "mach": [0.3, 0.6, 0.8],
    }

    # Optimization used as the child of the sweep
    opt_dict_for_sweep = {
        "type": "Optimization",
        "des_vars": {
            "x": {"lower": 0.0, "upper": 10.0, "indices": None},
        },
        "constraints": {
            "c1": {"lower": -1.0, "upper": 1.0, "indices": None},
        },
        "objective_name": "obj_sweep",
        "driver_options": {"optimizer": "SLSQP", "maxiter": 200, "tol": 1e-6},
        "optimizer_options": {},
        "optimizer": "ScipyOptimizeDriver",
    }

    # Standalone optimization trade
    opt_dict_alone = {
        "type": "Optimization",
        "des_vars": {
            "y": {"lower": [0.0, 1.0], "upper": [2.0, 3.0], "indices": None},
        },
        "constraints": {},
        "objective_name": "obj_opt_only",
        "driver_options": {"optimizer": "SLSQP", "maxiter": 200, "tol": 1e-6},
        "optimizer_options": {},
        "optimizer": "ScipyOptimizeDriver",
    }

    control_dict = {
        "title": "Test Control",
        "author": "Unit Tester",
        "description": "Control from_dict test",
        "trade_studies": {
            "sweep_trade": {
                "type": "ParameterSweep",
                "param_name_to_value_list": sweep_params,
                "child": opt_dict_for_sweep,
            },
            "opt_trade": opt_dict_alone,
        },
    }

    # ----- Rebuild Control from dict -----
    control = Control._from_dict(control_dict)

    # ----- Top-level checks -----
    assert isinstance(control, Control)
    assert control.title == "Test Control"
    assert control.author == "Unit Tester"
    assert control.description == "Control from_dict test"

    # Trade keys preserved
    assert set(control.trade_studies.keys()) == {"sweep_trade", "opt_trade"}

    # ----- Check sweep_trade (ParameterSweep) -----
    sweep_trade = control.trade_studies["sweep_trade"]
    assert isinstance(sweep_trade, ParameterSweep)

    # Parameter grid restored
    assert sweep_trade.param_name_to_value_list == sweep_params

    # Child optimization inside the sweep
    child_opt = sweep_trade.child
    assert isinstance(child_opt, Optimization)
    assert child_opt.objective_name == "obj_sweep"
    assert "x" in child_opt.des_vars
    assert child_opt.des_vars["x"]["lower"] == 0.0
    assert child_opt.des_vars["x"]["upper"] == 10.0
    assert "c1" in child_opt.constraints
    assert child_opt.constraints["c1"]["lower"] == -1.0
    assert child_opt.constraints["c1"]["upper"] == 1.0

    # ----- Check opt_trade (standalone Optimization) -----
    opt_trade = control.trade_studies["opt_trade"]
    assert isinstance(opt_trade, Optimization)
    assert opt_trade.objective_name == "obj_opt_only"
    assert "y" in opt_trade.des_vars

    # lower/upper were lists → Optimization._from_dict should have converted to np.ndarray
    y = opt_trade.des_vars["y"]
    assert isinstance(y["lower"], np.ndarray)
    assert isinstance(y["upper"], np.ndarray)
    assert np.array_equal(y["lower"], np.array([0.0, 1.0]))
    assert np.array_equal(y["upper"], np.array([2.0, 3.0]))
    assert opt_trade.constraints == {}

#Negative checks
def test_Control_from_dict_unknown_type_raises():
    # Minimal bad control dict with an unknown trade type
    bad_control_dict = {
        "title": "Bad Control",
        "author": "Unit Tester",
        "description": "Should fail",
        "trade_studies": {
            "weird_trade": {
                "type": "NotARealType",
            }
        },
    }

    with pytest.raises(ValueError, match="Unknown trade study type"):
        Control._from_dict(bad_control_dict)
