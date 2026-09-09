import os

os.environ.setdefault("UAM4X_DISABLE_REGISTRY_WRITES", "1")

import numpy as np
import pytest

from uam4x.study import Study
from uam4x.vehicle import Vehicle
from uam4x.subsystems import Wing
from uam4x.solution import Plugin, Group, Solution, register_plugin


@register_plugin
class TypeProbePlugin(Plugin):
    def setup_plugin(self, plugin_options=None):
        self.declare_input("x_float", float)
        self.declare_input("x_int", int)
        self.declare_output("y_float", float)
        self.declare_output("y_arr", np.ndarray, shape=(2,))

    def run_plugin(self, links, work_dir):
        return {"y_float": 1.0, "y_arr": np.array([1.0, 2.0])}


@pytest.fixture
def typed_study():
    study = Study()

    study.add_solution(Solution("test_solution"))
    wing = Wing("wing")
    wing.set_parameter("span", 12.5)
    mode = wing.add_mode("cruise")
    mode.set_parameter("span", 15.0)

    vehicle = Vehicle("veh")
    vehicle.add_component(wing)
    study.add_vehicle(vehicle)

    study.get_solution("test_solution").add_step(TypeProbePlugin, "probe")
    study.get_solution("test_solution").sort_steps()

    return study


def test_study_get_param_type_for_component_and_mode(typed_study):
    s = typed_study

    assert s.get_param_type("vehicles.veh.wing.span") is float
    assert s.get_param_type("vehicles.veh.wing.cruise.span") is float


def test_study_get_param_type_invalid_path_raises(typed_study):
    with pytest.raises(ValueError, match="Invalid vehicle parameter"):
        typed_study.get_param_type("vehicles.veh.wing")


def test_solution_get_var_type_for_plugin_io(typed_study):
    s = typed_study
    plugins = s.get_solution("test_solution").generate_plugins()

    assert s.get_solution("test_solution").get_var_type("probe.x_float", plugins) is float
    assert s.get_solution("test_solution").get_var_type("probe.x_int", plugins) is int
    assert s.get_solution("test_solution").get_var_type("probe.y_float", plugins) is float
    assert s.get_solution("test_solution").get_var_type("probe.y_arr", plugins) is np.ndarray


def test_solution_get_var_type_nested_group(typed_study):
    s = typed_study

    nested = Group()
    nested.add_step(TypeProbePlugin, "inner_probe")
    s.get_solution("test_solution").add_step(nested, "nested")
    s.get_solution("test_solution").sort_steps()

    plugins = s.get_solution("test_solution").generate_plugins()

    assert s.get_solution("test_solution").get_var_type("nested.inner_probe.x_float", plugins) is float
    assert s.get_solution("test_solution").get_var_type("nested.inner_probe.y_arr", plugins) is np.ndarray


def test_solution_get_var_type_missing_name_raises(typed_study):
    plugins = typed_study.get_solution("test_solution").generate_plugins()

    with pytest.raises(KeyError, match="not an input/output"):
        typed_study.get_solution("test_solution").get_var_type("probe.not_declared", plugins)
