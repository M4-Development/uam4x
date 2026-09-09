import os

os.environ.setdefault("UAM4X_DISABLE_REGISTRY_WRITES", "1")

import numpy as np
import pytest

pytest.importorskip("openmdao.api")

import uam4x.openmdao_optimization as omo
from uam4x.study import Study
from uam4x.vehicle import Vehicle
from uam4x.subsystems import Subsystem, register_subsystem
from uam4x.solution import Group, Plugin, Solution, register_plugin


@register_subsystem
class OMOSubsystem(Subsystem):
    def setup(self):
        self.add_parameter("x", default_value=2.0, dtype=float)


@register_plugin
class OMOPlugin(Plugin):
    def setup_plugin(self, plugin_options=None):
        self.declare_input("x", float)
        self.declare_output("y", float)
        self.declare_output("y_arr", np.ndarray, shape=(2,))

    def run_plugin(self, links, work_dir):
        x = float(links["x"])
        return {"y": x + 1.0, "y_arr": np.array([x, x + 2.0])}


@register_plugin
class OMOArrayProducer(Plugin):
    def setup_plugin(self, plugin_options=None):
        self.declare_output("values", np.ndarray, value=np.array([3.0, 4.0]))

    def run_plugin(self, links, work_dir):
        return {"values": np.array([3.0, 4.0])}


@register_plugin
class OMOArraySink(Plugin):
    def setup_plugin(self, plugin_options=None):
        self.declare_input(
            "values", np.ndarray, value=np.array([10.0, 20.0, 30.0])
        )
        self.declare_output("total", float)

    def run_plugin(self, links, work_dir):
        return {"total": float(np.sum(links["values"]))}


class _FakeComp:
    def __init__(self):
        self.options = {}


class _FakeDriver:
    def __init__(self, fail=False):
        self.options = {}
        self.opt_settings = {}
        self.fail = fail


class _FakeModel:
    def __init__(self):
        self.options = {}
        self.subsystems = {}
        self.connections = []
        self.design_vars = []
        self.constraints = []
        self.objective = None

    def add_subsystem(self, name, comp, promotes=None):
        self.subsystems[name] = comp
        return comp

    def connect(self, src, dst, src_indices=None):
        self.connections.append((src, dst, src_indices))

    def add_design_var(self, name, lower=None, upper=None, indices=None):
        self.design_vars.append((name, lower, upper, indices))

    def add_constraint(self, name, lower=None, upper=None, indices=None):
        self.constraints.append((name, lower, upper, indices))

    def add_objective(self, name):
        self.objective = name


class _FakeProblem:
    fail_driver = False
    values = {}

    def __init__(self):
        self.model = _FakeModel()
        self.driver = None

    def setup(self):
        return None

    def run_driver(self):
        if self.driver is not None:
            self.driver.fail = self.fail_driver

    def get_val(self, name):
        return self.values[name]


def _build_study_and_state(tmp_path, tracked_outputs):
    study = Study()
    study.add_solution(Solution("test_solution"))
    vehicle = Vehicle("v")
    vehicle.add_component(OMOSubsystem("core"))
    study.add_vehicle(vehicle)

    study.get_solution("test_solution").add_step(OMOPlugin, "analysis")
    study.get_solution("test_solution").add_link("vehicles.v.core.x", "analysis.x")
    for out in tracked_outputs:
        study.get_solution("test_solution").add_tracked_output(out)

    study.get_solution("test_solution").sort_steps()
    plugins = study.get_solution("test_solution").generate_plugins()

    state = {
        "optimization": True,
        "des_vars": {
            "vehicles.v.core.x": {"lower": 0.0, "upper": 5.0, "indices": None}
        },
        "constraints": {
            "analysis.y": {"lower": 0.0, "upper": 10.0, "indices": None}
        },
        "objective_name": "analysis.y",
        "driver": "ScipyOptimizeDriver",
        "driver_options": {"maxiter": 5},
        "optimizer_options": {"tol": 1e-8},
    }

    out_dir = tmp_path / "omo"
    out_dir.mkdir(exist_ok=True)
    return study, state, plugins, str(out_dir)


def test_run_problem_builds_model_and_collects_outputs(monkeypatch, tmp_path):
    study, state, plugins, out_dir = _build_study_and_state(tmp_path, ["analysis.y"])

    _FakeProblem.fail_driver = False
    _FakeProblem.values = {
        "analysis~y": np.array([3.5]),
        "vehicles~v~core~x": np.array([2.0]),
    }

    monkeypatch.setattr(omo.om, "Problem", _FakeProblem)
    monkeypatch.setitem(omo.DRIVER_NAME_TO_DRIVER, "ScipyOptimizeDriver", _FakeDriver)
    monkeypatch.setattr(omo, "OpenmdaoPreComp", _FakeComp)
    monkeypatch.setattr(omo, "UniversalOpenmdaoComp", _FakeComp)
    monkeypatch.setattr(omo, "UniversalOpenMDAOGroup", _FakeComp)

    opt = omo.Optimization(study, study.get_solution("test_solution"), state, plugins, out_dir)
    result = opt.run_problem()

    assert result.convergence_status is True
    assert result.outputs["analysis.y"] == pytest.approx(3.5)
    assert result.outputs["vehicles.v.core.x"] == pytest.approx(2.0)


def test_run_problem_handles_indexed_tracked_output_and_failed_driver(monkeypatch, tmp_path):
    study, state, plugins, out_dir = _build_study_and_state(tmp_path, ["analysis.y_arr[1]"])

    _FakeProblem.fail_driver = True
    _FakeProblem.values = {
        "analysis~y": np.array([3.5]),
        "analysis~y_arr": np.array([10.0, 20.0]),
        "vehicles~v~core~x": np.array([2.0]),
    }

    monkeypatch.setattr(omo.om, "Problem", _FakeProblem)
    monkeypatch.setitem(omo.DRIVER_NAME_TO_DRIVER, "ScipyOptimizeDriver", _FakeDriver)
    monkeypatch.setattr(omo, "OpenmdaoPreComp", _FakeComp)
    monkeypatch.setattr(omo, "UniversalOpenmdaoComp", _FakeComp)
    monkeypatch.setattr(omo, "UniversalOpenMDAOGroup", _FakeComp)

    opt = omo.Optimization(study, study.get_solution("test_solution"), state, plugins, out_dir)
    result = opt.run_problem()

    assert result.convergence_status is False
    assert result.outputs["analysis.y_arr[1]"] == pytest.approx(20.0)


def test_run_problem_returns_openmdao_problem_for_partial_checks(monkeypatch, tmp_path):
    study, state, plugins, out_dir = _build_study_and_state(tmp_path, ["analysis.y"])

    monkeypatch.setattr(omo.om, "Problem", _FakeProblem)
    monkeypatch.setitem(omo.DRIVER_NAME_TO_DRIVER, "ScipyOptimizeDriver", _FakeDriver)
    monkeypatch.setattr(omo, "OpenmdaoPreComp", _FakeComp)
    monkeypatch.setattr(omo, "UniversalOpenmdaoComp", _FakeComp)
    monkeypatch.setattr(omo, "UniversalOpenMDAOGroup", _FakeComp)

    opt = omo.Optimization(study, study.get_solution("test_solution"), state, plugins, out_dir, check_partials=True)
    problem = opt.run_problem()

    assert isinstance(problem, _FakeProblem)
    assert problem.driver is not None


@pytest.mark.parametrize(
    ("name", "group_name", "expected"),
    [
        ("vehicles.v.core.x", "array_group", "vehicles~v~core~x"),
        ("producer.values", "array_group", "array_group~producer~values"),
        ("producer.values", "", "producer~values"),
    ],
)
def test_qualify_name(name, group_name, expected):
    assert omo.qualify_name(name, group_name) == expected


def test_openmdao_group_assembles_indexed_destination(tmp_path):
    group = Group()
    group.add_step(OMOArrayProducer, "producer")
    group.add_step(OMOArraySink, "sink")
    group.add_link("producer.values[0]", "sink.values[1]")
    group.add_link("producer.values[1]", "sink.values[2]")
    group.sort_steps()

    plugins = group.generate_plugins()
    problem = omo.om.Problem()
    om_group = problem.model.add_subsystem(
        "array_group", omo.UniversalOpenMDAOGroup(), promotes=["*"]
    )
    om_group.options["group"] = group
    om_group.options["step_name"] = "array_group"
    om_group.options["sub_plugins"] = plugins
    om_group.options["work_dir"] = str(tmp_path / "group_work")
    om_group.options["links"] = group.links

    problem.model.options["auto_order"] = True
    problem.setup()
    problem.run_model()

    assert problem.get_val("array_group~sink~total").item() == pytest.approx(17.0)
