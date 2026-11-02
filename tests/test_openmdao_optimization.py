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


@register_plugin
class OMODeclaredDefaults(Plugin):
    def setup_plugin(self, plugin_options=None):
        self.declare_input("usupt", float, value=7.5)
        self.declare_input("coordinates", np.ndarray, value=np.array([2.0, 3.0]))
        self.declare_input("count", int, value=4)
        self.declare_output("boom_density", float, value=7.5)
        self.declare_output("positions", np.ndarray, value=np.array([4.0, 5.0]))
        self.declare_output("label", str, value="ready")

    def run_plugin(self, links, work_dir):
        return {
            "boom_density": links["usupt"],
            "positions": links["coordinates"],
            "label": "ready",
        }


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
            "vehicles.v.core.x": {"lower": 0.0, "upper": 5.0, "initial_value": None, "indices": None}
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
    assert (tmp_path / "array_group.producer").is_dir()
    assert (tmp_path / "array_group.sink").is_dir()


def test_plugin_defaults_initialize_openmdao_variables(tmp_path):
    plugin = OMODeclaredDefaults()
    plugin.setup_plugin()
    problem = omo.om.Problem()
    comp = problem.model.add_subsystem(
        "defaults", omo.UniversalOpenmdaoComp(), promotes=["*"]
    )
    comp.options["plugin"] = plugin
    comp.options["input_names"] = {
        f"defaults.{name}" for name in plugin.plugin_input_names
    }
    comp.options["output_names"] = {
        f"defaults.{name}" for name in plugin.plugin_output_names
    }
    comp.options["work_dir"] = str(tmp_path)
    comp.options["step_name"] = "defaults"

    problem.setup()

    assert problem.get_val("defaults~usupt").item() == pytest.approx(7.5)
    assert problem.get_val("defaults~boom_density").item() == pytest.approx(7.5)
    np.testing.assert_array_equal(problem.get_val("defaults~coordinates"), [2.0, 3.0])
    np.testing.assert_array_equal(problem.get_val("defaults~positions"), [4.0, 5.0])
    assert problem.get_val("defaults~count") == 4
    assert problem.get_val("defaults~label") == "ready"


def test_cycle_solver_options_applied(tmp_path):
    group = Group()
    group._is_cycle = True
    group.cycle_solver = "NonlinearBlockGS"

    problem = omo.om.Problem()
    om_group = problem.model.add_subsystem("cycle", omo.UniversalOpenMDAOGroup())
    om_group.options["group"] = group
    om_group.options["step_name"] = "cycle"
    om_group.options["sub_plugins"] = {}
    om_group.options["work_dir"] = str(tmp_path)
    om_group.options["links"] = {}
    om_group.options["cycle_solver_options"] = {"rtol": 0.01, "use_aitken": True}

    problem.setup()
    assert om_group.nonlinear_solver.options["rtol"] == 0.01
    assert om_group.nonlinear_solver.options["use_aitken"] is True


def test_run_problem_passes_cycle_options_to_group(monkeypatch, tmp_path):
    study, state, _, out_dir = _build_study_and_state(tmp_path, ["analysis.y"])
    solution = study.get_solution("test_solution")
    solution.add_step(Group(), "cycle")
    solution.sort_steps()
    plugins = solution.generate_plugins()
    state["cycle_solver_options"] = {"rtol": 0.01, "use_aitken": True}

    monkeypatch.setattr(omo.om, "Problem", _FakeProblem)
    monkeypatch.setitem(omo.DRIVER_NAME_TO_DRIVER, "ScipyOptimizeDriver", _FakeDriver)
    monkeypatch.setattr(omo, "OpenmdaoPreComp", _FakeComp)
    monkeypatch.setattr(omo, "UniversalOpenmdaoComp", _FakeComp)
    monkeypatch.setattr(omo, "UniversalOpenMDAOGroup", _FakeComp)

    opt = omo.Optimization(study, solution, state, plugins, out_dir, check_partials=True)
    problem = opt.run_problem()
    assert problem.model.subsystems["cycle"].options["cycle_solver_options"] == state["cycle_solver_options"]


@pytest.mark.parametrize("scaling", [
    {}, {"ref": 5.}, {"ref0": -5.}, {"ref": 5., "ref0": -5.},
])
def test_design_variable_scaling_in_openmdao(tmp_path, monkeypatch, scaling):
    monkeypatch.setenv("OPENMDAO_REPORTS", "0")
    study, state, plugins, out_dir = _build_study_and_state(tmp_path, ["analysis.y"])
    state["des_vars"]["vehicles.v.core.x"].update(scaling)
    state["constraints"] = {}
    state["optimizer_options"] = {}
    monkeypatch.chdir(tmp_path)
    solution = study.get_solution("test_solution")
    solution.approx_totals(method="fd")
    opt = omo.Optimization(study, solution, state, plugins, out_dir, check_partials=True)
    problem = opt.run_problem()
    try:
        name = "vehicles~v~core~x"
        physical = problem.get_val(name).item()
        scaled = problem.driver.get_design_var_values()[name].item()
        ref, ref0 = scaling.get("ref", 1.), scaling.get("ref0", 0.)
        assert scaled == pytest.approx((physical - ref0) / (ref - ref0))
        assert 0. <= physical <= 5.
    finally:
        problem.cleanup()


def test_wrapped_plugin_exact_partials_are_mapped_to_openmdao_names():
    class ExactPlugin(Plugin):
        def setup_plugin(self, plugin_options=None):
            self.declare_input("x", float)
            self.declare_output("y", float)
            self.declare_partials("y", "x", method="exact")

        def compute_partials(self, links):
            return {("y", "x"): 3.0}

    plugin = ExactPlugin()
    plugin.setup_plugin()
    plugin._declare_partials = True
    component = omo.UniversalOpenmdaoComp()
    component.plugin = plugin
    component.step_name = "nested.analysis"
    component.input_names = {"nested.analysis.x"}
    component.output_names = {"nested.analysis.y"}
    component.plugin_dict = {"x": 2.0}
    partials = {}

    component.compute_partials({}, partials)
    assert partials == {("nested~analysis~y", "nested~analysis~x"): 3.0}

    component.output_names = {"nested.analysis.other"}
    with pytest.raises(ValueError, match="Missing links"):
        component.compute_partials({}, {})
