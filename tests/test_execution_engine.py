import os
import base64
import json
import re
from pathlib import Path

os.environ.setdefault("UAM4X_DISABLE_REGISTRY_WRITES", "1")

import numpy as np
import pytest

pytest.importorskip("openmdao.api")

import uam4x.study as study_module
import uam4x.execution_engine as execution_engine_module
from uam4x.execution_engine import Engine
from uam4x.results import Result, TradestudyResults
from uam4x.study import Study
from uam4x.vehicle import Vehicle
from uam4x.subsystems import Subsystem, register_subsystem
from uam4x.solution import Plugin, register_plugin, Group, Solution
from uam4x.control import Optimization as ControlOptimization, ParameterSweep


@register_subsystem
class ExecSubsystem(Subsystem):
    def setup(self):
        self.add_parameter("x", default_value=2.0, dtype=float)
        self.add_parameter(
            "arr",
            default_value=np.array([[10.0, 20.0], [30.0, 40.0]]),
            dtype=np.ndarray,
        )
        self.add_parameter("n", default_value=3, dtype=int)


@register_plugin
class Stage1Plugin(Plugin):
    def setup_plugin(self, plugin_options=None):
        self.declare_input("x", float)
        self.declare_input("arr_row", np.ndarray, shape=(2,))
        self.declare_input("n", int)

        self.declare_output("y", float)
        self.declare_output("arr_out", np.ndarray, shape=(2, 2))
        self.declare_output("arr_scalar", float)
        self.declare_output("disc", int)

    def run_plugin(self, links, work_dir):
        x = float(links["x"])
        arr_row = links["arr_row"]
        n = int(links["n"])
        y = x + float(np.sum(arr_row)) + n
        return {
            "y": y,
            "arr_out": np.array([[1.0, 2.0], [3.0, 4.0]]) + x,
            "arr_scalar": x + 3.0,
            "disc": n * 2,
        }


@register_plugin
class Stage2Plugin(Plugin):
    def setup_plugin(self, plugin_options=None):
        self.declare_input("y", float)
        self.declare_input("arr_scalar", float)
        self.declare_output("objective", float)

    def run_plugin(self, links, work_dir):
        return {"objective": float(links["y"]) + float(links["arr_scalar"])}


@register_plugin
class NestedMatrixPlugin(Plugin):
    def setup_plugin(self, plugin_options=None):
        self.declare_input("seed", float)
        self.declare_output("matrix", np.ndarray, shape=(2, 2))

    def run_plugin(self, links, work_dir):
        seed = float(links["seed"])
        return {
            "matrix": np.array([[seed, seed + 1.0], [seed + 2.0, seed + 3.0]])
        }


@register_plugin
class NestedVectorSinkPlugin(Plugin):
    def setup_plugin(self, plugin_options=None):
        self.declare_input("in_vec", np.ndarray, shape=(2,))
        self.declare_output("scalar", float)

    def run_plugin(self, links, work_dir):
        return {"scalar": float(np.sum(links["in_vec"]))}


@register_plugin
class ArrayAssemblySinkPlugin(Plugin):
    def setup_plugin(self, plugin_options=None):
        self.declare_input(
            "values",
            np.ndarray,
            value=np.array([10.0, 20.0, 30.0]),
            shape=(3,),
        )
        self.declare_output("total", float)

    def run_plugin(self, links, work_dir):
        return {"total": float(np.sum(links["values"]))}


@register_plugin
class ZeroInitializedArraySinkPlugin(Plugin):
    def setup_plugin(self, plugin_options=None):
        self.declare_input("values", np.ndarray, shape=(3,))
        self.declare_output("values_out", np.ndarray, shape=(3,))

    def run_plugin(self, links, work_dir):
        return {"values_out": links["values"]}


@register_plugin
class ScalarDefaultArraySinkPlugin(Plugin):
    def setup_plugin(self, plugin_options=None):
        self.declare_input("values", np.ndarray, value=6.0, shape=(3,))
        self.declare_output("values_out", np.ndarray, shape=(3,))

    def run_plugin(self, links, work_dir):
        return {"values_out": links["values"]}


@pytest.fixture
def engine_setup(tmp_path):
    study = Study()

    study.add_solution(Solution("test_solution"))
    vehicle = Vehicle("exec_vehicle")
    vehicle.add_component(ExecSubsystem("core"))
    study.add_vehicle(vehicle)

    study.get_solution("test_solution").add_step(Stage1Plugin, "stage1")
    study.get_solution("test_solution").add_step(Stage2Plugin, "stage2")

    study.get_solution("test_solution").add_link("vehicles.exec_vehicle.core.x", "stage1.x")
    study.get_solution("test_solution").add_link("vehicles.exec_vehicle.core.arr[0,:]", "stage1.arr_row")
    study.get_solution("test_solution").add_link("vehicles.exec_vehicle.core.n", "stage1.n")

    study.get_solution("test_solution").add_link("stage1.y", "stage2.y")
    study.get_solution("test_solution").add_link("stage1.arr_scalar", "stage2.arr_scalar")

    study.get_solution("test_solution").add_tracked_output("stage2.objective")

    study.get_solution("test_solution").sort_steps()
    plugins = study.get_solution("test_solution").generate_plugins()

    engine = Engine(study, work_dir=str(tmp_path / "work"))
    return study, engine, plugins


def _mock_python_cmd(monkeypatch):
    monkeypatch.setattr(
        study_module.subprocess,
        "check_output",
        lambda *args, **kwargs: b"/usr/bin/python3\n",
    )


def test_check_links_to_plugins_and_link_types_and_indices_positive(engine_setup):
    study, engine, plugins = engine_setup

    engine.check_links_to_plugins(engine.study.get_solution("test_solution"), plugins)
    engine.check_link_types(engine.study.get_solution("test_solution"), plugins)
    engine.check_link_indices(engine.study.get_solution("test_solution"), plugins)


def test_check_links_to_plugins_missing_input_raises(engine_setup):
    study, engine, plugins = engine_setup
    study.get_solution("test_solution").add_link("stage1.y", "stage2.missing_input")

    with pytest.raises(ValueError, match="was not declared in the plugin"):
        engine.check_links_to_plugins(engine.study.get_solution("test_solution"), plugins)


def test_check_links_to_plugins_missing_output_raises(engine_setup):
    study, engine, plugins = engine_setup
    study.get_solution("test_solution").add_link("stage1.missing_output", "stage2.y")

    with pytest.raises(ValueError, match="was not declared in the plugin"):
        engine.check_links_to_plugins(engine.study.get_solution("test_solution"), plugins)


def test_check_link_types_mismatch_raises(engine_setup):
    study, engine, plugins = engine_setup
    study.get_solution("test_solution").add_link("vehicles.exec_vehicle.core.arr", "stage2.arr_scalar")

    with pytest.raises(TypeError, match="Link types must match"):
        engine.check_link_types(engine.study.get_solution("test_solution"), plugins)


def test_check_link_types_index_on_scalar_raises(engine_setup):
    study, engine, plugins = engine_setup
    study.get_solution("test_solution").add_link("stage1.y[0]", "stage2.y")

    with pytest.raises(ValueError, match="Indexes may only be specified for arrays"):
        engine.check_link_types(engine.study.get_solution("test_solution"), plugins)


def test_check_link_types_nested_group_allows_nested_array_index(engine_setup):
    study, engine, _ = engine_setup

    nested = Group()
    nested.add_step(NestedMatrixPlugin, "matrix")

    study.get_solution("test_solution").add_step(nested, "nested")
    study.get_solution("test_solution").add_step(NestedVectorSinkPlugin, "vector_sink")
    study.get_solution("test_solution").add_link("nested.matrix.matrix[0,:]", "vector_sink.in_vec")
    study.get_solution("test_solution").sort_steps()

    plugins = study.get_solution("test_solution").generate_plugins()
    engine.check_link_types(engine.study.get_solution("test_solution"), plugins)


def test_check_link_indices_nested_group_shape_mismatch_raises(engine_setup):
    study, engine, _ = engine_setup

    class BadVectorSinkPlugin(Plugin):
        def setup_plugin(self, plugin_options=None):
            self.declare_input("in_vec", np.ndarray, shape=(1,))
            self.declare_output("scalar", float)

        def run_plugin(self, links, work_dir):
            return {"scalar": float(np.sum(links["in_vec"]))}

    nested = Group()
    nested.add_step(NestedMatrixPlugin, "matrix")

    study.get_solution("test_solution").add_step(nested, "nested")
    study.get_solution("test_solution").add_step(BadVectorSinkPlugin, "vector_sink_bad")
    study.get_solution("test_solution").add_link("nested.matrix.matrix[0,:]", "vector_sink_bad.in_vec")
    study.get_solution("test_solution").sort_steps()

    plugins = study.get_solution("test_solution").generate_plugins()

    with pytest.raises(ValueError, match="source indices must match the target shape"):
        engine.check_link_indices(engine.study.get_solution("test_solution"), plugins)


def test_check_link_indices_bad_vehicle_index_raises(engine_setup):
    study, engine, plugins = engine_setup
    study.get_solution("test_solution").add_link("vehicles.exec_vehicle.core.arr[5,:]", "stage1.arr_row")

    with pytest.raises(ValueError, match="not a valid means to index"):
        engine.check_link_indices(engine.study.get_solution("test_solution"), plugins)


def test_check_model_full_optimization_positive(engine_setup):
    _, engine, plugins = engine_setup

    state = {
        "optimization": True,
        "des_vars": {
            "vehicles.exec_vehicle.core.x": {"lower": 0.0, "upper": 10.0, "initial_value": None, "indices": None}
        },
        "constraints": {
            "stage2.objective": {"lower": 0.0, "upper": 100.0, "indices": None}
        },
        "objective_name": "stage2.objective",
    }

    engine.check_model(engine.study.get_solution("test_solution"), plugins, state)


def test_check_desvars_non_continuous_raises(engine_setup):
    _, engine, plugins = engine_setup

    state = {
        "des_vars": {
            "vehicles.exec_vehicle.core.n": {"lower": 0.0, "upper": 10.0, "initial_value": None, "indices": None}
        }
    }

    with pytest.raises(TypeError, match="Only floats allowed as design variables"):
        engine.check_desvars(engine.study.get_solution("test_solution"), plugins, state)


def test_check_constraints_non_continuous_raises(engine_setup):
    _, engine, plugins = engine_setup

    state = {
        "constraints": {
            "stage1.disc": {"lower": 0.0, "upper": 10.0, "indices": None}
        }
    }

    with pytest.raises(TypeError, match="Only floats allowed as constraints"):
        engine.check_constraints(engine.study.get_solution("test_solution"), plugins, state)


def test_check_objective_non_continuous_raises(engine_setup):
    _, engine, plugins = engine_setup

    state = {"objective_name": "stage1.disc"}

    with pytest.raises(TypeError, match="Only continuous variables allowed as objectives"):
        engine.check_objective(engine.study.get_solution("test_solution"), plugins, state)


def test_check_upper_lower_array_bounds_positive_and_negative(engine_setup):
    study, engine, plugins = engine_setup
    var_data = engine._get_analysis_var_data(
        study.get_solution("test_solution"), "stage1.arr_out", plugins
    )

    engine.check_upper_lower(
        np.ndarray,
        0.0,
        1.0,
        "Constraint",
        "stage1.arr_out",
        var_data,
    )

    with pytest.raises(ValueError, match="shapes must match"):
        engine.check_upper_lower(
            np.ndarray,
            np.zeros((2, 2)),
            np.zeros((3, 2)),
            "Constraint",
            "stage1.arr_out",
            var_data,
        )


def test_run_single_point_returns_tracked_output(engine_setup, tmp_path):
    _, engine, plugins = engine_setup

    result = engine._run_single_point(solution=engine.study.get_solution("test_solution"), state={}, plugins=plugins, out_dir=str(tmp_path / "single"))

    assert result.outputs["stage2.objective"] == pytest.approx(40.0)


def test_run_single_point_applies_slice_to_runtime_plugin_output(tmp_path):
    study = Study()
    study.add_solution(Solution("test_solution"))
    vehicle = Vehicle("exec_vehicle")
    vehicle.add_component(ExecSubsystem("core"))
    study.add_vehicle(vehicle)

    study.get_solution("test_solution").add_step(Stage1Plugin, "stage1")
    study.get_solution("test_solution").add_step(NestedVectorSinkPlugin, "vector_sink")
    study.get_solution("test_solution").add_link("vehicles.exec_vehicle.core.x", "stage1.x")
    study.get_solution("test_solution").add_link("vehicles.exec_vehicle.core.arr[0,:]", "stage1.arr_row")
    study.get_solution("test_solution").add_link("vehicles.exec_vehicle.core.n", "stage1.n")
    study.get_solution("test_solution").add_link("stage1.arr_out[1,:]", "vector_sink.in_vec")
    study.get_solution("test_solution").add_tracked_output("vector_sink.scalar")
    study.get_solution("test_solution").sort_steps()

    engine = Engine(study, work_dir=str(tmp_path / "work"))
    plugins = study.get_solution("test_solution").generate_plugins()
    result = engine._run_single_point(
        solution=engine.study.get_solution("test_solution"),
        state={}, plugins=plugins, out_dir=str(tmp_path / "single")
    )

    # Stage1 adds x=2 to [[1, 2], [3, 4]], so row 1 is [5, 6].
    assert result.outputs["vector_sink.scalar"] == pytest.approx(11.0)


def test_run_single_point_applies_element_index_to_runtime_plugin_output(tmp_path):
    study = Study()
    study.add_solution(Solution("test_solution"))
    vehicle = Vehicle("exec_vehicle")
    vehicle.add_component(ExecSubsystem("core"))
    study.add_vehicle(vehicle)

    study.get_solution("test_solution").add_step(Stage1Plugin, "stage1")
    study.get_solution("test_solution").add_step(Stage2Plugin, "stage2")
    study.get_solution("test_solution").add_link("vehicles.exec_vehicle.core.x", "stage1.x")
    study.get_solution("test_solution").add_link("vehicles.exec_vehicle.core.arr[0,:]", "stage1.arr_row")
    study.get_solution("test_solution").add_link("vehicles.exec_vehicle.core.n", "stage1.n")
    study.get_solution("test_solution").add_link("stage1.arr_out[0,0]", "stage2.y")
    study.get_solution("test_solution").add_link("stage1.arr_out[0,1]", "stage2.arr_scalar")
    study.get_solution("test_solution").add_tracked_output("stage2.objective")
    study.get_solution("test_solution").sort_steps()

    engine = Engine(study, work_dir=str(tmp_path / "work"))
    plugins = study.get_solution("test_solution").generate_plugins()
    result = engine._run_single_point(
        solution=engine.study.get_solution("test_solution"),
        state={}, plugins=plugins, out_dir=str(tmp_path / "single")
    )

    # Stage1's first row is [3, 4]; Stage2 adds its two scalar inputs.
    assert result.outputs["stage2.objective"] == pytest.approx(7.0)


def test_run_single_point_assembles_scalar_links_into_array_input(tmp_path):
    study = Study()
    study.add_solution(Solution("test_solution"))
    vehicle = Vehicle("exec_vehicle")
    vehicle.add_component(ExecSubsystem("core"))
    study.add_vehicle(vehicle)

    study.get_solution("test_solution").add_step(ArrayAssemblySinkPlugin, "sink")
    study.get_solution("test_solution").add_link("vehicles.exec_vehicle.core.x", "sink.values[0]")
    study.get_solution("test_solution").add_link("vehicles.exec_vehicle.core.x", "sink.values[2]")
    study.get_solution("test_solution").add_tracked_output("sink.total")
    study.get_solution("test_solution").sort_steps()

    engine = Engine(study, work_dir=str(tmp_path / "work"))
    plugins = study.get_solution("test_solution").generate_plugins()
    result = engine._run_single_point(
        solution=engine.study.get_solution("test_solution"),
        state={}, plugins=plugins, out_dir=str(tmp_path / "single")
    )

    # The declared default [10, 20, 30] becomes [2, 20, 2].
    assert result.outputs["sink.total"] == pytest.approx(24.0)


@pytest.mark.parametrize(
    "plugin_class, expected",
    [
        (ZeroInitializedArraySinkPlugin, np.array([0.0, 2.0, 0.0])),
        (ScalarDefaultArraySinkPlugin, np.array([6.0, 2.0, 6.0])),
    ],
)
def test_run_single_point_initializes_partially_linked_array_input(
    tmp_path, plugin_class, expected
):
    study = Study()
    study.add_solution(Solution("test_solution"))
    vehicle = Vehicle("exec_vehicle")
    vehicle.add_component(ExecSubsystem("core"))
    study.add_vehicle(vehicle)

    study.get_solution("test_solution").add_step(plugin_class, "sink")
    study.get_solution("test_solution").add_link("vehicles.exec_vehicle.core.x", "sink.values[1]")
    study.get_solution("test_solution").add_tracked_output("sink.values_out")
    study.get_solution("test_solution").sort_steps()

    engine = Engine(study, work_dir=str(tmp_path / "work"))
    plugins = study.get_solution("test_solution").generate_plugins()
    result = engine._run_single_point(
        solution=engine.study.get_solution("test_solution"),
        state={}, plugins=plugins, out_dir=str(tmp_path / "single")
    )

    np.testing.assert_array_equal(result.outputs["sink.values_out"], expected)


def test_run_state_parallel_rebuilds_study_and_runs(tmp_path):
    study = Study()
    study.add_solution(Solution("test_solution"))
    vehicle = Vehicle("exec_vehicle")
    vehicle.add_component(ExecSubsystem("core"))
    study.add_vehicle(vehicle)

    study.get_solution("test_solution").add_step(Stage1Plugin, "stage1")
    study.get_solution("test_solution").add_step(Stage2Plugin, "stage2")
    study.get_solution("test_solution").add_link("vehicles.exec_vehicle.core.x", "stage1.x")
    study.get_solution("test_solution").add_link("vehicles.exec_vehicle.core.arr[0,:]", "stage1.arr_row")
    study.get_solution("test_solution").add_link("vehicles.exec_vehicle.core.n", "stage1.n")
    study.get_solution("test_solution").add_link("stage1.y", "stage2.y")
    study.get_solution("test_solution").add_link("stage1.arr_scalar", "stage2.arr_scalar")
    study.get_solution("test_solution").add_tracked_output("stage2.objective")

    trade_work_dir = tmp_path / "parallel"
    result = Engine._run_state_parallel(
        study._to_dict(),
        solution_name="test_solution",
        state={},
        iteration=0,
        trade_work_dir=str(trade_work_dir),
        create_step_work_dirs=False,
    )

    assert result.outputs["stage2.objective"] == pytest.approx(40.0)
    assert trade_work_dir.is_dir()
    assert not (trade_work_dir / "out_0").exists()
    assert not (trade_work_dir / "stage1").exists()
    assert not (trade_work_dir / "stage2").exists()


def test_run_solution_populates_results_and_runtime_metadata(engine_setup, monkeypatch):
    study, engine, _ = engine_setup
    _mock_python_cmd(monkeypatch)
    checked_states = []
    original_check_model = engine.check_model

    def record_check_model(solution, plugins, state=None):
        checked_states.append(state)
        return original_check_model(solution, plugins, state)

    monkeypatch.setattr(engine, "check_model", record_check_model)

    engine.run_solution("test_solution")

    step_dir = Path(engine.work_dir) / "test_solution" / "Default_State"
    assert (step_dir / "stage1").is_dir()
    assert not (step_dir / "solution.stage1").exists()
    assert checked_states == [{}]
    ts = study.results.get_tradestudy("test_solution", "Default State")
    assert len(ts._results) == 1
    assert ts._results[0].case_id == 0
    assert ts.get_case(0) is ts._results[0]
    assert ts._results[0].outputs["stage2.objective"] == pytest.approx(40.0)

    md = study._runtime_meta_data["test_solution"]["Default State"]
    assert "environment" in md
    assert "step_plugin_hashes" in md
    assert "Completion_Time" in md["time_info"]


def test_run_solution_checks_git_before_execution(engine_setup, monkeypatch):
    study, engine, _ = engine_setup

    def fail_git_check():
        raise RuntimeError("git check failed")

    monkeypatch.setattr(study, "check_git", fail_git_check)

    with pytest.raises(RuntimeError, match="git check failed"):
        engine.run_solution("test_solution", check_git=True)

    assert not (Path(engine.work_dir) / "test_solution" / "Default_State").exists()


def test_run_solution_keeps_results_separate_by_solution(engine_setup, monkeypatch):
    study, engine, _ = engine_setup
    _mock_python_cmd(monkeypatch)

    alternate = Solution("alternate")
    alternate.add_step(Stage1Plugin, "stage1")
    alternate.add_link("vehicles.exec_vehicle.core.x", "stage1.x")
    alternate.add_link("vehicles.exec_vehicle.core.arr[0,:]", "stage1.arr_row")
    alternate.add_link("vehicles.exec_vehicle.core.n", "stage1.n")
    alternate.add_tracked_output("stage1.y")
    study.add_solution(alternate)

    engine.run_solution("test_solution")
    engine.run_solution("alternate")

    primary = study.results.get_tradestudy("test_solution", "Default State")
    secondary = study.results.get_tradestudy("alternate", "Default State")

    assert primary._results[0].outputs["stage2.objective"] == pytest.approx(40.0)
    assert secondary._results[0].outputs["stage1.y"] == pytest.approx(35.0)
    assert set(study.results.get_all_solution_results()) == {
        "test_solution",
        "alternate",
    }
    assert "Default State" in study._runtime_meta_data["test_solution"]
    assert "Default State" in study._runtime_meta_data["alternate"]


def test_run_tradestudy_parameter_sweep(engine_setup, monkeypatch):
    study, engine, _ = engine_setup
    _mock_python_cmd(monkeypatch)

    sweep = ParameterSweep({"vehicles.exec_vehicle.core.x": [1.0, 2.0]})
    study.add_trade("sweep", sweep)

    engine.run_tradestudy("sweep", solution_name="test_solution", num_processors=1)

    ts = study.results.get_tradestudy("test_solution", "sweep")
    assert len(ts._results) == 2
    assert [result.case_id for result in ts._results] == [0, 1]
    assert ts.get_case(0) is ts._results[0]
    assert ts.get_case(1) is ts._results[1]

    values = [r.outputs["stage2.objective"] for r in ts._results]
    assert values == [pytest.approx(38.0), pytest.approx(40.0)]


def test_run_tradestudy_checks_git_before_execution(engine_setup, monkeypatch):
    study, engine, _ = engine_setup
    study.add_trade(
        "sweep", ParameterSweep({"vehicles.exec_vehicle.core.x": [1.0]})
    )

    def fail_git_check():
        raise RuntimeError("git check failed")

    monkeypatch.setattr(study, "check_git", fail_git_check)

    with pytest.raises(RuntimeError, match="git check failed"):
        engine.run_tradestudy(
            "sweep", solution_name="test_solution", check_git=True
        )

    assert not (Path(engine.work_dir) / "test_solution" / "sweep").exists()


def test_run_tradestudy_without_work_dirs_uses_shared_trade_directory(
    engine_setup, monkeypatch
):
    study, engine, _ = engine_setup
    _mock_python_cmd(monkeypatch)
    study.add_trade(
        "in_memory_sweep",
        ParameterSweep({"vehicles.exec_vehicle.core.x": [1.0, 2.0]}),
    )

    engine.run_tradestudy(
        "in_memory_sweep",
        solution_name="test_solution",
        num_processors=1,
        create_step_work_dirs=False,
    )

    trade_work_dir = (
        Path(engine.work_dir) / "test_solution" / "in_memory_sweep"
    )
    assert trade_work_dir.is_dir()
    assert not (trade_work_dir / "out_0").exists()
    assert not (trade_work_dir / "out_1").exists()
    assert not (trade_work_dir / "stage1").exists()
    assert not (trade_work_dir / "stage2").exists()

    ts = study.results.get_tradestudy("test_solution", "in_memory_sweep")
    assert [result.case_id for result in ts._results] == [0, 1]
    assert [result.outputs["stage2.objective"] for result in ts._results] == [
        pytest.approx(38.0),
        pytest.approx(40.0),
    ]


def test_run_tradestudy_optimization_uses_case_zero(engine_setup, monkeypatch):
    study, engine, _ = engine_setup
    _mock_python_cmd(monkeypatch)
    study.add_trade("optimization", ControlOptimization())
    optimization_result = Result(state={"optimization": True})
    monkeypatch.setattr(
        engine,
        "_run_optimization",
        lambda solution, state, plugins, out_dir: optimization_result,
    )

    engine.run_tradestudy("optimization", solution_name="test_solution", check=False)

    ts = study.results.get_tradestudy("test_solution", "optimization")
    assert len(ts._results) == 1
    assert ts.get_case(0) is optimization_result


def test_run_parameter_sweep_invalid_num_processors_raises(engine_setup):
    study, engine, _ = engine_setup
    study.add_trade("invalid_np", ParameterSweep({"vehicles.exec_vehicle.core.x": [1.0]}))

    with pytest.raises(ValueError, match="num_processors must be None or an integer >=1"):
        engine._run_parameter_sweep(
            study.get_solution("test_solution"),
            "invalid_np",
            trade_states=[{}],
            trade_work_dir=str(Path(engine.work_dir) / "test_solution" / "invalid_np"),
            num_processors=-1,
            trade_result=TradestudyResults(
                "invalid_np",
                "test_solution",
                executed_vehicles=study.vehicles,
                executed_tradestudy=study.control.trade_studies["invalid_np"],
                tracked_output_names=study.get_solution("test_solution").tracked_outputs,
            ),
        )


def test_check_partials_uses_named_solution_and_scoped_work_dir(
    engine_setup, monkeypatch
):
    study, engine, _ = engine_setup
    study.add_trade("optimization", ControlOptimization())
    captured = {}

    class FakeProblem:
        def check_partials(self, **kwargs):
            captured["partials"] = kwargs

    class FakeOptimization:
        def __init__(self, **kwargs):
            captured["optimization"] = kwargs

        def run_problem(self):
            return FakeProblem()

    monkeypatch.setattr(execution_engine_module, "Optimization", FakeOptimization)
    monkeypatch.setattr(engine, "check_model", lambda solution, plugins, state: None)
    original_cwd = os.getcwd()

    engine.check_partials(
        "optimization", "test_solution", form="central", step=1.0e-5
    )

    expected_dir = os.path.join(
        engine.work_dir, "test_solution", "optimization", "check_partials"
    )
    assert captured["optimization"]["solution"] is study.get_solution(
        "test_solution"
    )
    assert captured["optimization"]["out_dir"] == expected_dir
    assert captured["optimization"]["check_partials"] is True
    assert captured["partials"] == {
        "step": 1.0e-5,
        "form": "central",
        "compact_print": True,
    }
    assert os.getcwd() == original_cwd


def _read_n2_model(path):
    document = path.read_text(encoding="utf-8")
    match = re.search(r'const encodedModel="([^"]+)"', document)
    assert match is not None
    payload = base64.b64decode(match.group(1)).decode("utf-8")
    return document, json.loads(payload)


def test_generate_N2_creates_native_standalone_html(engine_setup, tmp_path):
    _, engine, _ = engine_setup
    outfile = tmp_path / "native_n2.html"

    path = engine.generate_N2("test_solution",
        outfile=outfile,
        show_browser=False,
        title="Execution engine model",
    )

    assert path == outfile.resolve()
    assert path.exists()

    document, model = _read_n2_model(path)
    assert "<!doctype html>" in document
    assert 'id="component-mode"' not in document
    assert 'id="variable-mode"' not in document
    assert "component-band" in document
    assert "variable-band" in document
    assert 'id="undo-graph"' in document
    assert 'id="collapse-element"' in document
    assert 'id="show-all-connections"' in document
    assert 'id="solver-button"' in document
    assert 'id="save-button"' not in document
    assert 'id="save-state-button"' not in document
    assert 'id="load-state-button"' not in document
    assert "MIN_CELL=1" in document
    assert "DETAIL_MIN_CELL=20" in document
    assert "function autoCollapsedFor(rootId)" in document
    assert "nodes.slice(0" not in document
    assert 'id="variables-button"' in document
    assert 'id="partials-button"' in document
    assert 'id="info-collapse"' in document
    assert 'id="info-reopen"' in document
    assert 'id="hierarchy-collapse"' in document
    assert 'id="hierarchy-reopen"' in document
    assert 'id="zoom-level" type="number"' in document
    assert 'id="matrix-controls-drag"' in document
    assert "function variableVisibilityDialog(componentId)" in document
    assert "function setZoomPercent(raw)" in document
    assert "34*percent/100" in document
    assert "Math.round(34*percent/100)" not in document
    assert "function moveMatrixControls(left,top)" in document
    assert "hiddenVariables:new Set()" in document
    assert "const compact=cell<state.font+6" in document
    assert "function applyPreferredScale()" in document
    assert "state.cell=Math.max(MIN_CELL" in document
    assert "function boundaryFor(cell)" in document
    assert '" · overview"' in document
    assert "OpenMDAO" not in model["title"]
    assert model["title"] == "Execution engine model"
    assert model["schema_version"] == 1
    assert model["solution_name"] == "test_solution"
    assert model["viewer"] == {
        "initial_path": "solution",
        "auto_collapse": True,
        "max_visible_rows": 120,
    }
    assert [component["id"] for component in model["components"]] == [
        "vehicles",
        "stage1",
        "stage2",
    ]

    connection = next(
        item
        for item in model["connections"]
        if item["target"] == "stage1.arr_row"
    )
    assert connection["source"] == "vehicles.exec_vehicle.core.arr"
    assert connection["source_indices"] == "[0,:]"

    variable = next(
        item for item in model["variables"] if item["id"] == "stage1.arr_out"
    )
    assert variable["direction"] == "output"
    assert variable["shape"] == [2, 2]
    assert variable["connected"] is False


def test_generate_N2_does_not_execute_plugins(engine_setup, monkeypatch, tmp_path):
    _, engine, plugins = engine_setup

    def fail_if_executed(*args, **kwargs):
        raise AssertionError("N2 generation must not execute plugins")

    for plugin in plugins.values():
        monkeypatch.setattr(plugin.__class__, "run_plugin", fail_if_executed)

    path = engine.generate_N2("test_solution",
        outfile=tmp_path / "no_execution.html",
        show_browser=False,
    )

    assert path.exists()


def test_generate_N2_includes_optimization_roles(engine_setup, tmp_path):
    study, engine, _ = engine_setup
    optimization = ControlOptimization()
    optimization.add_design_var(
        "vehicles.exec_vehicle.core.x",
        lower=0.0,
        upper=10.0,
    )
    optimization.add_constraint("stage1.arr_scalar", upper=10.0)
    optimization.add_objective("stage2.objective")
    study.add_trade("optimization", optimization)

    path = engine.generate_N2("test_solution",
        "optimization",
        outfile=tmp_path / "optimization_n2.html",
        show_browser=False,
    )

    _, model = _read_n2_model(path)
    assert model["optimization"] == {
        "enabled": True,
        "design_variables": ["vehicles.exec_vehicle.core.x"],
        "constraints": ["stage1.arr_scalar"],
        "objectives": ["stage2.objective"],
    }
    variables = {variable["id"]: variable for variable in model["variables"]}
    assert variables["vehicles.exec_vehicle.core.x"]["design_var"] is True
    assert variables["stage1.arr_scalar"]["constraint"] is True
    assert variables["stage2.objective"]["objective"] is True


def test_generate_N2_is_the_only_n2_generation_method(engine_setup):
    _, engine, _ = engine_setup

    path = engine.generate_N2("test_solution", show_browser=False)

    assert path == (
        Path(engine.work_dir) / "test_solution" / "n2.html"
    ).resolve()
    assert path.exists()
    assert not hasattr(engine, "generate_n2")


def test_generate_N2_supports_root_directory(engine_setup, tmp_path):
    _, engine, _ = engine_setup

    path = engine.generate_N2("test_solution", root_dir=tmp_path, show_browser=False)

    assert path == (tmp_path / "test_solution" / "n2.html").resolve()


def test_generate_N2_rejects_root_directory_and_outfile(engine_setup, tmp_path):
    _, engine, _ = engine_setup

    with pytest.raises(ValueError, match="either root_dir or outfile"):
        engine.generate_N2("test_solution",
            root_dir=tmp_path,
            outfile=tmp_path / "explicit.html",
            show_browser=False,
        )


def test_generate_N2_supports_initial_component_path(engine_setup, tmp_path):
    _, engine, _ = engine_setup

    path = engine.generate_N2("test_solution",
        outfile=tmp_path / "focused.html",
        path="solution.stage1",
        max_visible_rows=40,
        show_browser=False,
    )

    document, model = _read_n2_model(path)
    assert model["viewer"] == {
        "initial_path": "stage1",
        "auto_collapse": True,
        "max_visible_rows": 40,
    }
    assert "A focused component is never aggregated or truncated" in document


def test_generate_N2_rejects_unknown_path(engine_setup, tmp_path):
    _, engine, _ = engine_setup

    with pytest.raises(ValueError, match="not a unique uam4x group or component path"):
        engine.generate_N2("test_solution",
            outfile=tmp_path / "unknown.html",
            path="missing.system",
            show_browser=False,
        )


@pytest.mark.parametrize("row_limit", [True, 1, 9, 12.5])
def test_generate_N2_validates_visible_row_limit(engine_setup, tmp_path, row_limit):
    _, engine, _ = engine_setup

    expected = TypeError if isinstance(row_limit, (bool, float)) else ValueError
    with pytest.raises(expected):
        engine.generate_N2("test_solution",
            outfile=tmp_path / "bad_limit.html",
            max_visible_rows=row_limit,
            show_browser=False,
        )


def test_capture_execution_data_stores_plugin_hashes(engine_setup, monkeypatch):
    study, engine, _ = engine_setup
    _mock_python_cmd(monkeypatch)

    engine.capture_execution_data(
        "meta_trade", study.get_solution("test_solution")
    )

    hashes = study._runtime_meta_data["test_solution"]["meta_trade"]["step_plugin_hashes"]
    assert "stage1" in hashes
    assert "stage2" in hashes
    assert all(len(v) == 64 for v in hashes.values())
