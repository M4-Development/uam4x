import os

os.environ.setdefault("UAM4X_DISABLE_REGISTRY_WRITES", "1")

import numpy as np
import pytest

pytest.importorskip("openmdao.api")

import uam4x.study as study_module
from uam4x.execution_engine import Engine
from uam4x.study import Study
from uam4x.vehicle import Vehicle
from uam4x.subsystems import Subsystem, register_subsystem
from uam4x.solution import Plugin, register_plugin, Group
from uam4x.control import ParameterSweep


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


@pytest.fixture
def engine_setup(tmp_path):
    study = Study()

    vehicle = Vehicle("exec_vehicle")
    vehicle.add_component(ExecSubsystem("core"))
    study.add_vehicle(vehicle)

    study.solution.add_step(Stage1Plugin, "stage1")
    study.solution.add_step(Stage2Plugin, "stage2")

    study.solution.add_link("vehicles.exec_vehicle.core.x", "stage1.x")
    study.solution.add_link("vehicles.exec_vehicle.core.arr[0,:]", "stage1.arr_row")
    study.solution.add_link("vehicles.exec_vehicle.core.n", "stage1.n")

    study.solution.add_link("stage1.y", "stage2.y")
    study.solution.add_link("stage1.arr_scalar", "stage2.arr_scalar")

    study.solution.add_tracked_output("stage2.objective")

    study.solution.sort_steps()
    plugins = study.solution.generate_plugins()

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

    engine.check_links_to_plugins(plugins)
    engine.check_link_types(plugins)
    engine.check_link_indices(plugins)


def test_check_links_to_plugins_missing_input_raises(engine_setup):
    study, engine, plugins = engine_setup
    study.solution.add_link("stage1.y", "stage2.missing_input")

    with pytest.raises(ValueError, match="was not declared in the plugin"):
        engine.check_links_to_plugins(plugins)


def test_check_links_to_plugins_missing_output_raises(engine_setup):
    study, engine, plugins = engine_setup
    study.solution.add_link("stage1.missing_output", "stage2.y")

    with pytest.raises(ValueError, match="was not declared in the plugin"):
        engine.check_links_to_plugins(plugins)


def test_check_link_types_mismatch_raises(engine_setup):
    study, engine, plugins = engine_setup
    study.solution.add_link("vehicles.exec_vehicle.core.arr", "stage2.arr_scalar")

    with pytest.raises(TypeError, match="Link types must match"):
        engine.check_link_types(plugins)


def test_check_link_types_index_on_scalar_raises(engine_setup):
    study, engine, plugins = engine_setup
    study.solution.add_link("stage1.y[0]", "stage2.y")

    with pytest.raises(ValueError, match="Indexes may only be specified for arrays"):
        engine.check_link_types(plugins)


def test_check_link_types_nested_group_allows_nested_array_index(engine_setup):
    study, engine, _ = engine_setup

    nested = Group()
    nested.add_step(NestedMatrixPlugin, "matrix")

    study.solution.add_step(nested, "nested")
    study.solution.add_step(NestedVectorSinkPlugin, "vector_sink")
    study.solution.add_link("nested.matrix.matrix[0,:]", "vector_sink.in_vec")
    study.solution.sort_steps()

    plugins = study.solution.generate_plugins()
    engine.check_link_types(plugins)


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

    study.solution.add_step(nested, "nested")
    study.solution.add_step(BadVectorSinkPlugin, "vector_sink_bad")
    study.solution.add_link("nested.matrix.matrix[0,:]", "vector_sink_bad.in_vec")
    study.solution.sort_steps()

    plugins = study.solution.generate_plugins()

    with pytest.raises(ValueError, match="source indices must match the target shape"):
        engine.check_link_indices(plugins)


def test_check_link_indices_bad_vehicle_index_raises(engine_setup):
    study, engine, plugins = engine_setup
    study.solution.add_link("vehicles.exec_vehicle.core.arr[5,:]", "stage1.arr_row")

    with pytest.raises(ValueError, match="not a valid means to index"):
        engine.check_link_indices(plugins)


def test_check_model_full_optimization_positive(engine_setup):
    _, engine, plugins = engine_setup

    state = {
        "optimization": True,
        "des_vars": {
            "vehicles.exec_vehicle.core.x": {"lower": 0.0, "upper": 10.0, "indices": None}
        },
        "constraints": {
            "stage2.objective": {"lower": 0.0, "upper": 100.0, "indices": None}
        },
        "objective_name": "stage2.objective",
    }

    engine.check_model(plugins, state)


def test_check_desvars_non_continuous_raises(engine_setup):
    _, engine, plugins = engine_setup

    state = {
        "des_vars": {
            "vehicles.exec_vehicle.core.n": {"lower": 0.0, "upper": 10.0, "indices": None}
        }
    }

    with pytest.raises(TypeError, match="Only floats allowed as design variables"):
        engine.check_desvars(plugins, state)


def test_check_constraints_non_continuous_raises(engine_setup):
    _, engine, plugins = engine_setup

    state = {
        "constraints": {
            "stage1.disc": {"lower": 0.0, "upper": 10.0, "indices": None}
        }
    }

    with pytest.raises(TypeError, match="Only floats allowed as constraints"):
        engine.check_constraints(plugins, state)


def test_check_objective_non_continuous_raises(engine_setup):
    _, engine, plugins = engine_setup

    state = {"objective_name": "stage1.disc"}

    with pytest.raises(TypeError, match="Only continuous variables allowed as objectives"):
        engine.check_objective(plugins, state)


def test_check_upper_lower_array_bounds_positive_and_negative(engine_setup):
    study, engine, plugins = engine_setup
    var_data = engine._get_analysis_var_data("stage1.arr_out", plugins)

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

    result = engine._run_single_point(state={}, plugins=plugins, out_dir=str(tmp_path / "single"))

    assert result.outputs["stage2.objective"] == pytest.approx(40.0)


def test_run_state_parallel_rebuilds_study_and_runs(tmp_path):
    study = Study()
    vehicle = Vehicle("exec_vehicle")
    vehicle.add_component(ExecSubsystem("core"))
    study.add_vehicle(vehicle)

    study.solution.add_step(Stage1Plugin, "stage1")
    study.solution.add_step(Stage2Plugin, "stage2")
    study.solution.add_link("vehicles.exec_vehicle.core.x", "stage1.x")
    study.solution.add_link("vehicles.exec_vehicle.core.arr[0,:]", "stage1.arr_row")
    study.solution.add_link("vehicles.exec_vehicle.core.n", "stage1.n")
    study.solution.add_link("stage1.y", "stage2.y")
    study.solution.add_link("stage1.arr_scalar", "stage2.arr_scalar")
    study.solution.add_tracked_output("stage2.objective")

    result = Engine._run_state_parallel(
        study._to_dict(),
        state={},
        iteration=None,
        work_dir=str(tmp_path / "parallel"),
    )

    assert result.outputs["stage2.objective"] == pytest.approx(40.0)


def test_run_solution_populates_results_and_runtime_metadata(engine_setup, monkeypatch):
    study, engine, _ = engine_setup
    _mock_python_cmd(monkeypatch)

    engine.run_solution()

    ts = study.results.tradestudies["Default State"]
    assert len(ts._results) == 1
    assert ts._results[0].outputs["stage2.objective"] == pytest.approx(40.0)

    md = study._runtime_meta_data["Default State"]
    assert "environment" in md
    assert "step_plugin_hashes" in md
    assert "Completion_Time" in md["time_info"]


def test_run_tradestudy_parameter_sweep(engine_setup, monkeypatch):
    study, engine, _ = engine_setup
    _mock_python_cmd(monkeypatch)

    sweep = ParameterSweep({"vehicles.exec_vehicle.core.x": [1.0, 2.0]})
    study.add_trade("sweep", sweep)

    engine.run_tradestudy("sweep", num_processors=1)

    ts = study.results.tradestudies["sweep"]
    assert len(ts._results) == 2

    values = [r.outputs["stage2.objective"] for r in ts._results]
    assert values == [pytest.approx(38.0), pytest.approx(40.0)]


def test_run_parameter_sweep_invalid_num_processors_raises(engine_setup):
    study, engine, _ = engine_setup
    study.add_trade("invalid_np", ParameterSweep({"vehicles.exec_vehicle.core.x": [1.0]}))

    with pytest.raises(ValueError, match="num_processors must be None or an integer >=1"):
        engine._run_parameter_sweep("invalid_np", trade_states=[{}], num_processors=-1)


def test_capture_execution_data_stores_plugin_hashes(engine_setup, monkeypatch):
    study, engine, _ = engine_setup
    _mock_python_cmd(monkeypatch)

    engine.capture_execution_data("meta_trade")

    hashes = study._runtime_meta_data["meta_trade"]["step_plugin_hashes"]
    assert "stage1" in hashes
    assert "stage2" in hashes
    assert all(len(v) == 64 for v in hashes.values())
