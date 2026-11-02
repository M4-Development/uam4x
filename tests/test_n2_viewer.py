import os
from types import SimpleNamespace

os.environ.setdefault("UAM4X_DISABLE_REGISTRY_WRITES", "1")

import numpy as np
import pytest

from uam4x.n2_viewer import (
    _component_dependencies,
    _json_value,
    _optimization_sets,
    _strongly_connected_components,
    build_n2_data,
    render_n2_html,
)
from uam4x.solution import Plugin, Solution


class N2Source(Plugin):
    def setup_plugin(self, plugin_options=None):
        self.declare_input("x", float)
        self.declare_output("y", float)

    def run_plugin(self, links, work_dir):
        return {"y": links["x"] + 1.0}


class N2Sink(Plugin):
    def setup_plugin(self, plugin_options=None):
        self.declare_input("x", float)
        self.declare_output("y", float)

    def run_plugin(self, links, work_dir):
        return {"y": links["x"] * 2.0}


def test_json_value_handles_numpy_and_nonfinite_values():
    assert _json_value(np.float64(3.5)) == 3.5
    assert _json_value({1: (np.nan, np.inf, -np.inf)}) == {
        "1": ["nan", "inf", "-inf"]
    }
    assert _json_value(np.arange(101))["summary"] == "array with shape (101,)"
    assert _json_value(np.arange(101))["min"] == 0.0
    assert _json_value(np.array([1, 2])) == [1, 2]
    assert _json_value(object()).startswith("<object object")


def test_component_dependencies_expand_declarations_and_ignore_unknown_names():
    plugin = SimpleNamespace(
        plugin_input_names={"x": {}, "z": {}},
        plugin_output_names={"y": {}, "w": {}},
        initial_partials=[],
    )
    dense = _component_dependencies(plugin, "p")
    assert len(dense) == 4
    assert all(not edge["declared"] for edge in dense)

    plugin.initial_partials = [
        {"wrt": ["*"], "of": ["y"], "method": "fd"},
        {"wrt": ["missing"], "of": ["w"]},
        {"wrt": ["x"], "of": ["missing"]},
    ]
    sparse = _component_dependencies(plugin, "p")
    assert {(edge["source"], edge["target"]) for edge in sparse} == {
        ("p.x", "p.y"), ("p.z", "p.y")
    }
    assert all(edge["declared"] and edge["method"] == "fd" for edge in sparse)


def test_cycle_detection_and_optimization_name_normalization():
    connections = [
        {"source_component": "a", "target_component": "b"},
        {"source_component": "b", "target_component": "a"},
        {"source_component": "c", "target_component": "c"},
        {"source_component": "outside", "target_component": "a"},
    ]
    assert _strongly_connected_components(["a", "b", "c", "d"], connections) == [
        ["a", "b"], ["c"]
    ]
    assert _optimization_sets({
        "des_vars": {"solution.source.x[0]": {}},
        "constraints": {"solution.sink.y[1]": {}},
        "objective_name": "solution.sink.y",
    }) == ({"source.x"}, {"sink.y"}, {"sink.y"})


def test_build_and_render_n2_model(tmp_path):
    solution = Solution("n2_test")
    solution.add_step(N2Source, "source")
    solution.add_step(N2Sink, "sink")
    solution.add_link("vehicles.v.c.p", "source.x")
    solution.add_link("source.y", "sink.x")
    solution.add_tracked_output("sink.y")
    solution.sort_steps()
    study = SimpleNamespace(
        get_value=lambda name: 2.0,
        get_param_type=lambda name: float,
    )
    model = build_n2_data(
        study, solution, solution.generate_plugins(),
        state={"optimization": True, "des_vars": {"source.x": {}},
               "constraints": {"sink.y": {}}, "objective_name": "sink.y"},
        include_values=True, initial_path="source", title="N2 <view>",
    )
    assert model["viewer"]["initial_path"] == "source"
    assert [(edge["source"], edge["target"]) for edge in model["connections"]] == [
        ("vehicles.v.c.p", "source.x"), ("source.y", "sink.x")
    ]
    assert next(v for v in model["variables"] if v["id"] == "sink.y")["objective"]
    assert next(v for v in model["variables"] if v["id"] == "vehicles.v.c.p")["value"] == 2.0
    with pytest.raises(ValueError, match="N2 path"):
        build_n2_data(study, solution, solution.generate_plugins(), initial_path="missing")

    path = render_n2_html(model, tmp_path / "nested" / "n2.html")
    assert path.exists()
    assert "N2 &lt;view&gt;" in path.read_text(encoding="utf-8")


def test_build_n2_data_recurses_group_and_tolerates_missing_vehicle_value():
    from uam4x.solution import Group

    nested = Group()
    nested.add_step(N2Source, "source")
    solution = Solution("nested_n2")
    solution.add_step(nested, "subgroup")
    solution.add_link("vehicles.v.c.p[0]", "subgroup.source.x")
    solution.sort_steps()
    study = SimpleNamespace(
        get_value=lambda name: (_ for _ in ()).throw(KeyError(name)),
        get_param_type=lambda name: float,
    )

    model = build_n2_data(
        study, solution, solution.generate_plugins(),
        include_values=True, initial_path="source",
    )
    assert model["viewer"]["initial_path"] == "subgroup.source"
    assert model["tree"]["children"][1]["id"] == "subgroup"
    assert model["connections"][0]["source_indices"] == "[0]"
    vehicle_var = next(v for v in model["variables"] if v["id"] == "vehicles.v.c.p")
    assert vehicle_var["value"] is None
    assert vehicle_var["dtype"] == "NoneType"
