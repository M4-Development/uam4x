import os

os.environ.setdefault("UAM4X_DISABLE_REGISTRY_WRITES", "1")

import numpy as np
import pytest

from uam4x.solution import (
    Group,
    Solution,
    Plugin,
    register_plugin,
    get_plugin_class,
    get_plugin_names,
    _plugin_registry,
)
from uam4x.study import Study
from uam4x.vehicle import Vehicle
from uam4x.subsystems import Subsystem


class CoreSubsystem(Subsystem):
    def setup(self):
        self.add_parameter("x", default_value=2.0, dtype=float)
        self.add_parameter("arr", default_value=np.array([3.0, 4.0]), dtype=np.ndarray)
        self.add_parameter("n", default_value=2, dtype=int)


@register_plugin
class ProducerPlugin(Plugin):
    def setup_plugin(self, plugin_options=None):
        self.declare_input("x", float)
        self.declare_output("y", float)
        self.declare_output("arr_out", np.ndarray, shape=(2,))

    def run_plugin(self, links, work_dir):
        x = float(links["x"])
        return {"y": x + 1.0, "arr_out": np.array([x, x + 2.0])}


@register_plugin
class ConsumerPlugin(Plugin):
    def setup_plugin(self, plugin_options=None):
        self.declare_input("y_in", float)
        self.declare_input("arr_scalar", float)
        self.declare_output("z", float)

    def run_plugin(self, links, work_dir):
        return {"z": float(links["y_in"]) + float(links["arr_scalar"])}


@register_plugin
class IntSinkPlugin(Plugin):
    def setup_plugin(self, plugin_options=None):
        self.declare_input("n_in", int)
        self.declare_output("n_out", int)

    def run_plugin(self, links, work_dir):
        return {"n_out": int(links["n_in"]) + 1}


@pytest.fixture
def study_with_solution():
    study = Study()
    vehicle = Vehicle("veh")
    vehicle.add_component(CoreSubsystem("core"))
    study.add_vehicle(vehicle)

    study.solution.add_step(ProducerPlugin, "producer")
    study.solution.add_step(ConsumerPlugin, "consumer")

    study.solution.add_link("vehicles.veh.core.x", "producer.x")
    study.solution.add_link("producer.y", "consumer.y_in")
    study.solution.add_link("producer.arr_out[0]", "consumer.arr_scalar")
    study.solution.add_tracked_output("consumer.z")

    study.solution.sort_steps()
    plugins = study.solution.generate_plugins()
    return study, plugins


def test_group_add_step_and_step_name_defaults():
    g = Group()
    g.add_step(ProducerPlugin)
    g.add_step(ConsumerPlugin, "consumer")

    assert list(g.steps) == ["step1", "consumer"]
    assert g.get_step_names() == ["step1", "consumer"]


def test_group_add_step_rejects_invalid_inputs():
    g = Group()

    with pytest.raises(TypeError):
        g.add_step("not_a_plugin")

    with pytest.raises(ValueError):
        g.add_step(ProducerPlugin, step_name="bad[name")


def test_group_add_link_rejects_indexed_dependent():
    g = Group()
    with pytest.raises(ValueError):
        g.add_link("a.x", "b.y[0]")


def test_group_sort_steps_topological_and_unlinked():
    g = Group()
    g.add_step(ProducerPlugin, "a")
    g.add_step(ConsumerPlugin, "b")
    g.add_step(IntSinkPlugin, "unlinked")
    g.add_link("a.y", "b.y_in")

    g.sort_steps()

    assert "a" in g._execution_steps
    assert "b" in g._execution_steps
    assert "unlinked" in g._execution_steps
    assert list(g._execution_steps).index("a") < list(g._execution_steps).index("b")


def test_group_sort_steps_cycle_requires_solver():
    g = Group()
    g.add_step(ProducerPlugin, "a")
    g.add_step(ConsumerPlugin, "b")
    g.add_link("a.y", "b.y_in")
    g.add_link("b.z", "a.x")

    with pytest.raises(ValueError, match="Cycle solver must be selected"):
        g.sort_steps()


def test_group_sort_steps_cycle_with_solver_allowed():
    g = Group()
    g.add_step(ProducerPlugin, "a")
    g.add_step(ConsumerPlugin, "b")
    g.add_link("a.y", "b.y_in")
    g.add_link("b.z", "a.x")
    g.set_cycle_solver("NonlinearBlockGS")

    g.sort_steps()

    assert g._is_cycle is True
    assert list(g._execution_steps) == ["a", "b"]


def test_group_to_dict_from_dict_roundtrip_nested_group():
    inner = Group()
    inner.add_step(ProducerPlugin, "producer")

    sol = Solution()
    sol.add_step(inner, "inner")
    sol.add_step(ConsumerPlugin, "consumer")
    sol.add_link("inner.producer.y", "consumer.y_in")
    sol.add_link("inner.producer.arr_out[0]", "consumer.arr_scalar")
    sol.add_tracked_output("consumer.z")

    data = sol._to_dict()
    rebuilt = Solution._from_dict(data)

    assert isinstance(rebuilt.steps["inner"], Group)
    assert rebuilt.steps["inner"].group_name == "inner"
    assert rebuilt.steps["consumer"]["plugin"] is ConsumerPlugin
    assert rebuilt.tracked_outputs == ["consumer.z"]


def test_solution_add_remove_tracked_output_and_duplicate_rejection():
    sol = Solution()
    sol.add_tracked_output("consumer.z")

    with pytest.raises(RuntimeError):
        sol.add_tracked_output("consumer.z")

    sol.remove_tracked_output("consumer.z")
    assert sol.tracked_outputs == []


def test_solution_get_all_plugin_maps_with_nested_groups():
    inner = Group()
    inner.add_step(ProducerPlugin, "producer")

    sol = Solution()
    sol.add_step(inner, "inner")
    sol.add_step(ConsumerPlugin, "consumer")

    cls_map = sol.get_all_plugin_names_to_classes()
    hash_map = sol.get_all_plugin_names_to_hashes()

    assert cls_map["consumer"] is ConsumerPlugin
    assert cls_map["inner.producer"] is ProducerPlugin
    assert isinstance(hash_map["inner.producer"], str)
    assert len(hash_map["inner.producer"]) == 64


def test_solution_check_links_to_plugins_positive(study_with_solution):
    study, plugins = study_with_solution
    study.solution._check_links_to_plugins(plugins)


def test_solution_check_links_to_plugins_negative_missing_input(study_with_solution):
    study, plugins = study_with_solution
    study.solution.add_link("producer.y", "consumer.not_declared")

    with pytest.raises(ValueError, match="was not declared in the plugin"):
        study.solution._check_links_to_plugins(plugins)


def test_solution_check_links_to_plugins_negative_missing_output(study_with_solution):
    study, plugins = study_with_solution
    study.solution.add_link("producer.missing", "consumer.y_in")

    with pytest.raises(ValueError, match="was not declared in the plugin"):
        study.solution._check_links_to_plugins(plugins)


def test_solution_check_link_types_positive(study_with_solution):
    study, plugins = study_with_solution
    study.solution._check_link_types(plugins, study)


def test_solution_check_link_types_negative_mismatch(study_with_solution):
    study, plugins = study_with_solution
    study.solution.add_step(IntSinkPlugin, "int_sink")
    plugins = study.solution.generate_plugins()
    study.solution.add_link("vehicles.veh.core.arr", "int_sink.n_in")

    with pytest.raises(TypeError, match="Link types must match"):
        study.solution._check_link_types(plugins, study)


def test_solution_check_link_types_negative_index_on_scalar(study_with_solution):
    study, plugins = study_with_solution
    study.solution.add_link("producer.y[0]", "consumer.y_in")

    with pytest.raises(NameError, match="link"):
        study.solution._check_link_types(plugins, study)


def test_solution_get_all_used_vehicle_params(study_with_solution):
    study, _ = study_with_solution
    used = study.solution.get_all_used_vehicle_params()

    assert "vehicles.veh.core.x" in used
    assert all(name.startswith("vehicles.") for name in used)


def test_plugin_declare_io_positive_and_invalid_names():
    p = ProducerPlugin()
    p.setup_plugin({})

    assert p.plugin_input_names["x"]["type"] is float

    with pytest.raises(ValueError, match="not allowed in variable names"):
        p.declare_input("bad[name", float)

    with pytest.raises(ValueError, match="not allowed in variable names"):
        p.declare_output("bad]name", float)


def test_plugin_declare_array_shape_rules():
    p = Plugin()

    with pytest.raises(ValueError, match="only np.ndarray types are allowed"):
        p.declare_input("x", float, shape=(2,))

    with pytest.raises(ValueError, match="must have either the shape or value defined"):
        p.declare_output("x", np.ndarray)


def test_plugin_declare_partials_and_build_partials_positive():
    p = ProducerPlugin()
    p.setup_plugin({})
    p.declare_partials_plugin(of="y", wrt="x", method="fd", step=1e-6, form="forward")

    partials = p.build_partials(["producer.x"], ["producer.y"], "producer")
    categorized = p.categorize_partials()

    assert ("y", "x") in partials
    assert categorized["fd"] == [("y", "x")]
    assert categorized["exact"] == []


def test_plugin_build_partials_rejects_unknown_names():
    p = ProducerPlugin()
    p.setup_plugin({})
    p.declare_partials_plugin(of="y", wrt="missing", method="fd")

    with pytest.raises(KeyError, match="declared as a partial is not an input in the Plugin"):
        p.build_partials(["producer.x"], ["producer.y"], "producer")


def test_plugin_build_partials_requires_continuous_types():
    p = IntSinkPlugin()
    p.setup_plugin({})
    p.declare_partials_plugin(of="n_out", wrt="n_in", method="fd")

    with pytest.raises(TypeError, match="must be continuous"):
        p.build_partials(["int_sink.n_in"], ["int_sink.n_out"], "int_sink")


def test_plugin_declare_template_and_duplicate_key(tmp_path):
    p = Plugin()
    template = tmp_path / "template.txt"
    template.write_text("abc", encoding="utf-8")

    p.declare_template(str(template))
    assert str(template) in p._template_to_checksum

    with pytest.raises(KeyError, match="already in the template dictionary"):
        p.declare_template(str(template))


def test_get_plugin_class_and_names_and_missing(clean_plugin_registry):
    @register_plugin
    class LocalPlugin(Plugin):
        pass

    assert get_plugin_class("LocalPlugin") is LocalPlugin
    assert "LocalPlugin" in get_plugin_names()

    with pytest.raises(ImportError, match="was not registered"):
        get_plugin_class("DefinitelyMissingPlugin")


@pytest.fixture
def clean_plugin_registry():
    snapshot = dict(_plugin_registry)
    _plugin_registry.clear()
    try:
        yield
    finally:
        _plugin_registry.clear()
        _plugin_registry.update(snapshot)
