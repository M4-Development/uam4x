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


@register_plugin
class MatrixProducerPlugin(Plugin):
    def setup_plugin(self, plugin_options=None):
        self.declare_input("seed", float)
        self.declare_output("grid", np.ndarray, shape=(2, 2))

    def run_plugin(self, links, work_dir):
        seed = float(links["seed"])
        return {
            "grid": np.array([[seed, seed + 1.0], [seed + 2.0, seed + 3.0]])
        }


@register_plugin
class MatrixSinkPlugin(Plugin):
    def setup_plugin(self, plugin_options=None):
        self.declare_input("arr_in", np.ndarray, shape=(1,))
        self.declare_output("total", float)

    def run_plugin(self, links, work_dir):
        return {"total": float(np.sum(np.asarray(links["arr_in"])))}


@register_plugin
class VectorSinkPlugin(Plugin):
    def setup_plugin(self, plugin_options=None):
        self.declare_input("arr_in", np.ndarray, shape=(3,))
        self.declare_output("total", float)

    def run_plugin(self, links, work_dir):
        return {"total": float(np.sum(np.asarray(links["arr_in"])))}


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


def test_group_add_link_accepts_indexed_dependent():
    g = Group()
    g.add_link("a.x", "b.y[0]")

    assert g.links == [["a.x", "b.y[0]"]]


def test_group_get_plugin_names_strips_indices_from_both_sides():
    g = Group()
    g.add_link("producer.arr_out[0]", "sink.arr_in[0]")

    input_names, output_names = g.get_plugin_input_and_output_names("sink")
    assert input_names == {"sink.arr_in"}
    assert output_names == set()

    input_names, output_names = g.get_plugin_input_and_output_names("producer")
    assert input_names == set()
    assert output_names == {"producer.arr_out"}


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
    g.declare_cycle_step_order(["a", "b"])

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


def test_solution_check_links_to_plugins_accepts_indexed_input(study_with_solution):
    study, _ = study_with_solution
    study.solution.add_step(MatrixSinkPlugin, "sink")
    study.solution.add_link("producer.y", "sink.arr_in[0]")
    plugins = study.solution.generate_plugins()

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


def test_solution_check_link_types_rejects_vehicle_link_declared_in_group():
    study = Study()
    vehicle = Vehicle("veh")
    vehicle.add_component(CoreSubsystem("core"))
    study.add_vehicle(vehicle)

    nested = Group()
    nested.add_step(ProducerPlugin, "producer")
    nested.add_link("vehicles.veh.core.x", "producer.x")
    study.solution.add_step(nested, "nested")

    study.solution.sort_steps()
    plugins = study.solution.generate_plugins()

    with pytest.raises(
        ValueError,
        match="Vehicle links inside groups are not supported",
    ):
        study.solution._check_link_types(plugins, study)


def test_solution_check_link_types_allows_top_level_vehicle_link_to_group():
    study = Study()
    vehicle = Vehicle("veh")
    vehicle.add_component(CoreSubsystem("core"))
    study.add_vehicle(vehicle)

    nested = Group()
    nested.add_step(ProducerPlugin, "producer")
    study.solution.add_step(nested, "nested")
    study.solution.add_link("vehicles.veh.core.x", "nested.producer.x")

    study.solution.sort_steps()
    plugins = study.solution.generate_plugins()

    study.solution._check_link_types(plugins, study)


def test_solution_check_link_types_allows_scalar_to_indexed_array(study_with_solution):
    study, _ = study_with_solution
    study.solution.add_step(MatrixSinkPlugin, "sink")
    study.solution.add_link("producer.y", "sink.arr_in[0]")
    plugins = study.solution.generate_plugins()

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

    with pytest.raises(ValueError, match="Indexes may only be specified for arrays"):
        study.solution._check_link_types(plugins, study)


def test_solution_check_link_types_negative_index_on_scalar_input(study_with_solution):
    study, plugins = study_with_solution
    study.solution.add_link("producer.y", "consumer.y_in[0]")

    with pytest.raises(ValueError, match="Indexes may only be specified for arrays"):
        study.solution._check_link_types(plugins, study)


def test_solution_check_link_types_nested_group_allows_nested_array_index(study_with_solution):
    study, _ = study_with_solution
    nested = Group()
    nested.add_step(MatrixProducerPlugin, "matrix")

    outer_consumer = ConsumerPlugin
    study.solution.add_step(nested, "nested")
    study.solution.add_step(outer_consumer, "outer")
    study.solution.add_link("nested.matrix.grid[0,1]", "outer.arr_scalar")

    plugins = study.solution.generate_plugins()
    study.solution._check_link_types(plugins, study)


def test_solution_check_link_indices_nested_group_shape_mismatch_raises(study_with_solution):
    study, _ = study_with_solution
    nested = Group()
    nested.add_step(MatrixProducerPlugin, "matrix")

    study.solution.add_step(nested, "nested")
    study.solution.add_step(MatrixSinkPlugin, "sink")
    study.solution.add_link("nested.matrix.grid[0,:]", "sink.arr_in")

    plugins = study.solution.generate_plugins()

    with pytest.raises(ValueError, match="source indices must match the target shape"):
        study.solution._check_link_indices(plugins, study)


def test_solution_check_link_indices_allows_vehicle_scalar_to_array_element(
    study_with_solution,
):
    study, _ = study_with_solution
    study.solution.add_step(VectorSinkPlugin, "sink")
    study.solution.add_link("vehicles.veh.core.x", "sink.arr_in[1]")
    plugins = study.solution.generate_plugins()

    study.solution._check_link_indices(plugins, study)


def test_solution_check_link_indices_allows_plugin_scalar_to_array_element(
    study_with_solution,
):
    study, _ = study_with_solution
    study.solution.add_step(VectorSinkPlugin, "sink")
    study.solution.add_link("producer.y", "sink.arr_in[1]")
    plugins = study.solution.generate_plugins()

    study.solution._check_link_indices(plugins, study)


def test_solution_check_link_indices_allows_scalar_broadcast_to_array_slice(
    study_with_solution,
):
    study, _ = study_with_solution
    study.solution.add_step(VectorSinkPlugin, "sink")
    study.solution.add_link("vehicles.veh.core.x", "sink.arr_in[0:2]")
    plugins = study.solution.generate_plugins()

    study.solution._check_link_indices(plugins, study)


def test_solution_check_link_indices_allows_matching_source_and_target_slices(
    study_with_solution,
):
    study, _ = study_with_solution
    study.solution.add_step(MatrixProducerPlugin, "matrix")
    study.solution.add_step(VectorSinkPlugin, "sink")
    study.solution.add_link("matrix.grid[0,:]", "sink.arr_in[0:2]")
    plugins = study.solution.generate_plugins()

    study.solution._check_link_indices(plugins, study)


def test_solution_check_link_indices_rejects_mismatched_target_slice(
    study_with_solution,
):
    study, _ = study_with_solution
    study.solution.add_step(MatrixProducerPlugin, "matrix")
    study.solution.add_step(VectorSinkPlugin, "sink")
    study.solution.add_link("matrix.grid[0,:]", "sink.arr_in[0]")
    plugins = study.solution.generate_plugins()

    with pytest.raises(ValueError, match="source indices must match the target shape"):
        study.solution._check_link_indices(plugins, study)


def test_solution_check_link_indices_rejects_array_broadcasting(
    study_with_solution,
):
    study, _ = study_with_solution
    study.solution.add_step(VectorSinkPlugin, "sink")
    study.solution.add_link("producer.arr_out[0:1]", "sink.arr_in[0:2]")
    plugins = study.solution.generate_plugins()

    with pytest.raises(ValueError, match="source indices must match the target shape"):
        study.solution._check_link_indices(plugins, study)


def test_solution_check_link_indices_rejects_array_element_broadcast_to_slice(
    study_with_solution,
):
    study, _ = study_with_solution
    study.solution.add_step(VectorSinkPlugin, "sink")
    study.solution.add_link("producer.arr_out[0]", "sink.arr_in[0:2]")
    plugins = study.solution.generate_plugins()

    with pytest.raises(ValueError, match="source indices must match the target shape"):
        study.solution._check_link_indices(plugins, study)


def test_solution_check_link_indices_rejects_invalid_target_index(
    study_with_solution,
):
    study, _ = study_with_solution
    study.solution.add_step(VectorSinkPlugin, "sink")
    study.solution.add_link("vehicles.veh.core.x", "sink.arr_in[5]")
    plugins = study.solution.generate_plugins()

    with pytest.raises(ValueError, match="Invalid indices or array"):
        study.solution._check_link_indices(plugins, study)


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


def test_plugin_declare_io_sets_supported_scalar_defaults():
    p = Plugin()

    p.declare_input("float_input", float)
    p.declare_input("int_input", int)
    p.declare_output("bool_output", bool)
    p.declare_output("str_output", str)

    assert p.plugin_input_names["float_input"]["value"] == 0.0
    assert p.plugin_input_names["int_input"]["value"] == 0
    assert p.plugin_output_names["bool_output"]["value"] is False
    assert p.plugin_output_names["str_output"]["value"] == ""
    assert p.plugin_input_names["float_input"]["shape"] is None


def test_plugin_declare_array_defaults_and_shape_inference():
    p = Plugin()
    explicit_value = np.array([3.0, 4.0])

    p.declare_input("zeros", np.ndarray, shape=(2, 3))
    p.declare_output("explicit", np.ndarray, value=explicit_value, shape=(2,))
    p.declare_output("inferred", np.ndarray, value=np.ones((3, 1)))

    zeros = p.plugin_input_names["zeros"]
    explicit = p.plugin_output_names["explicit"]
    inferred = p.plugin_output_names["inferred"]

    assert zeros["shape"] == (2, 3)
    np.testing.assert_array_equal(zeros["value"], np.zeros((2, 3)))
    assert explicit["value"] is explicit_value
    assert explicit["shape"] == (2,)
    assert inferred["shape"] == (3, 1)


def test_plugin_declare_partials_and_build_partials_positive():
    p = ProducerPlugin()
    p.setup_plugin({})
    p.declare_partials(of="y", wrt="x", method="fd", step=1e-6, form="forward")

    partials = p.build_partials(["producer.x"], ["producer.y"], "producer")
    categorized = p.categorize_partials()

    assert ("y", "x") in partials
    assert categorized["fd"] == [("y", "x")]
    assert categorized["exact"] == []


def test_plugin_build_partials_rejects_unknown_names():
    p = ProducerPlugin()
    p.setup_plugin({})
    p.declare_partials(of="y", wrt="missing", method="fd")

    with pytest.raises(KeyError, match="declared as a partial is not an input in the Plugin"):
        p.build_partials(["producer.x"], ["producer.y"], "producer")


def test_plugin_build_partials_requires_continuous_types():
    p = IntSinkPlugin()
    p.setup_plugin({})
    p.declare_partials(of="n_out", wrt="n_in", method="fd")

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
