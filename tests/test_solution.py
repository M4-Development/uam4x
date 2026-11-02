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
    study.add_solution(Solution("test_solution"))
    vehicle = Vehicle("veh")
    vehicle.add_component(CoreSubsystem("core"))
    study.add_vehicle(vehicle)

    study.get_solution("test_solution").add_step(ProducerPlugin, "producer")
    study.get_solution("test_solution").add_step(ConsumerPlugin, "consumer")

    study.get_solution("test_solution").add_link("vehicles.veh.core.x", "producer.x")
    study.get_solution("test_solution").add_link("producer.y", "consumer.y_in")
    study.get_solution("test_solution").add_link("producer.arr_out[0]", "consumer.arr_scalar")
    study.get_solution("test_solution").add_tracked_output("consumer.z")

    study.get_solution("test_solution").sort_steps()
    plugins = study.get_solution("test_solution").generate_plugins()
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

    sol = Solution("test_solution")
    sol.add_step(inner, "inner")
    sol.add_step(ConsumerPlugin, "consumer")
    sol.add_link("inner.producer.y", "consumer.y_in")
    sol.add_link("inner.producer.arr_out[0]", "consumer.arr_scalar")
    sol.add_tracked_output("consumer.z")
    sol.approx_totals(method="fd", step=0.02, form="forward")

    data = sol._to_dict()
    rebuilt = Solution._from_dict(data)

    assert isinstance(rebuilt.steps["inner"], Group)
    assert rebuilt.steps["inner"].group_name == "inner"
    assert rebuilt.steps["consumer"]["plugin"] is ConsumerPlugin
    assert rebuilt.tracked_outputs == ["consumer.z"]
    assert rebuilt._approx_totals == {
        "compute": True, "method": "fd", "step": 0.02, "form": "forward"
    }

    del data["approx_totals"]
    assert Solution._from_dict(data)._approx_totals == {"compute": False}


def test_solution_add_remove_tracked_output_and_duplicate_rejection():
    sol = Solution("test_solution")
    sol.add_tracked_output("consumer.z")

    with pytest.raises(RuntimeError):
        sol.add_tracked_output("consumer.z")

    sol.remove_tracked_output("consumer.z")
    assert sol.tracked_outputs == []


def test_solution_get_all_plugin_maps_with_nested_groups():
    inner = Group()
    inner.add_step(ProducerPlugin, "producer")

    sol = Solution("test_solution")
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
    study.get_solution("test_solution")._check_links_to_plugins(plugins)


def test_solution_check_links_to_plugins_accepts_indexed_input(study_with_solution):
    study, _ = study_with_solution
    study.get_solution("test_solution").add_step(MatrixSinkPlugin, "sink")
    study.get_solution("test_solution").add_link("producer.y", "sink.arr_in[0]")
    plugins = study.get_solution("test_solution").generate_plugins()

    study.get_solution("test_solution")._check_links_to_plugins(plugins)


def test_solution_check_links_to_plugins_negative_missing_input(study_with_solution):
    study, plugins = study_with_solution
    study.get_solution("test_solution").add_link("producer.y", "consumer.not_declared")

    with pytest.raises(ValueError, match="was not declared in the plugin"):
        study.get_solution("test_solution")._check_links_to_plugins(plugins)


def test_solution_check_links_to_plugins_negative_missing_output(study_with_solution):
    study, plugins = study_with_solution
    study.get_solution("test_solution").add_link("producer.missing", "consumer.y_in")

    with pytest.raises(ValueError, match="was not declared in the plugin"):
        study.get_solution("test_solution")._check_links_to_plugins(plugins)


def test_solution_check_link_types_positive(study_with_solution):
    study, plugins = study_with_solution
    study.get_solution("test_solution")._check_link_types(plugins, study)


def test_solution_check_link_types_rejects_vehicle_link_declared_in_group():
    study = Study()
    study.add_solution(Solution("test_solution"))
    vehicle = Vehicle("veh")
    vehicle.add_component(CoreSubsystem("core"))
    study.add_vehicle(vehicle)

    nested = Group()
    nested.add_step(ProducerPlugin, "producer")
    nested.add_link("vehicles.veh.core.x", "producer.x")
    study.get_solution("test_solution").add_step(nested, "nested")

    study.get_solution("test_solution").sort_steps()
    plugins = study.get_solution("test_solution").generate_plugins()

    with pytest.raises(
        ValueError,
        match="Vehicle links inside groups are not supported",
    ):
        study.get_solution("test_solution")._check_link_types(plugins, study)


def test_solution_check_link_types_allows_top_level_vehicle_link_to_group():
    study = Study()
    study.add_solution(Solution("test_solution"))
    vehicle = Vehicle("veh")
    vehicle.add_component(CoreSubsystem("core"))
    study.add_vehicle(vehicle)

    nested = Group()
    nested.add_step(ProducerPlugin, "producer")
    study.get_solution("test_solution").add_step(nested, "nested")
    study.get_solution("test_solution").add_link("vehicles.veh.core.x", "nested.producer.x")

    study.get_solution("test_solution").sort_steps()
    plugins = study.get_solution("test_solution").generate_plugins()

    study.get_solution("test_solution")._check_link_types(plugins, study)


def test_solution_check_link_types_allows_scalar_to_indexed_array(study_with_solution):
    study, _ = study_with_solution
    study.get_solution("test_solution").add_step(MatrixSinkPlugin, "sink")
    study.get_solution("test_solution").add_link("producer.y", "sink.arr_in[0]")
    plugins = study.get_solution("test_solution").generate_plugins()

    study.get_solution("test_solution")._check_link_types(plugins, study)


def test_solution_check_link_types_negative_mismatch(study_with_solution):
    study, plugins = study_with_solution
    study.get_solution("test_solution").add_step(IntSinkPlugin, "int_sink")
    plugins = study.get_solution("test_solution").generate_plugins()
    study.get_solution("test_solution").add_link("vehicles.veh.core.arr", "int_sink.n_in")

    with pytest.raises(TypeError, match="Link types must match"):
        study.get_solution("test_solution")._check_link_types(plugins, study)


def test_solution_check_link_types_negative_index_on_scalar(study_with_solution):
    study, plugins = study_with_solution
    study.get_solution("test_solution").add_link("producer.y[0]", "consumer.y_in")

    with pytest.raises(ValueError, match="Indexes may only be specified for arrays"):
        study.get_solution("test_solution")._check_link_types(plugins, study)


def test_solution_check_link_types_negative_index_on_scalar_input(study_with_solution):
    study, plugins = study_with_solution
    study.get_solution("test_solution").add_link("producer.y", "consumer.y_in[0]")

    with pytest.raises(ValueError, match="Indexes may only be specified for arrays"):
        study.get_solution("test_solution")._check_link_types(plugins, study)


def test_solution_check_link_types_nested_group_allows_nested_array_index(study_with_solution):
    study, _ = study_with_solution
    nested = Group()
    nested.add_step(MatrixProducerPlugin, "matrix")

    outer_consumer = ConsumerPlugin
    study.get_solution("test_solution").add_step(nested, "nested")
    study.get_solution("test_solution").add_step(outer_consumer, "outer")
    study.get_solution("test_solution").add_link("nested.matrix.grid[0,1]", "outer.arr_scalar")

    plugins = study.get_solution("test_solution").generate_plugins()
    study.get_solution("test_solution")._check_link_types(plugins, study)


def test_solution_check_link_indices_nested_group_shape_mismatch_raises(study_with_solution):
    study, _ = study_with_solution
    nested = Group()
    nested.add_step(MatrixProducerPlugin, "matrix")

    study.get_solution("test_solution").add_step(nested, "nested")
    study.get_solution("test_solution").add_step(MatrixSinkPlugin, "sink")
    study.get_solution("test_solution").add_link("nested.matrix.grid[0,:]", "sink.arr_in")

    plugins = study.get_solution("test_solution").generate_plugins()

    with pytest.raises(ValueError, match="source indices must match the target shape"):
        study.get_solution("test_solution")._check_link_indices(plugins, study)


def test_solution_check_link_indices_allows_vehicle_scalar_to_array_element(
    study_with_solution,
):
    study, _ = study_with_solution
    study.get_solution("test_solution").add_step(VectorSinkPlugin, "sink")
    study.get_solution("test_solution").add_link("vehicles.veh.core.x", "sink.arr_in[1]")
    plugins = study.get_solution("test_solution").generate_plugins()

    study.get_solution("test_solution")._check_link_indices(plugins, study)


def test_solution_check_link_indices_allows_plugin_scalar_to_array_element(
    study_with_solution,
):
    study, _ = study_with_solution
    study.get_solution("test_solution").add_step(VectorSinkPlugin, "sink")
    study.get_solution("test_solution").add_link("producer.y", "sink.arr_in[1]")
    plugins = study.get_solution("test_solution").generate_plugins()

    study.get_solution("test_solution")._check_link_indices(plugins, study)


def test_solution_check_link_indices_allows_scalar_broadcast_to_array_slice(
    study_with_solution,
):
    study, _ = study_with_solution
    study.get_solution("test_solution").add_step(VectorSinkPlugin, "sink")
    study.get_solution("test_solution").add_link("vehicles.veh.core.x", "sink.arr_in[0:2]")
    plugins = study.get_solution("test_solution").generate_plugins()

    study.get_solution("test_solution")._check_link_indices(plugins, study)


def test_solution_check_link_indices_allows_matching_source_and_target_slices(
    study_with_solution,
):
    study, _ = study_with_solution
    study.get_solution("test_solution").add_step(MatrixProducerPlugin, "matrix")
    study.get_solution("test_solution").add_step(VectorSinkPlugin, "sink")
    study.get_solution("test_solution").add_link("matrix.grid[0,:]", "sink.arr_in[0:2]")
    plugins = study.get_solution("test_solution").generate_plugins()

    study.get_solution("test_solution")._check_link_indices(plugins, study)


def test_solution_check_link_indices_rejects_mismatched_target_slice(
    study_with_solution,
):
    study, _ = study_with_solution
    study.get_solution("test_solution").add_step(MatrixProducerPlugin, "matrix")
    study.get_solution("test_solution").add_step(VectorSinkPlugin, "sink")
    study.get_solution("test_solution").add_link("matrix.grid[0,:]", "sink.arr_in[0]")
    plugins = study.get_solution("test_solution").generate_plugins()

    with pytest.raises(ValueError, match="source indices must match the target shape"):
        study.get_solution("test_solution")._check_link_indices(plugins, study)


def test_solution_check_link_indices_rejects_array_broadcasting(
    study_with_solution,
):
    study, _ = study_with_solution
    study.get_solution("test_solution").add_step(VectorSinkPlugin, "sink")
    study.get_solution("test_solution").add_link("producer.arr_out[0:1]", "sink.arr_in[0:2]")
    plugins = study.get_solution("test_solution").generate_plugins()

    with pytest.raises(ValueError, match="source indices must match the target shape"):
        study.get_solution("test_solution")._check_link_indices(plugins, study)


def test_solution_check_link_indices_rejects_array_element_broadcast_to_slice(
    study_with_solution,
):
    study, _ = study_with_solution
    study.get_solution("test_solution").add_step(VectorSinkPlugin, "sink")
    study.get_solution("test_solution").add_link("producer.arr_out[0]", "sink.arr_in[0:2]")
    plugins = study.get_solution("test_solution").generate_plugins()

    with pytest.raises(ValueError, match="source indices must match the target shape"):
        study.get_solution("test_solution")._check_link_indices(plugins, study)


def test_solution_check_link_indices_rejects_invalid_target_index(
    study_with_solution,
):
    study, _ = study_with_solution
    study.get_solution("test_solution").add_step(VectorSinkPlugin, "sink")
    study.get_solution("test_solution").add_link("vehicles.veh.core.x", "sink.arr_in[5]")
    plugins = study.get_solution("test_solution").generate_plugins()

    with pytest.raises(ValueError, match="Invalid indices or array"):
        study.get_solution("test_solution")._check_link_indices(plugins, study)


def test_solution_get_all_used_vehicle_params(study_with_solution):
    study, _ = study_with_solution
    used = study.get_solution("test_solution").get_all_used_vehicle_params()

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


def test_group_plugin_options_can_be_added_edited_and_removed(capsys):
    group = Group()
    group.add_step(ProducerPlugin, "producer", {"scale": 2.0})

    group.add_plugin_option("producer", "offset", 1.0)
    group.add_plugin_option("producer", "scale", 3.0)
    assert "Overwriting value" in capsys.readouterr().out
    assert group.steps["producer"]["plugin_options"] == {
        "scale": 3.0, "offset": 1.0
    }

    group.edit_step("producer", {"offset": 4.0, "enabled": True})
    assert group.steps["producer"]["plugin_options"] == {
        "scale": 3.0, "offset": 4.0, "enabled": True
    }
    group.remove_plugin_option("producer", "enabled")
    assert "enabled" not in group.steps["producer"]["plugin_options"]

    group.edit_step("producer")
    assert group.steps["producer"]["plugin_options"] == {}


@pytest.mark.parametrize(
    "operation",
    [
        lambda group: group.edit_step("missing", {}),
        lambda group: group.add_plugin_option("missing", "x", 1),
        lambda group: group.remove_plugin_option("missing", "x"),
    ],
)
def test_group_plugin_option_operations_reject_missing_step(operation):
    with pytest.raises(KeyError, match="not a step"):
        operation(Group())


@pytest.mark.parametrize(
    "operation",
    [
        lambda group: group.edit_step("subgroup", {}),
        lambda group: group.add_plugin_option("subgroup", "x", 1),
        lambda group: group.remove_plugin_option("subgroup", "x"),
    ],
)
def test_group_plugin_option_operations_reject_subgroup(operation):
    group = Group()
    group.add_step(Group(), "subgroup")

    with pytest.raises(ValueError, match="Only plugins are editable"):
        operation(group)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"method": "cs"}, "other than exact or fd"),
        ({"step": 0.0}, "greater than zero"),
        ({"step": "small"}, "real number"),
        ({"form": "sideways"}, "finite difference forms"),
        ({"method": "exact", "step": 1e-6}, "does not take in a step"),
        ({"method": "exact", "form": "forward"}, "does not take in a form"),
    ],
)
def test_plugin_declare_partials_rejects_invalid_configuration(kwargs, message):
    plugin = ProducerPlugin()
    plugin.setup_plugin({})

    with pytest.raises(ValueError, match=message):
        plugin.declare_partials("y", "x", **kwargs)
    assert plugin.initial_partials == []


def test_plugin_exact_wildcard_partials_expand_to_declared_continuous_pairs():
    plugin = ProducerPlugin()
    plugin.setup_plugin({})
    plugin.declare_partials("*", "*", method="exact",
                            rows=[0], cols=[0])

    partials = plugin.build_partials(
        ["nested.producer.x"],
        ["nested.producer.y", "nested.producer.arr_out"],
        "nested.producer",
    )

    assert set(partials) == {("y", "x"), ("arr_out", "x")}
    assert all(data["method"] == "exact" for data in partials.values())
    assert all(data["rows"] == [0] and data["cols"] == [0]
               for data in partials.values())
    assert set(plugin.categorize_partials()["exact"]) == set(partials)
    assert plugin._declare_partials is True


@pytest.mark.parametrize(
    "input_names, output_names, message",
    [
        ([], ["producer.y"], "Inputs and partials do not match"),
        (["producer.x"], ["producer.arr_out"],
         "Outputs and partials do not match"),
    ],
)
def test_plugin_build_partials_requires_linked_declared_names(
    input_names, output_names, message
):
    plugin = ProducerPlugin()
    plugin.setup_plugin({})
    plugin.declare_partials("y", "x")

    with pytest.raises(ValueError, match=message):
        plugin.build_partials(input_names, output_names, "producer")


@pytest.mark.parametrize("input_type,output_type", [(int, float), (float, int)])
def test_plugin_wildcard_partials_reject_discrete_variables(
    input_type, output_type
):
    plugin = Plugin()
    plugin.declare_input("x", input_type)
    plugin.declare_output("y", output_type)
    plugin.declare_partials("*", "*")

    with pytest.raises(TypeError, match="not continuous"):
        plugin.build_partials(["sink.x"], ["sink.y"], "sink")


def test_solution_optimization_link_check_accepts_continuous_and_vehicle_links(
    study_with_solution,
):
    study, plugins = study_with_solution
    study.get_solution("test_solution")._check_optimization_discrete(plugins)


def test_solution_optimization_link_check_rejects_discrete_plugin_output(
    study_with_solution,
):
    study, _ = study_with_solution
    solution = study.get_solution("test_solution")
    solution.add_step(IntSinkPlugin, "first_int")
    solution.add_step(IntSinkPlugin, "second_int")
    solution.add_link("vehicles.veh.core.n", "first_int.n_in")
    solution.add_link("first_int.n_out", "second_int.n_in")
    plugins = solution.generate_plugins()

    with pytest.raises(TypeError, match="Discrete variables linked between plugins"):
        solution._check_optimization_discrete(plugins)


def test_solution_optimization_link_check_nested_group_rejects_discrete_output(
    study_with_solution,
):
    study, _ = study_with_solution
    solution = study.get_solution("test_solution")
    nested = Group()
    nested.add_step(IntSinkPlugin, "first")
    nested.add_step(IntSinkPlugin, "second")
    nested.add_link("first.n_out", "second.n_in")
    solution.add_step(nested, "nested")
    plugins = solution.generate_plugins()

    with pytest.raises(TypeError, match="first.n_out"):
        solution._check_optimization_discrete(plugins)


@pytest.mark.parametrize(
    "name,var_type,value,shape,error,match",
    [
        ("x", "float", None, None, TypeError, "Declared type must be a type"),
        ("bad~name", float, None, None, ValueError, "not allowed in variable names"),
        ("x", float, None, (2,), ValueError, "only np.ndarray types"),
        ("x", np.ndarray, None, (2.5,), ValueError, "shape must be declared"),
        ("x", np.ndarray, np.ones((3,)), (2,), ValueError, "shape must match"),
        ("x", np.ndarray, None, None, ValueError, "no shape or value"),
        ("x", np.ndarray, 2.0, None, ValueError, "shape must be declared"),
    ],
)
def test_plugin_check_declared_io_rejects_invalid_metadata(
    name, var_type, value, shape, error, match
):
    with pytest.raises(error, match=match):
        Plugin.check_declared_IO(name, var_type, value, shape)


def test_plugin_check_declared_io_broadcasts_scalar_and_infers_shape():
    value, shape = Plugin.check_declared_IO("x", np.ndarray, 2.0, 3)
    assert value == 2.0
    assert shape == (3,)

    arr = np.array([[1.0, 2.0]])
    value, shape = Plugin.check_declared_IO("y", np.ndarray, arr, None)
    assert value is arr
    assert shape == (1, 2)


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


def test_plugin_registration_only_updates_memory_when_writes_disabled(monkeypatch):
    monkeypatch.setenv("UAM4X_DISABLE_REGISTRY_WRITES", "1")

    @register_plugin
    class RegistryProbePlugin(Plugin):
        pass

    try:
        assert _plugin_registry["RegistryProbePlugin"] is RegistryProbePlugin
    finally:
        _plugin_registry.pop("RegistryProbePlugin", None)
