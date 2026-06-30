import os
from types import SimpleNamespace

os.environ.setdefault("UAM4X_DISABLE_REGISTRY_WRITES", "1")

import numpy as np
import pytest

import uam4x.study as study_module
from uam4x.study import Study, Variants
from uam4x.vehicle import Vehicle, VehicleManager
from uam4x.subsystems import Subsystem, register_subsystem
from uam4x.solution import Solution, Plugin, register_plugin


@register_subsystem
class StudySubsystem(Subsystem):
    def setup(self):
        self.add_parameter("span", default_value=12.0, dtype=float)
        self.add_parameter("count", default_value=3, dtype=int)
        self.add_parameter("vec", default_value=np.array([10.0, 20.0, 30.0]), dtype=np.ndarray)


@register_plugin
class StudyPlugin(Plugin):
    def setup_plugin(self, plugin_options=None):
        self.declare_input("x", float)
        self.declare_output("y", float)

    def run_plugin(self, links, work_dir):
        return {"y": float(links["x"]) + 5.0}


@pytest.fixture
def demo_study():
    s = Study()
    v = Vehicle("demo_vehicle")
    v.add_component(StudySubsystem("wing"))
    s.add_vehicle(v)

    s.solution.add_step(StudyPlugin, "step1")
    s.solution.add_link("vehicles.demo_vehicle.wing.span", "step1.x")
    s.solution.add_tracked_output("step1.y")

    return s


def test_study_initialization_types():
    s = Study()
    assert isinstance(s.vehicles, VehicleManager)
    assert isinstance(s.solution, Solution)


def test_study_add_get_vehicle(demo_study):
    s = demo_study
    got = s.get_vehicle("demo_vehicle")
    assert got.name == "demo_vehicle"


def test_study_capture_environment(monkeypatch):
    monkeypatch.setattr(study_module.subprocess, "check_output", lambda *args, **kwargs: b"/usr/bin/python3\n")

    s = Study()
    s._capture_environment("trade_a")

    assert "trade_a" in s._runtime_meta_data
    env = s._runtime_meta_data["trade_a"]["environment"]
    assert set(env) == {"Machine", "Python", "Packages"}
    assert "Start_Time" in s._runtime_meta_data["trade_a"]["time_info"]


def test_study_get_value_scalar_and_indexed_array(demo_study):
    s = demo_study

    assert s.get_value("vehicles.demo_vehicle.wing.span") == 12.0
    np.testing.assert_array_equal(
        s.get_value("vehicles.demo_vehicle.wing.vec"),
        np.array([10.0, 20.0, 30.0]),
    )
    assert s.get_value("vehicles.demo_vehicle.wing.vec[1]") == 20.0


def test_study_get_param_type_valid_and_invalid_path(demo_study):
    s = demo_study

    assert s.get_param_type("vehicles.demo_vehicle.wing.span") is float
    assert s.get_param_type("vehicles.demo_vehicle.wing.count") is int

    with pytest.raises(ValueError, match="Invalid vehicle parameter"):
        s.get_param_type("vehicles.demo_vehicle.wing")


def test_study_to_dict_from_dict_roundtrip(monkeypatch, demo_study):
    monkeypatch.setattr(study_module.subprocess, "check_output", lambda *args, **kwargs: b"/usr/bin/python3\n")

    s = demo_study
    s._capture_environment("roundtrip_trade")
    raw = s._to_dict()
    rebuilt = Study._from_dict(raw)

    assert rebuilt.get_vehicle("demo_vehicle").wing.span == 12.0
    assert rebuilt.solution.links == s.solution.links
    assert rebuilt.solution.tracked_outputs == s.solution.tracked_outputs
    assert "roundtrip_trade" in rebuilt._runtime_meta_data


def test_study_save_and_from_json_roundtrip(tmp_path, demo_study):
    path = tmp_path / "study_variant"
    path.mkdir()

    demo_study.save_model_to_json(str(path), commit=False)
    loaded = Study.from_json(str(path))

    assert loaded.get_vehicle("demo_vehicle").wing.span == 12.0
    assert loaded.solution.links == demo_study.solution.links
    assert loaded.solution.tracked_outputs == ["step1.y"]


def test_study_from_json_loads_study(tmp_path, demo_study):
    path = tmp_path / "study_load_variant"
    path.mkdir()
    demo_study.save_model_to_json(str(path), commit=False)

    fresh = Study.from_json(str(path))

    assert fresh.get_vehicle("demo_vehicle").wing.span == 12.0
    assert fresh.solution.links == demo_study.solution.links


def test_get_all_vehicle_param_names(demo_study):
    s = demo_study
    names = s.get_all_vehicle_param_names()

    assert names == sorted(names)
    assert len(names) == len(set(names))
    assert "vehicles.demo_vehicle.wing.span" in names
    assert "vehicles.demo_vehicle.wing.count" in names
    assert "vehicles.demo_vehicle.wing.vec" in names


def test_variants_generate_variants(tmp_path):
    root = tmp_path

    def add_vehicle_func(study):
        v = Vehicle("variant_vehicle")
        v.add_component(StudySubsystem("wing"))
        study.add_vehicle(v)

    def add_solution_func(study):
        study.solution.add_step(StudyPlugin, "step1")
        study.solution.add_link("vehicles.variant_vehicle.wing.span", "step1.x")
        study.solution.add_tracked_output("step1.y")

    variants = Variants(str(root))
    variants.add_variant("v1", [add_vehicle_func, add_solution_func])
    studies = variants.generate_variants(commit=False)

    expected_json = root / "variants" / "v1" / "model.json"
    assert len(studies) == 1
    assert expected_json.exists()
    assert studies[0].control.variant_name == "v1"


def test_study_runtime_metadata_helpers(demo_study):
    s = demo_study
    s._add_runtime_plugin_checksums("t1", {"step1": "abc"})
    s._add_runtime_external_executable_data("t1", {"exe": {"version": "1.0"}})

    assert s._runtime_meta_data["t1"]["step_plugin_hashes"] == {"step1": "abc"}
    assert s._runtime_meta_data["t1"]["step_external_executable_data"]["exe"]["version"] == "1.0"


def test_study_capture_environment_existing_trade_branch(monkeypatch):
    monkeypatch.setattr(study_module.subprocess, "check_output", lambda *args, **kwargs: b"/usr/bin/python3\n")
    s = Study()
    s._runtime_meta_data["trade_existing"] = {"seed": 1}
    s._capture_environment("trade_existing")

    assert s._runtime_meta_data["trade_existing"]["seed"] == 1
    assert "environment" in s._runtime_meta_data["trade_existing"]
    assert "time_info" in s._runtime_meta_data["trade_existing"]


def test_study_save_results_to_json_and_commit_hook(tmp_path, demo_study, monkeypatch):
    out = tmp_path / "variant"
    out.mkdir()
    calls = {}

    def fake_commit(json_paths, commit_message, new_branch):
        calls["paths"] = json_paths
        calls["msg"] = commit_message
        calls["new_branch"] = new_branch

    # save+reload to initialize model hash for results-only save path
    demo_study.save_model_to_json(str(out), commit=False)
    loaded = Study.from_json(str(out))

    data_with_results = loaded._to_dict()
    data_with_results["results"] = {"trade_a": {"points": [{"x": 1.0, "y": 6.0}]}}
    monkeypatch.setattr(loaded, "_to_dict", lambda: data_with_results)
    monkeypatch.setattr(loaded, "_commit_json_to_branch", fake_commit)

    loaded.save_results_to_json(str(out), commit=True)

    assert (out / "results.json").exists()
    assert calls["paths"] == [str(out / "results.json")]
    assert "Writing results for variant" in calls["msg"]
    assert calls["new_branch"] is False


def test_study_save_results_to_json_commit_uses_current_branch(tmp_path, demo_study, monkeypatch):
    out = tmp_path / "variant"
    out.mkdir()
    calls = {}

    demo_study.save_model_to_json(str(out), commit=False)
    loaded = Study.from_json(str(out))

    data_with_results = loaded._to_dict()
    data_with_results["results"] = {"trade_a": {"points": [{"x": 1.0, "y": 6.0}]}}
    monkeypatch.setattr(loaded, "_to_dict", lambda: data_with_results)
    monkeypatch.setattr(
        loaded,
        "_commit_json_to_branch",
        lambda json_paths, commit_message, new_branch: calls.update(
            {"paths": json_paths, "msg": commit_message, "new_branch": new_branch}
        ),
    )

    loaded.save_results_to_json(str(out), commit=True)

    assert calls["paths"] == [str(out / "results.json")]
    assert "Writing results for variant" in calls["msg"]
    assert calls["new_branch"] is False


def test_study_commit_json_to_branch_errors_no_dir(demo_study):
    demo_study.control.repo_root = "/definitely/not/a/repo"
    with pytest.raises(NotADirectoryError):
        demo_study._commit_json_to_branch(["dummy.json"], "msg")


def test_study_commit_json_to_branch_errors_when_no_repo(tmp_path, demo_study, monkeypatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()
    demo_study.control.repo_root = str(repo_root)

    monkeypatch.setattr(study_module.pygit2, "discover_repository", lambda *_: None)
    with pytest.raises(ValueError, match="No repository found"):
        demo_study._commit_json_to_branch([str(repo_root / "data.json")], "msg")


def test_study_commit_json_to_branch_errors_mismatch_repo_path(tmp_path, demo_study, monkeypatch):
    repo_root = tmp_path / "repo2"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()
    demo_study.control.repo_root = str(repo_root)

    monkeypatch.setattr(study_module.pygit2, "discover_repository", lambda *_: str(tmp_path / "other" / ".git"))
    with pytest.raises(ValueError, match="does not have a repo"):
        demo_study._commit_json_to_branch([str(repo_root / "data.json")], "msg")


def test_study_commit_json_to_branch_detached_head(tmp_path, demo_study, monkeypatch):
    repo_root = tmp_path / "repo3"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()
    demo_study.control.repo_root = str(repo_root)

    monkeypatch.setattr(study_module.pygit2, "discover_repository", lambda *_: str(repo_root / ".git"))
    fake_repo = SimpleNamespace(head_is_detached=True, head_is_unborn=False, head=SimpleNamespace(target="abc"))
    monkeypatch.setattr(study_module.pygit2, "Repository", lambda *_: fake_repo)

    with pytest.raises(RuntimeError, match="detached HEAD"):
        demo_study._commit_json_to_branch([str(repo_root / "data.json")], "msg")


def test_study_get_plugin_link_inputs_current_behavior_raises(demo_study):
    with pytest.raises(AttributeError):
        demo_study.get_plugin_link_inputs("step1", state={})
