import os
import subprocess
from types import SimpleNamespace

os.environ.setdefault("UAM4X_DISABLE_REGISTRY_WRITES", "1")

import numpy as np
import pytest

import uam4x.study as study_module
from uam4x.study import Study, Variants
from uam4x.vehicle import Vehicle, VehicleManager
from uam4x.subsystems import Subsystem, register_subsystem
from uam4x.solution import Solution, SolutionManager, Plugin, register_plugin


SOLUTION_NAME = "test_solution"


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
    solution = Solution(SOLUTION_NAME)
    s.add_solution(solution)
    v = Vehicle("demo_vehicle")
    v.add_component(StudySubsystem("wing"))
    s.add_vehicle(v)

    solution.add_step(StudyPlugin, "step1")
    solution.add_link("vehicles.demo_vehicle.wing.span", "step1.x")
    solution.add_tracked_output("step1.y")

    return s


def test_study_initialization_types():
    s = Study()
    assert isinstance(s.vehicles, VehicleManager)
    assert isinstance(s.solutions, SolutionManager)
    assert len(s.solutions) == 0


def test_study_manages_multiple_named_solutions():
    study = Study()
    first = Solution("first")
    second = Solution("second")

    study.add_solution(first)
    study.add_solution(second)

    assert len(study.solutions) == 2
    assert list(study.solutions) == [first, second]
    assert study.get_solution("first") is first
    assert study.get_solution("second") is second

    with pytest.raises(ValueError, match="Repeated Solution names"):
        study.add_solution(Solution("first"))


def test_study_add_get_vehicle(demo_study):
    s = demo_study
    got = s.get_vehicle("demo_vehicle")
    assert got.name == "demo_vehicle"


def test_study_capture_environment(monkeypatch):
    monkeypatch.setattr(study_module.subprocess, "check_output", lambda *args, **kwargs: b"/usr/bin/python3\n")

    s = Study()
    s._capture_environment(SOLUTION_NAME, "trade_a")

    assert "trade_a" in s._runtime_meta_data[SOLUTION_NAME]
    env = s._runtime_meta_data[SOLUTION_NAME]["trade_a"]["environment"]
    assert set(env) == {"Machine", "Python", "Packages"}
    assert "Start_Time" in s._runtime_meta_data[SOLUTION_NAME]["trade_a"]["time_info"]


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
    assert s.get_param_type("vehicles.demo_vehicle.wing.vec[1]") is np.ndarray

    with pytest.raises(ValueError, match="Invalid vehicle parameter"):
        s.get_param_type("vehicles.demo_vehicle.wing")


def test_study_to_dict_from_dict_roundtrip(monkeypatch, demo_study):
    monkeypatch.setattr(study_module.subprocess, "check_output", lambda *args, **kwargs: b"/usr/bin/python3\n")

    s = demo_study
    s.add_solution(Solution("alternate"))
    s._capture_environment(SOLUTION_NAME, "roundtrip_trade")
    raw = s._to_dict()
    rebuilt = Study._from_dict(raw)

    assert rebuilt.get_vehicle("demo_vehicle").wing.span == 12.0
    assert rebuilt.get_solution(SOLUTION_NAME).links == s.get_solution(SOLUTION_NAME).links
    assert rebuilt.get_solution(SOLUTION_NAME).tracked_outputs == s.get_solution(SOLUTION_NAME).tracked_outputs
    assert rebuilt.get_solution("alternate").name == "alternate"
    assert {solution.name for solution in rebuilt.solutions} == {
        SOLUTION_NAME,
        "alternate",
    }
    assert "roundtrip_trade" in rebuilt._runtime_meta_data[SOLUTION_NAME]
    assert rebuilt._runtime_meta_data[SOLUTION_NAME]["roundtrip_trade"]["time_info"]["Start_Time"]
    rebuilt._get_runtime_metadata(SOLUTION_NAME, "new_trade")["time_info"] = {"Start_Time": "now"}
    assert rebuilt._runtime_meta_data[SOLUTION_NAME]["new_trade"]["time_info"]["Start_Time"] == "now"


def test_study_save_and_from_json_roundtrip(tmp_path, demo_study):
    path = tmp_path / "study_variant"
    path.mkdir()

    demo_study.save_model_to_json(str(path), commit=False)
    loaded = Study.load_model_from_json(str(path / "model.json"))

    assert loaded.get_vehicle("demo_vehicle").wing.span == 12.0
    assert loaded.get_solution(SOLUTION_NAME).links == demo_study.get_solution(SOLUTION_NAME).links
    assert loaded.get_solution(SOLUTION_NAME).tracked_outputs == ["step1.y"]


def test_study_from_json_loads_study(tmp_path, demo_study):
    path = tmp_path / "study_load_variant"
    path.mkdir()
    demo_study.save_model_to_json(str(path), commit=False)

    fresh = Study.load_model_from_json(str(path / "model.json"))

    assert fresh.get_vehicle("demo_vehicle").wing.span == 12.0
    assert fresh.get_solution(SOLUTION_NAME).links == demo_study.get_solution(SOLUTION_NAME).links


def test_study_load_results_from_json_updates_existing_instance(tmp_path, demo_study, monkeypatch):
    out = tmp_path / "variant_results"
    out.mkdir()

    monkeypatch.setattr(study_module.subprocess, "check_output", lambda *args, **kwargs: b"/usr/bin/python3\n")

    demo_study._capture_environment(SOLUTION_NAME, "trade_a")
    demo_study.save_model_to_json(str(out), commit=False)

    loaded_model = Study.load_model_from_json(str(out / "model.json"))

    data_with_results = loaded_model._to_dict()
    data_with_results["results"] = {
        SOLUTION_NAME: {
            "trade_a": {
                "points": [
                    {
                        "input_state": {"step1": {"x": 1.0}},
                        "outputs": {"step1.y": 6.0},
                    }
                ]
            }
        }
    }
    data_with_results["execution_meta_data"] = {
        SOLUTION_NAME: {"trade_a": {"time_info": {"Start_Time": "saved"}}}
    }
    monkeypatch.setattr(loaded_model, "_to_dict", lambda: data_with_results)
    loaded_model.save_results_to_json(str(out), commit=False)

    fresh_model = Study.load_model_from_json(str(out / "model.json"))
    result = fresh_model.load_results_from_json(str(out / "results.json"))

    assert result is None
    assert fresh_model.results.get_tradestudy(SOLUTION_NAME, "trade_a") is not None
    assert fresh_model._runtime_meta_data[SOLUTION_NAME]["trade_a"]["time_info"]["Start_Time"] == "saved"
    fresh_model._get_runtime_metadata(SOLUTION_NAME, "new_trade")["time_info"] = {"Start_Time": "now"}
    assert fresh_model._runtime_meta_data[SOLUTION_NAME]["new_trade"]["time_info"]["Start_Time"] == "now"


def test_study_load_study_from_directory_loads_model_and_results(
    tmp_path, demo_study, monkeypatch
):
    out = tmp_path / "variant_directory"
    out.mkdir()

    monkeypatch.setattr(study_module.subprocess, "check_output", lambda *args, **kwargs: b"/usr/bin/python3\n")

    demo_study._capture_environment(SOLUTION_NAME, "trade_a")
    demo_study.save_model_to_json(str(out), commit=False)

    loaded_model = Study.load_model_from_json(str(out / "model.json"))
    data_with_results = loaded_model._to_dict()
    data_with_results["results"] = {
        SOLUTION_NAME: {
            "trade_a": {
                "points": [
                    {
                        "input_state": {"step1": {"x": 1.0}},
                        "outputs": {"step1.y": 6.0},
                    }
                ]
            }
        }
    }
    data_with_results["execution_meta_data"] = {
        SOLUTION_NAME: {"trade_a": {"time_info": {"Start_Time": "saved"}}}
    }
    monkeypatch.setattr(loaded_model, "_to_dict", lambda: data_with_results)
    loaded_model.save_results_to_json(str(out), commit=False)

    rebuilt = Study.load_study_from_directory(str(out))

    assert rebuilt.get_vehicle("demo_vehicle").wing.span == 12.0
    assert rebuilt.results.get_tradestudy(SOLUTION_NAME, "trade_a") is not None
    assert rebuilt._runtime_meta_data[SOLUTION_NAME]["trade_a"]["time_info"]["Start_Time"] == "saved"


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
        solution = Solution(SOLUTION_NAME)
        solution.add_step(StudyPlugin, "step1")
        solution.add_link("vehicles.variant_vehicle.wing.span", "step1.x")
        solution.add_tracked_output("step1.y")
        study.add_solution(solution)

    variants = Variants(str(root))
    variants.add_variant("v1", [add_vehicle_func, add_solution_func])
    studies = variants.generate_variants(commit=False)

    expected_json = root / "variants" / "v1" / "model.json"
    assert len(studies) == 1
    assert expected_json.exists()
    assert studies[0].control.variant_name == "v1"


def test_study_runtime_metadata_helpers(demo_study):
    s = demo_study
    s._add_runtime_plugin_checksums(SOLUTION_NAME, "t1", {"step1": "abc"})
    s._add_runtime_external_executable_data(
        SOLUTION_NAME, "t1", {"exe": {"version": "1.0"}}
    )

    assert s._runtime_meta_data[SOLUTION_NAME]["t1"]["step_plugin_hashes"] == {"step1": "abc"}
    assert s._runtime_meta_data[SOLUTION_NAME]["t1"]["step_external_executable_data"]["exe"]["version"] == "1.0"


def test_study_capture_environment_existing_trade_branch(monkeypatch):
    monkeypatch.setattr(study_module.subprocess, "check_output", lambda *args, **kwargs: b"/usr/bin/python3\n")
    s = Study()
    s._runtime_meta_data[SOLUTION_NAME] = {"trade_existing": {"seed": 1}}
    s._capture_environment(SOLUTION_NAME, "trade_existing")

    metadata = s._runtime_meta_data[SOLUTION_NAME]["trade_existing"]
    assert metadata["seed"] == 1
    assert "environment" in metadata
    assert "time_info" in metadata


def test_study_save_results_to_json_and_commit_hook(tmp_path, demo_study, monkeypatch):
    out = tmp_path / "variant"
    out.mkdir()
    calls = {}

    def fake_commit(json_paths, commit_message):
        calls["paths"] = json_paths
        calls["msg"] = commit_message

    # save+reload to initialize model hash for results-only save path
    demo_study.save_model_to_json(str(out), commit=False)
    loaded = Study.load_model_from_json(str(out / "model.json"))

    data_with_results = loaded._to_dict()
    data_with_results["results"] = {
        SOLUTION_NAME: {"trade_a": {"points": [{"x": 1.0, "y": 6.0}]}}
    }
    monkeypatch.setattr(loaded, "_to_dict", lambda: data_with_results)
    monkeypatch.setattr(loaded, "_commit_json_to_branch", fake_commit)
    monkeypatch.setattr(
        loaded, "_prepare_repository_for_autocommit",
        lambda new_branch, commit_code: calls.update({"prepare": (new_branch, commit_code)}),
    )

    loaded.save_results_to_json(str(out), commit=True, commit_code=True)

    assert (out / "results.json").exists()
    assert calls["paths"] == [str(out / "results.json")]
    assert "Writing results for variant" in calls["msg"]
    assert calls["prepare"] == (False, True)


def test_study_save_results_to_json_commit_uses_current_branch(tmp_path, demo_study, monkeypatch):
    out = tmp_path / "variant"
    out.mkdir()
    calls = {}

    demo_study.save_model_to_json(str(out), commit=False)
    loaded = Study.load_model_from_json(str(out / "model.json"))

    data_with_results = loaded._to_dict()
    data_with_results["results"] = {
        SOLUTION_NAME: {"trade_a": {"points": [{"x": 1.0, "y": 6.0}]}}
    }
    monkeypatch.setattr(loaded, "_to_dict", lambda: data_with_results)
    monkeypatch.setattr(
        loaded,
        "_commit_json_to_branch",
        lambda json_paths, commit_message: calls.update(
            {"paths": json_paths, "msg": commit_message}),
    )
    monkeypatch.setattr(
        loaded, "_prepare_repository_for_autocommit",
        lambda new_branch, commit_code: calls.update({"prepare": (new_branch, commit_code)}),
    )

    loaded.save_results_to_json(str(out), commit=True)

    assert calls["paths"] == [str(out / "results.json")]
    assert "Writing results for variant" in calls["msg"]
    assert calls["prepare"] == (False, False)


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


def _run_git(repo_root, *args):
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def initialized_git_repo(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _run_git(repo_root, "init")
    tracked_code = repo_root / "code.py"
    tracked_code.write_text("value = 1\n")
    _run_git(repo_root, "add", "code.py")
    _run_git(
        repo_root,
        "-c", "user.name=Test Author",
        "-c", "user.email=test@example.com",
        "commit", "-m", "Initial commit",
    )
    variant_dir = repo_root / "variants" / "demo"
    variant_dir.mkdir(parents=True)
    return repo_root, tracked_code, variant_dir


def _configure_study_repository(study, repo_root):
    study.control._set_repo_root(str(repo_root))
    study.control._set_variant_name("demo")
    study.control.author = "Test Author"


def test_autocommit_rejects_unstaged_tracked_changes_before_writing_model(
    initialized_git_repo, demo_study
):
    repo_root, tracked_code, variant_dir = initialized_git_repo
    _configure_study_repository(demo_study, repo_root)
    original_branch = _run_git(repo_root, "branch", "--show-current")
    tracked_code.write_text("value = 2\n")

    with pytest.raises(RuntimeError, match="unstaged changes to tracked files"):
        demo_study.save_model_to_json(
            str(variant_dir), commit=True, new_branch=True, commit_code=False
        )

    assert not (variant_dir / "model.json").exists()
    assert _run_git(repo_root, "branch", "--show-current") == original_branch
    assert _run_git(repo_root, "rev-list", "--count", "HEAD") == "1"


def test_commit_code_commits_code_before_model_on_new_branch(
    initialized_git_repo, demo_study
):
    repo_root, tracked_code, variant_dir = initialized_git_repo
    _configure_study_repository(demo_study, repo_root)
    tracked_code.write_text("value = 2\n")
    (repo_root / "untracked.txt").write_text("leave me untracked\n")

    demo_study.save_model_to_json(
        str(variant_dir), commit=True, new_branch=True, commit_code=True
    )

    assert _run_git(repo_root, "branch", "--show-current") == "AutoCommit1"
    subjects = _run_git(repo_root, "log", "-2", "--format=%s").splitlines()
    assert subjects == [
        "AutoCommit: Created data for variant demo",
        "AutoCommit: Committed code",
    ]
    assert _run_git(repo_root, "show", "--format=", "--name-only", "HEAD~1") == "code.py"
    assert _run_git(repo_root, "show", "--format=", "--name-only", "HEAD") == (
        "variants/demo/model.json"
    )
    assert _run_git(repo_root, "status", "--short") == "?? untracked.txt"


def test_untracked_files_do_not_block_json_autocommit(
    initialized_git_repo, demo_study
):
    repo_root, _, variant_dir = initialized_git_repo
    _configure_study_repository(demo_study, repo_root)
    (repo_root / "untracked.txt").write_text("leave me untracked\n")

    demo_study.save_model_to_json(
        str(variant_dir), commit=True, new_branch=False, commit_code=False
    )

    assert _run_git(repo_root, "log", "-1", "--format=%s") == (
        "AutoCommit: Created data for variant demo"
    )
    assert _run_git(repo_root, "status", "--short") == "?? untracked.txt"


def test_commit_code_requires_json_commit(demo_study, tmp_path):
    with pytest.raises(ValueError, match="Cannot commit code"):
        demo_study.save_model_to_json(
            str(tmp_path), commit=False, commit_code=True
        )
