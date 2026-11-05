"""Exercise configured component registration during package initialization."""

import runpy
from types import SimpleNamespace

import uam4x
import uam4x.solution as solution_module
import uam4x.subsystems as subsystem_module
import uam4x.utils as utils_module


def test_package_init_loads_configured_classes_and_reports_missing(monkeypatch):
    subsystem_registry = {"ExistingSubsystem": object()}
    plugin_registry = {"ExistingPlugin": object()}
    warnings = []
    loaded = []

    def load_class(name, path):
        loaded.append((name, path))
        if name.startswith("Missing"):
            return None, f"{name} could not be loaded"
        return type(name, (), {}), ""

    monkeypatch.setattr(subsystem_module, "_subsystem_registry", subsystem_registry)
    monkeypatch.setattr(solution_module, "_plugin_registry", plugin_registry)
    monkeypatch.setattr(
        utils_module, "check_and_get_config",
        lambda path: {
            "SUBSYSTEMS": {
                "ExistingSubsystem": "already.py",
                "NewSubsystem": "new.py",
                "MissingSubsystem": "missing.py",
            },
            "PLUGINS": {
                "ExistingPlugin": "already.py",
                "NewPlugin": "new.py",
                "MissingPlugin": "missing.py",
            },
        },
    )
    monkeypatch.setattr(utils_module, "_get_class_from_file", load_class)
    monkeypatch.setattr(
        utils_module, "log",
        SimpleNamespace(warning=warnings.append),
    )

    runpy.run_path(uam4x.__file__, run_name="uam4x_test_init")

    assert set(subsystem_registry) == {
        "ExistingSubsystem", "NewSubsystem"
    }
    assert set(plugin_registry) == {"ExistingPlugin", "NewPlugin"}
    assert ("ExistingSubsystem", "already.py") not in loaded
    assert ("ExistingPlugin", "already.py") not in loaded
    assert len(warnings) == 4
    assert any("MissingSubsystem" in message for message in warnings)
    assert any("MissingPlugin" in message for message in warnings)


def test_package_init_skips_saved_registry_for_documentation(monkeypatch):
    def fail_if_loaded(path):
        raise AssertionError("saved registry should not be read during docs builds")

    monkeypatch.setenv("UAM4X_DISABLE_REGISTRY_LOAD", "1")
    monkeypatch.setattr(utils_module, "check_and_get_config", fail_if_loaded)

    namespace = runpy.run_path(uam4x.__file__, run_name="uam4x_docs_init")

    assert namespace["_config"] == {"SUBSYSTEMS": {}, "PLUGINS": {}}
