import os

os.environ.setdefault("UAM4X_DISABLE_REGISTRY_WRITES", "1")

import inspect
import builtins

import pytest

import uam4x.node as node_mod
from uam4x.node import (
    Node,
    register_node,
    get_node,
    get_node_names,
    get_node_dict,
    validate_all_nodes,
    get_nodes_by_module,
    get_nodes_by_file,
    _node_registry,
)


@pytest.fixture
def clean_node_registry():
    snap = dict(_node_registry)
    _node_registry.clear()
    try:
        yield
    finally:
        _node_registry.clear()
        _node_registry.update(snap)


def test_node_execute_and_errors():
    called = {}

    def f(study):
        called["ok"] = study

    n = Node("n1", function=f)
    n.execute("study")
    assert called["ok"] == "study"

    with pytest.raises(ValueError, match="has no function"):
        Node("n2").execute("study")


def test_node_validate_signature_variants():
    def one(study):
        pass

    def two(a, b):
        pass

    def ann(study: "Study"):
        pass

    assert Node("one", function=one).validate_signature() is True
    assert Node("two", function=two).validate_signature() is False
    assert Node("ann", function=ann).validate_signature() is True
    assert Node("none", function=None).validate_signature() is False


def test_node_docstring_and_source_info(monkeypatch):
    def f(study):
        """doc text"""
        pass

    n = Node("n", function=f)
    assert n.get_docstring() == "doc text"

    info = n.get_source_info()
    assert "file" in info and "line" in info

    monkeypatch.setattr(inspect, "getfile", lambda *_: (_ for _ in ()).throw(OSError("x")))
    assert n.get_source_info() == {"file": "unknown", "line": "unknown"}


def test_register_node_and_registry_queries(clean_node_registry):
    @register_node(description="desc")
    def node_a(study):
        """A"""
        pass

    @register_node(name="custom_b")
    def node_b(study):
        pass

    assert get_node("node_a").name == "node_a"
    assert get_node("custom_b").name == "custom_b"
    assert set(get_node_names()) == {"node_a", "custom_b"}

    d = get_node_dict()
    assert set(d) == {"node_a", "custom_b"}

    v = validate_all_nodes()
    assert v["node_a"] is True

    mod_nodes = get_nodes_by_module(node_a.__module__)
    assert any(n.name == "node_a" for n in mod_nodes)

    file_nodes = get_nodes_by_file(get_node("node_a").source_file)
    assert any(n.name == "node_a" for n in file_nodes)


def test_register_node_invalid_signature_still_registers(clean_node_registry):
    @register_node()
    def bad_node(a, b):
        pass

    assert "bad_node" in _node_registry
    assert _node_registry["bad_node"].validate_signature() is False


def test_register_node_config_update_failure_path(clean_node_registry, monkeypatch):
    real_open = builtins.open

    def flaky_open(path, mode="r", *args, **kwargs):
        if path.endswith("conf.json") and "w" in mode:
            raise OSError("cannot write")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", flaky_open)

    @register_node()
    def still_registers(study):
        pass

    assert "still_registers" in _node_registry
