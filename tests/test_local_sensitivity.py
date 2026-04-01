import os

os.environ.setdefault("UAM4X_DISABLE_REGISTRY_WRITES", "1")

import numpy as np
import pytest

from uam4x.local_sensitivity import LocalSensitivity
from uam4x.post import LocalSensitivityPost
from uam4x.results import Results, Tradestudy, Result


class _DummySolution:
    def __init__(self):
        self.tracked_outputs = ["out.y"]


class _DummyControl:
    def __init__(self):
        self.trade_studies = {}


class _DummyStudy:
    def __init__(self):
        self.solution = _DummySolution()
        self.control = _DummyControl()

    def get_value(self, _):
        raise KeyError


def _build_sensitivity_tradestudy(method_states):
    study = _DummyStudy()
    ls = LocalSensitivity(step_size=0.1)
    ls.add_perturbation_var("x", evaluation_point=1.0)
    study.control.trade_studies["sens"] = ls

    ts = Tradestudy("sens")
    res = Results(study=study)
    res.add_tradestudy("sens", ts)

    for state, out in method_states:
        r = Result(state=state)
        r._add_result("out.y", out)
        ts._results.append(r)
        ts._hash_to_result[r.state_hash] = r

    return ls, ts


def test_local_sensitivity_basic_api():
    ls = LocalSensitivity(step_size=0.5)
    ls.add_perturbation_var("a", evaluation_point=2.0)
    ls.add_perturbation_var("b", evaluation_point=4.0, step_size=0.25)

    assert ls.variable_to_sensitivity_data["a"]["step_size"] == 0.5
    assert ls.variable_to_sensitivity_data["b"]["step_size"] == 0.25


def test_local_sensitivity_get_states_shape_and_values():
    ls = LocalSensitivity(step_size=0.1)
    ls.add_perturbation_var("x", evaluation_point=1.0)
    ls.add_perturbation_var("z", evaluation_point=10.0)

    states = ls.get_states()

    assert len(states) == 1 + 2 * 2
    assert states[0]["optimization"] is False
    assert states[0]["x"] == 1.0
    assert states[0]["z"] == 10.0


def test_local_sensitivity_to_dict_from_dict_roundtrip():
    ls = LocalSensitivity(step_size=0.2)
    ls.add_perturbation_var("x", evaluation_point=3.0)

    d = ls._to_dict()
    rebuilt = LocalSensitivity._from_dict(d)

    assert "post_spec" not in d
    assert rebuilt.step_size == 0.2
    assert rebuilt.variable_to_sensitivity_data["x"]["evaluation_point"] == 3.0
    assert not hasattr(rebuilt, "post_spec")


def test_local_sensitivity_post_compute_methods_and_invalid():
    # baseline x=1.0 -> y=3.0; down x=0.9 -> y=2.8; up x=1.1 -> y=3.2 => sensitivity 2.0
    method_states = [
        ({"x": 1.0}, 3.0),
        ({"x": 0.9}, 2.8),
        ({"x": 1.1}, 3.2),
    ]
    ls, ts = _build_sensitivity_tradestudy(method_states)

    post = LocalSensitivityPost(ts)

    for method in ["forward", "backward", "central"]:
        sens, maxd = post.compute_sensitivity("out.y", method=method)
        assert sens["out.y"]["x"] == pytest.approx(2.0)
        assert maxd["variable"] == "x"
        assert maxd["sensitivity"] == pytest.approx(2.0)

    with pytest.raises(ValueError, match="not recognized"):
        post.compute_sensitivity("out.y", method="weird")
