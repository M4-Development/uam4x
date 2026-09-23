import os

os.environ.setdefault("UAM4X_DISABLE_REGISTRY_WRITES", "1")

import numpy as np
import pytest

from uam4x.results import Results, SolutionResults, Tradestudy, Result
from uam4x.post import BasicPost


SOLUTION_NAME = "test_solution"


class _DummySolution:
    def __init__(self, tracked_outputs=None):
        self.tracked_outputs = tracked_outputs or []


class _DummySolutions:
    def __init__(self, solution):
        self._solution = solution

    def get_solution(self, name):
        if name != SOLUTION_NAME:
            raise KeyError(name)
        return self._solution


class _DummyControl:
    def __init__(self):
        self.trade_studies = {}


class _DummyStudy:
    def __init__(self, tracked_outputs=None, vehicle_value=123.0):
        self.solutions = _DummySolutions(_DummySolution(tracked_outputs))
        self.control = _DummyControl()
        self._vehicle_value = vehicle_value
        self._post_data = {}

    def get_value(self, name):
        if name == "vehicles.v.c.p":
            return self._vehicle_value
        raise KeyError(name)

    def _get_post_data(self, solution_name, trade_name):
        solution_data = self._post_data.setdefault(solution_name, {})
        return solution_data.setdefault(trade_name, {})


class _DummyTrade:
    def __init__(self):
        self.aleatory_var_to_dist = {"uq_x": "normal"}
        self.variable_to_sensitivity_data = {"sens_x": {"evaluation_point": 1.0}}

    def get_desvar_names(self):
        return ["dv_x"]

    def get_all_param_names(self):
        return {"sweep_x"}


class _DummyPost:
    def __init__(self, tradestudy):
        self.tradestudy = tradestudy


def _add_results(ts):
    r0 = Result(state={"sweep_x": 10.0, "uq_x": 0.1, "sens_x": 1.0})
    r0._add_result("tracked.y", 5.0)
    r0._add_result("dv_x", 2.0)

    r1 = Result(state={"sweep_x": 20.0, "uq_x": 0.2, "sens_x": 1.1})
    r1._add_result("tracked.y", 6.0)

    ts._results.extend([r0, r1])
    ts._hash_to_result[r0.state_hash] = r0
    ts._hash_to_result[r1.state_hash] = r1


def _attach_tradestudy(study, trade_name):
    ts = Tradestudy(trade_name, SOLUTION_NAME)
    results = Results(study=study)
    solution_results = SolutionResults(SOLUTION_NAME)
    results.add_solution_result(solution_results)
    solution_results.add_tradestudy(trade_name, ts)
    return results, ts


def test_result_roundtrip_and_equal_states():
    r = Result(state={"x": np.array([1.0, 2.0])})
    r._add_result("y", np.array([3.0, 4.0]))
    r._add_template_checksum("path", "abc")
    r._add_convergence_status(True)
    r._set_case_id(7)

    data = r._to_dict()
    rebuilt = Result._from_dict(data)

    assert data["case_id"] == 7
    assert rebuilt.case_id == 7
    np.testing.assert_allclose(rebuilt.state["x"], np.array([1.0, 2.0]))
    np.testing.assert_allclose(rebuilt.outputs["y"], np.array([3.0, 4.0]))
    assert rebuilt.template_hashes["path"] == "abc"
    assert rebuilt.convergence_status is True
    assert r.equal_states(rebuilt)


def test_tradestudy_get_case_by_case_id():
    ts = Tradestudy("trade", SOLUTION_NAME)
    first = Result(state={"x": 1.0})
    second = Result(state={"x": 2.0})
    first._set_case_id(0)
    second._set_case_id(1)
    ts._results.extend([first, second])

    assert ts.get_case(0) is first
    assert ts.get_case(1) is second

    with pytest.raises(ValueError, match="no case found for case id 2"):
        ts.get_case(2)


def test_result_get_var_val_from_state_and_outputs():
    result = Result(state={"sweep_x": 10.0, "shared": "state"})
    result._add_result("tracked.y", 5.0)
    result._add_result("shared", "output")

    assert result.get_var_val("sweep_x") == 10.0
    assert result.get_var_val("tracked.y") == 5.0
    assert result.get_var_val("shared") == "state"


def test_result_get_var_val_unknown_name_raises():
    result = Result(state={"sweep_x": 10.0})

    with pytest.raises(
        ValueError,
        match="Variable with name missing.variable not in result",
    ):
        result.get_var_val("missing.variable")


def test_tradestudy_get_variable_error_paths():
    ts = Tradestudy("trade", SOLUTION_NAME)

    with pytest.raises(ValueError, match="no parent Results"):
        ts.get_variable("x")

    owner = Results(study=_DummyStudy())
    solution_results = SolutionResults(SOLUTION_NAME)
    owner.add_solution_result(solution_results)
    solution_results.add_tradestudy("trade", ts)

    with pytest.raises(ValueError, match="No results available"):
        ts.get_variable("x")


def test_tradestudy_get_variable_all_branches():
    study = _DummyStudy(tracked_outputs=["tracked.y"], vehicle_value=777.0)
    trade = _DummyTrade()
    study.control.trade_studies["trade"] = trade

    _, ts = _attach_tradestudy(study, "trade")
    _add_results(ts)

    np.testing.assert_allclose(ts.get_variable("tracked.y"), [5.0, 6.0])
    np.testing.assert_allclose(ts.get_variable("dv_x"), [2.0, np.nan], equal_nan=True)
    np.testing.assert_allclose(ts.get_variable("sweep_x"), [10.0, 20.0])
    np.testing.assert_allclose(ts.get_variable("uq_x"), [0.1, 0.2])
    np.testing.assert_allclose(ts.get_variable("sens_x"), [1.0, 1.1])
    np.testing.assert_allclose(ts.get_variable("vehicles.v.c.p"), [777.0])

    with pytest.raises(ValueError, match="not found"):
        ts.get_variable("does.not.exist")


def test_tradestudy_default_state_branch():
    study = _DummyStudy(tracked_outputs=["tracked.y"])

    _, ts = _attach_tradestudy(study, "Default State")
    _add_results(ts)

    np.testing.assert_allclose(ts.get_variable("tracked.y"), [5.0, 6.0])


def test_tradestudy_to_dict_and_from_dict_roundtrip():
    study = _DummyStudy(tracked_outputs=["tracked.y"])
    study.control.trade_studies["trade"] = _DummyTrade()
    study._post_data[SOLUTION_NAME] = {
        "trade": {"user_note": "saved-only freeform post data"}
    }

    _, ts = _attach_tradestudy(study, "trade")

    r = Result(state={"x": 1.0})
    r._set_case_id(4)
    r._add_result("tracked.y", 10.0)
    ts._results.append(r)
    ts._hash_to_result[r.state_hash] = r

    d = ts._to_dict()
    rebuilt_ts = Tradestudy._from_dict("trade", d, SOLUTION_NAME)

    assert len(rebuilt_ts._results) == 1
    assert rebuilt_ts.get_case(4).case_id == 4
    assert rebuilt_ts._results[0].outputs["tracked.y"] == 10.0

    rebuilt_results = Results._from_dict({SOLUTION_NAME: {"trade": d}}, study)
    assert rebuilt_results.get_tradestudy(SOLUTION_NAME, "trade").trade_name == "trade"
    assert (
        study._post_data[SOLUTION_NAME]["trade"]["user_note"]
        == "saved-only freeform post data"
    )


def test_results_to_dict_and_scatter(tmp_path):
    study = _DummyStudy(tracked_outputs=["tracked.y", "tracked.x"])
    study._post_data[SOLUTION_NAME] = {"Default State": {}}
    res, ts = _attach_tradestudy(study, "Default State")

    r0 = Result(state={})
    r0._add_result("tracked.x", 1.0)
    r0._add_result("tracked.y", 2.0)
    r1 = Result(state={})
    r1._add_result("tracked.x", 2.0)
    r1._add_result("tracked.y", 4.0)
    ts._results.extend([r0, r1])

    ax = BasicPost(ts).scatter("tracked.x", "tracked.y", title="My Plot", save_dir=str(tmp_path))

    assert ax.get_title() == "My Plot"
    assert (tmp_path / "My_Plot.png").exists()

    d = res._to_dict()
    assert "Default State" in d[SOLUTION_NAME]


def test_results_get_tradestudy_uses_solution_and_trade_names():
    study = _DummyStudy()
    results, ts = _attach_tradestudy(study, "trade")

    assert results.get_tradestudy(SOLUTION_NAME, "trade") is ts
    assert results.get_tradestudy(SOLUTION_NAME, "missing") is None
    assert results.get_tradestudy("missing", "trade") is None


def test_results_roundtrip_keeps_same_trade_name_separate_by_solution():
    study = _DummyStudy()
    results = Results(study=study)

    for solution_name, value in (("first", 1.0), ("second", 2.0)):
        solution_results = SolutionResults(solution_name)
        results.add_solution_result(solution_results)
        tradestudy = Tradestudy("shared_trade", solution_name)
        result = Result(state={})
        result._add_result("out", value)
        tradestudy._results.append(result)
        tradestudy._hash_to_result[result.state_hash] = result
        solution_results.add_tradestudy("shared_trade", tradestudy)

    rebuilt = Results._from_dict(results._to_dict(), study)

    assert (
        rebuilt.get_tradestudy("first", "shared_trade")._results[0].outputs["out"]
        == 1.0
    )
    assert (
        rebuilt.get_tradestudy("second", "shared_trade")._results[0].outputs["out"]
        == 2.0
    )
