import numpy as np
import pytest

from uam4x.control import Optimization

DESIGN_VAR_NAME = "vehicles.multi_objective_vehicle.wing1.aspect_ratio"
CONSTRAINT_NAME = "analysis.wing.drag_constraint"


@pytest.fixture
def opt():
    """Minimal constructible Optimization instance."""
    return Optimization()


# -------------------- Happy paths --------------------


def test_add_dv_no_bounds(opt):
    opt.add_design_var(DESIGN_VAR_NAME)
    assert DESIGN_VAR_NAME in opt.des_vars


def test_add_dv_upper(opt):
    opt.add_design_var(DESIGN_VAR_NAME, upper=10.0)
    assert opt.des_vars[DESIGN_VAR_NAME]["upper"] == 10.0


def test_add_dv_lower(opt):
    opt.add_design_var(DESIGN_VAR_NAME, lower=1.0)
    assert opt.des_vars[DESIGN_VAR_NAME]["lower"] == 1.0


def test_add_dv_both_bounds(opt):
    opt.add_design_var(DESIGN_VAR_NAME, lower=1.0, upper=10.0)
    assert opt.des_vars[DESIGN_VAR_NAME]["lower"] == 1.0
    assert opt.des_vars[DESIGN_VAR_NAME]["upper"] == 10.0


def test_add_dv_array_bounds(opt):
    lower = np.array([0.0, 1.0], dtype=float)
    upper = np.array([1.0, 3.0], dtype=float)
    opt.add_design_var(DESIGN_VAR_NAME, lower=lower, upper=upper)
    np.testing.assert_array_equal(opt.des_vars[DESIGN_VAR_NAME]["lower"], lower)
    np.testing.assert_array_equal(opt.des_vars[DESIGN_VAR_NAME]["upper"], upper)


def test_add_constraint_lower_only(opt):
    opt.add_constraint(CONSTRAINT_NAME, lower=0.0)
    assert opt.constraints[CONSTRAINT_NAME]["lower"] == 0.0
    assert opt.constraints[CONSTRAINT_NAME]["upper"] is None


def test_add_constraint_upper_only(opt):
    opt.add_constraint(CONSTRAINT_NAME, upper=5.0)
    assert opt.constraints[CONSTRAINT_NAME]["upper"] == 5.0
    assert opt.constraints[CONSTRAINT_NAME]["lower"] is None


def test_add_constraint_array_bounds(opt):
    lower = np.array([0.0, 1.0], dtype=float)
    upper = np.array([1.0, 3.0], dtype=float)
    opt.add_constraint(CONSTRAINT_NAME, lower=lower, upper=upper)
    np.testing.assert_array_equal(opt.constraints[CONSTRAINT_NAME]["lower"], lower)
    np.testing.assert_array_equal(opt.constraints[CONSTRAINT_NAME]["upper"], upper)


# -------------------- Failure paths: shared validator branches --------------------


@pytest.mark.parametrize(
    ("adder", "kwargs", "exc_type", "match"),
    [
        ("add_design_var", {"lower": "one"}, TypeError, r"The lower bound.*not continuos"),
        ("add_design_var", {"upper": "two"}, TypeError, r"The upper bound.*not continuos"),
        ("add_constraint", {"lower": "three"}, TypeError, r"The lower bound.*not continuos"),
        ("add_constraint", {"upper": "four"}, TypeError, r"The upper bound.*not continuos"),
        ("add_design_var", {"lower": [0.0, 1.0]}, TypeError, r"The lower bound.*not continuos"),
        ("add_design_var", {"upper": [0.0, 1.0]}, TypeError, r"The upper bound.*not continuos"),
        ("add_constraint", {"lower": [0.0, 1.0]}, TypeError, r"The lower bound.*not continuos"),
        ("add_constraint", {"upper": [0.0, 1.0]}, TypeError, r"The upper bound.*not continuos"),
        ("add_design_var", {"lower": (0.0, 1.0)}, TypeError, r"The lower bound.*not continuos"),
        ("add_design_var", {"upper": (0.0, 1.0)}, TypeError, r"The upper bound.*not continuos"),
        ("add_constraint", {"lower": (0.0, 1.0)}, TypeError, r"The lower bound.*not continuos"),
        ("add_constraint", {"upper": (0.0, 1.0)}, TypeError, r"The upper bound.*not continuos"),
        ("add_design_var", {"lower": {0.0, 1.0}}, TypeError, r"The lower bound.*not continuos"),
        ("add_design_var", {"upper": {0.0, 1.0}}, TypeError, r"The upper bound.*not continuos"),
        ("add_constraint", {"lower": {0.0, 1.0}}, TypeError, r"The lower bound.*not continuos"),
        ("add_constraint", {"upper": {0.0, 1.0}}, TypeError, r"The upper bound.*not continuos"),
        ("add_design_var", {"lower": 1}, TypeError, r"The lower bound.*not continuos"),
        ("add_design_var", {"upper": 1}, TypeError, r"The upper bound.*not continuos"),
        ("add_constraint", {"lower": 1}, TypeError, r"The lower bound.*not continuos"),
        ("add_constraint", {"upper": 1}, TypeError, r"The upper bound.*not continuos"),
        (
            "add_design_var",
            {"lower": 0.0, "upper": np.array([1.0, 2.0], dtype=float)},
            TypeError,
            r"Upper and lower bounds must be of the same type",
        ),
        (
            "add_constraint",
            {"lower": 0.0, "upper": np.array([1.0, 2.0], dtype=float)},
            TypeError,
            r"Upper and lower bounds must be of the same type",
        ),
        (
            "add_design_var",
            {"lower": np.array([0.0, 1.0], dtype=float), "upper": np.array([1.0], dtype=float)},
            ValueError,
            r"shapes must match",
        ),
        (
            "add_constraint",
            {"lower": np.array([0.0, 1.0], dtype=float), "upper": np.array([1.0], dtype=float)},
            ValueError,
            r"shapes must match",
        ),
        ("add_design_var", {"lower": 2.0, "upper": 1.0}, ValueError, r"upper bounds must be greater than or equal to the lower bounds"),
        ("add_constraint", {"lower": 2.0, "upper": 1.0}, ValueError, r"upper bounds must be greater than or equal to the lower bounds"),
        (
            "add_design_var",
            {"lower": np.array([0.0, 1.0], dtype=float), "upper": np.array([0.5, 0.0], dtype=float)},
            ValueError,
            r"upper bounds must be greater than or equal to the lower bounds",
        ),
        (
            "add_constraint",
            {"lower": np.array([0.0, 1.0], dtype=float), "upper": np.array([0.5, 0.0], dtype=float)},
            ValueError,
            r"upper bounds must be greater than or equal to the lower bounds",
        ),
    ],
)
def test_shared_bound_validation_failures(opt, adder, kwargs, exc_type, match):
    name = DESIGN_VAR_NAME if "design" in adder else CONSTRAINT_NAME
    method = getattr(opt, adder)
    with pytest.raises(exc_type, match=match):
        method(name, **kwargs)


# -------------------- Constraint-only special rule --------------------


def test_add_constraint_requires_bound(opt):
    with pytest.raises(ValueError, match="At least one constraint must be declared"):
        opt.add_constraint(CONSTRAINT_NAME)


# -------------------- Indices passthrough --------------------


def test_add_design_var_indices_passthrough(opt):
    indices = [0, 2]
    opt.add_design_var(DESIGN_VAR_NAME, lower=0.0, upper=1.0, indices=indices)
    assert opt.des_vars[DESIGN_VAR_NAME]["indices"] is indices


def test_add_constraint_indices_passthrough(opt):
    indices = np.array([1, 3])
    opt.add_constraint(CONSTRAINT_NAME, lower=0.0, upper=1.0, indices=indices)
    assert opt.constraints[CONSTRAINT_NAME]["indices"] is indices