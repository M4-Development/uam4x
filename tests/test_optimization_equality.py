import pytest

from uam4x.control import Optimization


def test_equality_constraint_survives_serialization():
    optimization = Optimization()
    optimization.add_constraint("boom_density_consistency.residual", equals=0.0)

    restored = Optimization._from_dict(optimization._to_dict())
    constraint = restored.constraints["boom_density_consistency.residual"]
    assert constraint["equals"] == 0.0
    assert constraint["lower"] is None
    assert constraint["upper"] is None


def test_equality_constraint_cannot_have_bounds():
    optimization = Optimization()

    with pytest.raises(ValueError, match="equals cannot be combined"):
        optimization.add_constraint("residual", lower=0.0, equals=0.0)
