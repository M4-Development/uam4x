import numpy as np
import pytest

from uam4x.UQ import (
    BasicAleatory,
    DiscreteUniformDistribution,
    UniformDistribution,
)


class FixedSampler:
    """Return deterministic OpenTURNS-style floating-point samples."""

    @staticmethod
    def sample(distributions, num_samples):
        assert num_samples == 1
        assert list(distributions) == ["discrete", "continuous"]
        return np.array([[2.0, 0.25]]), {"discrete": 0, "continuous": 1}


def test_discrete_uniform_cast_sample_returns_native_integer():
    distribution = DiscreteUniformDistribution(1, 3)

    value = distribution.cast_sample(2.0)

    assert value == 2
    assert type(value) is int


@pytest.mark.parametrize("value", [1.5, 0.0, 4.0])
def test_discrete_uniform_cast_sample_rejects_values_outside_support(value):
    distribution = DiscreteUniformDistribution(1, 3)

    with pytest.raises(ValueError):
        distribution.cast_sample(value)


def test_basic_aleatory_casts_each_state_value_with_its_distribution():
    workflow = BasicAleatory(num_samples=1, SamplingClass=FixedSampler)
    workflow.add_aleatory_variable("discrete", DiscreteUniformDistribution(1, 3))
    workflow.add_aleatory_variable("continuous", UniformDistribution(0.0, 1.0))

    state = workflow.get_states()[0]

    assert state == {"discrete": 2, "continuous": 0.25}
    assert type(state["discrete"]) is int
    assert type(state["continuous"]) is float
