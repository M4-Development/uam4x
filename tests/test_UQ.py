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


# Small workflows keep these tests independent of the 100,000-point examples.
from uam4x.UQ import (
    AleatoryCSVPoints,
    BasicEpistemic,
    BasicMixedIVP,
    BasicSOP,
    EpistemicCSVPoints,
    GaussQuadraturePCE,
    HaltonSampler,
    LatinHypercubeSampler,
    MixedCSVPoints,
    MixedRegressionPCE,
    MonteCarloSampler,
    NormalDistribution,
    SobolSampler,
    gen_csv_states,
)


class EndpointSampler:
    """Sample each distribution at its bounds, retaining column order."""

    @staticmethod
    def sample(distributions, num_samples):
        assert num_samples == 2
        names = list(distributions)
        rows = []
        for endpoint in ("low", "high"):
            rows.append([getattr(distributions[name], endpoint) for name in names])
        return np.asarray(rows, dtype=float), {
            name: index for index, name in enumerate(names)
        }


class SOPAleatorySampler:
    """Make the updated distribution bounds visible in each inner sample."""

    @staticmethod
    def sample(distributions, num_samples):
        assert num_samples == 2
        dist = distributions["flow"]
        return np.array([[dist.low], [dist.high]]), {"flow": 0}


@pytest.mark.parametrize(
    "interval, message",
    [
        ([0.0, 1.0], "numpy array"),
        (np.array([[0.0, 1.0]]), "shape"),
        (np.array(["a", "b"]), "numeric"),
        (np.array([0.0, np.inf]), "finite"),
        (np.array([2.0, 1.0]), "low <= high"),
    ],
)
def test_epistemic_interval_validation(interval, message):
    analysis = BasicEpistemic(2, EndpointSampler)
    with pytest.raises(ValueError, match=message):
        analysis.add_epistemic_variable("x", interval)


def test_basic_epistemic_casts_integer_and_continuous_intervals():
    analysis = BasicEpistemic(2, EndpointSampler, seed=17)
    analysis.add_epistemic_variable("count", np.array([1, 3]))
    analysis.add_epistemic_variable("fraction", np.array([0.25, 0.75]))

    assert analysis.get_states() == [
        {"count": 1, "fraction": 0.25},
        {"count": 3, "fraction": 0.75},
    ]
    assert type(analysis.get_states()[0]["count"]) is int


def test_basic_aleatory_roundtrip_preserves_seed_sampler_and_distribution():
    original = BasicAleatory(4, MonteCarloSampler, seed=71)
    original.add_aleatory_variable("x", NormalDistribution(2.0, 0.5))

    restored = BasicAleatory._from_dict(original._to_dict())

    assert restored._to_dict() == original._to_dict()
    assert restored.SamplingClass is MonteCarloSampler
    assert restored.get_states() == original.get_states()


@pytest.mark.parametrize(
    "sampler",
    [MonteCarloSampler, LatinHypercubeSampler, SobolSampler, HaltonSampler],
)
@pytest.mark.parametrize("variable_count", [1, 2])
def test_sampler_preserves_shape_columns_and_uniform_support(sampler, variable_count):
    distributions = {"x": UniformDistribution(-2.0, -1.0)}
    if variable_count == 2:
        distributions["y"] = UniformDistribution(10.0, 20.0)

    samples, columns = sampler.sample(distributions, 8)

    assert samples.shape == (8, variable_count)
    assert columns == {name: index for index, name in enumerate(distributions)}
    assert np.all((-2.0 <= samples[:, columns["x"]]) &
                  (samples[:, columns["x"]] <= -1.0))
    if variable_count == 2:
        assert np.all((10.0 <= samples[:, columns["y"]]) &
                      (samples[:, columns["y"]] <= 20.0))


def test_gauss_quadrature_tensor_grid_and_roundtrip():
    analysis = GaussQuadraturePCE(poly_order=1)
    analysis.add_aleatory_variable("x", UniformDistribution(0.0, 1.0))
    analysis.add_aleatory_variable("y", UniformDistribution(2.0, 4.0))

    states = analysis.get_states()
    restored = GaussQuadraturePCE._from_dict(analysis._to_dict())

    assert len(states) == 4
    assert sum(state["weight"] for state in states) == pytest.approx(1.0)
    assert all(0.0 <= state["x"] <= 1.0 and
               2.0 <= state["y"] <= 4.0 for state in states)
    assert restored._to_dict() == analysis._to_dict()
    assert len(restored.get_states()) == 4


def test_gauss_quadrature_rejects_more_than_five_variables():
    analysis = GaussQuadraturePCE(poly_order=1)
    for index in range(6):
        analysis.add_aleatory_variable(
            f"x{index}", UniformDistribution(0.0, 1.0)
        )
    with pytest.raises(ValueError, match="more than 5"):
        analysis._check_analysis()


def test_mixed_ivp_repeats_outer_samples_and_keeps_inner_values():
    analysis = BasicMixedIVP(2, EndpointSampler, 2, EndpointSampler, seed=11)
    analysis.add_epistemic_variable("count", np.array([1, 2]))
    analysis.add_aleatory_variable("speed", UniformDistribution(10.0, 20.0))

    assert analysis.get_states() == [
        {"count": 1, "speed": 10.0},
        {"count": 1, "speed": 20.0},
        {"count": 2, "speed": 10.0},
        {"count": 2, "speed": 20.0},
    ]


def test_sop_updates_aleatory_distribution_for_each_outer_sample():
    analysis = BasicSOP(2, EndpointSampler, 2, SOPAleatorySampler, seed=11)
    analysis.add_epistemic_variable("flow.low", np.array([0.5, 1.5]))
    analysis.add_aleatory_variable("flow", UniformDistribution(0.0, 3.0))

    assert analysis.get_states() == [
        {"flow.low": 0.5, "flow": 0.5},
        {"flow.low": 0.5, "flow": 3.0},
        {"flow.low": 1.5, "flow": 1.5},
        {"flow.low": 1.5, "flow": 3.0},
    ]
    assert analysis.aleatory_var_to_dist["flow"].low == 0.0


def test_mixed_regression_samples_integer_and_float_values():
    analysis = MixedRegressionPCE(
        EndpointSampler, num_samples=2, seed=9
    )
    analysis.add_epistemic_variable("count", np.array([1, 3]))
    analysis.add_aleatory_variable("fraction", UniformDistribution(0.25, 0.75))

    assert analysis.get_states() == [
        {"count": 1, "fraction": 0.25},
        {"count": 3, "fraction": 0.75},
    ]


def test_csv_point_workflows_select_declared_columns_and_roundtrip(tmp_path):
    csv_path = tmp_path / "points.csv"
    csv_path.write_text("extra,e,a\n99,1,0.25\n98,2,0.75\n")
    assert gen_csv_states(csv_path, ["e", "a"]) == [
        {"e": 1, "a": 0.25}, {"e": 2, "a": 0.75}
    ]

    epistemic = EpistemicCSVPoints(str(csv_path))
    epistemic.add_epistemic_variable("e", np.array([1, 2]))
    aleatory = AleatoryCSVPoints(str(csv_path))
    aleatory.add_aleatory_variable("a", UniformDistribution(0.0, 1.0))
    mixed = MixedCSVPoints(str(csv_path))
    mixed.add_epistemic_variable("e", np.array([1, 2]))
    mixed.add_aleatory_variable("a", UniformDistribution(0.0, 1.0))

    for analysis, expected in [
        (epistemic, [{"e": 1}, {"e": 2}]),
        (aleatory, [{"a": 0.25}, {"a": 0.75}]),
        (mixed, [{"e": 1, "a": 0.25}, {"e": 2, "a": 0.75}]),
    ]:
        analysis.add_polynomial_order(2)
        analysis.set_compute_second_order(False)
        restored = type(analysis)._from_dict(analysis._to_dict())
        assert restored.get_states() == expected
        assert restored.poly_order == 2
        assert restored.computeSecondOrder is False


@pytest.mark.parametrize("value", [-1, 1.5, "2"])
def test_csv_points_reject_invalid_polynomial_order(value):
    analysis = AleatoryCSVPoints("unused.csv")
    with pytest.raises(ValueError, match="Polynomial order"):
        analysis.add_polynomial_order(value)


@pytest.mark.parametrize("value", [0, 1, "false"])
def test_csv_points_reject_non_boolean_second_order(value):
    analysis = AleatoryCSVPoints("unused.csv")
    with pytest.raises(ValueError, match="boolean"):
        analysis.set_compute_second_order(value)


from uam4x.UQ import (
    DiscreteUniformDistribution as _DiscreteUniform,
    LogNormalDistribution,
    RegressionPCE,
    SobolSensitivity,
)


@pytest.mark.parametrize(
    "distribution",
    [
        NormalDistribution(2.0, 0.5),
        UniformDistribution(-1.0, 3.0),
        _DiscreteUniform(1, 4),
        LogNormalDistribution(0.0, 0.2, 1.0),
    ],
)
def test_distribution_roundtrip_preserves_parameters_and_sample_type(distribution):
    data = distribution._to_dict()
    restored = type(distribution)._from_dict(data)

    assert restored._to_dict() == data
    sample = 2.0
    assert restored.cast_sample(sample) == sample
    assert type(restored.cast_sample(sample)) is (
        int if isinstance(restored, _DiscreteUniform) else float
    )


@pytest.mark.parametrize("holdout", [-0.01, 1.01])
def test_regression_pce_rejects_invalid_holdout_fraction(holdout):
    with pytest.raises(ValueError, match="Hold out fraction"):
        RegressionPCE(8, MonteCarloSampler, hold_out_fraction=holdout)
    with pytest.raises(ValueError, match="Hold out fraction"):
        MixedRegressionPCE(MonteCarloSampler, 8,
                           hold_out_fraction=holdout)


def test_regression_pce_roundtrip_preserves_holdout_configuration():
    analysis = RegressionPCE(
        8, MonteCarloSampler, seed=12,
        hold_out_fraction=0.25, hold_out_seed=93
    )
    analysis.add_aleatory_variable("x", UniformDistribution(0.0, 1.0))

    restored = RegressionPCE._from_dict(analysis._to_dict())

    assert restored._to_dict() == analysis._to_dict()
    assert restored.get_states() == analysis.get_states()


def test_mixed_workflow_roundtrips_preserve_both_variable_kinds():
    for analysis in (
        BasicMixedIVP(2, MonteCarloSampler, 3, MonteCarloSampler, seed=19),
        BasicSOP(2, MonteCarloSampler, 3, MonteCarloSampler, seed=19),
    ):
        analysis.add_aleatory_variable("flow", UniformDistribution(0.0, 1.0))
        analysis.add_epistemic_variable("flow.low", np.array([0.1, 0.2]))

        restored = type(analysis)._from_dict(analysis._to_dict())

        assert restored.num_epistemic_samples == 2
        assert restored.num_aleatory_samples == 3
        assert restored.seed == 19
        assert restored.aleatory_var_to_dist["flow"]._to_dict() == (
            analysis.aleatory_var_to_dist["flow"]._to_dict()
        )
        np.testing.assert_array_equal(
            restored.epistemic_var_to_interval["flow.low"],
            analysis.epistemic_var_to_interval["flow.low"],
        )


def test_mixed_regression_roundtrip_rejects_polynomial_order():
    analysis = MixedRegressionPCE(
        MonteCarloSampler, num_samples=8, seed=19,
        hold_out_fraction=0.25, hold_out_seed=93
    )
    analysis.add_aleatory_variable("a", UniformDistribution(0.0, 1.0))
    analysis.add_epistemic_variable("e", np.array([1, 3]))

    serialized = analysis._to_dict()
    assert "polynomial_order" not in serialized
    restored = MixedRegressionPCE._from_dict(serialized)

    assert restored.num_samples == analysis.num_samples
    assert restored.seed == analysis.seed
    assert restored.hold_out_seed == 93
    assert restored.hold_out_fraction == 0.25
    assert restored.SamplingClass is MonteCarloSampler
    assert restored.aleatory_var_to_dist["a"]._to_dict() == (
        analysis.aleatory_var_to_dist["a"]._to_dict()
    )
    np.testing.assert_array_equal(restored.epistemic_var_to_interval["e"], [1, 3])
    original_states = analysis.get_states()
    restored_states = restored.get_states()
    for name in ("a", "e"):
        np.testing.assert_array_equal(
            [state[name] for state in restored_states],
            [state[name] for state in original_states],
        )

    with pytest.raises(ValueError, match="polynomial_order"):
        MixedRegressionPCE._from_dict({**serialized, "polynomial_order": 1})
    with pytest.raises(TypeError):
        MixedRegressionPCE(MonteCarloSampler, 8, 1)


@pytest.mark.parametrize(
    "second_order, variable_count, expected_rows",
    [(False, 2, 16), (True, 2, 16), (True, 3, 32)],
)
def test_sobol_design_row_count_and_roundtrip(
    second_order, variable_count, expected_rows
):
    analysis = SobolSensitivity(
        4, MonteCarloSampler, seed=19, computeSecondOrder=second_order
    )
    for index in range(variable_count):
        analysis.add_aleatory_variable(
            f"x{index}", UniformDistribution(0.0, 1.0)
        )

    states = analysis.get_states()
    restored = SobolSensitivity._from_dict(analysis._to_dict())

    assert len(states) == expected_rows
    assert all(
        set(state) == set(analysis.aleatory_var_to_dist)
        and all(0.0 <= value <= 1.0 for value in state.values())
        for state in states
    )
    assert restored._to_dict() == analysis._to_dict()


def test_regression_samples_roundtrip_rejects_polynomial_order():
    from uam4x.UQ import RegressionPCE

    trade = RegressionPCE(10, LatinHypercubeSampler)
    trade.add_aleatory_variable("x", UniformDistribution(-1.0, 1.0))
    data = trade._to_dict()
    assert "polynomial_order" not in data
    restored = RegressionPCE._from_dict(data)
    np.testing.assert_array_equal(
        [state["x"] for state in restored.get_states()],
        [state["x"] for state in trade.get_states()],
    )
    with pytest.raises(ValueError, match="polynomial_order"):
        RegressionPCE._from_dict({**data, "polynomial_order": 4})
    with pytest.raises(TypeError):
        RegressionPCE(10, LatinHypercubeSampler, 4)
