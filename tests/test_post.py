"""Tests for uam4x.post statistics, plots, and surrogate models."""
import os
os.environ.setdefault("UAM4X_DISABLE_REGISTRY_WRITES", "1")

from types import SimpleNamespace
import warnings

import numpy as np
import pytest
import matplotlib.pyplot as plt

from uam4x.post import (
    BasicPost, BasicAleatoryPost, BasicEpistemicPost, BasicMixedPost,
    RegressionPCEPost, GaussQuadtraturePCEPost, MixedRegressionPCEPost,
)
from uam4x.UQ import (
    RegressionPCE, LatinHypercubeSampler, UniformDistribution,
    GaussQuadraturePCE, LogNormalDistribution, MixedCSVPoints,
    MixedRegressionPCE,
)

SOLUTION_NAME = "test_solution"

class _ArrayTradeStudy:
    """Expose small completed result arrays to postprocessing methods."""

    def __init__(self, values):
        self.study = SimpleNamespace(_get_post_data=lambda **kwargs: {})
        self.solution_name = SOLUTION_NAME
        self.trade_name = "array_trade"
        self.values = {name: np.asarray(value) for name, value in values.items()}

    def get_variable(self, name):
        return self.values[name]


def test_aleatory_statistics_use_sample_and_population_conventions():
    post = BasicAleatoryPost(
        _ArrayTradeStudy({"x": [1.0, 2.0, 3.0, 4.0]})
    )

    assert post.compute_mean("x") == pytest.approx(2.5)
    assert post.compute_variance("x") == pytest.approx(5.0 / 3.0)
    assert post.compute_variance("x", method="population") == pytest.approx(1.25)
    assert post.compute_std("x") == pytest.approx(np.sqrt(5.0 / 3.0))
    assert post.compute_std("x", method="pop") == pytest.approx(np.sqrt(1.25))
    assert post.compute_skewness("x") == pytest.approx(0.0)
    assert post.compute_kurtosis("x") == pytest.approx(-1.36)

    with pytest.raises(ValueError, match="Unsupported variance method"):
        post.compute_variance("x", method="unknown")
    with pytest.raises(ValueError, match="Unsupported standard deviation method"):
        post.compute_std("x", method="unknown")


def test_aleatory_cdf_and_quantile_methods_handle_values_and_errors():
    post = BasicAleatoryPost(
        _ArrayTradeStudy({"x": [1.0, 2.0, 3.0, 4.0]})
    )

    assert post.compute_cdf_value("x", 2.0) == pytest.approx(0.5)
    assert post.compute_cdf_value("x", 2.5, method="smooth") == pytest.approx(0.625)
    assert post.get_cdf_function("x")(3.0) == pytest.approx(0.75)
    assert post.get_cdf_function("x", method="smooth")(0.0) == pytest.approx(0.0)
    assert post.compute_quantile("x", 0.5, method="linear") == pytest.approx(2.5)

    with pytest.raises(ValueError, match="Quantile must be between"):
        post.compute_quantile("x", 1.1)
    with pytest.raises(ValueError, match="Unsupported quantile method"):
        post.compute_quantile("x", 0.5, method="unknown")
    with pytest.raises(ValueError, match="Unsupported CDF method"):
        post.get_cdf_function("x", method="unknown")
    with pytest.raises(ValueError, match="Unsupported CDF method"):
        post.compute_cdf_value("x", 2.0, method="unknown")


def test_aleatory_correlations_on_monotonic_and_linear_arrays():
    post = BasicAleatoryPost(
        _ArrayTradeStudy({
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [2.0, 4.0, 6.0, 8.0],
            "z": [8.0, 6.0, 4.0, 2.0],
        })
    )

    pearson, pearson_p = post.compute_pearson_correlation("x", "y")
    spearman, spearman_p = post.compute_spearman_correlation("x", "z")
    assert pearson == pytest.approx(1.0)
    assert spearman == pytest.approx(-1.0)
    assert pearson_p < 0.05
    assert spearman_p < 0.05


def test_basic_post_regression_drops_nonfinite_outputs():
    post = BasicPost(
        _ArrayTradeStudy({
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [2.0, 4.0, np.nan, 8.0],
        })
    )

    slope, intercept, correlation, _, _ = post.linear_regression("x", "y")

    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(0.0)
    assert correlation == pytest.approx(1.0)


def test_aleatory_plots_render_and_save_without_external_data(tmp_path):
    plt.switch_backend("Agg")
    post = BasicAleatoryPost(
        _ArrayTradeStudy({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    )
    try:
        histogram = post.plot_pdf("x", method="histogram", save_dir=tmp_path)
        kernel = post.plot_pdf("x", method="kde")
        cdf_points = post.plot_cdf("x", save_dir=tmp_path)

        assert len(histogram.patches) == 20
        assert len(kernel.lines) == 1
        np.testing.assert_allclose(cdf_points[:5], [1, 2, 3, 4, 5])
        np.testing.assert_allclose(cdf_points[5:], [0.2, 0.4, 0.6, 0.8, 1.0])
        assert (tmp_path / "PDF_x.png").is_file()
        assert (tmp_path / "CDF_of_x.png").is_file()
        with pytest.raises(ValueError, match="Unsupported PDF plotting method"):
            post.plot_pdf("x", method="unknown")
    finally:
        plt.close("all")


def _mixed_array_trade(values, epistemic_states):
    trade = _ArrayTradeStudy({"output": values})
    trade.executed_tradestudy = SimpleNamespace(
        epistemic_var_to_interval={"e": np.array([0, 1])}
    )
    trade._results = [
        SimpleNamespace(state={"e": state})
        for state in epistemic_states
    ]
    return trade


def test_epistemic_bounds_include_states_at_extrema():
    trade = _mixed_array_trade(
        [3.0, -2.0, 5.0], [0, 1, 2]
    )

    bounds = BasicEpistemicPost(trade).compute_bounds("output")

    assert bounds["minimum"] == {"value": -2.0, "state": {"e": 1}}
    assert bounds["maximum"] == {"value": 5.0, "state": {"e": 2}}


def test_mixed_post_groups_results_by_epistemic_state_and_interpolates_bounds():
    trade = _mixed_array_trade(
        [1.0, 2.0, 3.0, 4.0], [0, 0, 1, 1]
    )
    post = BasicMixedPost(trade)

    cdfs, quantiles, probabilities = post._generate_output_var_data("output")

    assert set(cdfs) == {(0,), (1,)}
    np.testing.assert_allclose(quantiles, [[1.0, 3.0], [2.0, 4.0]])
    np.testing.assert_allclose(probabilities, [[0.5, 0.5], [1.0, 1.0]])
    assert post.get_plevel_bounds("output", 0.0) == pytest.approx((1.0, 3.0))
    assert post.get_plevel_bounds("output", 0.5) == pytest.approx((1.0, 3.0))
    assert post.get_plevel_bounds("output", 0.75) == pytest.approx((1.5, 3.5))
    assert post.get_plevel_bounds("output", 1.0) == pytest.approx((2.0, 4.0))
    with pytest.raises(ValueError, match="p_level must be between"):
        post.get_plevel_bounds("output", 1.1)


def test_mixed_post_rejects_unequal_aleatory_group_sizes():
    trade = _mixed_array_trade(
        [1.0, 2.0, 3.0], [0, 0, 1]
    )

    with pytest.raises(ValueError, match="Inconsistent number"):
        BasicMixedPost(trade)._generate_output_var_data("output")


def test_mixed_post_plots_pbox_and_horsetails(tmp_path):
    plt.switch_backend("Agg")
    post = BasicMixedPost(
        _mixed_array_trade([1.0, 2.0, 3.0, 4.0], [0, 0, 1, 1])
    )
    try:
        post.plot_pbox("output", save_dir=tmp_path)
        post.plot_horsetails("output", save_dir=tmp_path)

        assert (tmp_path / "P-box_Plot_of_output.png").is_file()
        assert (tmp_path / "Horsetail_Plot_of_output.png").is_file()
    finally:
        plt.close("all")


def make_post(num_samples=30, hold_out_fraction=0.0, invalid=0):
    trade = RegressionPCE(num_samples, LatinHypercubeSampler,
                          hold_out_fraction=hold_out_fraction)
    trade.add_aleatory_variable('x', UniformDistribution(-1.0, 1.0))
    x = np.linspace(-1.0, 1.0, num_samples)
    y = 1.0 + x + x**2
    y[:invalid] = np.nan
    values = {'x': x, 'y': y}
    results = SimpleNamespace(
        executed_tradestudy=trade,
        _results=[None] * num_samples,
        get_variable=values.__getitem__,
        solution_name='basic', trade_name='pce',
        study=SimpleNamespace(_get_post_data=lambda **kwargs: {}),
    )
    return RegressionPCEPost(results)


def test_same_samples_fit_different_orders_with_same_holdout():
    post = make_post(hold_out_fraction=0.2)
    linear = post.generate_surrogate('y', poly_order=1)
    quadratic = post.generate_surrogate('y', poly_order=2)
    np.testing.assert_array_equal(linear.X_test, quadratic.X_test)
    np.testing.assert_array_equal(linear.y_test, quadratic.y_test)
    assert linear.get_surrogate_rmse() > 0.01
    assert quadratic.get_surrogate_rmse() < 1e-10


def test_warning_counts_only_finite_training_points():
    post = make_post(num_samples=12, hold_out_fraction=0.25, invalid=2)
    with pytest.warns(RuntimeWarning) as caught:
        post.generate_surrogate('y', poly_order=2)
    messages = [str(w.message) for w in caught]
    assert any('Dropping 2' in message for message in messages)
    assert any('training_samples=7' in message and '6-9 training points' in message
               for message in messages)


@pytest.mark.parametrize('num_samples, expected_warning', [(8, True), (9, False)])
def test_warning_threshold_is_three_times_basis_size(num_samples, expected_warning):
    post = make_post(num_samples=num_samples)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        post.generate_surrogate('y', poly_order=2)
    assert any('lightly sampled' in str(w.message) for w in caught) == expected_warning


def test_insufficient_training_points_still_raise():
    post = make_post(num_samples=4, hold_out_fraction=0.5)
    with pytest.raises(ValueError, match='Need at least 3 samples'):
        post.generate_surrogate('y', poly_order=2)


@pytest.mark.parametrize('order', [-1, 1.5, True, '3'])
def test_invalid_orders_rejected(order):
    with pytest.raises(ValueError, match='nonnegative integer'):
        make_post().generate_surrogate('y', poly_order=order)


@pytest.mark.parametrize("invalid_value", [1.0, 0.5])
@pytest.mark.parametrize(
    "workflow_type, post_type",
    [
        (RegressionPCE, RegressionPCEPost),
        (GaussQuadraturePCE, GaussQuadtraturePCEPost),
        (MixedRegressionPCE, MixedRegressionPCEPost),
    ],
)
def test_pce_rejects_lognormal_values_at_or_below_shift(
    workflow_type, post_type, invalid_value
):
    if workflow_type is RegressionPCE:
        workflow = RegressionPCE(2, LatinHypercubeSampler)
    elif workflow_type is GaussQuadraturePCE:
        workflow = GaussQuadraturePCE(poly_order=1)
    else:
        workflow = MixedRegressionPCE(
            LatinHypercubeSampler, num_samples=2
        )
    workflow.add_aleatory_variable(
        "x", LogNormalDistribution(0.0, 0.25, gamma=1.0)
    )
    values = {
        "x": np.array([invalid_value, 2.0]),
        "y": np.array([1.0, 2.0]),
    }
    results = SimpleNamespace(
        executed_tradestudy=workflow,
        _results=[
            SimpleNamespace(state={"weight": 0.5}),
            SimpleNamespace(state={"weight": 0.5}),
        ],
        get_variable=values.__getitem__,
        solution_name="basic",
        trade_name="pce",
        study=SimpleNamespace(_get_post_data=lambda **kwargs: {}),
    )
    post = post_type(results)

    with pytest.raises(ValueError, match="at or below its shift"):
        if post_type is GaussQuadtraturePCEPost:
            post.generate_surrogate("y")
        else:
            post.generate_surrogate("y", poly_order=1)


def test_quadratic_surrogate_predicts_and_reports_analytical_moments():
    surrogate = make_post(num_samples=30).generate_surrogate("y", poly_order=2)

    assert surrogate.predict({"x": 0.0})[0] == pytest.approx(1.0, abs=1e-10)
    assert surrogate.analytical_mean == pytest.approx(4.0 / 3.0, abs=1e-10)
    assert surrogate.analytical_variance == pytest.approx(19.0 / 45.0, abs=1e-10)
    assert surrogate.analytical_std == pytest.approx(np.sqrt(19.0 / 45.0), abs=1e-10)
    assert surrogate.get_first_order_sobol("x") == pytest.approx(1.0)
    assert surrogate.get_total_order_sobol("x") == pytest.approx(1.0)


def test_quadratic_surrogate_validates_external_predictions_and_metrics():
    surrogate = make_post(num_samples=30).generate_surrogate("y", poly_order=2)
    x = np.array([[-0.8], [-0.3], [0.2], [0.7]])
    y = 1.0 + x + x**2

    assert surrogate.get_surrogate_mse(x, y) == pytest.approx(0.0, abs=1e-20)
    assert surrogate.get_surrogate_rmse(x, y) == pytest.approx(0.0, abs=1e-10)
    assert surrogate.get_surrogate_nrmse(x, y) == pytest.approx(0.0, abs=1e-10)
    assert surrogate.get_surrogate_r2(x, y) == pytest.approx(1.0, abs=1e-10)
    assert surrogate.get_surrogate_rel_mse(x, y) == pytest.approx(0.0, abs=1e-20)

    with pytest.raises(ValueError, match="No hold out"):
        surrogate.get_surrogate_mse()
    with pytest.raises(ValueError, match="Both X and y"):
        surrogate.get_surrogate_r2(x, None)
    with pytest.raises(ValueError, match="2D array"):
        surrogate.get_surrogate_mse(x[:, 0], y)
    with pytest.raises(ValueError, match="one validation output"):
        surrogate.get_surrogate_mse(x, y[:-1])


@pytest.mark.parametrize("method", ["MC", "LHS", "Sobol", "Halton"])
def test_quadratic_surrogate_sampling_methods_follow_fitted_curve(method):
    surrogate = make_post(num_samples=30).generate_surrogate("y", poly_order=2)

    points, outputs = surrogate.sample(8, method=method)

    assert points.shape == (8, 1)
    assert outputs.shape == (8, 1)
    assert np.all((-1.0 <= points) & (points <= 1.0))
    np.testing.assert_allclose(outputs, 1.0 + points + points**2, atol=1e-10)


def test_surrogate_plots_require_samples_and_render_after_sampling(tmp_path):
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")
    surrogate = make_post(num_samples=30).generate_surrogate("y", poly_order=2)
    try:
        with pytest.raises(ValueError, match="Must generate samples"):
            surrogate.plot_cdf()
        with pytest.raises(ValueError, match="Must generate samples"):
            surrogate.plot_pdf()

        surrogate.sample(16, method="LHS")
        cdf = surrogate.plot_cdf(save_dir=tmp_path)
        pdf = surrogate.plot_pdf(bins=8, save_dir=tmp_path)
        surrogate.plot_total_order_sobol(save_dir=tmp_path)

        assert len(cdf) == 32
        assert len(pdf.patches) == 8
        assert (tmp_path / "CDF_of_surrogate_output.png").is_file()
        assert (tmp_path / "PDF_of_surrogate_output.png").is_file()
        assert (tmp_path / "Total_Sensitivity_Indices_of_Surrogate.png").is_file()
    finally:
        plt.close("all")


def make_mixed_linear_post(num_samples=40):
    workflow = MixedRegressionPCE(
        LatinHypercubeSampler, num_samples=num_samples, seed=17
    )
    workflow.add_epistemic_variable("e", np.array([0.0, 1.0]))
    workflow.add_aleatory_variable("a", UniformDistribution(-1.0, 1.0))
    states = workflow.get_states()
    e = np.array([state["e"] for state in states])
    a = np.array([state["a"] for state in states])
    values = {"e": e, "a": a, "y": 1.0 + 2.0 * e + 3.0 * a}
    results = SimpleNamespace(
        executed_tradestudy=workflow,
        _results=[None] * num_samples,
        values=values,
        get_variable=values.__getitem__,
        solution_name="basic",
        trade_name="mixed_linear",
        study=SimpleNamespace(_get_post_data=lambda **kwargs: {}),
    )
    return MixedRegressionPCEPost(results)


def make_mixed_linear_surrogate(num_samples=40):
    return make_mixed_linear_post(num_samples).generate_surrogate("y", poly_order=1)


def test_mixed_regression_fits_multiple_orders_and_validates_order():
    post = make_mixed_linear_post()
    for order in (1, 2):
        surrogate = post.generate_surrogate("y", poly_order=order)
        assert surrogate.predict({"e": 0.4, "a": 0.2})[0] == pytest.approx(
            2.4, abs=1e-8
        )
    for invalid_order in (-1, 1.5, True):
        with pytest.raises(ValueError, match="Polynomial order"):
            post.generate_surrogate("y", poly_order=invalid_order)


def test_mixed_regression_checks_valid_training_count_after_filtering():
    post = make_mixed_linear_post(num_samples=6)
    post.tradestudy.values["y"][:4] = np.nan
    with pytest.warns(RuntimeWarning, match="Dropping 4"):
        with pytest.raises(ValueError, match="Need at least 3 samples"):
            post.generate_surrogate("y", poly_order=1)


def test_mixed_regression_checks_training_count_after_holdout():
    post = make_mixed_linear_post(num_samples=4)
    post.tradestudy.executed_tradestudy.hold_out_fraction = 0.5
    post.tradestudy.executed_tradestudy.hold_out_seed = 5
    with pytest.raises(ValueError, match="Need at least 3 samples"):
        post.generate_surrogate("y", poly_order=1)


def test_mixed_csv_results_fit_without_holdout_configuration():
    post = make_mixed_linear_post()
    workflow = MixedCSVPoints("unused.csv")
    workflow.add_epistemic_variable("e", np.array([0.0, 1.0]))
    workflow.add_aleatory_variable("a", UniformDistribution(-1.0, 1.0))
    post.tradestudy.executed_tradestudy = workflow
    surrogate = post.generate_surrogate("y", poly_order=1)
    assert surrogate.predict({"e": 0.4, "a": 0.2})[0] == pytest.approx(
        2.4, abs=1e-8
    )


def test_mixed_linear_surrogate_predicts_and_validates_external_points():
    surrogate = make_mixed_linear_surrogate()
    x = np.array([[0.1, -0.8], [0.4, 0.2], [0.9, 0.7]])
    y = (1.0 + 2.0 * x[:, 0] + 3.0 * x[:, 1])[:, None]

    assert surrogate.predict({"e": 0.4, "a": 0.2})[0] == pytest.approx(
        2.4, abs=1e-8
    )
    assert surrogate.get_surrogate_mse(x, y) == pytest.approx(0.0, abs=1e-16)
    assert surrogate.get_surrogate_rmse(x, y) == pytest.approx(0.0, abs=1e-8)
    assert surrogate.get_surrogate_nrmse(x, y) == pytest.approx(0.0, abs=1e-8)
    assert surrogate.get_surrogate_r2(x, y) == pytest.approx(1.0, abs=1e-8)
    assert surrogate.get_surrogate_rel_mse(x, y) == pytest.approx(0.0, abs=1e-16)
    with pytest.raises(ValueError, match="Both X and y"):
        surrogate.get_surrogate_mse(x, None)


@pytest.mark.parametrize("method", ["MC", "LHS", "Sobol", "Halton"])
def test_mixed_surrogate_nested_samples_keep_outer_state(method):
    surrogate = make_mixed_linear_surrogate()

    points, outputs = surrogate.sample(
        num_epistemic_samples=3, num_aleatory_samples=4, method=method
    )

    assert points.shape == (12, 2)
    assert outputs.shape == (12, 1)
    np.testing.assert_allclose(
        outputs[:, 0], 1.0 + 2.0 * points[:, 0] + 3.0 * points[:, 1],
        atol=1e-8,
    )
    for first in (0, 4, 8):
        np.testing.assert_allclose(
            points[first:first + 4, 0], points[first, 0]
        )
    _, quantiles, probabilities = surrogate._aggregate_predicted_vals()
    assert quantiles.shape == (4, 3)
    np.testing.assert_allclose(probabilities[:, 0], [0.25, 0.5, 0.75, 1.0])
    lower, upper = surrogate.get_plevel_bounds(0.5)
    assert lower <= upper


def test_mixed_surrogate_rejects_unknown_sampling_method():
    surrogate = make_mixed_linear_surrogate()

    with pytest.raises(ValueError, match="Unsupported sampling method"):
        surrogate.sample(2, 3, method="unknown")


def test_basic_post_scatter_saves_plot(tmp_path):
    plt.switch_backend("Agg")
    post = BasicPost(_ArrayTradeStudy({"x": [1.0, 2.0], "y": [2.0, 4.0]}))
    try:
        ax = post.scatter("x", "y", title="My Plot", save_dir=str(tmp_path))
        assert ax.get_title() == "My Plot"
        assert (tmp_path / "My_Plot.png").exists()
    finally:
        plt.close("all")
