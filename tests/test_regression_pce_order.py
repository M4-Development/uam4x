"""Regression order is selected when fitting completed sample results."""
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from uam4x.UQ import RegressionPCE, LatinHypercubeSampler, UniformDistribution
from uam4x.post import RegressionPCEPost


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


def test_samples_roundtrip_without_order_and_read_legacy_order():
    trade = make_post().tradestudy.executed_tradestudy
    data = trade._to_dict()
    assert 'polynomial_order' not in data
    restored = RegressionPCE._from_dict(data)
    assert restored.get_states() == trade.get_states()
    data['polynomial_order'] = 4
    legacy = RegressionPCE._from_dict(data)
    assert legacy.get_states() == trade.get_states()
    assert 'polynomial_order' not in legacy._to_dict()


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
