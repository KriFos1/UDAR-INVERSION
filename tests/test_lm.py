import numpy as np
import pytest

from udarinv import lm
from udarinv.prior import SeparableMatern


def setup(seed=0, channels=6):
    rng = np.random.default_rng(seed)
    axes = [np.arange(0., 12., 2.), np.arange(0., 10., 2.)]
    cells = 30
    prior = SeparableMatern(axes, [[6., 5.], [8., 4.]], rng.normal(scale=.2, size=2 * cells), np.full(2 * cells, .7))
    operator = rng.normal(size=(channels, 2 * cells))
    truth = prior.mean + prior.color(rng.normal(size=2 * cells))
    variance = np.full(channels, .01)
    return prior, operator, truth, variance


def linear(operator):
    def evaluate(models):
        return [(operator @ model, operator.copy()) for model in models]
    return evaluate


def test_linear_problem_reaches_the_analytic_map():
    prior, operator, truth, variance = setup()
    target = operator @ truth
    result = lm.optimize(prior, target, variance, linear(operator), candidates=3, max_iterations=20)
    covariance = prior.color(prior.color(np.eye(len(prior.mean)), transpose=True))
    gain = covariance @ operator.T @ np.linalg.inv(operator @ covariance @ operator.T + np.diag(variance))
    expected = prior.mean + gain @ (target - operator @ prior.mean)
    np.testing.assert_allclose(result['model'], expected, atol=1e-5)
    assert result['converged'] and result['reason'] in ('relative_gradient', 'objective_stagnation')
    assert result['jacobian'].shape == operator.shape and result['misfit'] < 1.
    objectives = [row['objective'] for row in result['history']]
    assert all(b < a for a, b in zip(objectives, objectives[1:]))


def nonlinear(operator):
    def predict(model):
        return np.tanh(operator @ model / 3.) * 3.

    def evaluate(models):
        out = []
        for model in models:
            inner = operator @ model / 3.
            out.append((np.tanh(inner) * 3., (1. - np.tanh(inner) ** 2)[:, None] * operator))
        return out
    return predict, evaluate


def test_forward_only_candidates_follow_the_same_path_and_linearize_once_per_step():
    prior, operator, truth, variance = setup(seed=3)
    predict, evaluate = nonlinear(operator)
    target = predict(truth)
    counts = {'jacobian': 0, 'forward': 0}

    def counted(models):
        counts['jacobian'] += len(models)
        return evaluate(models)

    def forward(models):
        counts['forward'] += len(models)
        return [predict(model) for model in models]
    eager = lm.optimize(prior, target, variance, evaluate, candidates=4, max_iterations=15)
    lazy = lm.optimize(prior, target, variance, counted, evaluate_forward=forward, candidates=4, max_iterations=15)
    np.testing.assert_allclose(lazy['model'], eager['model'], atol=1e-10)
    assert lazy['accepted_iterations'] == eager['accepted_iterations'] >= 2
    assert counts['jacobian'] == 1 + lazy['accepted_iterations'] and counts['forward'] >= 4 * lazy['accepted_iterations']


def test_start_model_target_stop_and_gradient_gate():
    prior, operator, truth, variance = setup(seed=5)
    predict, evaluate = nonlinear(operator)
    target = predict(truth)
    stopped = lm.optimize(prior, target, variance, evaluate, target_misfit=5.)
    assert stopped['reason'] == 'target_misfit' and stopped['misfit'] <= 5.
    full = lm.optimize(prior, target, variance, evaluate, max_iterations=40)
    assert full['misfit'] < stopped['misfit']
    warm = lm.optimize(prior, target, variance, evaluate, start=stopped['model'], target_misfit=5.)
    assert warm['accepted_iterations'] == 0 and warm['reason'] == 'target_misfit'
    np.testing.assert_allclose(warm['model'], stopped['model'], atol=1e-9)
    np.testing.assert_allclose(warm['objective'], stopped['objective'], rtol=1e-9)     # same objective, other start
    # A loose gradient tolerance stops early unless the misfit gate forbids it.
    early = lm.optimize(prior, target, variance, evaluate, relative_gradient_tolerance=.5)
    gated = lm.optimize(prior, target, variance, evaluate, relative_gradient_tolerance=.5, gradient_stop_misfit=1e-9,
                        max_iterations=6)
    assert early['reason'] == 'relative_gradient' and gated['reason'] != 'relative_gradient'
    assert gated['accepted_iterations'] > early['accepted_iterations']
    with pytest.raises(ValueError):
        lm.optimize(prior, target, variance, evaluate, start=np.zeros(3))


def test_failed_start_raises_and_failed_candidates_are_skipped():
    prior, operator, truth, variance = setup(seed=7)
    target = operator @ truth
    with pytest.raises(RuntimeError, match='starting model'):
        lm.optimize(prior, target, variance, lambda models: [None for _ in models])
    calls = {'n': 0}

    def flaky(models):
        calls['n'] += 1
        results = linear(operator)(models)
        if calls['n'] == 2:
            results[0] = None                       # the boldest candidate of the first round fails
        return results
    result = lm.optimize(prior, target, variance, flaky, candidates=3)
    assert result['converged'] and result['trials'][0]['failed'] and not result['trials'][1]['failed']

    def stopping(models):
        calls['n'] += 1
        if calls['n'] > 12:
            raise lm.SearchStopped('too many solver failures')
        return linear(operator)(models)
    calls['n'] = 10
    halted = lm.optimize(prior, target, variance, stopping, candidates=1, max_iterations=50,
                         relative_gradient_tolerance=1e-12, stagnation_tolerance=1e-15)
    assert halted['reason'] == 'stopped' and halted['accepted_iterations'] >= 1 and np.isfinite(halted['misfit'])


def test_tolerance_schedule_tightens_with_the_misfit_and_reevaluates():
    prior, operator, truth, variance = setup(seed=9)
    predict, evaluate = nonlinear(operator)
    target = predict(truth)
    state = {'tolerance': None, 'log': []}

    def set_tolerance(value):
        state['tolerance'] = value

    def noisy(models):
        """A looser solve returns a slightly different response (deterministic in the model)."""
        out = []
        for prediction, jacobian in evaluate(models):
            state['log'].append(state['tolerance'])
            out.append((prediction + .3 * state['tolerance'] * np.cos(np.arange(len(prediction))), jacobian))
        return out
    start = lm.optimize(prior, target, variance, noisy, tolerance_schedule=[[1e9, 1e-2]], set_tolerance=set_tolerance,
                        max_iterations=1)['history'][0]['misfit']
    schedule = [[start / 5., 1e-2], [start / 50., 1e-3], [0., 1e-5]]
    state['log'].clear()
    result = lm.optimize(prior, target, variance, noisy, tolerance_schedule=schedule, set_tolerance=set_tolerance,
                         max_iterations=40, candidates=3)
    assert state['log'][0] == 1e-2 and result['solver_tolerance'] == 1e-5 == state['tolerance']
    used = [t for i, t in enumerate(state['log']) if i == 0 or t != state['log'][i - 1]]
    assert used == [1e-2, 1e-3, 1e-5]                                     # never loosens, every level reached in order
    refined = [row for row in result['history'] if 'refined_tolerance' in row]
    assert [row['refined_tolerance'] for row in refined] == [1e-3, 1e-5]
    assert all(row['refined_misfit'] != row['misfit'] for row in refined)     # the incumbent was solved again
    assert result['blocked_tolerances'] == [] and result['misfit'] < start / 50.
    with pytest.raises(ValueError):
        lm.optimize(prior, target, variance, noisy, tolerance_schedule=schedule)
    for bad in ([[5., 1e-3], [10., 1e-4]], [[10., 1e-4], [5., 1e-3]], [[10., 2.]], []):
        with pytest.raises(ValueError):
            lm.validate_tolerance_schedule(bad)


@pytest.mark.parametrize('retries, expected_final', [(0, 1e-2), (3, 1e-4)])
def test_a_tolerance_that_fails_is_retried_after_later_steps(retries, expected_final):
    prior, operator, truth, variance = setup(seed=11)
    predict, evaluate = nonlinear(operator)
    target = predict(truth)
    state = {'tolerance': None, 'tight_calls': 0}

    def set_tolerance(value):
        state['tolerance'] = value

    def fragile(models):
        if state['tolerance'] == 1e-4:
            state['tight_calls'] += 1
            if state['tight_calls'] <= 2:                 # the first two solves at the tight tolerance fail
                return [None for _ in models]
        return evaluate(models)
    start = float(np.mean((predict(prior.mean) - target) ** 2 / variance))
    result = lm.optimize(prior, target, variance, fragile, tolerance_schedule=[[start / 3., 1e-2], [0., 1e-4]],
                         set_tolerance=set_tolerance, tolerance_retries=retries, max_iterations=40, candidates=2)
    assert result['solver_tolerance'] == expected_final
    assert result['blocked_tolerances'] == ([1e-4] if retries == 0 else [])
    assert result['tolerance_failures'] == {'0.0001': 1 if retries == 0 else 2}
    assert any(row.get('tolerance_failed') == 1e-4 for row in result['history'])


def test_time_budget_and_settings_validation():
    prior, operator, truth, variance = setup(seed=13)
    target = operator @ truth
    clock = {'calls': 0}

    def slow(models):
        clock['calls'] += 1
        return linear(operator)(models)
    result = lm.optimize(prior, target, variance, slow, max_seconds=1e-9)
    assert result['reason'] == 'time_limit' and result['accepted_iterations'] == 0 and clock['calls'] == 1
    for bad in (dict(candidates=0), dict(initial_trust=5.), dict(acceptance=.5), dict(tolerance_retries=-1)):
        with pytest.raises(ValueError):
            lm.optimize(prior, target, variance, linear(operator), **bad)
