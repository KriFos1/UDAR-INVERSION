"""Whitened trust-region Levenberg-Marquardt search for the MAP model.

The model is m = centre + L z with C = L L^T the prior covariance, so the
objective is 0.5 |z|^2 + 0.5 sum((g(m) - d)^2 / variance). Steps are solved in
data space. The solver tolerance can follow the misfit (``tolerance_schedule``).
"""

import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve

MINIMUM_DAMPING = 1e-6


class SearchStopped(Exception):
    """Raised by an evaluator to end a search early; the best model so far is returned (reason 'stopped')."""


def state_from(z, prediction, jacobian, target, variance, prior):
    """Loss, gradient, prediction, prior loss, likelihood loss and whitened sensitivity at an evaluated point."""
    prediction, jacobian = np.asarray(prediction), np.asarray(jacobian)
    if (prediction.shape != target.shape or jacobian.shape != (len(target), len(z))
            or not np.isfinite(prediction).all() or not np.isfinite(jacobian).all()):
        raise ValueError('Invalid prediction or Jacobian')
    residual = prediction - target
    prior_loss = .5 * float(z @ z)
    likelihood_loss = .5 * float(np.sum(residual ** 2 / variance))
    sensitivity = prior.color(jacobian.T, transpose=True).T / np.sqrt(variance)[:, None]
    gradient = z + sensitivity.T @ (residual / np.sqrt(variance))
    if not np.isfinite(gradient).all() or not np.isfinite(prior_loss + likelihood_loss):
        raise ValueError('Non-finite objective or gradient')
    return prior_loss + likelihood_loss, gradient, prediction, prior_loss, likelihood_loss, sensitivity


def gauss_newton_step(z, residual, sensitivity, damping=0., gram=None):
    """Solve [(1 + lambda) I + A^T A] s = -(z + A^T r) in data space."""
    if not np.isfinite(damping) or damping < 0:
        raise ValueError('Damping must be finite and nonnegative')
    alpha = 1. + damping
    if gram is None:
        gram = sensitivity @ sensitivity.T
    system = gram + alpha * np.eye(len(residual))
    rhs = residual - sensitivity @ z / alpha
    return -z / alpha - sensitivity.T @ cho_solve(cho_factor(system, lower=True), rhs)


def step_size(prior, step):
    """Largest change of a state entry (ln Rh or latent) of a whitened step."""
    return float(np.max(np.abs(prior.color(step))))


def trust_damping(prior, z, residual, sensitivity, gram, radius, floor):
    """Smallest damping >= floor whose step changes no state entry by more than radius."""
    if radius <= 0 or floor <= 0:
        raise ValueError('Trust radius and damping floor must be positive')

    def size(damping):
        return step_size(prior, gauss_newton_step(z, residual, sensitivity, damping, gram))
    if size(floor) <= radius:
        return floor
    low, high = floor, floor * 10.
    while size(high) > radius:
        low, high = high, high * 10.
        if high > 1e14:
            raise ValueError('Trust radius cannot be met by damping')
    while high / low > 1.01:
        middle = np.sqrt(low * high)
        low, high = (middle, high) if size(middle) > radius else (low, middle)
    return high


def candidate_dampings(prior, z, residual, sensitivity, gram, radius, floor, count):
    """Dampings for the nested trust radii radius, radius / 2, ...; strictly increasing."""
    dampings = []
    for index in range(count):
        damping = trust_damping(prior, z, residual, sensitivity, gram, radius / 2 ** index, floor)
        dampings.append(max(damping, 2. * dampings[-1]) if dampings else damping)
    return dampings


def validate_tolerance_schedule(schedule):
    """[[misfit threshold, solver tolerance], ...]: thresholds descend, tolerances do not increase.

    The tolerance of the first pair whose threshold the misfit exceeds applies;
    below the last threshold the last tolerance applies.
    """
    try:
        pairs = [(float(threshold), float(tolerance)) for threshold, tolerance in schedule]
    except (TypeError, ValueError):
        raise ValueError('A tolerance schedule is a list of [misfit threshold, solver tolerance] pairs')
    if not pairs or any(not np.isfinite(t) or t < 0 or not 0 < tol < 1 for t, tol in pairs):
        raise ValueError('Schedule thresholds must be finite and nonnegative, tolerances in (0, 1)')
    if any(a[0] <= b[0] or a[1] < b[1] for a, b in zip(pairs, pairs[1:])):
        raise ValueError('Schedule thresholds must descend and tolerances must not increase')
    return pairs


def optimize(prior, target, variance, evaluate, evaluate_forward=None, start=None, candidates=4,
             initial_trust=2., max_trust=2., min_trust=1e-3, max_iterations=30, max_rounds=4,
             max_seconds=2400., stagnation_tolerance=1e-3, stagnation_iterations=2,
             relative_gradient_tolerance=1e-3, acceptance=1e-4, target_misfit=None,
             gradient_stop_misfit=None, tolerance_schedule=None, set_tolerance=None, tolerance_retries=0):
    """Search for the MAP of N(prior.mean, C) given ``target`` with ``variance``.

    ``evaluate(models)`` returns, per model, (prediction, Jacobian) or None for
    a failed solve. Each round tries ``candidates`` damped Gauss-Newton steps
    whose dampings meet nested trust radii on the largest state change. With
    ``evaluate_forward(models)`` (predictions or None) the candidates are
    forward-only and only the accepted one is linearized. Damping follows
    Nielsen's rule and the trust radius adapts to the gain ratio; ``max_rounds``
    rounds without an acceptable candidate end the search (``nonlinearity_limited``).

    ``start`` (a model) begins the search away from the prior mean; the objective
    is unchanged. ``target_misfit`` stops once mean(r^2 / variance) is at or
    below it. ``gradient_stop_misfit`` accepts the relative-gradient stop only at
    or below that misfit.

    ``tolerance_schedule`` with ``set_tolerance(tol)`` tightens the solver as the
    misfit falls: after each accepted step the current model is solved again at
    the tighter tolerance before steps are compared. A tolerance at which the
    model cannot be solved is tried again after each of the next
    ``tolerance_retries`` accepted steps and then given up for this search.

    Raises RuntimeError when the starting model cannot be evaluated.
    """
    if (candidates < 1 or max_iterations < 1 or max_rounds < 1 or max_seconds <= 0
            or not 0 < min_trust <= initial_trust <= max_trust or stagnation_iterations < 1
            or stagnation_tolerance <= 0 or relative_gradient_tolerance <= 0 or not 0 < acceptance < .25):
        raise ValueError('Invalid Levenberg-Marquardt settings')
    schedule = validate_tolerance_schedule(tolerance_schedule) if tolerance_schedule is not None else None
    if schedule is not None and set_tolerance is None:
        raise ValueError('A tolerance schedule needs a set_tolerance callable')
    if int(tolerance_retries) != tolerance_retries or tolerance_retries < 0:
        raise ValueError('Tolerance retries must be a non-negative integer')
    center, target, variance = prior.mean, np.asarray(target, dtype=float), np.asarray(variance, dtype=float)
    if (target.ndim != 1 or variance.shape != target.shape or not np.isfinite(target).all()
            or not np.isfinite(variance).all() or np.any(variance <= 0)):
        raise ValueError('Require finite data with positive variances')
    z = np.zeros_like(center)
    if start is not None:
        start = np.asarray(start, dtype=float)
        if start.shape != center.shape or not np.isfinite(start).all():
            raise ValueError('The start model must be finite and match the prior mean')
        z = prior.whiten(start - center)
    history, trials = [], []
    damping, trust, stagnant = MINIMUM_DAMPING, float(initial_trust), 0
    began, started = time.monotonic(), None       # the time budget starts once the start model is evaluated
    evaluations = 0
    current_tolerance = None
    blocked_tolerances = set()
    tolerance_failures = {}       # tolerance -> failed solves of the current model at it
    if schedule is not None:
        current_tolerance = schedule[0][1]
        set_tolerance(current_tolerance)

    def check_time():
        if started is not None and time.monotonic() - started >= max_seconds:
            raise TimeoutError('Search time budget exhausted')

    def linearize(points):
        nonlocal evaluations
        check_time()
        results = evaluate([center + prior.color(point) for point in points])
        evaluations += len(points)
        if len(results) != len(points):
            raise ValueError('The evaluator returned the wrong number of results')
        states = []
        for point, result in zip(points, results):
            try:
                states.append(None if result is None else
                              (state_from(point, *result, target, variance, prior), np.asarray(result[1])))
            except ValueError:
                states.append(None)
        return states

    def forward_objectives(points):
        """((objective,), None) per point, or None for a failed solve."""
        nonlocal evaluations
        check_time()
        predictions = evaluate_forward([center + prior.color(point) for point in points])
        evaluations += len(points)
        outcomes = []
        for point, prediction in zip(points, predictions):
            if prediction is None or not np.all(np.isfinite(prediction)):
                outcomes.append(None)
                continue
            residual = np.asarray(prediction) - target
            outcomes.append(((.5 * float(point @ point) + .5 * float(np.sum(residual ** 2 / variance)),), None))
        return outcomes

    def record(**extra):
        loss, gradient, _, prior_loss, likelihood_loss, _ = state
        gradient_inf = float(np.max(np.abs(gradient)))
        history.append({'objective': loss, 'prior_loss': prior_loss, 'likelihood_loss': likelihood_loss,
                        'misfit': 2. * likelihood_loss / target.size, 'gradient_inf': gradient_inf,
                        'relative_gradient': gradient_inf / initial_gradient, 'trust': trust,
                        'next_damping': damping,
                        **({'solver_tolerance': current_tolerance} if schedule is not None else {}), **extra})

    def tighten_tolerance():
        nonlocal state, jacobian, current_tolerance
        if schedule is None:
            return
        misfit = 2. * state[4] / target.size
        desired = next((tol for threshold, tol in schedule if misfit > threshold), schedule[-1][1])
        allowed = [tol for _, tol in schedule if tol >= desired and tol not in blocked_tolerances]
        desired = min(allowed) if allowed else current_tolerance
        if desired >= current_tolerance:
            return
        previous = current_tolerance
        set_tolerance(desired)
        try:
            refined = linearize([z])[0]
        except (TimeoutError, SearchStopped):
            set_tolerance(previous)
            raise
        if refined is None:
            tolerance_failures[desired] = tolerance_failures.get(desired, 0) + 1
            if tolerance_failures[desired] > tolerance_retries:
                blocked_tolerances.add(desired)
            set_tolerance(previous)
            history[-1]['tolerance_failed'] = desired
        else:
            current_tolerance = desired
            state, jacobian = refined
            history[-1].update(refined_tolerance=desired, refined_objective=state[0],
                               refined_misfit=2. * state[4] / target.size)

    first = linearize([z])[0]
    if first is None:
        raise RuntimeError('The starting model could not be evaluated')
    state, jacobian = first
    started = time.monotonic()
    initial_gradient = max(float(np.max(np.abs(state[1]))), 1e-300)
    record(step_inf=0.)
    reason, message = 'iteration_limit', ''
    try:
        tighten_tolerance()
        while True:
            loss, gradient, prediction, _, _, sensitivity = state
            misfit = np.mean((prediction - target) ** 2 / variance)
            if (np.max(np.abs(gradient)) <= relative_gradient_tolerance * initial_gradient
                    and (gradient_stop_misfit is None or misfit <= gradient_stop_misfit)):
                reason = 'relative_gradient'
                break
            if target_misfit is not None and misfit <= target_misfit:
                reason = 'target_misfit'
                break
            if stagnant >= stagnation_iterations:
                reason = 'objective_stagnation'
                break
            if len(history) - 1 >= max_iterations:
                break
            residual = (prediction - target) / np.sqrt(variance)
            gram = sensitivity @ sensitivity.T
            accepted = False
            for round_index in range(max_rounds):
                dampings = candidate_dampings(prior, z, residual, sensitivity, gram, trust, damping, candidates)
                steps = [gauss_newton_step(z, residual, sensitivity, value, gram) for value in dampings]
                predicted = [float(-gradient @ step - .5 * (step @ step + np.sum((sensitivity @ step) ** 2)))
                             for step in steps]
                sizes = [step_size(prior, step) for step in steps]
                points = [z + step for step in steps]
                outcomes = linearize(points) if evaluate_forward is None else forward_objectives(points)
                rows = []
                for value, step_inf, gain_model, outcome in zip(dampings, sizes, predicted, outcomes):
                    row = {'iteration': len(history), 'round': round_index + 1, 'damping': value,
                           'step_inf': step_inf, 'trust': trust, 'predicted_reduction': gain_model,
                           'failed': outcome is None, 'accepted_candidate': False}
                    if outcome is not None:
                        actual = loss - outcome[0][0]
                        row.update(objective=outcome[0][0], actual_reduction=actual,
                                   gain_ratio=actual / gain_model if gain_model > 0 else None,
                                   accepted_candidate=bool(gain_model > 0 and actual > 0
                                                           and actual / gain_model >= acceptance))
                    rows.append(row)
                trials.extend(rows)
                good = [index for index, row in enumerate(rows) if row['accepted_candidate']]
                if not good:
                    trust = max(min_trust, min(sizes) / 4.)
                    damping = max(dampings) * 2.
                    continue
                if evaluate_forward is not None:
                    best = None
                    for candidate in sorted(good, key=lambda index: rows[index]['objective']):
                        linearized = linearize([points[candidate]])[0]
                        if linearized is None:
                            rows[candidate]['linearization_failed'] = True
                            continue
                        actual = loss - linearized[0][0]
                        ratio = actual / predicted[candidate] if predicted[candidate] > 0 else None
                        if ratio is None or actual <= 0 or ratio < acceptance:
                            rows[candidate]['linearization_rejected'] = True
                            continue
                        rows[candidate].update(objective=linearized[0][0], actual_reduction=actual, gain_ratio=ratio)
                        outcomes[candidate] = linearized
                        best = candidate
                        break
                    if best is None:
                        trust = max(min_trust, min(sizes[index] for index in good) / 4.)
                        continue
                else:
                    best = min(good, key=lambda index: rows[index]['objective'])
                ratio, size = rows[best]['gain_ratio'], sizes[best]
                damping = max(MINIMUM_DAMPING, dampings[best] * max(1 / 3, 1 - (2 * ratio - 1) ** 3))
                if ratio > .75:
                    trust = min(max_trust, max(trust, 2. * size))
                elif ratio < .25:
                    trust = max(min_trust, size / 2.)
                decrease = (loss - rows[best]['objective']) / max(abs(loss), 1e-300)
                stagnant = stagnant + 1 if decrease < stagnation_tolerance else 0
                z = points[best]
                state, jacobian = outcomes[best]
                record(step_inf=size, damping=dampings[best], gain_ratio=ratio, relative_decrease=decrease,
                       round=round_index + 1, candidate=best)
                tighten_tolerance()
                accepted = True
                break
            if not accepted:
                reason = 'nonlinearity_limited'
                message = f'{max_rounds} rounds of {candidates} candidates without an accepted step'
                break
    except TimeoutError as error:
        reason, message = 'time_limit', str(error)
    except SearchStopped as error:
        reason, message = 'stopped', str(error)
    residual = state[2] - target
    return {'model': center + prior.color(z), 'z': z, 'prediction': state[2], 'jacobian': jacobian,
            'misfit': float(np.mean(residual ** 2 / variance)), 'objective': float(state[0]),
            'solver_tolerance': current_tolerance, 'blocked_tolerances': sorted(blocked_tolerances),
            'tolerance_failures': {str(key): value for key, value in sorted(tolerance_failures.items())},
            'accepted_iterations': len(history) - 1, 'evaluations': evaluations,
            'converged': reason in ('relative_gradient', 'objective_stagnation', 'target_misfit'),
            'reason': reason, 'message': message, 'history': history, 'trials': trials,
            'seconds': time.monotonic() - began}
