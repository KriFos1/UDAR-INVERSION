"""One logging point: prior from the carried moments, MAP search, closed-form posterior, handoff.

Starts of the search:

1. From the prior mean, or from the previous point's MAP on the moved window
   (``warm``). The prior, and so the objective, is the same in both cases.
2. A warm search that ends above the target misfit, or whose start cannot be
   solved, is followed by one search from the prior mean.
3. Of the searches that produced a model, the one that ended at the tighter
   solver tolerance is kept, and among equals the one with the lower objective
   (objectives computed at different solver tolerances are not comparable).
4. If no search produced a model, the point is searched once more with the
   background prior instead of the carried one (``background_reset``).

Files written to the point folder: ``report.json``, ``map.npz`` (MAP, prior
and posterior std on the solve grid) and ``handoff.npz`` (moments on the carry
lattice, data and prediction), which the next point reads.
"""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time

import numpy as np

from . import carry, forward, laplace
from .grid import CarryLattice, centres
from .lm import SearchStopped, optimize
from .prior import LatticeMatern, SeparableMatern
from .solver import SolverError, log_tail


def background_lattice(config):
    """The climatological field on the carry lattice (build once per chain: it holds a factorisation)."""
    prior, lattice = config.prior, config.lattice
    return LatticeMatern((lattice.nx, lattice.nz), lattice.cell, prior.lattice_ranges(),
                         [prior.rh_mean, prior.latent_mean], [prior.rh_std, prior.latent_std])


def background_prior(config, axes):
    """The climatological prior on a solve grid: constant mean and std."""
    cells = int(np.prod([len(axis) for axis in axes]))
    prior = config.prior
    return SeparableMatern(axes, prior.ranges(len(axes)),
                           np.r_[np.full(cells, prior.rh_mean), np.full(cells, prior.latent_mean)],
                           np.r_[np.full(cells, prior.rh_std), np.full(cells, prior.latent_std)])


def prepare(config, data, index, background, previous=None):
    """Prior, grids and start model of a point. ``previous`` is the folder of the preceding point."""
    dimension = config.run.dimension
    bit = data.bit(index)
    origin = data.bit(0)                                  # fixes both lattices for the whole well
    lattice = CarryLattice(config.lattice.nx, config.lattice.nz, config.lattice.cell, anchor=(origin[0], origin[2]))
    lattice.update(bit[0], bit[2])
    mean, std = background.mean.copy(), background.std.copy()
    movement, rho, entered, previous_map = (0, 0), 1., 0, None
    if previous is not None:
        with np.load(Path(previous) / 'handoff.npz') as saved:
            if int(saved['index']) != index - 1:
                raise ValueError('The handoff must come from the immediately preceding logging point')
            movement = tuple(int(a - b) for a, b in zip(lattice.offsets, saved['offsets']))
            step = data.md(index) - float(saved['md_ft'])
            if step <= 0:
                raise ValueError('Measured depth must increase along the chain')
            rho = float(np.exp(-step / config.carry.memory_ft))
            mean, std, entered = carry.carry_moments(background, saved['mean'], saved['std'], movement, rho)
            previous_y = float(saved['well_y_ft'])
        with np.load(Path(previous) / 'map.npz') as saved:
            previous_map = saved['map'].astype(float)
            previous_centres = [saved[f'centres_{axis}'] for axis in ('xz' if dimension == 2 else 'xyz')]
    edges = config.grid.edges(bit, origin[2], dimension)
    axes = [centres(edge) for edge in edges]
    floors = np.repeat(np.asarray(config.prior.std_floors, dtype=float), background.cells)
    lifted_mean, lifted_std = carry.lifted_moments(
        lattice.edges(), mean, np.maximum(std, floors), (config.prior.rh_std, config.prior.latent_std), edges,
        config.carry.offplane_relax_ft, well_y=bit[1])
    warm = None
    if previous_map is not None:
        anomaly_rho = float(np.exp(-step / config.carry.anomaly_memory_ft))
        lifted_mean = lifted_mean + carry.carried_anomaly(previous_map, previous_centres, edges, anomaly_rho,
                                                          previous_well_y=previous_y)
    prior = SeparableMatern(axes, config.prior.ranges(dimension), lifted_mean.ravel(), lifted_std.ravel())
    if previous_map is not None:
        warm = carry.carried_state(previous_map, previous_centres, edges, prior.mean)
    return {'prior': prior, 'warm': warm, 'edges': edges, 'axes': axes, 'lattice': lattice, 'bit': bit,
            'movement': movement, 'rho': rho, 'entered': entered, 'carried_mean': mean, 'carried_std': std}


def select_attempt(attempts):
    """Index of the search to keep among those that produced a model; None when none did."""
    finished = [index for index, attempt in enumerate(attempts) if attempt.get('objective') is not None]
    return min(finished, key=lambda i: (attempts[i]['solver_tolerance'], attempts[i]['objective'])) if finished else None


def needs_prior_mean_retry(attempts, target):
    """Whether the warm search (the only one so far) could not start or ended above the target."""
    return len(attempts) == 1 and (attempts[0].get('misfit') is None or attempts[0]['misfit'] > target)


class Evaluator:
    """Solver calls of one point in the optimizer's variables [ln Rh | latent], with the failure record."""

    def __init__(self, config, data, solver, index, axes, output):
        self.config, self.solver, self.axes, self.output = config, solver, axes, Path(output)
        self.well, self.tools = data.well(index), data.tools
        self.observations, self.variance, self.mask = data.point(index)
        self.phase = forward.is_phase(solver.dimension, len(self.tools))[self.mask]
        self.cells = int(np.prod([len(axis) for axis in axes]))
        self.tolerance = float(config.search.tolerance_schedule[0][1])
        self.count, self.failures, self.repeated, self.evaluations = 0, [], {}, []
        self.lock = threading.Lock()

    def set_tolerance(self, value):
        self.tolerance = float(value)

    def new_search(self):
        self.repeated.clear()

    def _one(self, state, jacobian):
        with self.lock:
            self.count += 1
            number, tolerance = self.count, self.tolerance
            self.evaluations.append({'evaluation': number, 'jacobian': jacobian, 'tolerance': tolerance})
        directory = self.output / f'evaluation-{number:04d}'
        n = self.cells
        try:
            result = forward.evaluate(directory, self.solver, state[:n], state[:n] + state[n:], self.axes, self.well,
                                      self.tools, jacobian=jacobian, tolerance=tolerance,
                                      keep=self.config.solver.keep_scratch)
            prediction, gradient = result if jacobian else (result, None)
            prediction = prediction[self.mask]
            # Phases are compared on the branch of the observation.
            prediction[self.phase] = (self.observations[self.phase]
                                      + (prediction[self.phase] - self.observations[self.phase] + 180.) % 360. - 180.)
            if not jacobian:
                return prediction
            gradient = gradient[self.mask]
            gradient[:, :n] += gradient[:, n:]             # ln Rv = ln Rh + latent
            return prediction, gradient
        except (SolverError, ValueError, subprocess.SubprocessError, OSError) as error:
            tail = log_tail(directory)
            if tail:
                (self.output / 'failed-logs').mkdir(exist_ok=True)
                (self.output / 'failed-logs' / f'evaluation-{number:04d}.solver.log').write_text(tail)
            if not self.config.solver.keep_scratch:
                shutil.rmtree(directory, ignore_errors=True)
            key = f"{type(error).__name__}:{getattr(error, 'returncode', '')}"
            with self.lock:
                self.failures.append({'evaluation': number, 'jacobian': jacobian, 'tolerance': tolerance,
                                      'error': repr(error)[:300]})
                self.repeated[key] = self.repeated.get(key, 0) + 1
                exceeded = self.repeated[key] > self.config.search.failure_limit
            if exceeded:
                raise SearchStopped(f'solver failure {key} occurred {self.repeated[key]} times in this search') from error
            return None

    def _batch(self, states, jacobian):
        workers = min(self.config.search.concurrency, len(states))
        if workers <= 1:
            return [self._one(state, jacobian) for state in states]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda state: self._one(state, jacobian), states))

    def linearize(self, states):
        return self._batch(states, True)

    def predict(self, states):
        return self._batch(states, False)


def finish(context, result, variance, output, index, data, evaluator, cell):
    """Closed-form posterior of the kept search and the two files the next point and the plots read."""
    prior, edges, axes, lattice, bit = (context[key] for key in ('prior', 'edges', 'axes', 'lattice', 'bit'))
    weights = carry.lattice_weights(edges, lattice.edges(), well_y=bit[1], cell=cell)
    factor, resolved = laplace.posterior_factor(prior, result['jacobian'], variance)
    cell_std, correlation = laplace.marginals(prior, factor)
    mean = laplace.to_lattice(result['model'][:, None], prior.grid, weights)[:, 0]
    std = laplace.lattice_std(prior, weights, factor)
    del factor
    names = 'xz' if len(axes) == 2 else 'xyz'
    np.savez_compressed(
        output / 'map.npz', map=result['model'].reshape(2, *prior.grid).astype(np.float32),
        posterior_std=cell_std.reshape(2, *prior.grid).astype(np.float32),
        posterior_correlation=correlation.reshape(prior.grid).astype(np.float16),
        prior_mean=prior.mean.reshape(2, *prior.grid).astype(np.float32),
        prior_std=prior.std.reshape(2, *prior.grid).astype(np.float16),
        bit_ft=np.asarray(bit), **{f'edges_{name}': edge for name, edge in zip(names, edges)},
        **{f'centres_{name}': axis for name, axis in zip(names, axes)})
    lx, lz = lattice.edges()
    np.savez_compressed(
        output / 'handoff.npz', index=index, mean=mean, std=std, prior_mean=context['carried_mean'],
        prior_std=context['carried_std'], offsets=np.asarray(lattice.offsets), edges_x=lx, edges_z=lz,
        md_ft=data.md(index), well_y_ft=bit[1], bit_ft=np.asarray(bit),
        observations=evaluator.observations, variance=variance, mask=evaluator.mask, prediction=result['prediction'])
    return resolved


def run_point(config, data, solver, index, output, background, previous=None, warm=False, context=None):
    """Invert one logging point into the new folder ``output``; returns the report (also in report.json).

    ``context`` replaces the result of ``prepare`` (a prior built elsewhere, for tests of single points).
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    search, started = config.search, time.monotonic()
    context = context or prepare(config, data, index, background, previous)
    evaluator = Evaluator(config, data, solver, index, context['axes'], output)
    observations, variance = evaluator.observations, evaluator.variance
    attempts = []
    report = {'index': index, 'status': 'started', 'dimension': config.run.dimension,
              'grid': list(context['prior'].grid), 'channels': int(evaluator.mask.sum()),
              'previous': str(Path(previous).resolve()) if previous else None,
              'requested_start': 'warm' if warm and previous else 'prior_mean',
              'tolerance_schedule': search.tolerance_schedule, 'target_misfit': search.target_misfit,
              'movement': list(context['movement']), 'memory_rho': context['rho'],
              'entered_cells': context['entered'], 'attempts': attempts}

    def run_search(name, prior, start):
        """One search; its record goes to ``attempts``. None when the start could not be solved."""
        evaluator.new_search()
        first, failed, clock = evaluator.count + 1, len(evaluator.failures), time.monotonic()
        common = {'start': name, 'evaluations': [first, None], 'solver_failures': None, 'seconds': None}
        try:
            found = optimize(prior, observations, variance, evaluator.linearize, evaluate_forward=evaluator.predict,
                             start=start, candidates=search.candidates, initial_trust=search.initial_trust,
                             max_trust=max(2., search.initial_trust), max_iterations=search.max_iterations,
                             max_rounds=search.max_rounds, max_seconds=search.max_seconds,
                             stagnation_tolerance=search.stagnation_tolerance,
                             stagnation_iterations=search.stagnation_iterations,
                             target_misfit=search.target_misfit, gradient_stop_misfit=search.target_misfit,
                             tolerance_schedule=search.tolerance_schedule, set_tolerance=evaluator.set_tolerance,
                             tolerance_retries=search.tolerance_retries)
        except (RuntimeError, SearchStopped) as error:
            if isinstance(error, RuntimeError) and 'starting model' not in str(error):
                raise
            found = None
        common.update(evaluations=[first, evaluator.count], solver_failures=len(evaluator.failures) - failed,
                      seconds=time.monotonic() - clock)
        if found is None:
            attempts.append({**common, 'reason': 'start_failed', 'misfit': None, 'objective': None})
            return None
        attempts.append({**common, 'reason': found['reason'], 'message': found['message'], 'misfit': found['misfit'],
                         'objective': found['objective'], 'prior_term': .5 * float(found['z'] @ found['z']),
                         'start_misfit': found['history'][0]['misfit'],
                         'solver_tolerance': found['solver_tolerance'],
                         'blocked_tolerances': found['blocked_tolerances'],
                         'accepted_iterations': found['accepted_iterations'], 'history': found['history']})
        return found

    try:
        prior = context['prior']
        use_warm = bool(warm and context['warm'] is not None)
        results = [run_search('previous_map' if use_warm else 'prior_mean', prior, context['warm'] if use_warm else None)]
        if use_warm and needs_prior_mean_retry(attempts, search.target_misfit):
            results.append(run_search('prior_mean', prior, None))
        if select_attempt(attempts) is None and previous is not None and config.start.background_reset:
            # Every search with the carried prior failed to start: fall back on the background prior.
            prior = context['prior'] = background_prior(config, context['axes'])
            context['carried_mean'], context['carried_std'] = background.mean.copy(), background.std.copy()
            report['prior_reset'] = 'background mean and covariance: no search with the carried prior could start'
            results.append(run_search('background_prior_mean', prior, None))
        chosen = select_attempt(attempts)
        if chosen is None:
            raise RuntimeError('No search could start: ' + ', '.join(a['start'] for a in attempts))
        attempts[chosen]['selected'] = True
        result = results[chosen]
        del results
        final_tolerance = float(search.tolerance_schedule[-1][1])
        model = result['model']
        cells = len(model) // 2
        report.update(start=attempts[chosen]['start'], reason=result['reason'], misfit=result['misfit'],
                      start_misfit=attempts[chosen]['start_misfit'],
                      accepted_iterations=result['accepted_iterations'],
                      solver_tolerance=result['solver_tolerance'],
                      at_target=bool(result['misfit'] <= search.target_misfit),
                      at_final_tolerance=bool(result['solver_tolerance'] <= final_tolerance),
                      rh_min=float(np.exp(model[:cells].min())), rh_max=float(np.exp(model[:cells].max())))
        report['resolved_dimensions'] = finish(context, result, variance, output, index, data, evaluator,
                                               config.lattice.cell)
        report['status'] = 'finished'
    except Exception as error:                              # the report records why the point did not finish
        report.update(status='failed', error=repr(error)[:500])
    report.update(failures=evaluator.failures, solver_evaluations=len(evaluator.evaluations),
                  seconds=time.monotonic() - started)
    (output / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    return report
