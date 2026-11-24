"""The chain: one point after another along the well, resumable.

Start rule: a point is searched from the prior mean until the last
``warm_streak`` points all ended below the target misfit; while that holds the
search starts from the previous MAP (``point.run_point`` keeps the prior-mean
retry). A point at or above the target resets the streak.

The output folder holds ``point-NNN/`` per logging point, ``summary.json``
(one row per finished point), ``progress.json`` (state and the points that
need attention) and ``config.json``. Creating a file ``PAUSE`` in it stops
the chain between points. Running the same command again continues after the
last finished point.
"""

import itertools
import json
from pathlib import Path
import time

from . import point as point_module
from .data import Benchmark
from .solver import Solver

EXIT_COMPLETE, EXIT_PAUSED, EXIT_TIME, EXIT_FAILED = 0, 0, 75, 1


def write_json(path, value):
    temporary = Path(path).with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=1) + '\n')
    temporary.replace(path)


def finished_report(folder, index):
    """The report of a finished point folder, or None when it is missing, failed or incomplete."""
    try:
        report = json.loads((folder / 'report.json').read_text())
    except (OSError, ValueError):
        return None
    complete = (report.get('status') == 'finished' and report.get('index') == index
                and (folder / 'map.npz').is_file() and (folder / 'handoff.npz').is_file())
    return report if complete else None


def set_aside(folder):
    """Rename a point folder that did not finish; returns its new path."""
    for count in itertools.count(1):
        target = folder.with_name(f'{folder.name}.incomplete-{count}')
        if not target.exists():
            folder.rename(target)
            return target


def row(report):
    """What the chain records of one finished point."""
    attempts = report.get('attempts', [])
    return {'index': report['index'], 'misfit': float(report['misfit']), 'start': report['start'],
            'searches': [a['start'] for a in attempts], 'reason': report.get('reason'),
            'start_misfit': report.get('start_misfit'), 'iterations': report.get('accepted_iterations'),
            'solver_tolerance': report.get('solver_tolerance'), 'at_final_tolerance': report.get('at_final_tolerance'),
            'prior_reset': bool(report.get('prior_reset')), 'solver_failures': len(report.get('failures', [])),
            'solver_evaluations': report.get('solver_evaluations'), 'rh_min': report.get('rh_min'),
            'rh_max': report.get('rh_max'), 'resolved_dimensions': report.get('resolved_dimensions'),
            'seconds': report.get('seconds')}


def warm_next(rows, config):
    """Whether the next point starts from the previous MAP: the last ``warm_streak`` points met the target."""
    streak = config.start.warm_streak
    return len(rows) >= streak and all(r['misfit'] < config.search.target_misfit for r in rows[-streak:])


def attention(rows, target):
    return {'above_target': [r['index'] for r in rows if r['misfit'] > target],
            'not_at_final_tolerance': [r['index'] for r in rows if not r['at_final_tolerance']],
            'started_from_prior_mean': [r['index'] for r in rows if r['start'] != 'previous_map'],
            'prior_resets': [r['index'] for r in rows if r['prior_reset']]}


def run(config, data=None, solver=None, log=print):
    """Run or continue the chain of ``config``; returns the process exit code."""
    output = Path(config.run.output)
    output.mkdir(parents=True, exist_ok=True)
    data = data or Benchmark(config.data.path, config.data.outputs, config.run.dimension, config.data.noise_fraction,
                             config.data.noise_floor, config.data.floors)
    if solver is None:
        settings = {key: value for key, value in vars(config.solver).items() if key not in ('keep_scratch',)}
        solver = Solver(**settings)
    if solver.dimension != config.run.dimension:
        raise ValueError(f'A {config.run.dimension}D run needs the {config.run.dimension}D solver, '
                         f'not {solver.binary}')
    stop = len(data) if config.run.stop is None else config.run.stop
    if not config.run.start < stop <= len(data):
        raise ValueError(f'The run covers points {config.run.start}..{stop - 1} but the data have {len(data)} points')
    recorded = output / 'config.json'
    settings = config.as_dict()
    if recorded.exists():
        earlier = json.loads(recorded.read_text())
        ignore = ('run', 'solver')                          # range, time budget and machine may change between calls
        changed = [name for name in settings if name not in ignore
                   and json.loads(json.dumps(settings[name])) != earlier.get(name)]
        if changed or earlier['run']['start'] != settings['run']['start']:
            raise ValueError(f'{output} was started with other settings ({", ".join(changed) or "start"}); '
                             'use a new output folder')
    write_json(recorded, settings)
    background = point_module.background_lattice(config)
    began = time.monotonic()
    rows, previous = [], None

    def progress(status, **extra):
        write_json(output / 'summary.json', rows)
        streak = sum(1 for _ in itertools.takewhile(lambda r: r['misfit'] < config.search.target_misfit, reversed(rows)))
        write_json(output / 'progress.json',
                   {'status': status, 'first': config.run.start, 'stop': stop,
                    'completed_through': rows[-1]['index'] if rows else None, 'points': len(rows),
                    **attention(rows, config.search.target_misfit), 'success_streak': streak,
                    'next_start': 'previous_map' if warm_next(rows, config) else 'prior_mean', **extra})

    for index in range(config.run.start, stop):
        folder = output / f'point-{index:04d}'
        report = finished_report(folder, index)
        if report is None:
            if (output / 'PAUSE').exists():
                progress('paused', next_index=index)
                log(f'paused before point {index}')
                return EXIT_PAUSED
            if config.run.seconds is not None and time.monotonic() - began >= config.run.seconds:
                progress('time_budget', next_index=index)
                log(f'time budget reached before point {index}')
                return EXIT_TIME
            if folder.exists():
                log(f'point {index}: an earlier run did not finish, set aside as {set_aside(folder).name}')
            warm = previous is not None and warm_next(rows, config)
            log(f"point {index} ({index - config.run.start + 1}/{stop - config.run.start}): "
                f"start {'previous MAP' if warm else 'prior mean'}")
            report = point_module.run_point(config, data, solver, index, folder, background, previous, warm)
            if report['status'] != 'finished':
                progress('failed', index=index, error=report.get('error'))
                log(f"point {index} failed: {report.get('error')}")
                return EXIT_FAILED
            log(f"point {index}: chi2/n {report['start_misfit']:.3g} -> {report['misfit']:.3g} "
                f"({report['start']}, {report['accepted_iterations']} steps, tolerance {report['solver_tolerance']:g}, "
                f"{len(report['failures'])} failed solves, {report['seconds']:.0f} s)")
        elif (report.get('previous') or None) != (str(previous.resolve()) if previous else None):
            raise ValueError(f'{folder} does not follow {previous}')
        rows.append(row(report))
        previous = folder
        progress('running')
    progress('complete')
    log(f'chain complete through point {stop - 1}')
    return EXIT_COMPLETE
