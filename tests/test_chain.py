import json

import numpy as np
import pytest

from udarinv import chain, point
from udarinv.data import Benchmark
from udarinv.grid import centres

from conftest import true_fields


def load(world):
    config = world.config
    return Benchmark(config.data.path, config.data.outputs, world.dimension, config.data.noise_fraction,
                     config.data.noise_floor, config.data.floors)


def run(world, **kwargs):
    messages = []
    code = chain.run(world.config, data=load(world), solver=world.solver, log=messages.append, **kwargs)
    return code, messages


def read(world, name):
    return json.loads((world.tmp / 'run' / name).read_text())


def test_chain_fits_every_point_and_follows_the_start_rule(world):
    code, messages = run(world)
    assert code == 0 and 'chain complete' in messages[-1]
    rows, progress = read(world, 'summary.json'), read(world, 'progress.json')
    count = 8 if world.dimension == 2 else 4
    assert [r['index'] for r in rows] == list(range(count)) and progress['status'] == 'complete'
    assert all(r['misfit'] <= world.config.search.target_misfit for r in rows)
    assert all(r['solver_tolerance'] == 1e-3 and r['at_final_tolerance'] for r in rows)
    # Prior mean until two points in a row are below the target, the previous MAP afterwards.
    assert [r['start'] for r in rows[:2]] == ['prior_mean', 'prior_mean']
    assert all(r['start'] == 'previous_map' for r in rows[2:])
    assert progress['started_from_prior_mean'] == [0, 1] and progress['above_target'] == []
    assert progress['next_start'] == 'previous_map' and progress['success_streak'] == count
    # No solver work directory is left behind.
    assert not list((world.tmp / 'run').glob('point-*/evaluation-*'))
    # The solver tolerance followed the misfit: loose first, tight at the end of every point.
    for index in range(count):
        used = [c['tolerance'] for c in world.fake.calls if c['point'] == f'point-{index:04d}']
        assert used[-1] == 1e-3 and (used[0] == 1e-2 or rows[index]['start_misfit'] <= 12.)


def test_outputs_of_a_point(world):
    run(world)
    folder = world.tmp / 'run' / 'point-0001'
    data = load(world)
    report = json.loads((folder / 'report.json').read_text())
    assert report['status'] == 'finished' and report['movement'] == [1, 0] and report['entered_cells'] == 8
    assert report['memory_rho'] == pytest.approx(np.exp(-5. / 50.)) and report['channels'] == len(data.point(1)[0])
    assert 0. < report['resolved_dimensions'] <= report['channels']
    with np.load(folder / 'map.npz') as saved:
        names = 'xz' if world.dimension == 2 else 'xyz'
        grid = tuple(len(saved[f'centres_{n}']) for n in names)
        assert saved['map'].shape == (2, *grid) and saved['posterior_std'].shape == (2, *grid)
        assert np.all(saved['posterior_std'] <= saved['prior_std'].astype(float) * (1 + 1e-3) + 1e-3)
        assert np.all(saved['posterior_std'] > 0) and np.abs(saved['posterior_correlation']).max() <= 1.
        axes = [saved[f'centres_{n}'] for n in names]
        np.testing.assert_allclose(saved['bit_ft'], data.bit(1))
        truth = true_fields(axes)[0]
        near = np.ones(grid, dtype=bool)
        for axis, (centre, bit) in enumerate(zip(axes, (data.bit(1)[0], data.bit(1)[2]) if world.dimension == 2
                                                 else data.bit(1))):
            shape = [1] * len(grid)
            shape[axis] = -1
            near &= (np.abs(centre - bit) < 6.).reshape(shape)
        # The data moved the model towards the synthetic earth near the tool.
        assert np.abs(saved['map'][0][near] - truth[near]).mean() < np.abs(saved['prior_mean'][0][near] - truth[near]).mean()
    with np.load(folder / 'handoff.npz') as saved:
        cells = 7 * 8
        assert saved['mean'].shape == (2 * cells,) and np.all(saved['std'] > 0) and int(saved['index']) == 1
        assert np.all(saved['std'] <= saved['prior_std'] * (1 + 1e-9))               # data only reduce the variance
        assert saved['std'][:cells].min() < .9 * saved['prior_std'][:cells].max()
        assert len(saved['edges_x']) == 8 and len(saved['edges_z']) == 9
        residual = saved['prediction'] - saved['observations']
        assert np.mean(residual ** 2 / saved['variance']) == pytest.approx(report['misfit'])
        assert saved['mask'].sum() == len(saved['observations'])


def test_chain_resumes_pauses_and_respects_the_time_budget(world):
    world.config.run.stop = 2
    assert run(world)[0] == 0
    calls = len(world.fake.calls)
    assert run(world)[0] == 0 and len(world.fake.calls) == calls                     # nothing to do
    world.config.run.stop = 3
    (world.tmp / 'run' / 'PAUSE').touch()
    code, messages = run(world)
    assert code == 0 and 'paused before point 2' in messages[-1] and read(world, 'progress.json')['status'] == 'paused'
    (world.tmp / 'run' / 'PAUSE').unlink()
    world.config.run.seconds = 0.
    assert run(world)[0] == 75 and read(world, 'progress.json')['next_index'] == 2
    world.config.run.seconds = None
    # A folder left by an interrupted point is set aside and the point is run again.
    broken = world.tmp / 'run' / 'point-0002'
    broken.mkdir()
    (broken / 'report.json').write_text('{"status": "started", "index": 2}')
    code, messages = run(world)
    assert code == 0 and any('set aside as point-0002.incomplete-1' in m for m in messages)
    rows = read(world, 'summary.json')
    assert [r['index'] for r in rows] == [0, 1, 2] and rows[2]['start'] == 'previous_map'
    # Other settings need another output folder.
    world.config.carry.memory_ft = 80.
    with pytest.raises(ValueError, match='other settings'):
        run(world)


def test_a_warm_start_that_cannot_be_solved_is_retried_from_the_prior_mean(world):
    world.config.run.stop = 3
    run(world)
    state = {'armed': False, 'failed': 0}

    def fail(call):
        # The first evaluation of point 3 is its warm start.
        if call['point'] == 'point-0003' and not state['armed']:
            state['armed'] = True
            state['failed'] += 1
            return True
        return False
    world.fake.fail = fail
    world.config.run.stop = 4
    assert run(world)[0] == 0
    report = json.loads((world.tmp / 'run' / 'point-0003' / 'report.json').read_text())
    assert [a['start'] for a in report['attempts']] == ['previous_map', 'prior_mean']
    assert report['attempts'][0]['reason'] == 'start_failed' and report['attempts'][1].get('selected')
    assert report['start'] == 'prior_mean' and report['requested_start'] == 'warm' and len(report['failures']) == 1
    assert (world.tmp / 'run' / 'point-0003' / 'failed-logs').exists() is False      # the fake solver writes no log
    assert read(world, 'progress.json')['started_from_prior_mean'] == [0, 1, 3]


def test_background_reset_when_no_search_can_start(world):
    world.config.run.stop = 3
    run(world)
    with np.load(world.tmp / 'run' / 'point-0002' / 'handoff.npz') as saved:
        carried = saved['mean'].copy()
    seen = {'count': 0}

    def fail(call):
        # Both searches with the carried prior fail at their start; the background prior is solvable.
        if call['point'] != 'point-0003':
            return False
        seen['count'] += 1
        return seen['count'] <= 2
    world.fake.fail = fail
    world.config.run.stop = 4
    assert run(world)[0] == 0
    report = json.loads((world.tmp / 'run' / 'point-0003' / 'report.json').read_text())
    assert [a['start'] for a in report['attempts']] == ['previous_map', 'prior_mean', 'background_prior_mean']
    assert 'background' in report['prior_reset'] and report['start'] == 'background_prior_mean'
    assert read(world, 'progress.json')['prior_resets'] == [3]
    with np.load(world.tmp / 'run' / 'point-0003' / 'handoff.npz') as saved:
        background = point.background_lattice(world.config)
        np.testing.assert_array_equal(saved['prior_mean'], background.mean)        # the carried state was dropped
        assert not np.allclose(saved['prior_mean'], carried)
    # Without the reset the chain stops at that point.
    world.config.run.stop = 4
    world.config.start.background_reset = False
    world.config.run.output = str(world.tmp / 'run2')
    seen['count'] = 0
    world.fake.fail = lambda call: call['point'] == 'point-0001'
    assert chain.run(world.config, data=load(world), solver=world.solver, log=lambda m: None) == 1
    progress = json.loads((world.tmp / 'run2' / 'progress.json').read_text())
    assert progress['status'] == 'failed' and progress['index'] == 1 and 'No search could start' in progress['error']


def test_repeated_solver_failures_end_a_search_with_its_best_model(world):
    world.config.search.failure_limit = 2
    world.config.search.target_misfit = 1e-6            # unreachable: the search keeps going
    world.config.search.max_iterations = 50
    world.config.run.stop = 1
    count = {'forward': 0}

    def fail(call):
        if call['jacobian']:
            return False
        count['forward'] += 1
        return count['forward'] > 4                      # trial solves start to fail after two rounds
    world.fake.fail = fail
    assert run(world)[0] == 0
    report = json.loads((world.tmp / 'run' / 'point-0000' / 'report.json').read_text())
    assert report['reason'] == 'stopped' and len(report['failures']) == 3 and report['accepted_iterations'] >= 1
    assert not report['at_target'] and np.isfinite(report['misfit'])


def test_select_attempt_prefers_the_tighter_tolerance_then_the_lower_objective():
    attempts = [{'objective': 10., 'solver_tolerance': 1e-2}, {'objective': 50., 'solver_tolerance': 1e-3},
                {'objective': 40., 'solver_tolerance': 1e-3}, {'objective': None}]
    assert point.select_attempt(attempts) == 2
    assert point.select_attempt([{'objective': None}]) is None
    assert point.needs_prior_mean_retry([{'misfit': None}], 5.) and point.needs_prior_mean_retry([{'misfit': 7.}], 5.)
    assert not point.needs_prior_mean_retry([{'misfit': 4.}], 5.)
    assert not point.needs_prior_mean_retry([{'misfit': 7.}, {'misfit': 9.}], 5.)
