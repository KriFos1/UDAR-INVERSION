import json

import numpy as np
import pytest

from udarinv import chain, scoring
from udarinv.cli import main
from udarinv.data import Benchmark

from conftest import true_fields


def test_truth_is_upscaled_as_layers(world):
    truth = scoring.Truth(world.folder)
    edges_x, edges_z = np.array([0., 8., 16.]), np.array([1000., 1005., 1010.])
    log_rh, log_rv = truth.on_cells(edges_x, edges_z)
    fine_x, fine_z = truth.x, truth.z
    rh, rv = (np.exp(field) for field in true_fields([fine_x, np.array([0.]), fine_z]))
    cell = (fine_x > 0) & (fine_x < 8)
    rows = (fine_z > 1000) & (fine_z < 1005)
    harmonic = 1. / np.mean(1. / rh[cell][:, 0][:, rows])
    assert np.exp(log_rh[0, 0]) == pytest.approx(harmonic, rel=.05)
    assert np.exp(log_rh[0, 0]) <= np.mean(rh[cell][:, 0][:, rows]) + 1e-12            # harmonic <= arithmetic
    assert np.all(log_rv > log_rh)


def test_summarize_and_command_line(world, capsys):
    config = world.config
    data = Benchmark(config.data.path, config.data.outputs, world.dimension, config.data.noise_fraction,
                     config.data.noise_floor)
    assert chain.run(config, data=data, solver=world.solver, log=lambda m: None) == 0
    summary = scoring.summarize(world.tmp / 'run')
    assert summary['points'] == len(data) and summary['at_target'] == len(data) and summary['solver_failures'] == 0
    assert summary['starts'] == {'previous_map': len(data) - 2, 'prior_mean': 2}
    truth = summary['truth']
    assert truth['rh_rmse_near_bit'] < truth['prior_rh_rmse'] and 0 < truth['mean_std_ln_rh'] < config.prior.rh_std
    assert len(summary['per_point']) == len(data) and all(0. <= v <= 1. for v in truth['within_1_2_3_std'])
    text = scoring.format_summary(summary)
    assert 'at target' in text and 'against the true model' in text
    assert main(['summarize', str(world.tmp / 'run'), '--no-truth', '--json']) == 0
    printed = json.loads(capsys.readouterr().out)
    assert 'truth' not in printed and printed['misfit']['max'] <= config.search.target_misfit
