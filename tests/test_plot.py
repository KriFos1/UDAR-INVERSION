import numpy as np
import pytest

pytest.importorskip('matplotlib')
pytest.importorskip('PIL')

from udarinv import chain
from udarinv.data import Benchmark
from udarinv_plot.cli import main
from udarinv_plot.results import Results, painted


@pytest.fixture
def finished(world):
    config = world.config
    data = Benchmark(config.data.path, config.data.outputs, world.dimension, config.data.noise_fraction,
                     config.data.noise_floor)
    assert chain.run(config, data=data, solver=world.solver, log=lambda m: None) == 0
    return world


def test_painted_curtain_takes_each_column_from_the_nearest_bit(finished):
    results = Results(finished.tmp / 'run')
    canvas = painted(results)
    count = len(results.indices)
    assert canvas['rh'].shape == (7 + count - 1, len(canvas['edges_z']) - 1)
    assert canvas['truth'] is not None and canvas['truth'][0].shape == canvas['rh'].shape
    column = int(np.argmin(np.abs(.5 * (canvas['edges_x'][1:] + canvas['edges_x'][:-1]) - canvas['bits'][2, 0])))
    handoff = results.handoff(results.indices[2])
    own = handoff['mean'][:7 * 8].reshape(7, 8)[3]                       # the bit column of point 2
    rows = int(round((handoff['edges_z'][0] - canvas['edges_z'][0]) / 5.))
    np.testing.assert_allclose(canvas['rh'][column, rows:rows + 8], own)
    part = painted(results, upto=results.indices[1])
    assert part['rh'].shape[0] == 8 and len(part['indices']) == 2


@pytest.mark.parametrize('command, extra, name', [
    ('curtain', [], 'curtain.png'), ('curtain', ['--no-truth', '--linear', '--limits', '1', '20', '--no-wash'], 'curtain.png'),
    ('sections', [], None), ('sections', ['--index', '1'], 'sections-0001.png'), ('residuals', [], 'residuals.png'),
    ('signals', ['--tool', '1'], 'signals-tool1.png'), ('progress', [], 'progress.png'),
    ('animate', ['--every', '2', '--dpi', '40'], 'curtain.gif')])
def test_every_figure_is_written(finished, command, extra, name):
    run = finished.tmp / 'run'
    assert main([command, str(run), *extra, '--dpi', '50'] if command != 'animate' else [command, str(run), *extra]) == 0
    if name is None:
        name = f"sections-{Results(run).indices[-1]:04d}.png"
    assert (run / name).stat().st_size > 2000
