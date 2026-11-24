import json

import h5py
import numpy as np
import pytest

from udarinv import config as configuration
from udarinv import forward
from udarinv.data import Benchmark, read_tools
from udarinv.grid import FT

from conftest import TOOLS, write_benchmark


def test_benchmark_loader_error_model_and_mask(tmp_path):
    folder = write_benchmark(tmp_path / 'b', 3, points=4)
    data = Benchmark(folder, 'outputs', 3, noise_fraction=.02, noise_floor=1e-3)
    assert len(data) == 4 and data.tools == TOOLS and data.values.shape == (4, 2, 16)
    assert data.bit(2) == pytest.approx((10., 0., 1004.)) and data.md(2) == pytest.approx(10.)
    assert data.well(2)['Inclination'] == 85. and data.well(2)['X'] == pytest.approx(10. * FT)
    observations, variance, mask = data.point(1)
    assert mask.shape == (32,) and observations.shape == variance.shape == (mask.sum(),)
    # The std of a component is the same for every tool: a fraction of its mean absolute value over the tools.
    expected = .02 * np.mean(np.abs(data.values[1]), axis=0)
    np.testing.assert_allclose(data.std[1][0], np.maximum(expected, 1e-12))
    np.testing.assert_array_equal(data.std[1][0], data.std[1][1])
    np.testing.assert_array_equal(mask, np.tile(expected >= 1e-3, 2))
    floored = Benchmark(folder, 'outputs', 3, noise_fraction=.02, floors={'USDP': .5, 'USDYA': 7.})
    assert np.all(floored.std[:, :, 0] >= .5) and np.all(floored.std[:, :, 9] == 7.)
    assert floored.point(1)[2].sum() >= mask.sum()                       # a floored channel is never masked
    two = Benchmark(folder, 'outputs', 2)
    assert two.values.shape == (4, 2, 8) and two.names == forward.IN_PLANE
    with pytest.raises(ValueError, match='Unknown component'):
        Benchmark(folder, 'outputs', 2, floors={'USDYP': 1.})
    with pytest.raises(ValueError, match='below the noise floor'):
        Benchmark(folder, 'outputs', 2, noise_floor=1e9).point(0)
    axes, rh, rv = data.truth()
    assert rh.shape == tuple(len(a) for a in axes) and np.all(rv > rh)


def test_tool_files_with_commas_or_blanks(tmp_path):
    (tmp_path / 'a.inp').write_text('freq,sub\r\n2,83\r\n96,43\r\n')
    (tmp_path / 'b.inp').write_text('freq (kHz), sub (ft)\r\n   2.0000000E+00   8.3000000E+01\r\n   9.6E+01 4.3E+01\r\n\r\n')
    assert read_tools(tmp_path / 'a.inp') == read_tools(tmp_path / 'b.inp') == [(2., 83.), (96., 43.)]


@pytest.mark.parametrize('dimension', [2, 3])
def test_solver_inputs(tmp_path, dimension):
    axes = [np.array([10., 12., 15.]), np.array([-2., 0., 2., 5.]), np.array([100., 102., 104., 106., 110.])]
    if dimension == 2:
        axes = [axes[0], axes[2]]
    shape = tuple(len(a) for a in axes)
    log_rh = np.arange(np.prod(shape), dtype=float).reshape(shape) / 10.
    well = {'X': 12. * FT, 'Y': 1. * FT, 'Z': 104. * FT, 'Distance': 50., 'Inclination': 85., 'Azimuth': 0.}
    forward.write_inputs(tmp_path, log_rh, log_rh + .3, axes, well, [(6., 83.), (24., 43.)], 1e-4, True)
    with h5py.File(tmp_path / 'globalmodel.h5') as handle:
        rh = handle['formation/Rh'][:]
        ny = forward.Y_LAYERS if dimension == 2 else shape[1]
        assert rh.shape == (shape[-1], ny, shape[0])                              # stored as (z, y, x)
        np.testing.assert_allclose(handle['formation/x'][:], axes[0] * FT)
        np.testing.assert_allclose(handle['formation/z'][:], axes[-1] * FT)
        assert handle['formation/y'].shape == (ny,) and handle['wellpath/Z'].shape == (1, 1)
        if dimension == 2:
            np.testing.assert_allclose(rh[:, 2, :], np.exp(log_rh).T)
            assert np.all(rh == rh[:, :1, :])                                     # identical y layers
            assert np.median(handle['formation/y'][:]) == pytest.approx(1. * FT)  # centred on the well
        else:
            np.testing.assert_allclose(rh, np.exp(log_rh).transpose(2, 1, 0))
        np.testing.assert_allclose(handle['formation/Rv'][:], rh * np.exp(.3))
        assert not handle['formation/Inclination'][:].any()
    assert (tmp_path / 'tool.inp').read_text() == 'freq,sub\n6,83\n24,43\n'
    assert (tmp_path / 'selection.inp').read_text() == '1\n1\n0\n0.0001\n'
    assert (tmp_path / 'outputsettings.inp').read_text().split()[:3] == ['5' if dimension == 3 else '3', '1', '1']
    assert (tmp_path / 'orientation.inp').read_text() == '0\n'
    with pytest.raises(ValueError, match='overflow'):
        forward.write_inputs(tmp_path / 'bad', log_rh + 1e4, log_rh, axes, well, [(6., 83.)], 1e-4, False)
    with pytest.raises(ValueError, match='collide'):
        forward.write_inputs(tmp_path / 'bad', log_rh, log_rh, axes, well, [(6., 83.2), (6., 83.7)], 1e-4, False)


@pytest.mark.parametrize('dimension', [2, 3])
def test_predictions_and_jacobian_layout(tmp_path, monkeypatch, dimension):
    names = forward.components(dimension)
    tools = [(6., 83.), (24., 43.)]
    (tmp_path / 'outputs').mkdir()
    shuffled = list(reversed(names))
    for number, tool in enumerate(tools):
        values = [100. * number + names.index(name) for name in shuffled]
        text = '%  MD_ft  ' + '  '.join(shuffled) + '\n  1.0D+03  ' + '  '.join(f'{v:.4E}' for v in values) + '\n'
        (tmp_path / 'outputs' / forward.output_name(tool, dimension)).write_text(text)
    prediction = forward.read_predictions(tmp_path, tools, dimension)
    np.testing.assert_allclose(prediction, np.r_[np.arange(len(names)), 100. + np.arange(len(names))])
    assert forward.is_phase(dimension, 2).sum() == len(names) and forward.is_phase(dimension, 2)[0]
    assert forward.output_name((24., 43.), 2) == 'UDAR_43ft_24kHz.las'
    assert forward.output_name((24., 43.), 3) == 'UDAR3D_43ft_24kHz.las'
    shape = (3, 5) if dimension == 2 else (3, 4, 5)
    rng = np.random.default_rng(0)
    ny = forward.Y_LAYERS if dimension == 2 else shape[1]
    pulled = {index: rng.normal(size=(2, len(names), shape[-1], ny, shape[0])) for index in (1, 2)}
    monkeypatch.setattr(forward, 'pullback', lambda directory, spec, count: {k: v.copy() for k, v in pulled.items()})
    jacobian = forward.read_jacobian(tmp_path, {}, tools, shape)
    cells = int(np.prod(shape))
    assert jacobian.shape == (2 * len(names), 2 * cells)
    component, block, ix, iz = 3, 1, 2, 4
    if dimension == 2:
        expected = pulled[2][block, component, iz, :, ix].sum()
        column = block * cells + ix * shape[1] + iz
    else:
        iy = 1
        expected = pulled[2][block, component, iz, iy, ix]
        column = block * cells + (ix * shape[1] + iy) * shape[2] + iz
    assert jacobian[len(names) + component, column] == pytest.approx(expected)
    if dimension == 3:                                               # the cross-plane UAD derivatives change sign
        row = names.index('UADYP')
        np.testing.assert_allclose(jacobian[row, :cells], -pulled[1][0, row].transpose(2, 1, 0).ravel())
        row = names.index('UHRYP')
        np.testing.assert_allclose(jacobian[row, :cells], pulled[1][0, row].transpose(2, 1, 0).ravel())


def test_config_file(tmp_path, monkeypatch):
    (tmp_path / 'c.toml').write_text('''
[run]
dimension = 2
start = 3
output = "runs/a"
[data]
path = "../data/B3"
outputs = "outputs_2D"
floors = { USDP = 0.5 }
[solver]
binary = "$MY_SOLVER"
[search]
tolerance_schedule = [[0.0, 1e-6]]
[grid]
x_half = 100.0
''')
    monkeypatch.setenv('MY_SOLVER', '/opt/build/fwd2D')
    loaded = configuration.load(tmp_path / 'c.toml', stop=9, seconds=None)
    assert loaded.run.dimension == 2 and loaded.run.start == 3 and loaded.run.stop == 9
    assert loaded.data.path == str((tmp_path.parent / 'data/B3').resolve()) and loaded.solver.binary == '/opt/build/fwd2D'
    assert loaded.run.output == str(tmp_path / 'runs/a') and loaded.data.floors == {'USDP': .5}
    assert loaded.search.tolerance_schedule == [[0., 1e-6]] and loaded.grid.x_half == 100. and loaded.grid.z_half == 200.
    assert loaded.prior.ranges(2) == [[100., 50.], [160., 35.]] and loaded.prior.ranges(3)[1] == [160., 160., 35.]
    json.dumps(loaded.as_dict())                                      # the chain records it as JSON
    for text, message in (('[run]\ndimenson = 2\n', 'Unknown key'), ('[runs]\n', 'Unknown table'),
                          ('[run]\ndimension = 4\n[data]\npath="a"\noutputs="b"\n[solver]\nbinary="c"\n', 'dimension'),
                          ('[run]\ndimension = 2\n', 'required'),
                          ('[data]\npath="a"\noutputs="b"\n[solver]\nbinary="c"\n[search]\ntolerance_schedule=[[1, 1e-3],[5, 1e-4]]\n',
                           'descend')):
        (tmp_path / 'bad.toml').write_text(text)
        with pytest.raises(ValueError, match=message):
            configuration.load(tmp_path / 'bad.toml')
