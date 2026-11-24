"""A small synthetic world: a benchmark folder and a fast stand-in for the Fortran solver."""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from udarinv import forward
from udarinv.config import Config, Data, Lattice, Prior, Run, Search, SolverSettings, Start
from udarinv.grid import FT, SolveGrid, centres
from udarinv.solver import SolverError

TOOLS = [(6., 83.), (24., 43.)]


def true_fields(axes):
    """ln Rh and ln Rv of the synthetic earth at cell centres (absolute coordinates in ft)."""
    grids = np.meshgrid(*axes, indexing='ij')
    x, z = grids[0], grids[-1]
    log_rh = 1.5 + .5 * np.sin(x / 30.) + .02 * (z - 1000.)
    if len(axes) == 3:
        log_rh = log_rh + .01 * grids[1]
    return log_rh, log_rh + .3


def kernels(axes, well, dimension):
    """Linear response operators (channels x cells) for ln Rh and ln Rv, fixed in tool coordinates."""
    grids = np.meshgrid(*axes, indexing='ij')
    relative = [grid - well[key] / FT for grid, key in zip(grids, 'XZ' if dimension == 2 else 'XYZ')]
    volume = np.ones_like(grids[0])
    for axis, centre in enumerate(axes):
        shape = [1] * len(axes)
        shape[axis] = -1
        volume = volume * np.gradient(centre).reshape(shape) if len(centre) > 1 else volume
    radius2 = sum(d ** 2 for d in relative)
    names = forward.components(dimension)
    rows_h, rows_v = [], []
    for tool, (frequency, spacing) in enumerate(TOOLS):
        for index, name in enumerate(names):
            length = 8. + 3. * tool + index % 4
            kernel = np.exp(-radius2 / length ** 2) * (1. + .5 * np.cos(.3 * relative[0] + .2 * index + .4 * relative[-1]))
            if dimension == 3 and 'Y' in name:
                kernel = kernel * relative[1] / length
            kernel = kernel * volume / (np.pi * length ** 2) ** (len(axes) / 2.)
            scale = 20. if index % 2 == 0 else 2.                      # phases in degrees, attenuations in dB
            rows_h.append((scale * kernel).ravel())
            rows_v.append((.5 * scale * kernel * np.cos(.1 * index)).ravel())
    return np.array(rows_h), np.array(rows_v)


class FakeSolver:
    """Replaces ``forward.evaluate``: linear in the log fields, with programmable failures."""

    def __init__(self, dimension):
        self.dimension, self.calls, self.fail = dimension, [], lambda call: False

    def __call__(self, directory, solver, log_rh, log_rv, centres_ft, well, tools, jacobian=True, tolerance=1e-6,
                 keep=False):
        call = {'number': len(self.calls) + 1, 'jacobian': jacobian, 'tolerance': tolerance,
                'point': directory.parent.name, 'log_rh': np.array(log_rh, dtype=float)}
        self.calls.append(call)
        if self.fail(call):
            raise SolverError('convergence error')
        operator_h, operator_v = kernels(centres_ft, well, self.dimension)
        prediction = operator_h @ np.ravel(log_rh) + operator_v @ np.ravel(log_rv)
        return (prediction, np.hstack([operator_h, operator_v])) if jacobian else prediction


def write_benchmark(folder, dimension, points=10, grid=None):
    """A benchmark folder whose data are the fake solver's response of the synthetic earth."""
    folder.mkdir(parents=True)
    distance = 5. * np.arange(points)
    well = {'X': distance * FT, 'Y': np.zeros(points), 'Z': (1000. + .4 * distance) * FT, 'Distance': distance * FT,
            'Inclination': np.full(points, 85.), 'Azimuth': np.zeros(points)}
    with h5py.File(folder / 'globalmodel.h5', 'w') as handle:
        for key, values in well.items():
            handle[f'wellpath/{key}'] = values[:, None]
        x, y, z = np.arange(-60., 120., 4.), np.arange(-30., 31., 4.), np.arange(950., 1071., 1.)
        log_rh, log_rv = true_fields([x, y, z])
        for name, axis in zip('xyz', (x, y, z)):
            handle[f'formation/{name}'] = (axis * FT)[:, None]
        handle['formation/Rh'] = np.exp(log_rh).transpose(2, 1, 0)
        handle['formation/Rv'] = np.exp(log_rv).transpose(2, 1, 0)
    (folder / 'tool.inp').write_text('freq,sub\n' + ''.join(f'{int(f)},{int(s)}\n' for f, s in TOOLS))
    grid = grid or small_grid()
    names = forward.IN_PLANE + forward.CROSS_PLANE
    tables = np.zeros((points, len(TOOLS), 16))
    for index in range(points):
        point_well = {key: float(values[index]) for key, values in well.items()}
        bit = tuple(point_well[key] / FT for key in 'XYZ')
        axes = [centres(edge) for edge in grid.edges(bit, 1000., dimension)]
        log_rh, log_rv = true_fields(axes)
        operator_h, operator_v = kernels(axes, point_well, dimension)
        count = len(forward.components(dimension))
        tables[index, :, :count] = (operator_h @ log_rh.ravel() + operator_v @ log_rv.ravel()).reshape(len(TOOLS), count)
    (folder / 'outputs').mkdir()
    for tool, (frequency, spacing) in enumerate(TOOLS):
        lines = ['%  MD_ft  ' + '  '.join(names)]
        lines += ['  '.join(f'{value:.7E}' for value in np.r_[distance[index], tables[index, tool]])
                  for index in range(points)]
        (folder / 'outputs' / f'UDAR3D_{int(spacing)}ft_{int(frequency)}kHz.las').write_text('\n'.join(lines) + '\n')
    return folder


def small_grid():
    return SolveGrid(x_half=30., x_grade='10:2:1.5:8', y_half=16., y_grade='4:2:1.5:6', z_half=30., z_grade='12:2:1.5:8',
                     z_lattice=2.)


def small_config(folder, dimension, output, **search):
    return Config(run=Run(dimension=dimension, start=0, stop=None, output=str(output)),
                  data=Data(path=str(folder), outputs='outputs', noise_fraction=.02, noise_floor=1e-6),
                  grid=small_grid(), lattice=Lattice(nx=7, nz=8, cell=5.),
                  prior=Prior(rh_mean=1., rh_std=.8, latent_mean=.3, latent_std=.3, rh_ranges=(20., 20., 12.),
                              latent_ranges=(30., 30., 10.), std_floors=(.05, .03)),
                  solver=SolverSettings(binary='fake'), start=Start(warm_streak=2),
                  search=Search(**{'candidates': 2, 'max_iterations': 12, 'tolerance_schedule': [[12., 1e-2], [0., 1e-3]],
                                   **search})).validate()


@pytest.fixture(params=[2, 3])
def world(request, tmp_path, monkeypatch):
    """Benchmark folder, config and fake solver for a 2D and a 3D run."""
    dimension = request.param
    folder = write_benchmark(tmp_path / 'benchmark', dimension, points=8 if dimension == 2 else 4)
    fake = FakeSolver(dimension)
    monkeypatch.setattr(forward, 'evaluate', fake)
    config = small_config(folder, dimension, tmp_path / 'run')
    solver = SimpleNamespace(dimension=dimension, spec={}, binary='fake')
    return SimpleNamespace(dimension=dimension, folder=folder, fake=fake, config=config, solver=solver, tmp=tmp_path)
