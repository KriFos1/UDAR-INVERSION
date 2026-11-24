"""Adjoint Jacobian against central differences at one logging point (needs the solver).

The model is the benchmark model averaged onto the solve grid, so the check
runs on a heterogeneous model. Each direction perturbs ln Rh or ln Rv in a
region; J v is compared with (g(m + h v) - g(m - h v)) / 2h.
"""

from pathlib import Path
import tempfile
import time

import numpy as np

from . import forward
from .data import Benchmark
from .grid import centres, overlap_weights
from .solver import Solver


def averaged_truth(data, edges, well_y=0.):
    """ln Rh (harmonic) and ln Rv (arithmetic) of the benchmark model averaged onto solve cells."""
    (tx, ty, tz), rh, rv = data.truth()
    wx, wz = overlap_weights(tx, edges[0]), overlap_weights(tz, edges[-1])
    if len(edges) == 2:
        plane = int(np.argmin(np.abs(ty - well_y)))
        return np.log(1. / (wx @ (1. / rh[:, plane]) @ wz.T)), np.log(wx @ rv[:, plane] @ wz.T)
    wy = overlap_weights(ty, edges[1])
    return (np.log(1. / np.einsum('ax,by,cz,xyz->abc', wx, wy, wz, 1. / rh, optimize=True)),
            np.log(np.einsum('ax,by,cz,xyz->abc', wx, wy, wz, rv, optimize=True)))


def directions(axes, bit):
    """Named perturbation fields and the block (0: ln Rh, 1: ln Rv) each applies to."""
    dimension = len(axes)
    grids = np.meshgrid(*axes, indexing='ij')
    relative = [grid - b for grid, b in zip(grids, (bit[0], bit[2]) if dimension == 2 else bit)]
    radius = np.sqrt(sum(d ** 2 for d in relative))

    def box(half, centre=None):
        centre = centre or [0.] * dimension
        return np.all([np.abs(d - c) <= half for d, c in zip(relative, centre)], axis=0).astype(float)
    ahead = [150.] + [0.] * (dimension - 1)
    above = [0.] * (dimension - 1) + [-60.]
    found = {'smooth whole window, ln Rh': (np.cos(relative[0] / 90.) * np.exp(-radius / 250.), 0),
             'smooth whole window, ln Rv': (np.cos(relative[-1] / 70.) * np.exp(-radius / 250.), 1),
             '20 ft box at the bit, ln Rh': (box(10.), 0),
             '20 ft box at the bit, ln Rv': (box(10.), 1),
             '40 ft box 60 ft above the bit, ln Rh': (box(20., above), 0),
             '80 ft box 150 ft ahead, ln Rh': (box(40., ahead), 0)}
    if dimension == 3:
        found['40 ft box 80 ft off the well plane, ln Rh'] = (box(20., [0., 80., 0.]), 0)
        found['40 ft box 80 ft off the well plane, ln Rv'] = (box(20., [0., 80., 0.]), 1)
    return {name: value for name, value in found.items() if value[0].any()}


def jacobian(config, index, tolerance=1e-6, step=.02, log=print):
    """Relative error and cosine of J v against central differences for each direction."""
    dimension = config.run.dimension
    data = Benchmark(config.data.path, config.data.outputs, dimension, config.data.noise_fraction,
                     config.data.noise_floor, config.data.floors)
    solver = Solver(**{key: value for key, value in vars(config.solver).items() if key != 'keep_scratch'})
    bit = data.bit(index)
    edges = config.grid.edges(bit, data.bit(0)[2], dimension)
    axes = [centres(edge) for edge in edges]
    log_rh, log_rv = averaged_truth(data, edges, bit[1])
    phase = forward.is_phase(dimension, len(data.tools))
    rows = []
    with tempfile.TemporaryDirectory(prefix='udarinv-check-') as scratch:
        work = Path(scratch) / 'solve'

        def solve(rh, rv, with_jacobian):
            return forward.evaluate(work, solver, rh, rv, axes, data.well(index), data.tools, jacobian=with_jacobian,
                                    tolerance=tolerance)
        clock = time.monotonic()
        _, matrix = solve(log_rh, log_rv, True)
        log(f'{dimension}D point {index}: grid {tuple(len(a) for a in axes)}, tolerance {tolerance:g}, '
            f'forward and Jacobian in {time.monotonic() - clock:.1f} s')
        cells = log_rh.size
        for name, (field, block) in directions(axes, bit).items():
            vector = np.zeros(2 * cells)
            vector[block * cells:(block + 1) * cells] = field.ravel()
            shift = [step * field * (block == 0), step * field * (block == 1)]
            difference = solve(log_rh + shift[0], log_rv + shift[1], False) - solve(log_rh - shift[0], log_rv - shift[1], False)
            difference[phase] = (difference[phase] + 180.) % 360. - 180.
            finite, adjoint = difference / (2. * step), matrix @ vector
            row = {'direction': name,
                   'relative_error': float(np.linalg.norm(finite - adjoint) / np.linalg.norm(finite)),
                   'cosine': float(finite @ adjoint / (np.linalg.norm(finite) * np.linalg.norm(adjoint)))}
            rows.append(row)
            log(f"{name:45s} relative error {row['relative_error']:.4f}   cosine {row['cosine']:.6f}")
    return rows
