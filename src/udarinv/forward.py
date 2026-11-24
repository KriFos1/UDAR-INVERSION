"""Forward response and adjoint Jacobian of one logging point on a rectilinear window.

A model is two log fields (ln Rh, ln Rv) on the solve grid: shape (nx, nz) with
centres [x, z] in 2D, (nx, ny, nz) with [x, y, z] in 3D, centres in feet.
``evaluate`` writes the solver inputs (one logging point, vertical TI axis),
runs the solver, reads the geosignals of each tool and, on request, pulls the
adjoint tapes back onto the window cells (``adjoint``).

Data vector: tool-major, components in ``components(dimension)`` order, phases
in degrees and attenuations in dB. Jacobian columns: [ln Rh | ln Rv] in state
order (x-major, z fastest).

2D: the solver takes the model in the tool's vertical plane and extends it
across; it is written with a few identical y layers centred on the well.
"""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil

import h5py
import numpy as np

from . import adjoint
from .grid import FT

IN_PLANE = ('USDP', 'USDA', 'UADP', 'UADA', 'UHRP', 'UHRA', 'UHAP', 'UHAA')
CROSS_PLANE = ('USDYP', 'USDYA', 'UADYP', 'UADYA', 'UHRYP', 'UHRYA', 'UHAYP', 'UHAYA')
WELL_KEYS = ('X', 'Y', 'Z', 'Inclination', 'Azimuth', 'Distance')
Y_LAYERS, Y_LAYER_FT = 5, 5.          # dummy y axis of a 2D model
# The adjoint's H tensor holds -zy and -yz where the forward has yz and zy. USDY, UHRY and
# UHAY are invariant under that; the UADY ratio is inverted, so its derivative changes sign.
ADJOINT_SIGN = {'UADYP': -1., 'UADYA': -1.}


def components(dimension):
    """Geosignal names per tool: 8 in-plane in 2D, plus the 8 cross-plane ones in 3D."""
    return IN_PLANE if dimension == 2 else IN_PLANE + CROSS_PLANE


def is_phase(dimension, count):
    """Boolean mask over a data vector of ``count`` tools: phases (degrees) alternate with attenuations."""
    return np.tile(np.arange(len(components(dimension))) % 2 == 0, count)


def output_name(tool, dimension):
    frequency, spacing = tool
    return f"{'UDAR3D' if dimension == 3 else 'UDAR'}_{int(spacing)}ft_{int(frequency)}kHz.las"


def write_inputs(directory, log_rh, log_rv, centres_ft, well, tools, tolerance, jacobian):
    """Solver inputs for one point. ``well`` holds X, Y, Z, Distance [m], Inclination, Azimuth [deg];
    ``tools`` are (frequency kHz, spacing ft) pairs."""
    directory = Path(directory)
    (directory / 'outputs').mkdir(parents=True, exist_ok=True)
    dimension = len(centres_ft)
    shape = tuple(len(c) for c in centres_ft)
    fields = []
    for values in (log_rh, log_rv):
        with np.errstate(over='ignore'):
            values = np.exp(np.asarray(values, dtype=float).reshape(shape))
        if not (np.isfinite(values).all() and (values > 0).all()):
            raise ValueError('Resistivity overflow or underflow')
        if dimension == 2:
            values = np.repeat(values[:, None, :], Y_LAYERS, axis=1)
        fields.append(np.ascontiguousarray(values.transpose(2, 1, 0)))          # stored as (z, y, x)
    if dimension == 2:
        y = well['Y'] / FT + Y_LAYER_FT * (np.arange(Y_LAYERS) - Y_LAYERS // 2)
        axes = (centres_ft[0], y, centres_ft[1])
    else:
        axes = tuple(centres_ft)
    with h5py.File(directory / 'globalmodel.h5', 'w') as handle:
        formation = handle.create_group('formation')
        formation['Rh'], formation['Rv'] = fields
        formation['Inclination'] = np.zeros_like(fields[0])                     # vertical TI axis everywhere
        formation['Azimuth'] = np.zeros_like(fields[0])
        for name, axis in zip('xyz', axes):
            formation[name] = np.asarray(axis, dtype=float) * FT
        path = handle.create_group('wellpath')
        for key in WELL_KEYS:
            path[key] = np.array([[float(well[key])]])
    names = [output_name(tool, dimension) for tool in tools]
    if len(set(names)) != len(names):
        raise ValueError('Tool output file names collide')
    (directory / 'tool.inp').write_text('freq,sub\n' + ''.join(f'{int(f)},{float(s):g}\n' for f, s in tools))
    (directory / 'selection.inp').write_text(f'1\n1\n0\n{tolerance:.16g}\n')
    (directory / 'outputsettings.inp').write_text(f'{5 if dimension == 3 else 3}\n1\n{1 if jacobian else 0}\n1e-10\n')
    (directory / 'orientation.inp').write_text('0\n')


def read_las_row(path, names):
    """The single data row of a geosignal file, in the order of ``names``."""
    header, rows = None, []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        if line.lstrip().startswith(('%', '#', '~')):
            found = line.lstrip('%#~ ').split()
            if all(name in found for name in names):
                header = found
            continue
        rows.append([float(token.replace('D', 'E')) for token in line.split()])
    if len(rows) != 1:
        raise ValueError(f'Expected one logging point in {path}, found {len(rows)}')
    values = np.asarray(rows[0], dtype=float)
    if header is None or len(header) != len(values):
        raise ValueError(f'No component header in {path}')
    values = values[[header.index(name) for name in names]]
    if not np.isfinite(values).all():
        raise ValueError(f'Non-finite response in {path}')
    return values


def read_predictions(directory, tools, dimension):
    names = components(dimension)
    return np.concatenate([read_las_row(Path(directory) / 'outputs' / output_name(tool, dimension), names)
                           for tool in tools])


def pullback(directory, spec, count, workers=None):
    """Adjoint tapes pulled back onto the model cells: {tool number: (2, components, nz, ny, nx)}."""
    directory = Path(directory)
    groups = {}
    for index in range(1, count + 1):
        with (directory / f'outputs/adjoint/kernel_tool{index}.bin').open('rb') as stream:
            np.fromfile(stream, dtype='<i8', count=5)
            groups.setdefault(int(np.fromfile(stream, dtype='<i8', count=1)[0]), []).append(index)
    # Tools of one receiver spacing share a material grid: prepare it once, then one thread per tool.
    tasks = [(receiver, index) for receiver, indices in groups.items() for index in indices]
    threads = max(1, min(len(tasks), workers or len(os.sched_getaffinity(0))))
    with ThreadPoolExecutor(max_workers=threads) as pool:
        prepared = dict(zip(groups, pool.map(lambda receiver: adjoint._prepare(directory, spec, receiver), groups)))

        def tool_jacobian(task):
            receiver, index = task
            kernel = adjoint.read_array(directory / f'outputs/adjoint/kernel_tool{index}.bin', True)[0]
            return index, adjoint._pullback_prepared(prepared[receiver], kernel[..., None])[:, :, 0]
        return dict(pool.map(tool_jacobian, tasks))


def read_jacobian(directory, spec, tools, shape):
    """(components * tools, 2 * cells): rows as the data vector, columns [ln Rh | ln Rv]."""
    dimension = len(shape)
    names = components(dimension)
    sign = np.array([ADJOINT_SIGN.get(name, 1.) for name in names])
    pulled = pullback(directory, spec, len(tools))
    cells = int(np.prod(shape))
    blocks = []
    for index in range(1, len(tools) + 1):
        jac = pulled.pop(index)                                                 # (2, comp, z, y, x)
        if dimension == 2:
            if jac.shape != (2, len(names), shape[1], Y_LAYERS, shape[0]):
                raise ValueError(f'Unexpected 2D adjoint Jacobian shape {jac.shape}')
            jac = jac.sum(axis=3).transpose(1, 0, 3, 2)                         # the solver samples one y layer
        else:
            if jac.shape != (2, len(names), shape[2], shape[1], shape[0]):
                raise ValueError(f'Unexpected 3D adjoint Jacobian shape {jac.shape}')
            jac = jac.transpose(1, 0, 4, 3, 2)
        blocks.append(jac.reshape(len(names), 2 * cells) * sign[:, None])
    jacobian = np.vstack(blocks)
    if not np.isfinite(jacobian).all():
        raise ValueError('Non-finite Jacobian')
    return jacobian


def evaluate(directory, solver, log_rh, log_rv, centres_ft, well, tools, jacobian=True, tolerance=1e-6, keep=False):
    """Prediction and, if requested, (prediction, Jacobian) for one model.

    The work directory is created, used and removed (unless ``keep``); on an
    error it is left for the caller, which may want the solver log.
    """
    directory = Path(directory)
    if len(centres_ft) != solver.dimension:
        raise ValueError(f'A {len(centres_ft)}D model was given to the {solver.dimension}D solver')
    if directory.exists():
        shutil.rmtree(directory)
    write_inputs(directory, log_rh, log_rv, centres_ft, well, tools, tolerance, jacobian)
    solver.run(directory)
    result = read_predictions(directory, tools, solver.dimension)
    if jacobian:
        result = (result, read_jacobian(directory, solver.spec, tools, tuple(len(c) for c in centres_ft)))
    if not keep:
        shutil.rmtree(directory, ignore_errors=True)
    return result
