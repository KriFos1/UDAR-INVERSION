"""Observations of a UDAR benchmark folder (https://gitlab.norceresearch.no/krfo/udar-benchmarks).

A benchmark folder holds ``globalmodel.h5`` (model and well path), ``tool.inp``
and one ``outputs_*`` folder per simulator with a ``UDAR3D_{spacing}ft_{freq}kHz.las``
table per tool: one row per logging point, measured depth and 16 geosignals.

Error model: the standard deviation of a component is ``noise_fraction`` of
its mean absolute value over the tools at that logging point, raised to the
``floors`` given per component. Channels whose standard deviation is still below
``noise_floor`` carry no information and are left out of the inversion.
"""

from pathlib import Path

import h5py
import numpy as np

from .forward import WELL_KEYS, components
from .grid import FT


def read_tools(path):
    """(frequency kHz, spacing ft) pairs of a ``tool.inp``."""
    tools = []
    for line in Path(path).read_text().splitlines()[1:]:           # values separated by commas or blanks
        values = line.replace(',', ' ').split()
        if values:
            if len(values) != 2:
                raise ValueError(f'{path}: expected a frequency and a spacing per line')
            tools.append((float(values[0]), float(values[1])))
    if not tools:
        raise ValueError(f'{path}: no tools')
    return tools


def read_las(path, names):
    """Measured depth [ft] and the named components, one row per logging point."""
    lines = Path(path).read_text().splitlines()
    header = lines[0].lstrip('%#~ ').split()
    missing = [name for name in names if name not in header]
    if missing:
        raise ValueError(f'{path}: missing components {missing}')
    values = np.array([[float(token.replace('D', 'E')) for token in line.split()]
                       for line in lines[1:] if line.strip()], dtype=float)
    if values.shape[1] != len(header):
        raise ValueError(f'{path}: rows do not match the header')
    return values[:, 0], values[:, [header.index(name) for name in names]]


class Benchmark:
    """Data, error model and well path of one benchmark for a 2D or a 3D inversion."""

    def __init__(self, path, outputs, dimension, noise_fraction=.025, noise_floor=1e-3, floors=None):
        self.path = Path(path).expanduser().resolve()
        self.dimension = int(dimension)
        self.names = components(self.dimension)
        self.tools = read_tools(self.path / 'tool.inp')
        with h5py.File(self.path / 'globalmodel.h5', 'r') as handle:
            self.wellpath = {key: handle[f'wellpath/{key}'][:, 0].astype(float) for key in WELL_KEYS}
        tables = []
        for frequency, spacing in self.tools:
            md, values = read_las(self.path / outputs / f'UDAR3D_{int(spacing)}ft_{int(frequency)}kHz.las', self.names)
            if md.shape != self.wellpath['Distance'].shape or not np.allclose(md, self.wellpath['Distance'] / FT,
                                                                              rtol=0, atol=.02):
                raise ValueError(f'The logging points of {frequency:g} kHz / {spacing:g} ft do not match the well path')
            tables.append(values)
        self.values = np.stack(tables, axis=1)                                  # (points, tools, components)
        if not np.isfinite(self.values).all():
            raise ValueError('Non-finite observations')
        if not (np.isfinite(noise_fraction) and noise_fraction > 0 and noise_floor >= 0):
            raise ValueError('Require a positive noise fraction and a non-negative noise floor')
        std = np.maximum(noise_fraction * np.mean(np.abs(self.values), axis=1), 1e-12)   # (points, components)
        for name, floor in (floors or {}).items():
            if name not in self.names:
                raise ValueError(f'Unknown component {name!r} in the floors')
            std[:, self.names.index(name)] = np.maximum(std[:, self.names.index(name)], float(floor))
        self.std = np.repeat(std[:, None, :], len(self.tools), axis=1)
        self.noise = {'fraction': float(noise_fraction), 'floor': float(noise_floor), 'floors': dict(floors or {})}

    def __len__(self):
        return len(self.values)

    def well(self, index):
        """Well path of one logging point in solver units (m, degrees)."""
        return {key: float(values[index]) for key, values in self.wellpath.items()}

    def bit(self, index):
        """Tool position (x, y, z) [ft]."""
        return tuple(float(self.wellpath[key][index]) / FT for key in 'XYZ')

    def md(self, index):
        return float(self.wellpath['Distance'][index]) / FT

    def point(self, index):
        """Observations and variances of the channels used, and the mask over the full data vector."""
        std = self.std[index].ravel()
        mask = std >= self.noise['floor']
        if not mask.any():
            raise ValueError(f'Logging point {index}: every channel is below the noise floor')
        return self.values[index].ravel()[mask], std[mask] ** 2, mask

    def truth(self):
        """True model: cell centres x, y, z [ft] and Rh, Rv [ohm m] as (x, y, z) arrays."""
        with h5py.File(self.path / 'globalmodel.h5', 'r') as handle:
            axes = [handle[f'formation/{name}'][:].ravel() / FT for name in 'xyz']
            rh, rv = (handle[f'formation/{name}'][:].transpose(2, 1, 0) for name in ('Rh', 'Rv'))
        return axes, rh, rv
