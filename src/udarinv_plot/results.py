"""Reading a chain's output folder for plotting."""

import json
from pathlib import Path

import numpy as np

from udarinv import forward, scoring


class Results:
    """Summary rows, per-point files and (for a benchmark) the true model of one chain."""

    def __init__(self, run, truth=True):
        self.run = Path(run)
        self.config = json.loads((self.run / 'config.json').read_text())
        self.rows = json.loads((self.run / 'summary.json').read_text())
        if not self.rows:
            raise ValueError(f'{self.run} has no finished point yet')
        self.indices = [row['index'] for row in self.rows]
        self.dimension = self.config['run']['dimension']
        self.truth = None
        if truth and (Path(self.config['data']['path']) / 'globalmodel.h5').is_file():
            self.truth = scoring.Truth(self.config['data']['path'])

    def folder(self, index):
        return self.run / f'point-{index:04d}'

    def handoff(self, index):
        with np.load(self.folder(index) / 'handoff.npz') as saved:
            return {key: saved[key] for key in saved.files}

    def map(self, index):
        with np.load(self.folder(index) / 'map.npz') as saved:
            return {key: saved[key] for key in saved.files}

    def component_names(self):
        return forward.components(self.dimension)

    def tools(self):
        """(frequency kHz, spacing ft) of each tool, from the data folder's tool.inp."""
        from udarinv.data import read_tools
        return read_tools(Path(self.config['data']['path']) / 'tool.inp')


def painted(results, upto=None):
    """At-bit curtain: every lattice column shows the estimate of the logging point whose bit was nearest.

    Returns a dict with the canvas edges and (x, z) arrays ``mean`` / ``std`` of ln Rh and of the latent,
    the true ln Rh / ln Rv on the canvas (or None), the well path and the bit positions. Cells no point
    covered are NaN.
    """
    indices = results.indices if upto is None else [i for i in results.indices if i <= upto]
    handoffs = [results.handoff(index) for index in indices]
    cell = float(results.config['lattice']['cell'])
    x0 = min(h['edges_x'][0] for h in handoffs)
    x1 = max(h['edges_x'][-1] for h in handoffs)
    z0 = min(h['edges_z'][0] for h in handoffs)
    z1 = max(h['edges_z'][-1] for h in handoffs)
    nx, nz = int(round((x1 - x0) / cell)), int(round((z1 - z0) / cell))
    fields = {name: np.full((nx, nz), np.nan) for name in ('rh', 'rh_std', 'latent', 'latent_std')}
    distance = np.full((nx, nz), np.inf)
    centre_x = x0 + cell * (np.arange(nx) + .5)
    for h in handoffs:
        lx, lz = len(h['edges_x']) - 1, len(h['edges_z']) - 1
        ix = int(round((h['edges_x'][0] - x0) / cell))
        iz = int(round((h['edges_z'][0] - z0) / cell))
        mean, std = h['mean'].reshape(2, lx, lz), h['std'].reshape(2, lx, lz)
        away = np.abs(centre_x[ix:ix + lx] - h['bit_ft'][0])[:, None] * np.ones((1, lz))
        closer = away < distance[ix:ix + lx, iz:iz + lz]
        for name, values in (('rh', mean[0]), ('rh_std', std[0]), ('latent', mean[1]), ('latent_std', std[1])):
            fields[name][ix:ix + lx, iz:iz + lz][closer] = values[closer]
        distance[ix:ix + lx, iz:iz + lz][closer] = away[closer]
    edges_x, edges_z = x0 + cell * np.arange(nx + 1), z0 + cell * np.arange(nz + 1)
    bits = np.array([h['bit_ft'] for h in handoffs])
    truth = None
    if results.truth is not None:
        truth = results.truth.on_cells(edges_x, edges_z, float(np.median(bits[:, 1])))
    return {'edges_x': edges_x, 'edges_z': edges_z, **fields, 'truth': truth, 'bits': bits, 'indices': indices}
