"""The two grids of the inversion.

Solve grid: a graded rectilinear window around the bit, (x, z) in 2D and
(x, y, z) in 3D. x runs along the well, y across it, z is TVD (positive down).
The z edges sit on a fixed lattice, so bed boundaries do not move inside the
cells from point to point.

Carry lattice: uniform cells in the well plane that shift by whole cells with
the bit. Only the cellwise moments carried between logging points live on it.

State order everywhere: the two blocks (ln Rh, latent), then x-major with z
fastest: cell index ``(ix * ny + iy) * nz + iz`` in 3D, ``ix * nz + iz`` in 2D.
"""

from dataclasses import dataclass

import numpy as np

FT = .3048        # metres per foot


def parse_grade(text):
    """'INNER:CELL:GROWTH:MAX', or tiered 'CELLxHALF+CELLxHALF+...+GROWTH:MAX'.

    Uniform cells of each tier out to its half-width, then cells that grow
    geometrically by GROWTH up to MAX. 'CELL' alone gives a uniform axis.
    """
    text = str(text)
    if 'x' not in text:
        values = tuple(float(v) for v in text.split(':'))
        if len(values) == 1:
            return ((values[0], np.inf),), 1., values[0]
        if len(values) != 4:
            raise ValueError(f'Invalid grade {text!r}')
        inner, cell, growth, largest = values
        return ((cell, inner),), growth, largest
    *tiers, tail = text.split('+')
    growth, largest = (float(v) for v in tail.split(':'))
    return tuple(tuple(float(v) for v in tier.split('x')) for tier in tiers), growth, largest


def graded_edges(half, grade, offset):
    """Symmetric cell edges about 0 out to ``half``.

    ``offset`` .5 centres a cell on 0 (x: the bit, y: the well plane); 0 puts an
    edge on 0 (z: the lattice). Tier cells that are multiples of the first keep
    the edges on the first tier's lattice.
    """
    tiers, growth, largest = parse_grade(grade) if isinstance(grade, str) else grade
    if half <= 0 or growth < 1 or min(cell for cell, _ in tiers) <= 0 or largest <= 0:
        raise ValueError('Require positive extent and cells, and growth >= 1')
    first = tiers[0][0]
    positive = [offset * first]
    for cell, extent in tiers:
        while positive[-1] + cell <= extent + offset * first + 1e-9 and positive[-1] + cell <= half + 1e-9:
            positive.append(positive[-1] + cell)
    fine = len(positive)
    step = cell
    while positive[-1] < half - 1e-9:
        step = min(step * growth, largest)
        positive.append(min(positive[-1] + step, half))
    if len(positive) > fine + 1 and half - positive[-2] < .5 * step:
        positive.pop(-2)                      # merge a sliver last cell into its neighbour
    positive = np.array(positive)
    return np.unique(np.r_[-positive[::-1], positive])


def centres(edges):
    edges = np.asarray(edges, dtype=float)
    return .5 * (edges[1:] + edges[:-1])


def overlap_weights(source_centres, target_edges):
    """Averaging weights, rows: target cells, columns: source cells (overlap length, rows sum to 1).

    Source cell bounds are the midpoints between neighbouring centres, which is
    how the solver sees a graded axis. A target cell outside the source axis
    takes the nearest source cell.
    """
    source = np.asarray(source_centres, dtype=float)
    target = np.asarray(target_edges, dtype=float)
    if len(source) == 1:
        return np.ones((len(target) - 1, 1))
    middle = .5 * (source[1:] + source[:-1])
    lo = np.r_[source[0] - (middle[0] - source[0]), middle]
    hi = np.r_[middle, source[-1] + (source[-1] - middle[-1])]
    weights = np.maximum(0., np.minimum(target[1:, None], hi) - np.maximum(target[:-1, None], lo))
    for index in np.flatnonzero(weights.sum(axis=1) == 0):
        weights[index, np.argmin(np.abs(source - .5 * (target[index] + target[index + 1])))] = 1.
    return weights / weights.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class SolveGrid:
    """Graded window of one solve. The defaults are the window of the Benchmark-2 3D runs."""
    x_half: float = 320.
    x_grade: str = '2x40+1.5:40'
    y_half: float = 300.
    y_grade: str = '2x20+4x130+1.5:40'
    z_half: float = 200.
    z_grade: str = '100:2:1.3:20'
    z_lattice: float = 2.

    def edges(self, bit, z_origin, dimension):
        """Cell edges [ft] around the bit (x, y, z in ft): [x, z] in 2D, [x, y, z] in 3D.

        ``z_origin`` fixes the TVD lattice (the first logging point's TVD): the
        z edges are those of the lattice line nearest the bit.
        """
        if dimension not in (2, 3):
            raise ValueError('The dimension is 2 or 3')
        bit_x, bit_y, bit_z = (float(v) for v in bit)
        x = bit_x + graded_edges(self.x_half, self.x_grade, .5)
        anchor = z_origin + self.z_lattice * np.floor((bit_z - z_origin) / self.z_lattice + .5)
        z = anchor + graded_edges(self.z_half, self.z_grade, 0.)
        if dimension == 2:
            return [x, z]
        return [x, bit_y + graded_edges(self.y_half, self.y_grade, .5), z]


class CarryLattice:
    """Uniform (x, z) cells in the well plane, moving with the bit by whole cells."""

    def __init__(self, nx=33, nz=80, cell=5., anchor=None, offsets=(0, 0)):
        if nx < 1 or nz < 1 or cell <= 0:
            raise ValueError('Require a positive lattice size and cell')
        self.nx, self.nz, self.cell = int(nx), int(nz), float(cell)
        self.anchor = None if anchor is None else (float(anchor[0]), float(anchor[1]))
        self.offsets = (int(offsets[0]), int(offsets[1]))

    def update(self, bit_x, bit_z):
        """Move to the bit. Returns the lower corner (x, z) [ft] and the movement in cells."""
        if self.anchor is None:
            self.anchor = (float(bit_x), float(bit_z))
        offsets = tuple(int(np.floor((position - anchor) / self.cell + .5))
                        for position, anchor in zip((bit_x, bit_z), self.anchor))
        movement = (offsets[0] - self.offsets[0], offsets[1] - self.offsets[1])
        self.offsets = offsets
        return self.corner(), movement

    def corner(self):
        return (self.anchor[0] + (self.offsets[0] - self.nx // 2 - .5) * self.cell,
                self.anchor[1] + (self.offsets[1] - self.nz // 2 - .5) * self.cell)

    def edges(self):
        x0, z0 = self.corner()
        return x0 + self.cell * np.arange(self.nx + 1), z0 + self.cell * np.arange(self.nz + 1)

    def state(self):
        return {'anchor': list(self.anchor), 'offsets': list(self.offsets)}
