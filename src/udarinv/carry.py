"""What passes from one logging point to the next, all in closed form.

The carried state is the cellwise mean and std of both blocks on the carry
lattice (uniform cells in the well plane). ``carry_moments`` moves it to the
next point; ``lifted_moments`` turns it into the prior mean and std of the
graded solve grid; ``lattice_weights`` averages a solve back onto the lattice.
The previous MAP on the solve grid supplies the start of the next search
(``carried_state``) and, in 3D, its off-plane structure (``carried_anomaly``).
"""

import numpy as np

from .grid import centres, overlap_weights


def entered_cells(nx, nz, movement):
    """Lattice cells that are new after the lattice moved by ``movement`` cells (x, z)."""
    entered = np.zeros((nx, nz), dtype=bool)
    move_x, move_z = (int(step) for step in movement)
    if move_x > 0:
        entered[max(nx - move_x, 0):] = True
    elif move_x < 0:
        entered[:min(-move_x, nx)] = True
    if move_z > 0:
        entered[:, max(nz - move_z, 0):] = True
    elif move_z < 0:
        entered[:, :min(-move_z, nz)] = True
    return entered


def reindex(values, nx, nz, movement):
    """Shift both blocks of a lattice vector with the lattice; entered cells take the edge value."""
    x = np.clip(np.arange(nx) + int(movement[0]), 0, nx - 1)
    z = np.clip(np.arange(nz) + int(movement[1]), 0, nz - 1)
    return np.asarray(values, dtype=float).reshape(2, nx, nz)[:, x][:, :, z].reshape(-1)


def carry_moments(background, mean, std, movement, rho):
    """Carried mean and std on the moved lattice, and the number of entered cells.

    Kept cells are reindexed and relaxed towards the background with
    ``rho`` = exp(-dMD / memory): m <- mu + rho (m - mu), s^2 <- rho^2 s^2 +
    (1 - rho^2) sigma^2, what an Ornstein-Uhlenbeck step does to the marginals.
    Entered cells get the moments of the conditional Matern fill
    (``LatticeMatern.entered_moments``).
    """
    if not 0. <= rho <= 1.:
        raise ValueError('Require 0 <= rho <= 1')
    nx, nz = background.shape
    mean, std = np.asarray(mean, dtype=float), np.asarray(std, dtype=float)
    if mean.shape != background.mean.shape or std.shape != mean.shape or np.any(std <= 0):
        raise ValueError('Mean and positive std must cover both blocks of the lattice')
    entered = entered_cells(nx, nz, movement)
    relaxed = background.mean + rho * (reindex(mean, nx, nz, movement) - background.mean)
    filled, variance = background.entered_moments(relaxed, reindex(std, nx, nz, movement), entered.ravel())
    variance = rho ** 2 * variance + (1. - rho ** 2) * background.std ** 2
    return filled, np.sqrt(variance), int(entered.sum())


def _split(edges):
    """(x, y or None, z) from the 2 or 3 edge arrays of a solve grid."""
    return (edges[0], None, edges[1]) if len(edges) == 2 else tuple(edges)


def lifted_moments(lattice_edges, mean, std, background_std, edges, relax_ft, well_y=0.):
    """Prior mean and std on the solve grid, shape (2, *grid), from the lattice moments.

    The lattice moments are averaged onto the solve cells (cells beyond the
    lattice take its edge values) and, in 3D, extended across the well. The std
    returns to the background std with distance d from the lattice extent and
    from the well plane: s^2 = s_c^2 + (s_bg^2 - s_c^2)(1 - exp(-d / relax_ft)).
    """
    lx, lz = lattice_edges
    ex, ey, ez = _split(edges)
    x, z = centres(ex), centres(ez)
    nx, nz = len(lx) - 1, len(lz) - 1
    wx, wz = overlap_weights(centres(lx), ex), overlap_weights(centres(lz), ez)
    outside_x = np.maximum(0., np.maximum(lx[0] - x, x - lx[-1]))
    outside_z = np.maximum(0., np.maximum(lz[0] - z, z - lz[-1]))
    if ey is None:
        distance = np.sqrt(outside_x[:, None] ** 2 + outside_z[None, :] ** 2)
    else:
        distance = np.sqrt(outside_x[:, None, None] ** 2 + (centres(ey)[None, :, None] - well_y) ** 2
                           + outside_z[None, None, :] ** 2)
    relax = 1. - np.exp(-distance / relax_ft)
    mean, std = np.asarray(mean).reshape(2, nx, nz), np.asarray(std).reshape(2, nx, nz)
    lifted_mean, lifted_std = [], []
    for block in (0, 1):
        m, s = wx @ mean[block] @ wz.T, wx @ std[block] @ wz.T
        if ey is not None:
            m, s = m[:, None, :], s[:, None, :]
        variance = s ** 2 + np.maximum(float(background_std[block]) ** 2 - s ** 2, 0.) * relax
        lifted_mean.append(np.broadcast_to(m, distance.shape))
        lifted_std.append(np.sqrt(variance))
    return np.stack(lifted_mean), np.stack(lifted_std)


def lattice_weights(edges, lattice_edges, well_y=0., cell=5.):
    """Weights (wx, wy, wz) that average a solve-grid field onto the lattice cells.

    In 3D the lattice is the layer of thickness ``cell`` centred on the well
    plane; in 2D ``wy`` is None.
    """
    ex, ey, ez = _split(edges)
    lx, lz = lattice_edges
    wy = None if ey is None else overlap_weights(centres(ey), np.array([well_y - .5 * cell, well_y + .5 * cell]))[0]
    return overlap_weights(centres(ex), lx), wy, overlap_weights(centres(ez), lz)


def _lookup(previous_centres, edges):
    """Nearest previous cell of every new cell per axis, and whether it lies inside the previous window."""
    found = []
    for new, old in zip((centres(e) for e in edges), previous_centres):
        if len(old) == 1:
            found.append((np.zeros(len(new), dtype=int), np.ones(len(new), dtype=bool)))
            continue
        index = np.clip(np.searchsorted(old, new), 1, len(old) - 1)
        index = np.where(new - old[index - 1] <= old[index] - new, index - 1, index)
        half = .5 * np.diff(old).max()
        found.append((index, (new >= old[0] - half) & (new <= old[-1] + half)))
    indices = np.ix_(*(index for index, _ in found))
    inside = np.ones(tuple(len(index) for index, _ in found), dtype=bool)
    for axis, (_, mask) in enumerate(found):
        inside &= mask.reshape([-1 if a == axis else 1 for a in range(len(found))])
    return indices, inside


def carried_state(previous_map, previous_centres, edges, fallback):
    """The previous MAP (2, *grid) looked up on the new window; ``fallback`` (state vector) outside."""
    indices, inside = _lookup(previous_centres, edges)
    values = np.stack([np.asarray(block, dtype=float)[indices] for block in previous_map])
    return np.where(inside[None], values, np.asarray(fallback).reshape(values.shape)).reshape(-1)


def carried_anomaly(previous_map, previous_centres, edges, rho, previous_well_y=0.):
    """Off-plane part of the previous 3D MAP on the new window: the MAP minus its well-plane layer.

    Zero for new cells outside the previous window; scaled by ``rho`` (memory
    along the well). The well plane itself is carried by the lattice moments.
    """
    if len(edges) != 3 or rho <= 0:
        return 0.
    fields = np.asarray(previous_map, dtype=float)
    plane = int(np.argmin(np.abs(np.asarray(previous_centres[1]) - previous_well_y)))
    anomaly = fields - fields[:, :, plane:plane + 1, :]
    indices, inside = _lookup(previous_centres, edges)
    values = np.stack([block[indices] for block in anomaly])
    return rho * np.where(inside[None], values, 0.)
