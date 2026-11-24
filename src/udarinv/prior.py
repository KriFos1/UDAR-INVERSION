"""Gaussian Matern priors of the two state blocks (ln Rh, latent).

``SeparableMatern`` is the prior of a solve on the graded window (2 or 3 axes).
``LatticeMatern`` is the background field on the uniform carry lattice; its
sparse precision gives the moments of cells that enter the lattice.

Both expose ``mean``, ``std``, ``color`` (L z or L^T v for C = L L^T) and
``whiten`` (L^-1 v): the prior term of the objective is 0.5 |L^-1 (m - mean)|^2.
"""

import math

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cholesky, solve_triangular
from scipy.sparse.linalg import splu


def axis_correlation(centres, length, nu=1.5):
    """Unit-variance 1D Matern correlation (SPDE, alpha = 2) on cells with these centres."""
    centres = np.asarray(centres, dtype=float)
    if len(centres) == 1:
        return np.ones((1, 1))
    middle = .5 * (centres[1:] + centres[:-1])
    edges = np.r_[centres[0] - (middle[0] - centres[0]), middle, centres[-1] + (centres[-1] - middle[-1])]
    volume = np.diff(edges)
    kappa = math.sqrt(8. * nu) / float(length)
    transmissibility = 1. / np.diff(centres)
    stiffness = np.diag(np.r_[transmissibility, 0.] + np.r_[0., transmissibility]) \
        - np.diag(transmissibility, 1) - np.diag(transmissibility, -1)
    operator = kappa ** 2 * np.diag(volume) + stiffness
    covariance = np.linalg.inv(operator @ np.diag(1. / volume) @ operator)
    scale = 1. / np.sqrt(np.diag(covariance))
    correlation = scale[:, None] * covariance * scale[None, :]
    return .5 * (correlation + correlation.T)


class SeparableMatern:
    """N(mean, S R S) with R the Kronecker product of one Matern correlation per axis.

    ``centres`` are the cell centres of the 2 or 3 axes, ``ranges`` the practical
    correlation ranges per block and axis. Every operation uses the small dense
    Cholesky factors of the axes: C = S (L1 (x) L2 (x) ...)(...)^T S per block.
    """

    def __init__(self, centres, ranges, mean, std, nu=1.5):
        self.centres = [np.asarray(c, dtype=float) for c in centres]
        self.grid = tuple(len(c) for c in self.centres)
        self.cells = int(np.prod(self.grid))
        self.ranges = [[float(r) for r in block] for block in ranges]
        if len(self.ranges) != 2 or any(len(block) != len(self.grid) for block in self.ranges):
            raise ValueError('Need one range per axis for each of the two blocks')
        self.mean = np.asarray(mean, dtype=float).ravel().copy()
        self.std = np.asarray(std, dtype=float).ravel().copy()
        if self.mean.shape != (2 * self.cells,) or self.std.shape != self.mean.shape or np.any(self.std <= 0):
            raise ValueError('Mean and positive std must cover both blocks of the window')
        self.factors = [[cholesky(axis_correlation(c, r, nu), lower=True) for c, r in zip(self.centres, block)]
                        for block in self.ranges]

    def _apply(self, values, transpose=False, inverse=False):
        values = np.asarray(values, dtype=float)
        fields = values.reshape(2, *self.grid, -1).copy()
        for block, factors in enumerate(self.factors):
            field = fields[block]
            for axis, factor in enumerate(factors):
                moved = np.moveaxis(field, axis, 0)
                flat = moved.reshape(moved.shape[0], -1)
                if inverse:
                    flat = solve_triangular(factor, flat, lower=True, check_finite=False)
                else:
                    flat = (factor.T if transpose else factor) @ flat
                field = np.moveaxis(flat.reshape(moved.shape), 0, axis)
            fields[block] = field
        result = fields.reshape(2 * self.cells, -1)
        return result[:, 0] if values.ndim == 1 else result

    def color(self, values, transpose=False):
        values = np.asarray(values, dtype=float)
        if transpose:
            return self._apply(values * (self.std if values.ndim == 1 else self.std[:, None]), transpose=True)
        result = self._apply(values)
        return result * (self.std if result.ndim == 1 else self.std[:, None])

    def whiten(self, values):
        values = np.asarray(values, dtype=float)
        return self._apply(values / (self.std if values.ndim == 1 else self.std[:, None]), inverse=True)

    def axis_correlations(self, block):
        return [factor @ factor.T for factor in self.factors[block]]


class _LatticeBlock:
    """One block of the lattice field: SPDE (kappa^2 - div H grad) x = W, nu = 1, Robin boundary."""

    ROBIN_SCALE = 1.42

    def __init__(self, shape, cell, ranges, mean, std):
        nx, nz = shape
        range_x, range_z = (float(r) for r in ranges)
        rho = math.sqrt(range_x * range_z)
        kappa = math.sqrt(8.) / rho
        lam_x, lam_z = (range_x / rho) ** 2, (range_z / rho) ** 2
        self.volume = cell * cell
        index = np.arange(nx * nz).reshape(nx, nz)
        diagonal = np.full(nx * nz, self.volume * kappa ** 2)
        rows, cols, values = [], [], []
        # Square cells: every face has length ``cell`` and centre distance ``cell``.
        for left, right, transmissibility in ((index[:-1].ravel(), index[1:].ravel(), lam_x),
                                              (index[:, :-1].ravel(), index[:, 1:].ravel(), lam_z)):
            np.add.at(diagonal, left, transmissibility)
            np.add.at(diagonal, right, transmissibility)
            rows += [left, right]
            cols += [right, left]
            values += [np.full(left.size, -transmissibility)] * 2
        for cells in (index[0], index[-1], index[:, 0], index[:, -1]):
            np.add.at(diagonal, cells.ravel(), cell * kappa / self.ROBIN_SCALE)
        self.n = nx * nz
        self.K = (sp.coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                                shape=(self.n, self.n)) + sp.diags(diagonal)).tocsc()
        self.lu = splu(self.K)
        inverse = self.lu.solve(np.eye(self.n))
        self.mean = np.full(self.n, float(mean))
        self.std = np.full(self.n, float(std))
        # The base field K^-1 C^1/2 xi has covariance V K^-2; S sets the marginal std cell by cell.
        self.scale = self.std / np.sqrt(self.volume * np.sum(inverse ** 2, axis=1))

    def covariance_columns(self, cells):
        """Columns ``cells`` of the block covariance S K^-1 V K^-1 S."""
        unit = np.zeros((self.n, len(cells)))
        unit[cells, np.arange(len(cells))] = self.scale[cells]
        return self.scale[:, None] * self.lu.solve(self.volume * self.lu.solve(unit))

    def precision(self):
        inverse_scale = sp.diags(1. / self.scale)
        return (inverse_scale @ self.K @ self.K @ inverse_scale / self.volume).tocsr()

    def entered_moments(self, mean, std, entered):
        """Mean and variance of the entered cells (A) given the kept cells (B).

        Conditional redraw x_A = mu_A + W (x_B - mu_B) + xi, W = -Q_AA^-1 Q_AB,
        xi ~ N(0, Q_AA^-1): E[x_A] = mu_A + W (m_B - mu_B) and
        Var[x_A] = diag(W Sigma_B W^T) + diag(Q_AA^-1). Only kept cells in the
        stencil of A have nonzero weights. Sigma_B is their carried ``std`` with
        the prior correlation: exact where they are at their prior spread, an
        approximation of the posterior correlation otherwise.
        """
        kept = ~entered
        precision = self.precision()
        block = splu(precision[entered][:, entered].tocsc())
        coupling = precision[entered][:, kept].tocsc()
        cells = np.flatnonzero(kept)[np.flatnonzero(np.diff(coupling.indptr))]
        weights = -block.solve(precision[entered][:, cells].toarray())
        correlation = self.covariance_columns(cells)[cells] / np.outer(self.std[cells], self.std[cells])
        spread = np.asarray(std, dtype=float)[cells]
        covariance = spread[:, None] * correlation * spread[None, :]
        conditional = np.diag(block.solve(np.eye(int(entered.sum()))))
        return (self.mean[entered] + weights @ (np.asarray(mean, dtype=float)[cells] - self.mean[cells]),
                np.einsum('ij,jk,ik->i', weights, covariance, weights) + conditional)


class LatticeMatern:
    """Background (climatological) Matern field of both blocks on the carry lattice."""

    def __init__(self, shape, cell, ranges, means, stds):
        self.shape = (int(shape[0]), int(shape[1]))
        self.cells = self.shape[0] * self.shape[1]
        self.blocks = [_LatticeBlock(self.shape, float(cell), block_ranges, mean, std)
                       for block_ranges, mean, std in zip(ranges, means, stds)]
        if len(self.blocks) != 2:
            raise ValueError('Provide ranges, means and stds for both blocks')
        self.mean = np.concatenate([block.mean for block in self.blocks])
        self.std = np.concatenate([block.std for block in self.blocks])

    def entered_moments(self, mean, std, entered):
        """Mean and variance of the whole lattice after the entered cells are filled from the kept ones."""
        mean = np.array(mean, dtype=float, copy=True)
        variance = np.asarray(std, dtype=float) ** 2
        entered = np.asarray(entered, dtype=bool).ravel()
        if entered.shape != (self.cells,) or mean.shape != self.mean.shape or variance.shape != mean.shape:
            raise ValueError('Mean, std and the entered-cell mask must cover the lattice')
        if not entered.any():
            return mean, variance
        if entered.all():
            return self.mean.copy(), self.std ** 2
        for index, block in enumerate(self.blocks):
            rows = slice(index * self.cells, (index + 1) * self.cells)
            filled_mean, filled_variance = block.entered_moments(mean[rows], np.sqrt(variance[rows]), entered)
            mean[rows][entered] = filled_mean
            variance[rows][entered] = filled_variance
        return mean, variance
