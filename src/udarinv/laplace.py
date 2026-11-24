"""Closed-form moments of the Laplace posterior. Nothing is sampled.

For a Gaussian prior N(mean, C), C = L L^T, data variances Cd and the Jacobian
G at the MAP, the Gauss-Newton Laplace covariance is

    C_post = (G^T Cd^-1 G + C^-1)^-1 = C - B B^T,
    J = Cd^-1/2 G L,   B = C G^T Cd^-1/2 K^-T,   K K^T = I + J J^T,

a correction of rank at most the number of data. Cellwise variances, the
variance of any average of cells and the cellwise covariance between the two
blocks follow exactly from the Jacobian the optimizer already holds.
"""

import numpy as np
from scipy.linalg import cholesky, solve_triangular

VARIANCE_FLOOR = 1e-12        # relative to the prior variance: guards the subtraction against rounding


def posterior_factor(prior, jacobian, variance):
    """B with C_post = C - B B^T (state rows, data columns), and the resolved dimensions.

    The second value, trace(J^T J (I + J^T J)^-1), counts the directions the data
    determined: 0 without data, the number of data when all are far above noise.
    """
    variance = np.asarray(variance, dtype=float)
    whitened = prior.color(np.asarray(jacobian, dtype=float).T, transpose=True).T / np.sqrt(variance)[:, None]
    count = len(variance)
    factor = cholesky(np.eye(count) + whitened @ whitened.T, lower=True)
    inverse = solve_triangular(factor, np.eye(count), lower=True)
    cross = prior.color(whitened.T)                                # C G^T Cd^-1/2
    del whitened
    return cross @ inverse.T, float(count - np.sum(inverse ** 2))


def marginals(prior, factor):
    """Posterior std of every state entry and the cellwise correlation between the two blocks.

    The prior blocks are independent, so the posterior covariance between ln Rh
    and the latent of one cell is the correction term alone.
    """
    cells = len(prior.mean) // 2
    prior_variance = prior.std ** 2
    variance = np.maximum(prior_variance - np.einsum('ij,ij->i', factor, factor), VARIANCE_FLOOR * prior_variance)
    std = np.sqrt(variance)
    covariance = -np.einsum('ij,ij->i', factor[:cells], factor[cells:])
    return std, np.clip(covariance / (std[:cells] * std[cells:]), -1., 1.)


def to_lattice(values, grid, weights):
    """Average solve-grid states (2 * cells, k) onto the carry lattice (2 * nx * nz, k).

    ``weights`` = (wx, wy, wz) from ``carry.lattice_weights``: wx and wz are
    (lattice cells, solve cells) matrices, wy a vector over the solve y cells
    (None in 2D).
    """
    wx, wy, wz = weights
    values = np.asarray(values, dtype=float)
    fields = values.reshape(2, *grid, -1)
    if wy is None:
        averaged = np.einsum('ax,cz,bxzk->back', wx, wz, fields, optimize=True)
    else:
        averaged = np.einsum('ax,y,cz,bxyzk->back', wx, wy, wz, fields, optimize=True)
    return averaged.reshape(-1, fields.shape[-1])


def lattice_prior_variance(prior, weights):
    """Prior variance of the lattice-cell averages of a separable prior, (2 * nx * nz,).

    A lattice cell averages the few solve cells it overlaps; its variance is the
    quadratic form of those weights (times the cellwise std) in the Kronecker
    correlation of that small box.
    """
    wx, wy, wz = weights
    three = wy is not None
    std = prior.std.reshape(2, *prior.grid)
    xs = [np.flatnonzero(row) for row in wx]
    zs = [np.flatnonzero(row) for row in wz]
    ys = np.flatnonzero(wy) if three else None
    variance = np.empty((2, len(xs), len(zs)))
    for block in (0, 1):
        correlations = prior.axis_correlations(block)
        rx, rz = correlations[0], correlations[-1]
        ry = correlations[1][np.ix_(ys, ys)] if three else None
        cz = [rz[np.ix_(z, z)] for z in zs]
        for a, x in enumerate(xs):
            cx = rx[np.ix_(x, x)]
            for c, z in enumerate(zs):
                if three:
                    scaled = (wx[a, x][:, None, None] * wy[ys][None, :, None] * wz[c, z][None, None, :]
                              * std[block][np.ix_(x, ys, z)])
                    variance[block, a, c] = np.einsum('xyz,xu,yv,zw,uvw->', scaled, cx, ry, cz[c], scaled,
                                                      optimize=True)
                else:
                    scaled = wx[a, x][:, None] * wz[c, z][None, :] * std[block][np.ix_(x, z)]
                    variance[block, a, c] = np.einsum('xz,xu,zw,uw->', scaled, cx, cz[c], scaled, optimize=True)
    return variance.reshape(-1)


def lattice_std(prior, weights, factor):
    """Posterior std of the lattice-cell averages (exact for the Gaussian of the stage)."""
    prior_variance = lattice_prior_variance(prior, weights)
    averaged = to_lattice(factor, prior.grid, weights)
    reduction = np.einsum('ij,ij->i', averaged, averaged)
    return np.sqrt(np.maximum(prior_variance - reduction, VARIANCE_FLOOR * prior_variance))


def resistivity_moments(model, std, correlation):
    """Mean and std of Rh and Rv [ohm m] of the cellwise Gaussian (ln Rh, latent = ln(Rv/Rh)).

    Both resistivities are lognormal: ln Rv = ln Rh + latent has variance
    s_h^2 + s_l^2 + 2 rho s_h s_l. Returns (rh_mean, rh_std, rv_mean, rv_std), one value per cell.
    """
    model, std = np.asarray(model, dtype=float), np.asarray(std, dtype=float)
    cells = len(model) // 2
    var_rh = std[:cells] ** 2
    var_rv = np.maximum(var_rh + std[cells:] ** 2 + 2. * correlation * std[:cells] * std[cells:], 0.)
    rh_mean = np.exp(model[:cells] + .5 * var_rh)
    rv_mean = np.exp(model[:cells] + model[cells:] + .5 * var_rv)
    return rh_mean, rh_mean * np.sqrt(np.expm1(var_rh)), rv_mean, rv_mean * np.sqrt(np.expm1(var_rv))
