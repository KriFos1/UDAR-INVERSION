import numpy as np
import pytest

from udarinv import laplace
from udarinv.carry import lattice_weights
from udarinv.grid import centres
from udarinv.prior import SeparableMatern


def problem(dimension, seed=0, channels=9):
    rng = np.random.default_rng(seed)
    edges = [np.array([0., 2., 4., 6., 9., 14.]), np.array([-6., -2., 0., 2., 6.]), np.array([0., 2., 4., 6., 8., 12.])]
    if dimension == 2:
        edges = [edges[0], edges[2]]
    axes = [centres(e) for e in edges]
    cells = int(np.prod([len(a) for a in axes]))
    ranges = [[10., 6., 8.], [16., 9., 5.]] if dimension == 3 else [[10., 8.], [16., 5.]]
    prior = SeparableMatern(axes, ranges, rng.normal(size=2 * cells), rng.uniform(.3, 1.2, 2 * cells))
    jacobian = rng.normal(size=(channels, 2 * cells)) * rng.uniform(.1, 2., (channels, 1))
    variance = rng.uniform(.5, 2., channels)
    covariance = prior.color(prior.color(np.eye(2 * cells), transpose=True))
    posterior = np.linalg.inv(np.linalg.inv(covariance) + jacobian.T @ (jacobian / variance[:, None]))
    return edges, prior, jacobian, variance, covariance, posterior


@pytest.mark.parametrize('dimension', [2, 3])
def test_cellwise_moments_equal_the_dense_laplace_covariance(dimension):
    _, prior, jacobian, variance, covariance, posterior = problem(dimension)
    cells = prior.cells
    factor, resolved = laplace.posterior_factor(prior, jacobian, variance)
    np.testing.assert_allclose(covariance - factor @ factor.T, posterior, atol=1e-9)
    std, correlation = laplace.marginals(prior, factor)
    np.testing.assert_allclose(std, np.sqrt(np.diag(posterior)), rtol=1e-9)
    cross = np.diag(posterior[:cells, cells:])
    np.testing.assert_allclose(correlation, cross / (std[:cells] * std[cells:]), atol=1e-9)
    whitened = prior.color(jacobian.T, transpose=True).T / np.sqrt(variance)[:, None]
    gram = whitened.T @ whitened
    assert resolved == pytest.approx(np.trace(gram @ np.linalg.inv(np.eye(len(gram)) + gram)), rel=1e-9)
    assert 0. < resolved < len(variance)


@pytest.mark.parametrize('dimension', [2, 3])
def test_lattice_moments_are_those_of_the_averaged_gaussian(dimension):
    edges, prior, jacobian, variance, covariance, posterior = problem(dimension, seed=4)
    lattice_edges = (np.array([1., 5., 9., 13.]), np.array([0., 4., 8.]))
    weights = lattice_weights(edges, lattice_edges, well_y=0., cell=4.)
    assert (weights[1] is None) == (dimension == 2)
    average = laplace.to_lattice(np.eye(2 * prior.cells), prior.grid, weights)       # (2 * lattice, state)
    assert average.shape == (2 * 3 * 2, 2 * prior.cells) and np.allclose(average.sum(axis=1), 1.)
    np.testing.assert_allclose(laplace.lattice_prior_variance(prior, weights),
                               np.diag(average @ covariance @ average.T), rtol=1e-9)
    factor, _ = laplace.posterior_factor(prior, jacobian, variance)
    np.testing.assert_allclose(laplace.lattice_std(prior, weights, factor),
                               np.sqrt(np.diag(average @ posterior @ average.T)), rtol=1e-8)
    model = np.random.default_rng(2).normal(size=2 * prior.cells)
    np.testing.assert_allclose(laplace.to_lattice(model[:, None], prior.grid, weights)[:, 0], average @ model)


def test_resistivity_moments_are_lognormal():
    rng = np.random.default_rng(5)
    model, std = np.array([1., .3]), np.array([.4, .25])
    for correlation in (0., -.6, .8):
        covariance = np.array([[std[0] ** 2, correlation * std.prod()], [correlation * std.prod(), std[1] ** 2]])
        draws = rng.multivariate_normal(model, covariance, size=400_000)
        rh, rv = np.exp(draws[:, 0]), np.exp(draws[:, 0] + draws[:, 1])
        moments = laplace.resistivity_moments(model, std, np.array([correlation]))
        for value, reference in zip(moments, (rh.mean(), rh.std(), rv.mean(), rv.std())):
            assert value[0] == pytest.approx(reference, rel=.02)
