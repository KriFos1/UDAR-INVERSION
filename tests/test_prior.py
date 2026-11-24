import numpy as np
import pytest

from udarinv.prior import LatticeMatern, SeparableMatern, axis_correlation


def dense_covariance(prior):
    return prior.color(prior.color(np.eye(len(prior.mean)), transpose=True))


@pytest.mark.parametrize('axes', [[np.array([0., 2., 4., 7., 12.]), np.arange(0., 8., 2.)],
                                  [np.array([0., 2., 4., 7.]), np.array([-3., 0., 3.]), np.arange(0., 6., 2.)]])
def test_separable_prior_is_the_kronecker_product_with_cellwise_std(axes):
    cells = int(np.prod([len(a) for a in axes]))
    rng = np.random.default_rng(1)
    std = rng.uniform(.3, 1.5, 2 * cells)
    ranges = [[10., 6., 8.][:len(axes)], [16., 9., 5.][:len(axes)]]
    prior = SeparableMatern(axes, ranges, rng.normal(size=2 * cells), std)
    covariance = dense_covariance(prior)
    for block in (0, 1):
        expected = np.ones((1, 1))
        for axis, length in zip(axes, ranges[block]):
            expected = np.kron(expected, axis_correlation(axis, length))
        rows = slice(block * cells, (block + 1) * cells)
        np.testing.assert_allclose(covariance[rows, rows], std[rows, None] * expected * std[None, rows], atol=1e-12)
    assert np.allclose(covariance[:cells, cells:], 0.)                      # independent blocks
    np.testing.assert_allclose(np.sqrt(np.diag(covariance)), std)
    vector = rng.normal(size=2 * cells)
    np.testing.assert_allclose(prior.whiten(prior.color(vector)), vector, atol=1e-10)
    matrix = rng.normal(size=(2 * cells, 3))
    np.testing.assert_allclose(prior.color(matrix)[:, 1], prior.color(matrix[:, 1]))
    np.testing.assert_allclose(vector @ prior.color(matrix[:, 0]), prior.color(vector, transpose=True) @ matrix[:, 0])


def test_axis_correlation_decays_with_the_range():
    centres = np.arange(0., 200., 2.)
    short, long = axis_correlation(centres, 20.), axis_correlation(centres, 80.)
    assert np.allclose(np.diag(short), 1.) and short[50, 60] < long[50, 60] < 1.
    assert .05 < long[50, 90] < .3                               # about 0.14 at the practical range (Matern 3/2)
    assert axis_correlation([3.], 10.).tolist() == [[1.]]
    with pytest.raises(ValueError):
        SeparableMatern([centres], [[10.], [10.]], np.zeros(200), np.zeros(200))


def test_lattice_field_has_the_requested_std_and_dense_conditional_moments():
    nx, nz = 7, 6
    prior = LatticeMatern((nx, nz), 5., [[20., 10.], [30., 8.]], [1., .5], [1.1, .4])
    assert prior.mean.shape == (2 * nx * nz,) and np.allclose(prior.std[:nx * nz], 1.1)
    rng = np.random.default_rng(3)
    entered = np.zeros((nx, nz), dtype=bool)
    entered[-2:] = True
    entered[:, :1] = True
    entered = entered.ravel()
    mean = prior.mean + rng.normal(scale=.3, size=prior.mean.shape)
    for std in (prior.std.copy(), prior.std * rng.uniform(.2, .9, prior.std.shape)):
        filled, variance = prior.entered_moments(mean, std, entered)
        for index, block in enumerate(prior.blocks):
            rows = slice(index * nx * nz, (index + 1) * nx * nz)
            covariance = np.linalg.inv(block.precision().toarray())
            np.testing.assert_allclose(np.sqrt(np.diag(covariance)), block.std, rtol=1e-9)
            kept = ~entered
            gain = covariance[np.ix_(entered, kept)] @ np.linalg.inv(covariance[np.ix_(kept, kept)])
            conditional = covariance[np.ix_(entered, entered)] - gain @ covariance[np.ix_(kept, entered)]
            correlation = covariance[np.ix_(kept, kept)] / np.outer(block.std[kept], block.std[kept])
            sigma = std[rows][kept, None] * correlation * std[rows][None, kept]
            np.testing.assert_allclose(filled[rows][entered],
                                       block.mean[entered] + gain @ (mean[rows][kept] - block.mean[kept]), atol=1e-9)
            np.testing.assert_allclose(variance[rows][entered], np.diag(gain @ sigma @ gain.T + conditional),
                                       rtol=1e-8)
            np.testing.assert_array_equal(filled[rows][kept], mean[rows][kept])
            np.testing.assert_allclose(variance[rows][kept], std[rows][kept] ** 2)
    # Kept cells at their prior spread: the entered cells come back at the prior variance.
    _, variance = prior.entered_moments(mean, prior.std, entered)
    np.testing.assert_allclose(variance, prior.std ** 2, rtol=1e-8)
    everything = prior.entered_moments(mean, prior.std, np.ones(nx * nz, dtype=bool))
    np.testing.assert_array_equal(everything[0], prior.mean)
