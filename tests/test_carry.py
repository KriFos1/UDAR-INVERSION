import numpy as np
import pytest

from udarinv import carry
from udarinv.grid import CarryLattice, SolveGrid, centres
from udarinv.prior import LatticeMatern


def test_entered_cells_and_reindex():
    assert carry.entered_cells(5, 4, (0, 0)).sum() == 0
    entered = carry.entered_cells(5, 4, (2, -1))
    assert entered[3:].all() and entered[:, 0].all() and entered.sum() == 2 * 4 + 5 - 2
    values = np.arange(2 * 5 * 4, dtype=float)
    moved = carry.reindex(values, 5, 4, (1, 0)).reshape(2, 5, 4)
    np.testing.assert_array_equal(moved[:, :4], values.reshape(2, 5, 4)[:, 1:])
    np.testing.assert_array_equal(moved[:, 4], values.reshape(2, 5, 4)[:, 4])        # edge value repeated


def test_carry_moments_relaxes_kept_cells_and_fills_entered_ones():
    nx, nz = 9, 8
    background = LatticeMatern((nx, nz), 5., [[20., 10.], [30., 8.]], [1., .5], [1.1, .4])
    rng = np.random.default_rng(0)
    mean = background.mean + rng.normal(scale=.5, size=background.mean.shape)
    std = background.std * rng.uniform(.2, .8, background.std.shape)
    same_mean, same_std, entered = carry.carry_moments(background, mean, std, (0, 0), 1.)
    np.testing.assert_allclose(same_mean, mean)
    np.testing.assert_allclose(same_std, std)
    assert entered == 0
    rho = np.exp(-5. / 50.)
    moved_mean, moved_std, entered = carry.carry_moments(background, mean, std, (1, 0), rho)
    assert entered == nz
    kept = ~carry.entered_cells(nx, nz, (1, 0)).ravel()
    for block in (0, 1):
        rows = slice(block * nx * nz, (block + 1) * nx * nz)
        shifted_mean = carry.reindex(mean, nx, nz, (1, 0))[rows][kept]
        shifted_std = carry.reindex(std, nx, nz, (1, 0))[rows][kept]
        mu, sigma = background.mean[rows][kept], background.std[rows][kept]
        np.testing.assert_allclose(moved_mean[rows][kept], mu + rho * (shifted_mean - mu))
        np.testing.assert_allclose(moved_std[rows][kept] ** 2, rho ** 2 * shifted_std ** 2 + (1 - rho ** 2) * sigma ** 2)
        new = moved_std[rows][~kept]
        assert np.all(new > 0) and np.all(new <= background.std[rows][~kept] + 1e-9)       # dense check: test_prior
    # Without memory the whole lattice returns to the background.
    gone_mean, gone_std, _ = carry.carry_moments(background, mean, std, (1, 0), 0.)
    np.testing.assert_allclose(gone_mean[:nx * nz][kept], background.mean[:nx * nz][kept])
    np.testing.assert_allclose(gone_std, background.std, rtol=1e-8)
    with pytest.raises(ValueError):
        carry.carry_moments(background, mean, std, (1, 0), 1.5)


@pytest.mark.parametrize('dimension', [2, 3])
def test_lift_and_average_back_are_consistent(dimension):
    lattice = CarryLattice(33, 80, 5.)
    lattice.update(1000., 17000.)
    edges = SolveGrid().edges((1000., 3., 17000.), 17000., dimension)
    rng = np.random.default_rng(1)
    mean = rng.normal(size=(2, 33, 80))
    std = rng.uniform(.3, .9, (2, 33, 80))
    background_std = (1.1, .5)
    lifted_mean, lifted_std = carry.lifted_moments(lattice.edges(), mean, std, background_std, edges, 100., well_y=3.)
    grid = tuple(len(e) - 1 for e in edges)
    assert lifted_mean.shape == (2, *grid) and lifted_std.shape == (2, *grid)
    assert np.all(lifted_std[0] <= 1.1 + 1e-12) and np.all(lifted_std >= std.min() - 1e-12)
    x, z = centres(edges[0]), centres(edges[-1])
    if dimension == 3:
        y = centres(edges[1])
        plane, far = int(np.argmin(np.abs(y - 3.))), int(np.argmax(np.abs(y - 3.)))
        np.testing.assert_array_equal(lifted_mean[:, :, plane], lifted_mean[:, :, far])      # mean extended across
        # 300 ft off the plane, 100 ft scale: 1 - exp(-3) = 95 % of the way to the background variance
        assert lifted_std[0][:, far].min() > .95 * 1.1
        inside = lifted_std[0][:, plane]
    else:
        inside = lifted_std[0]
    lx, lz = lattice.edges()
    core = np.ix_((x > lx[0] + 5) & (x < lx[-1] - 5), (z > lz[0] + 5) & (z < lz[-1] - 5))
    assert inside[core].max() <= .9 + 1e-9                                                  # no relaxation inside
    outside = inside[np.abs(x - 1000.) > 300.]
    assert outside.min() > inside[core].mean()
    # Averaging the lifted mean back reproduces a lattice field that is constant per block.
    constant = np.stack([np.full((33, 80), 2.), np.full((33, 80), -.5)])
    lifted, _ = carry.lifted_moments(lattice.edges(), constant, std, background_std, edges, 100., well_y=3.)
    weights = carry.lattice_weights(edges, lattice.edges(), well_y=3.)
    from udarinv.laplace import to_lattice
    back = to_lattice(lifted.reshape(-1, 1), grid, weights)[:, 0].reshape(2, 33, 80)
    np.testing.assert_allclose(back, constant, atol=1e-12)


def test_previous_map_lookup_and_off_plane_anomaly():
    grid = SolveGrid()
    old_edges = grid.edges((1000., 0., 17000.), 17000., 3)
    new_edges = grid.edges((1005., 0., 17000.4), 17000., 3)
    old_centres = [centres(e) for e in old_edges]
    shape = tuple(len(c) for c in old_centres)
    x, y, z = np.meshgrid(*old_centres, indexing='ij')
    previous = np.stack([.01 * x + .02 * y, np.full(shape, .3) + .001 * (z - 17000.)])
    fallback = np.full(2 * int(np.prod(shape)), 9.)
    state = carry.carried_state(previous, old_centres, new_edges, fallback).reshape(2, *shape)
    nx, ny, nz = (centres(e) for e in new_edges)
    inside = nx <= old_centres[0][-1] + 20.
    np.testing.assert_allclose(state[0][inside][:, :, 60], (.01 * nx[inside, None] + .02 * ny[None, :]), atol=.25)
    assert np.all(state[0][nx > old_edges[0][-1] + 25.] == 9.)                       # new ground: the fallback
    anomaly = carry.carried_anomaly(previous, old_centres, new_edges, .5, previous_well_y=0.)
    plane = int(np.argmin(np.abs(ny)))
    assert np.allclose(anomaly[:, :, plane], 0.) and np.allclose(anomaly[1], 0.)     # no off-plane latent structure
    np.testing.assert_allclose(anomaly[0][inside][:, :, 10], .5 * .02 * np.broadcast_to(ny, (inside.sum(), len(ny))),
                               atol=1e-9)
    assert carry.carried_anomaly(previous[:, :, 0], [old_centres[0], old_centres[2]],
                                 [new_edges[0], new_edges[2]], .5) == 0.
    two = carry.carried_state(previous[:, :, plane], [old_centres[0], old_centres[2]],
                              [new_edges[0], new_edges[2]], np.full(2 * shape[0] * shape[2], 9.))
    assert two.shape == (2 * shape[0] * shape[2],)
