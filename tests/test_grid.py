import numpy as np
import pytest

from udarinv.grid import CarryLattice, SolveGrid, centres, graded_edges, overlap_weights, parse_grade


def test_default_window_is_the_benchmark_window():
    edges = SolveGrid().edges((1000., 0., 17012.3), 17000., 3)
    assert [len(e) - 1 for e in edges] == [65, 89, 120]
    x, y, z = edges
    assert x[0] == pytest.approx(680.) and x[-1] == pytest.approx(1320.)
    assert y[0] == pytest.approx(-300.) and y[-1] == pytest.approx(300.)
    # A cell is centred on the bit in x and on the well plane in y; 2 ft cells near the tool.
    assert np.min(np.abs(centres(x) - 1000.)) < 1e-9 and np.min(np.abs(centres(y))) < 1e-9
    assert np.allclose(np.diff(x)[np.abs(centres(x) - 1000.) < 39.], 2.)
    assert np.allclose(np.diff(z)[np.abs(centres(z) - 17012.) < 99.], 2.)
    two = SolveGrid().edges((1000., 0., 17012.3), 17000., 2)
    assert len(two) == 2 and np.array_equal(two[0], x) and np.array_equal(two[1], z)


def test_z_edges_stay_on_the_tvd_lattice():
    grid = SolveGrid()
    first = grid.edges((0., 0., 17000.), 17000., 3)[2]
    for bit_z in (17000.4, 17000.99, 17001.01, 17003.7, 16994.2):
        z = grid.edges((0., 0., bit_z), 17000., 3)[2]
        inner = z[np.abs(z - bit_z) < 90.]
        assert np.allclose(np.mod(inner - 17000., 2.), 0.)            # fine edges on the 2 ft lattice
        assert abs(.5 * (z[0] + z[-1]) - bit_z) <= 1. + 1e-9           # window within one cell of the bit
    assert np.allclose(np.mod(first[np.abs(first - 17000.) < 90.], 2.), 0.)


def test_grades():
    assert parse_grade('100:2:1.3:20') == (((2., 100.),), 1.3, 20.)
    assert parse_grade('2x20+4x130+1.5:40') == (((2., 20.), (4., 130.)), 1.5, 40.)
    uniform = graded_edges(50., '5', .5)
    assert np.allclose(np.diff(uniform)[1:-1], 5.) and uniform[0] == -50. and uniform[-1] == 50.
    tiered = graded_edges(300., '2x20+4x130+1.5:40', .5)
    widths = np.diff(tiered)
    # The outermost cell absorbs a sliver, so it may exceed the largest growth cell (by less than half of it).
    assert np.allclose(tiered, -tiered[::-1]) and widths.min() == pytest.approx(2.) and widths.max() < 60.
    assert np.allclose(widths[np.abs(.5 * (tiered[1:] + tiered[:-1])) < 19.], 2.)
    with pytest.raises(ValueError):
        graded_edges(-1., '5', 0.)


def test_overlap_weights_average_and_extend():
    source = np.array([1., 3., 5., 7.])                        # cells [0,2], [2,4], [4,6], [6,8]
    weights = overlap_weights(source, np.array([0., 4., 8., 20.]))
    assert np.allclose(weights.sum(axis=1), 1.)
    assert np.allclose(weights[0], [.5, .5, 0., 0.]) and np.allclose(weights[1], [0., 0., .5, .5])
    assert np.allclose(weights[2], [0., 0., 0., 1.])           # beyond the source: nearest cell
    assert overlap_weights(np.array([2.]), np.array([0., 1., 5.])).tolist() == [[1.], [1.]]


def test_carry_lattice_moves_by_whole_cells():
    lattice = CarryLattice(33, 80, 5.)
    corner, movement = lattice.update(1000., 17000.)
    assert movement == (0, 0) and corner == (1000. - 16.5 * 5., 17000. - 40.5 * 5.)
    x, z = lattice.edges()
    assert len(x) == 34 and len(z) == 81 and x[16] < 1000. < x[17]
    assert lattice.update(1002.4, 17000.4)[1] == (0, 0)
    corner, movement = lattice.update(1012.6, 16994.)
    assert movement == (3, -1) and corner[0] == pytest.approx(1000. - 16.5 * 5. + 15.)
    restored = CarryLattice(33, 80, 5., **lattice.state())
    assert restored.update(1012.6, 16994.) == (corner, (0, 0))
